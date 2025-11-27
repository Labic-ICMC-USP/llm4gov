import os
import csv
import json
import logging
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from tqdm import tqdm
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class ClassItem(BaseModel):
    """
    One class that is highly relevant to the document.
    """
    class_id: str
    justification: str


class LLMClassificationResponse(BaseModel):
    """
    Expected response from the LLM for a SINGLE document.
    """
    relevant_classes: List[ClassItem]


class ICLClassifier:
    """
    ICLClassifier (In-Context Learning Text Classification).

    This class:

    - Loads a YAML configuration file.
    - Reads a CSV file with at least two columns: ID and TEXT.
    - Sends parallel requests to a LLM using an in-context learning prompt.
    - Returns a list of JSON-like dictionaries, one per document, in the form:

        {
            "doc_id": "<original_document_id>",
            "relevant_classes": [
            {"class_id": "...", "justification": "..."},
            ...
            ]
        }
    """

    def __init__(self, config_path: str):
        logger.info("Initializing ICLClassifier", extra={"config_path": config_path})
        self.config = self._load_config(config_path)

        # Model / endpoint configuration
        model_cfg: Dict[str, Any] = self.config.get("model", {})
        self.model_name: str = model_cfg.get("name", "mistralai/mistral-nemo")
        self.base_url: str = model_cfg.get("base_url", "https://openrouter.ai/api/v1")
        self.temperature: float = float(model_cfg.get("temperature", 0.0))

        # API key is read directly from YAML (no environment variable).
        self.api_key: str | None = model_cfg.get("api_key")
        if not self.api_key:
            logger.error(
                "API key not provided in YAML",
                extra={"yaml_key": "model.api_key"},
            )
            raise ValueError(
                "API key not found. Please set 'model.api_key' in your config.yaml."
            )

        logger.info(
            "Model configuration loaded",
            extra={
                "model_name": self.model_name,
                "base_url": self.base_url,
                "temperature": self.temperature,
            },
        )

        # Classification configuration
        cls_cfg: Dict[str, Any] = self.config.get("classification", {})
        self.system_prompt_template: str = cls_cfg.get(
            "system_prompt",
            (
                "You are an expert text classifier.\n"
                "You will receive a text and a list of possible classes.\n"
                "Return only the classes that are highly relevant to the text."
            ),
        )

        self.classes: List[Dict[str, str]] = cls_cfg.get("classes", [])
        if not self.classes:
            logger.error("No classes found in configuration under classification.classes")
            raise ValueError("No classes defined in classification.classes in the YAML file.")

        self.csv_input_path: str = cls_cfg.get("csv_input_path", "input.csv")
        self.id_column: str = cls_cfg.get("id_column", "ID")
        self.text_column: str = cls_cfg.get("text_column", "TEXT")

        self.num_threads: int = int(cls_cfg.get("num_threads", 4))
        self.max_tries: int = int(cls_cfg.get("max_tries", 3))

        # Output path for JSONL results (with a safe default)
        self.output_path: str = cls_cfg.get("output_path", "output.jsonl")

        logger.info(
            "Classification configuration loaded",
            extra={
                "csv_input_path": self.csv_input_path,
                "id_column": self.id_column,
                "text_column": self.text_column,
                "num_threads": self.num_threads,
                "max_tries": self.max_tries,
                "num_classes": len(self.classes),
                "output_path": self.output_path,
            },
        )

        # Build the textual description of classes for the prompt
        self.classes_description: str = self._build_classes_description()

        # Build the final system prompt (with or without the placeholder)
        self.system_prompt: str = self._build_system_prompt()

        # Instantiate the LLM client once, using JSON schema in strict mode
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            model_kwargs={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "classification_response",
                        "strict": True,
                        "schema": LLMClassificationResponse.model_json_schema(),
                    },
                }
            },
        )

        self.system_message = SystemMessage(content=self.system_prompt)
        logger.info("LLM client initialized successfully")

    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        """
        Load YAML configuration from the given path.
        """
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.debug("Configuration file loaded", extra={"path": path})
        return config

    def _build_classes_description(self) -> str:
        """
        Build a plain-text description of the classes defined in the YAML file.
        """
        lines = []
        for c in self.classes:
            cid = c.get("id", "").strip()
            desc = c.get("description", "").strip()
            lines.append(f"- {cid}: {desc}")

        description = "\n".join(lines)
        logger.debug(
            "Classes description built",
            extra={"num_classes": len(self.classes)},
        )
        return description

    def _build_system_prompt(self) -> str:
        """
        Construct the final system prompt.
        """
        if "{CLASSES_DESCRIPTION}" in self.system_prompt_template:
            prompt = self.system_prompt_template.replace(
                "{CLASSES_DESCRIPTION}", self.classes_description
            )
        else:
            prompt = (
                self.system_prompt_template
                + "\n\nPredefined classes:\n"
                + self.classes_description
            )

        logger.debug("System prompt built")
        return prompt

    def _build_user_prompt(self, text: str) -> str:
        """
        Build the user prompt for a single document.
        """
        return f"""
    You must decide which of the predefined classes are highly relevant for the following text.

    Predefined classes (class_id : description):
    {self.classes_description}

    Instructions:
    - Analyze the text carefully.
    - Return ONLY the classes that are highly relevant to the text.
    - You MAY assign more than one class if they are all highly relevant.
    - For each returned class, provide a short justification.
    - Use only class_id values that appear in the predefined list.
    - If no class is clearly relevant, return an empty list.

    Text:
    \"\"\"{text}\"\"\"

    Return a JSON object with the following structure:
    {{
    "relevant_classes": [
        {{
        "class_id": "<one_of_the_defined_class_ids>",
        "justification": "<short explanation>"
        }}
    ]
    }}
    """


    def _classify_single(
        self,
        idx: int,
        doc_id: str,
        text: str,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Classify a single document, with retry logic.
        """
        for attempt in range(1, self.max_tries + 1):
            try:
                user_prompt = self._build_user_prompt(text)
                logger.debug(
                    "Sending request to LLM",
                    extra={"doc_id": doc_id, "attempt": attempt},
                )

                response = self.llm.invoke(
                    [self.system_message, HumanMessage(content=user_prompt)]
                )

                parsed = json.loads(response.content)
                validated = LLMClassificationResponse(**parsed)

                result = {
                    "doc_id": doc_id,
                    "relevant_classes": [
                        {"class_id": c.class_id, "justification": c.justification}
                        for c in validated.relevant_classes
                    ],
                }

                logger.info(
                    "Document classified successfully",
                    extra={
                        "doc_id": doc_id,
                        "attempt": attempt,
                        "num_relevant_classes": len(validated.relevant_classes),
                    },
                )
                return idx, result

            except Exception as e:
                logger.warning(
                    "Classification attempt failed",
                    extra={
                        "doc_id": doc_id,
                        "attempt": attempt,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )

                if attempt == self.max_tries:
                    logger.error(
                        "Max attempts exceeded for document",
                        extra={"doc_id": doc_id, "max_tries": self.max_tries},
                    )
                    result = {
                        "doc_id": doc_id,
                        "relevant_classes": [],
                    }
                    return idx, result

    def _read_csv_docs(self) -> List[Tuple[int, str, str]]:
        """
        Read documents from the CSV input file.
        """
        docs: List[Tuple[int, str, str]] = []

        logger.info("Reading input CSV", extra={"csv_input_path": self.csv_input_path})
        with open(self.csv_input_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if self.id_column not in reader.fieldnames or self.text_column not in reader.fieldnames:
                logger.error(
                    "CSV does not contain required columns",
                    extra={
                        "required_id_column": self.id_column,
                        "required_text_column": self.text_column,
                        "fieldnames": reader.fieldnames,
                    },
                )
                raise ValueError(
                    f"CSV must contain columns '{self.id_column}' and '{self.text_column}'. "
                    f"Found: {reader.fieldnames}"
                )

            for idx, row in enumerate(reader):
                doc_id = str(row[self.id_column]).strip()
                text = str(row[self.text_column]).strip()
                if not doc_id or not text:
                    logger.debug(
                        "Skipping row with empty ID or TEXT",
                        extra={"row_index": idx, "doc_id": doc_id},
                    )
                    continue
                docs.append((idx, doc_id, text))

        logger.info("CSV reading completed", extra={"num_documents": len(docs)})
        return docs

    def run(self) -> List[Dict[str, Any]]:
        """
        Run classification for ALL documents in the input CSV.
        """
        docs = self._read_csv_docs()
        if not docs:
            logger.warning("No documents found to classify")
            return []

        logger.info("Starting parallel classification", extra={"num_documents": len(docs)})
        results_map: Dict[int, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self._classify_single, idx, doc_id, text): (idx, doc_id)
                for idx, doc_id, text in docs
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Classifying documents",
            ):
                idx, doc_id = futures[future]
                try:
                    idx_result, result = future.result()
                    results_map[idx_result] = result
                except Exception as e:
                    logger.error(
                        "Unexpected error when processing future",
                        extra={
                            "doc_id": doc_id,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        },
                    )
                    results_map[idx] = {
                        "doc_id": doc_id,
                        "relevant_classes": [],
                    }

        final_results: List[Dict[str, Any]] = [
            results_map[idx] for idx, _, _ in sorted(docs, key=lambda x: x[0])
        ]

        logger.info(
            "Classification completed",
            extra={"num_documents": len(final_results)},
        )
        return final_results

    def save_results(self, results: List[Dict[str, Any]], output_path: str | None = None) -> None:
        """
        Save classification results to a JSONL file.
        """
        path = output_path or self.output_path
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        logger.info("Saving results to file", extra={"output_path": path})
        with open(path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Results saved successfully", extra={"output_path": path})
