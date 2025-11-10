from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from tqdm import tqdm
from pydantic import BaseModel, create_model
from typing import Any as TypingAny, Optional as TypingOptional, List as TypingList, Dict as TypingDict
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


# =========================
# Config
# =========================
@dataclass
class TeacherConfig:
    # Core LLM params
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.0

    # IO
    input_json: str = "training_unlabeled.json"    # array or JSONL
    output_json: str = "training_labeled.json"
    log_file: str = "teacher.log"

    # Parallelism
    num_workers: int = 4

    # Response-format (json_schema) params
    schema_name: str = "response_schema"
    schema_strict: bool = True

    # New: path to the JSON example that defines the desired output structure
    schema_example_path: str = "structure_example.json"

    # How to store teacher output in the final dataset
    teacher_output_mode: str = "dict"       # "dict" or "field"
    teacher_output_field: Optional[str] = None

    @staticmethod
    def from_yaml(path: str) -> "TeacherConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "schema_example_path" not in data or not data["schema_example_path"]:
            raise ValueError("teacher_config.yaml must include 'schema_example_path' pointing to a JSON example.")

        return TeacherConfig(
            model=data["model"],
            api_key=data.get("api_key", ""),
            base_url=data.get("base_url", "https://openrouter.ai/api/v1"),
            temperature=float(data.get("temperature", 0.0)),
            input_json=data.get("input_json", "training_unlabeled.json"),
            output_json=data.get("output_json", "training_labeled.json"),
            log_file=data.get("log_file", "teacher.log"),
            num_workers=int(data.get("num_workers", 4)),
            schema_name=data.get("schema_name", "response_schema"),
            schema_strict=bool(data.get("schema_strict", True)),
            schema_example_path=data["schema_example_path"],
            teacher_output_mode=data.get("teacher_output_mode", "dict"),
            teacher_output_field=data.get("teacher_output_field"),
        )


# =========================
# Utilities
# =========================
def _strip_json_comments(text: str) -> str:
    """
    Allows 'commented JSON' by removing // line comments and /* ... */ block comments.
    Simple heuristic that ignores common cases; for complex strings with // or /* inside quotes,
    prefer plain JSON instead. Good enough for config/structure examples.
    """
    # remove /* ... */ blocks
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    # remove // ... end of line
    text = re.sub(r"//.*?$", "", text, flags=re.M)
    return text


def _load_json_allowing_comments(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    cleaned = _strip_json_comments(raw).strip()
    return json.loads(cleaned)


# =========================
# Dynamic Pydantic model builder from example JSON
# =========================
class _ModelRegistry:
    """Keeps named submodels to avoid duplicating structures."""
    def __init__(self):
        self.models: Dict[str, Type[BaseModel]] = {}

    def get_or_create(self, name: str, fields: Dict[str, Tuple[Any, Any]]) -> Type[BaseModel]:
        if name in self.models:
            return self.models[name]
        model = create_model(name, **fields)  # type: ignore[arg-type]
        self.models[name] = model
        return model


def _infer_type_from_value(val: Any, submodel_name: str, registry: _ModelRegistry) -> Tuple[Any, Any]:
    """
    Returns a (type_annotation, default_or_required) tuple for Pydantic create_model.
    - Supports: str, int, float, bool, dict, list, None
    - Dicts become nested BaseModels (recursively inferred)
    - Lists become List[T] inferred from first non-null element; fallback List[Any]
    - None makes the field Optional with default None
    """
    if val is None:
        return (TypingOptional[TypingAny], None)

    if isinstance(val, bool):
        return (bool, ...)
    if isinstance(val, int) and not isinstance(val, bool):
        return (int, ...)
    if isinstance(val, float):
        return (float, ...)
    if isinstance(val, str):
        return (str, ...)
    if isinstance(val, dict):
        # Build submodel
        fields: Dict[str, Tuple[Any, Any]] = {}
        for k, v in val.items():
            t, d = _infer_type_from_value(v, f"{submodel_name}_{k.capitalize()}", registry)
            fields[k] = (t, d)
        sub = registry.get_or_create(submodel_name, fields)
        return (sub, ...)
    if isinstance(val, list):
        # Infer element type from first non-null element
        elem_type: Any = TypingAny
        default_required: Any = ...
        for el in val:
            if el is not None:
                elem_type, default_required = _infer_type_from_value(el, f"{submodel_name}Item", registry)
                # If element type is a BaseModel class or simple type, extract raw py type
                # create List[elem_pytype]
                break
        # If list is empty or only nulls, use List[Any]
        return (TypingList[elem_type if elem_type is not None else TypingAny], ...)

    # Fallback
    return (TypingAny, ...)


def build_pydantic_model_from_example(root_name: str, example: Dict[str, Any]) -> Type[BaseModel]:
    """
    Given a root object example (dict), build a Pydantic model class dynamically.
    """
    if not isinstance(example, dict):
        raise ValueError("The structure example must be a JSON object (a dict at the root).")

    registry = _ModelRegistry()
    fields: Dict[str, Tuple[Any, Any]] = {}

    for key, value in example.items():
        t, d = _infer_type_from_value(value, f"{root_name}_{key.capitalize()}", registry)
        fields[key] = (t, d)

    Model = registry.get_or_create(root_name, fields)
    return Model


# =========================
# LLMExecutor (structured JSON via json_schema)
# =========================
class LLMExecutor:
    """
    Forces JSON output via response_format=json_schema; validates with the provided Pydantic model.
    Also appends the structure example at the end of the system prompt to reinforce formatting.
    """

    def __init__(self):
        self.model_name = "mistralai/mistral-nemo"
        self.api_key: Optional[str] = None
        self.base_url: str = "https://openrouter.ai/api/v1"
        self.system_prompt: Optional[str] = None
        self.user_prompt: Optional[str] = None
        self.schema_class: Optional[Type[BaseModel]] = None
        self.schema_name = "response_schema"
        self.strict_mode = True
        self.temperature: float = 0.0
        self.structure_example: Optional[Dict[str, Any]] = None  # injected in system prompt

    def set_model(self, model_name: str):
        self.model_name = model_name

    def set_api_key(self, api_key: str):
        self.api_key = api_key

    def set_base_url(self, base_url: str):
        self.base_url = base_url

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def set_user_prompt(self, prompt: str):
        self.user_prompt = prompt

    def set_schema(self, schema_class: Type[BaseModel], name: str = "response_schema", strict: bool = True):
        self.schema_class = schema_class
        self.schema_name = name
        self.strict_mode = strict

    def set_temperature(self, temperature: float):
        self.temperature = temperature


    def set_structure_example(self, structure_example: Dict[str, Any]):
        self.structure_example = structure_example

    def _compose_system_prompt(self) -> Optional[str]:
        base = self.system_prompt or ""
        if self.structure_example is not None:
            appended = (
                "\n\nYou MUST respond strictly as a JSON object matching the structure below. "
                "Do not include any prose outside JSON. Use the same keys and types. "
                "Fill values appropriately.\n\n"
                "### REQUIRED JSON OUTPUT STRUCTURE (EXAMPLE)\n"
                + json.dumps(self.structure_example, ensure_ascii=False, indent=2)
            )
            return (base + appended) if base else appended
        return base if base else None

    def run(self) -> BaseModel:
        if not self.api_key:
            raise ValueError("API key not configured.")
        if not self.schema_class:
            raise ValueError("Validation Pydantic class not defined.")
        if not self.user_prompt:
            raise ValueError("User prompt not defined.")

        json_schema = self.schema_class.model_json_schema()
        # Strict mode: refuse unknown keys
        json_schema.setdefault("additionalProperties", False)

        llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            model_kwargs={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.schema_name,
                        "strict": self.strict_mode,
                        "schema": json_schema,
                    },
                }
            },
        )

        messages = []
        sys_msg = self._compose_system_prompt()
        if sys_msg:
            messages.append(SystemMessage(content=sys_msg))
        messages.append(HumanMessage(content=self.user_prompt))

        try:
            response = llm.invoke(messages)
            parsed_json = json.loads(response.content)
            return self.schema_class(**parsed_json)
        except Exception as e:
            raise ValueError(f"Failed to parse/validate model response: {e}")


# =========================
# Teacher
# =========================
class Teacher:
    """
    Input items:  { "system_prompt": str, "user_prompt": str }
    Output items: { "system_prompt": str, "user_prompt": str, "teacher_output": <dict|field> }
    """

    def __init__(self, cfg: TeacherConfig):
        self.cfg = cfg
        self._setup_logging()
        # Load the example structure JSON (allows comments)
        self.structure_example: Dict[str, Any] = _load_json_allowing_comments(self.cfg.schema_example_path)
        # Build a Pydantic model from this example
        self.schema_model = build_pydantic_model_from_example("TeacherOutput", self.structure_example)
        logging.info(
            "Teacher initialized | model=%s base_url=%s temp=%s workers=%d",
            cfg.model, cfg.base_url, cfg.temperature, cfg.num_workers
        )

    # ---------- public ----------
    def run(self) -> None:
        items = self._load_input(self.cfg.input_json)
        logging.info("Loaded %d items from %s", len(items), self.cfg.input_json)

        results: List[Dict[str, Any]] = [None] * len(items)  # type: ignore
        errors = 0

        with ThreadPoolExecutor(max_workers=self.cfg.num_workers) as pool:
            futures = {
                pool.submit(self._label_one, i, item): i
                for i, item in enumerate(items)
            }

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Labeling"):
                idx = futures[fut]
                try:
                    labeled = fut.result()
                    results[idx] = labeled
                except Exception as e:
                    errors += 1
                    results[idx] = {
                        **items[idx],
                        "teacher_output": None,
                        "error": f"{type(e).__name__}: {e}",
                    }
                    logging.exception("Error labeling index=%d", idx)

        self._save_output(results, self.cfg.output_json)
        logging.info("Saved %d items to %s (errors=%d)", len(results), self.cfg.output_json, errors)

    # ---------- internals ----------
    def _label_one(self, idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
        sys_prompt = item.get("system_prompt")
        usr_prompt = item.get("user_prompt")

        # system_prompt pode ser None ou str
        if sys_prompt is not None and not isinstance(sys_prompt, str):
            raise ValueError(f"Invalid record at index={idx}. 'system_prompt' must be a string or null.")

        # user_prompt pode ser str OU dict {"document_id":..., "texto":...}
        if isinstance(usr_prompt, dict):
            # (opcional) validação leve da estrutura esperada
            if "document_id" not in usr_prompt or "texto" not in usr_prompt:
                raise ValueError(
                    f"Invalid record at index={idx}. 'user_prompt' dict must have 'document_id' and 'texto' keys."
                )
            # Serialize dict -> string (o conteúdo enviado à LLM será exatamente esse JSON)
            usr_prompt_str = json.dumps(usr_prompt, ensure_ascii=False)
        elif isinstance(usr_prompt, str):
            usr_prompt_str = usr_prompt
        else:
            raise ValueError(
                f"Invalid record at index={idx}. 'user_prompt' must be a string or a dict with 'document_id' and 'texto'."
            )

        exec_ = LLMExecutor()
        exec_.set_model(self.cfg.model)
        exec_.set_api_key(self._resolve_api_key(self.cfg.api_key))
        exec_.set_base_url(self.cfg.base_url)
        if sys_prompt:
            exec_.set_system_prompt(sys_prompt)
        exec_.set_user_prompt(usr_prompt_str)

        exec_.set_schema(self.schema_model, name=self.cfg.schema_name, strict=self.cfg.schema_strict)
        exec_.set_temperature(self.cfg.temperature)

        # Injeta o exemplo de estrutura no final do system prompt
        exec_.set_structure_example(self.structure_example)

        last_err: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                model_obj = exec_.run()  # -> Pydantic object
                payload = model_obj.model_dump()

                if self.cfg.teacher_output_mode == "field":
                    field = self.cfg.teacher_output_field
                    if not field:
                        raise ValueError("teacher_output_mode='field' requires 'teacher_output_field' in config.")
                    if field not in payload:
                        raise ValueError(f"Field '{field}' not found in model output.")
                    return {**item, "teacher_output": payload[field]}
                else:
                    return {**item, "teacher_output": payload}
            except Exception as e:
                last_err = e
                logging.warning("Attempt %d/3 failed idx=%d (%s). Retrying...", attempt, idx, type(e).__name__)

        assert last_err is not None
        raise last_err


    @staticmethod
    def _resolve_api_key(api_key: str) -> str:
        if api_key:
            return api_key
        env = os.environ.get("OPENAI_API_KEY", "")
        if env:
            return env
        raise ValueError("Missing API key. Provide in teacher_config.yaml or set OPENAI_API_KEY.")

    @staticmethod
    def _setup_logging() -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

    @staticmethod
    def _load_input(path: str) -> List[Dict[str, Any]]:
        """
        Accepts:
          1) JSON array of objects
          2) JSON Lines (JSONL)
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text.startswith("["):
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("Input JSON must be a list.")
            return data
        items: List[Dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("Each JSONL line must be an object.")
            items.append(obj)
        return items

    @staticmethod
    def _save_output(items: List[Dict[str, Any]], path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)


# =========================
# Entrypoint (CLI)
# =========================
import argparse

def main():
    parser = argparse.ArgumentParser(description="Teacher labeling runner (JSON-schema constrained).")
    parser.add_argument(
        "--config",
        type=str,
        default="teacher_config.yaml",
        help="Path to teacher_config.yaml (default: teacher_config.yaml)",
    )
    args = parser.parse_args()

    # Load config
    cfg = TeacherConfig.from_yaml(args.config)

    # Optional: attach file logger using the path from config
    root = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        fh = logging.FileHandler(cfg.log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
        root.addHandler(fh)

    # Run teacher
    teacher = Teacher(cfg)
    teacher.run()


if __name__ == "__main__":
    main()

