from __future__ import annotations
import json
import re
import time
import logging
from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional, Protocol, Any, Tuple, runtime_checkable
from tqdm import tqdm

# ------------------------------
# Logging (uses structlog if available; falls back to logging)
# ------------------------------
try:
    import structlog  # type: ignore

    _STRUCTLOG = True
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        cache_logger_on_first_use=True,
    )
    log = structlog.get_logger("TextAnonymizer")
except Exception:  # pragma: no cover
    _STRUCTLOG = False
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("TextAnonymizer")


# ------------------------------
# Pydantic schemas (structured I/O)
# ------------------------------
from pydantic import BaseModel, Field, ValidationError, field_validator


class AnonymizedSentence(BaseModel):
    original_index: int = Field(..., ge=0)
    text: str = Field(..., min_length=0)


class AnonymizedBatchOutput(BaseModel):
    sentences: List[AnonymizedSentence]
    warnings: Optional[List[str]] = None
    errors: Optional[List[str]] = None

    @field_validator("sentences")
    @classmethod
    def _unique_indices(cls, v: List[AnonymizedSentence]) -> List[AnonymizedSentence]:
        idxs = [s.original_index for s in v]
        if len(idxs) != len(set(idxs)):
            raise ValueError("Duplicated original_index in sentences.")
        return v



class BatchMeta(BaseModel):
    batch_id: int
    sent_count: int
    llm_latency_s: float
    prompt_chars: int
    returned_count: int
    provider: Optional[str] = None
    model: Optional[str] = None
    token_usage: Optional[Dict[str, Any]] = None
    retries: int = 0
    warnings: Optional[List[str]] = None
    errors: Optional[List[str]] = None


# ------------------------------
# Exceptions
# ------------------------------
class TextAnonymizerError(Exception):
    pass


class LLMResponseParseError(TextAnonymizerError):
    pass


class LLMCallError(TextAnonymizerError):
    pass


# ------------------------------
# LLM protocol and configs
# ------------------------------
@runtime_checkable
class LLMClient(Protocol):
    """Minimal protocol for an LLM client that returns a JSON string matching AnonymizedBatchOutput."""

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Must return a JSON **string** strictly matching the AnonymizedBatchOutput schema.
        The client should enforce JSON mode when possible (e.g., response_format={"type": "json_object"}).
        """
        ...


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: int = 60
    max_retries: int = 3
    provider: Optional[str] = None           # e.g., "openrouter", "openai", "ollama", "vllm"
    extra_params: Optional[Dict[str, Any]] = None


@dataclass
class AnonymizationConfig:
    """Anonymization options referenced by the prompt."""
    preserve_formatting: bool = True
    replace_person_names: bool = True
    replace_locations: bool = True
    replace_organizations: bool = True
    replace_ids: bool = True            # national IDs/passport/etc.
    replace_contacts: bool = True       # email/phone
    custom_rules: Optional[Dict[str, str]] = None  # e.g., {"Plate": "<PLATE_REDACTED>"}
    force_same_language: bool = True    # keep output in the same language as the input sentence
    explicit_language: Optional[str] = None  # if set, force this language label (e.g., "Portuguese")


# ------------------------------
# Utilities
# ------------------------------
def _now() -> float:
    return time.perf_counter()


def _log_debug(event: str, **fields: Any) -> None:
    if _STRUCTLOG:
        log.debug(event, **fields)
    else:
        log.debug("%s | %s", event, fields)


# ------------------------------
# OpenAI-like client (works with OpenRouter, OpenAI-compatible servers incl. vLLM/ollama)
# ------------------------------
class OpenAICompatibleClient:
    """
    Chat Completions client compatible with OpenRouter/OpenAI-like servers.
    Expects /chat/completions and returns **JSON string** content.
    """
    def __init__(self, base_url: str, api_key: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        import requests

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        # Strict system instruction: JSON only + language preservation
        system_msg = (
            "You ALWAYS answer with compact JSON only. No prose. "
            'Return a JSON object with "sentences": [ { "original_index": int, "text": str } ], '
            '"warnings": [str]? and "errors": [str]?. '
            "CRITICAL: Do NOT translate or paraphrase across languages. "
            "For each input sentence, output the anonymized sentence in the SAME LANGUAGE as the input. "
            "Preserve punctuation and casing when possible. "
            "If a language is explicitly provided in the user prompt, strictly use that language."
        )

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop
        if extra_params:
            payload.update(extra_params)

        _log_debug("llm_request", url=url, preview=prompt[:300], max_tokens=max_tokens)

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        except Exception as e:  # pragma: no cover
            raise LLMCallError(f"Request error: {e}") from e

        if resp.status_code >= 400:
            raise LLMCallError(f"HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        # Token usage (if provided) can be attached by the caller if needed.
        _log_debug("llm_response", status=resp.status_code, content_preview=content[:300])
        return content


# ------------------------------
# Core anonymizer
# ------------------------------
class TextAnonymizer:
    """
    Pipeline:
      1) Accept an input text;
      2) Split into sentences;
      3) Send N-sentence batches to an LLM with a configurable prompt;
      4) Expect JSON output per batch, validate with Pydantic, keep sentence order;
      5) Postprocess and join sentences back to a single text.

    Public API:
      - anonymize_text(text) -> str
      - set_prompt_template(template: str)
      - set_sentences_per_batch(n: int)
      - set_llm_config(llm_config: LLMConfig)
      - set_anonymization_config(config: AnonymizationConfig)
      - get_last_batches_info() -> List[Dict[str, Any]]
      - dry_run_prompt(text, limit_sentences=5) -> str
    """

    def __init__(
        self,
        llm_client: LLMClient,
        llm_config: LLMConfig,
        *,
        sentences_per_batch: int = 8,
        sentence_split_pattern: Optional[str] = None,
        prompt_template: str = (
            "You are an anonymization assistant for Portuguese and English texts. "
            "Given the following sentences, you MUST return a JSON object with the schema:\n"
            "{{\n"
            '  "sentences": [ {{"original_index": <int>, "text": "<anonymized>"}} ],\n'
            '  "warnings": [ "<optional>" ],\n'
            '  "errors": [ "<optional>" ]\n'
            "}}\n\n"
            "Rules:\n"
            "- Replace person names, locations, organizations, IDs, and contacts with generic tags.\n"
            "- Keep non-sensitive content intact and keep sentence order by original_index.\n"
            "- Do NOT translate across languages; preserve the input sentence language in the output.\n"
            "- Preserve punctuation and casing as much as possible.\n"
            f"- Preserve formatting: {{preserve_formatting}}\n"
            f"- Replace person names: {{replace_person_names}}\n"
            f"- Replace locations: {{replace_locations}}\n"
            f"- Replace organizations: {{replace_organizations}}\n"
            f"- Replace IDs: {{replace_ids}}\n"
            f"- Replace contacts: {{replace_contacts}}\n"
            "{custom_rules_block}"   # <-- entra aqui naturalmente, ou fica vazio
            "{language_constraints_block}"  # <-- idem, vazio se não houver
            "\nReturn ONLY compact JSON. No extra text.\n\n"
            "Sentences (one per line):\n{sentences}\n"
        ),
        anonymization_config: Optional[AnonymizationConfig] = None,
        join_separator: str = " ",
    ) -> None:
        if sentences_per_batch <= 0:
            raise ValueError("sentences_per_batch must be > 0")

        self._llm_client = llm_client
        self._llm_config = llm_config
        self._sentences_per_batch = sentences_per_batch
        # A pragmatic regex for PT/EN sentence boundaries (punctuation + whitespace + capital/number start)
        self._sentence_split_pattern = (
            sentence_split_pattern
            or r"(?<=[\.\!\?;:])\s+(?=[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÀÄËÏÖÜÇ0-9])"
        )
        self._prompt_template = prompt_template
        self._anonymization_config = anonymization_config or AnonymizationConfig()
        self._join_separator = join_separator
        self._last_batches_info: List[BatchMeta] = []

        _log_debug(
            "init",
            provider=self._llm_config.provider,
            model=self._llm_config.model,
            sentences_per_batch=self._sentences_per_batch,
            join_separator=self._join_separator,
        )

    # -----------
    # Public API
    # -----------

    def _format_custom_rules_for_prompt(self, rules: Dict[str, str]) -> str:
        if not rules:
            return ""
        lines = []
        for label in sorted(rules.keys(), key=lambda s: s.lower()):
            repl = rules[label]
            lines.append(f"- Replace {str(label).strip()}: {str(repl).strip()}")
        # prefixa com quebra de linha para “colar” naturalmente nas Rules
        return "\n" + "\n".join(lines) + "\n"


    def anonymize_text(self, text: str) -> str:
        """Process the whole text and return the anonymized version."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        self._last_batches_info.clear()
        sentences = self._split_sentences(text)
        _log_debug("split_done", count=len(sentences))

        indexed: List[Tuple[int, str]] = list(enumerate(sentences))
        output_map: Dict[int, str] = {}

        batches = list(self._batch(indexed, self._sentences_per_batch))
        for batch_id, batch in enumerate(tqdm(batches, desc="Anonymizing", unit="batch")):
            idxs, sents = zip(*batch) if batch else ([], [])
            prompt = self._build_prompt(list(sents), self._anonymization_config)
            t0 = _now()
            retries = 0
            warnings: List[str] = []
            errors: List[str] = []

            while True:
                try:
                    raw_json = self._call_llm(prompt)
                    parsed = self._parse_llm_output(raw_json, expected_count=len(sents))
                    dt = _now() - t0

                    for s in parsed.sentences:
                        # original_index is batch-relative; map it to the global index.
                        global_idx = idxs[s.original_index]
                        output_map[global_idx] = s.text

                    warnings = parsed.warnings or []
                    errors = parsed.errors or []
                    self._last_batches_info.append(
                        BatchMeta(
                            batch_id=batch_id,
                            sent_count=len(sents),
                            llm_latency_s=dt,
                            prompt_chars=len(prompt),
                            returned_count=len(parsed.sentences),
                            provider=self._llm_config.provider,
                            model=self._llm_config.model,
                            token_usage=None,
                            retries=retries,
                            warnings=warnings or None,
                            errors=errors or None,
                        )
                    )
                    break
                except (LLMCallError, LLMResponseParseError, ValidationError) as e:
                    retries += 1
                    _log_debug("batch_error", batch_id=batch_id, retries=retries, error=str(e))
                    if retries > self._llm_config.max_retries:
                        raise
                    time.sleep(min(2.0 * retries, 5.0))  # simple backoff

        anonymized = [output_map[i] for i in range(len(sentences))]
        anonymized = self._postprocess_sentences(anonymized)
        return self._join_separator.join(anonymized)

    def set_prompt_template(self, template: str) -> None:
        """Update the prompt template (must contain {sentences})."""
        if "{sentences}" not in template:
            raise ValueError("template must contain '{sentences}' placeholder")
        self._prompt_template = template
        _log_debug("prompt_template_updated")

    def set_sentences_per_batch(self, n: int) -> None:
        """Update the batch size."""
        if n <= 0:
            raise ValueError("sentences_per_batch must be > 0")
        self._sentences_per_batch = n
        _log_debug("sentences_per_batch_updated", value=n)

    def set_llm_config(self, llm_config: LLMConfig) -> None:
        """Update LLM settings."""
        self._llm_config = llm_config
        _log_debug("llm_config_updated")

    def set_anonymization_config(self, config: AnonymizationConfig) -> None:
        """Update anonymization rules."""
        self._anonymization_config = config
        _log_debug("anonymization_config_updated")

    def get_last_batches_info(self) -> List[Dict[str, Any]]:
        """Return metadata for the last anonymization run."""
        return [m.dict() for m in self._last_batches_info]

    def dry_run_prompt(self, text: str, limit_sentences: int = 5) -> str:
        """Preview the constructed prompt for the first `limit_sentences` sentences without calling the LLM."""
        sents = self._split_sentences(text)[: max(1, limit_sentences)]
        return self._build_prompt(sents, self._anonymization_config)

    # -----------------
    # Internal helpers
    # -----------------

    def _split_sentences(self, text: str) -> List[str]:
        """Split `text` into sentences using a pragmatic regex suited for PT/EN."""
        text = text.strip()
        if not text:
            return []
        parts = re.split(self._sentence_split_pattern, text)
        sentences = [s.strip() for s in parts if s and s.strip()]
        return sentences

    def _batch(self, items: List[Tuple[int, str]], size: int) -> Iterable[List[Tuple[int, str]]]:
        """Yield sublists of `items` with chunk size `size`."""
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _build_prompt(
        self,
        sentences: List[str],
        anonymization_config: Optional[AnonymizationConfig],
    ) -> str:
        cfg = anonymization_config or self._anonymization_config
        cfg_dict = {
            "preserve_formatting": cfg.preserve_formatting,
            "replace_person_names": cfg.replace_person_names,
            "replace_locations": cfg.replace_locations,
            "replace_organizations": cfg.replace_organizations,
            "replace_ids": cfg.replace_ids,
            "replace_contacts": cfg.replace_contacts,
        }

        # bloco de idioma (opcional)
        if cfg.explicit_language:
            language_constraints_block = (
                "\n- LANGUAGE: Always output in **"
                + cfg.explicit_language
                + "**; do NOT translate away from it.\n"
            )
        elif cfg.force_same_language:
            language_constraints_block = (
                "\n- LANGUAGE: For each input sentence, output in the SAME LANGUAGE as that sentence. "
                "Do NOT translate across languages.\n"
            )
        else:
            language_constraints_block = ""

        # bloco de regras customizadas (opcional)
        custom_rules_block = self._format_custom_rules_for_prompt(cfg.custom_rules or {})

        block = "\n".join(s.strip() for s in sentences)

        prompt = self._prompt_template.format(
            sentences=block,
            custom_rules_block=custom_rules_block,              # << injeta “dentro” de Rules
            language_constraints_block=language_constraints_block,  # << idem
            **cfg_dict,
        )

        _log_debug("prompt_built", preview=prompt[:400])
        return prompt



    def _call_llm(self, prompt: str) -> str:
        """Send the prompt to the LLM and return a JSON string."""
        return self._llm_client.generate(
            model=self._llm_config.model,
            prompt=prompt,
            temperature=self._llm_config.temperature,
            max_tokens=self._llm_config.max_tokens,
            extra_params=self._llm_config.extra_params,
        )

    def _parse_llm_output(self, llm_text: str, expected_count: int) -> AnonymizedBatchOutput:
        """
        Parse the LLM output (JSON string) and ensure we have `expected_count` items.
        If fewer, fill missing entries with empty strings (maintain order).
        """
        try:
            data = json.loads(llm_text)
        except json.JSONDecodeError as e:
            raise LLMResponseParseError(f"Invalid JSON from LLM: {e}\nText: {llm_text[:500]}")

        try:
            parsed = AnonymizedBatchOutput(**data)
        except ValidationError as e:
            raise LLMResponseParseError(f"Pydantic validation error: {e}\nData: {data}")

        if len(parsed.sentences) != expected_count:
            _log_debug("count_mismatch", expected=expected_count, got=len(parsed.sentences))
            existing = {s.original_index: s.text for s in parsed.sentences}
            completed = [
                AnonymizedSentence(original_index=i, text=existing.get(i, "")) for i in range(expected_count)
            ]
            parsed = AnonymizedBatchOutput(
                sentences=completed, warnings=parsed.warnings, errors=parsed.errors
            )

        parsed.sentences.sort(key=lambda s: s.original_index)
        return parsed


    def _postprocess_sentences(self, sentences: List[str]) -> List[str]:
        cfg = self._anonymization_config
        out = []
        for s in sentences:
            t = s
            if cfg.custom_rules:
                for patt, repl in cfg.custom_rules.items():
                    t = re.sub(re.escape(patt), repl, t, flags=re.IGNORECASE)
            t = re.sub(r"\s{2,}", " ", t).strip()
            out.append(t)
        return out


