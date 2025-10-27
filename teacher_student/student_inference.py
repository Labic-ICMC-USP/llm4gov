"""
Minimal, fast batched student inference using the Alpaca template (no chat roles).

Configuration exposes:
- model_path
- max_seq_length
- max_new_tokens
- batch_size
- test_file
- output_file
- log_file  (optional; if present, logs are written to this file as well)

Run:
  python student_inference.py --config student_inference_config.yaml
"""

import argparse
import json
import os
import time
import logging
from typing import Any, Dict, List, Union

from transformers import GenerationConfig
import yaml
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm

# Environment for tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAYON_NUM_THREADS"] = str(os.cpu_count() or 1)

# --- Speed toggles ------------------------------------------------------------
torch.set_num_threads(max(1, os.cpu_count() or 1))
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# --- Logging setup ------------------------------------------------------------
def setup_logging(log_file: str | None = None) -> None:
    """
    Configure root logger with console handler and (optionally) file handler.
    If called multiple times, it won't duplicate handlers.
    """
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
        root.addHandler(ch)

    if log_file:
        # Avoid adding duplicate file handlers for the same path
        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_file)
                   for h in logging.getLogger().handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
            logging.getLogger().addHandler(fh)


# --- Load YAML configuration --------------------------------------------------
def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --- Read JSON/JSONL test set -------------------------------------------------
def read_unlabeled(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    out: List[Dict[str, Any]] = []

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON test file must be an array of records.")
        out = data
    else:
        raise ValueError("Test file must be .json or .jsonl")

    # Keep only valid rows containing both fields
    valid = []
    for r in out:
        if isinstance(r, dict) and "system_prompt" in r and "user_prompt" in r:
            valid.append(r)
    return valid


# --- Write final outputs ------------------------------------------------------
def write_outputs(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


# --- Convert object to string or JSON string ---------------------------------
def to_str_or_json(v: Union[str, Dict[str, Any], List[Any]]) -> str:
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


# --- Build prompt using Alpaca template --------------------------------------
def build_alpaca_prompt(system_prompt: str, user_prompt: Union[str, Dict[str, Any]]) -> str:
    """
    Maps:
      system_prompt -> Instruction
      user_prompt   -> Input
    """
    instr = to_str_or_json(system_prompt).strip()
    utext = to_str_or_json(user_prompt).strip()

    # Full Alpaca template (with both instruction and input)
    tmpl = (
        "### Instruction:\n"
        f"{instr}\n\n"
        "### Input:\n"
        f"{utext}\n\n"
        "### Response:\n"
    )
    return tmpl


# --- Try to parse model output as JSON, fallback to text ---------------------
def extract_json_or_text(s: str):
    s_strip = s.strip()
    starts = [i for i in [s_strip.find("{"), s_strip.find("[")] if i != -1]
    if starts:
        start = min(starts)
        end = max(s_strip.rfind("}"), s_strip.rfind("]"))
        if end > start:
            cand = s_strip[start: end + 1]
            try:
                return json.loads(cand)
            except Exception:
                pass
    return s_strip


# --- Main inference routine ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Minimal, fast batched student inference (Alpaca).")
    parser.add_argument("--config", type=str, required=True, help="Path to student_inference_config.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    # Optional log file (like teacher.py)
    log_file = cfg.get("log_file")
    setup_logging(log_file)

    logger = logging.getLogger("student")

    model_path     = cfg["model_path"]
    max_seq_length = int(cfg.get("max_seq_length", 8192))
    max_new_tokens = int(cfg.get("max_new_tokens", 4096))
    batch_size     = int(cfg.get("batch_size", 16))
    test_file      = cfg["test_file"]
    output_file    = cfg["output_file"]

    logger.info("Loading model=%s | max_seq_length=%d | 4bit=True", model_path, max_seq_length)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    # Decoder-only models should use left padding (Alpaca has no chat roles)
    tok.padding_side = "left"
    tok.truncation_side = "left"

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # safest: reuse eos as pad

    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id

    model.generation_config = GenerationConfig(
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        num_beams=1,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )

    FastLanguageModel.for_inference(model)
    logger.info("Tokenizer padding_side=%s | truncation_side=%s", tok.padding_side, tok.truncation_side)

    # Load data
    data = read_unlabeled(test_file)
    logger.info("Loaded %d items from %s", len(data), test_file)
    outputs: List[Dict[str, Any]] = []

    # Batched generation
    for i in tqdm(range(0, len(data), batch_size), desc="Infer"):
        batch = data[i: i + batch_size]
        prompts = [build_alpaca_prompt(rec["system_prompt"], rec["user_prompt"]) for rec in batch]

        # Tokenize batch
        t0_tok = time.time()
        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            padding_side="left",   # explicit override
        )
        if torch.cuda.is_available():
            enc = {k: v.to("cuda") for k, v in enc.items()}
        t_tok = time.time() - t0_tok

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        input_lens = attention_mask.sum(dim=1).tolist()

        # --- Generate
        t0_gen = time.time()
        with torch.inference_mode():
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gen = time.time() - t0_gen

        # --- Batch-level metrics (Opção A – simples)
        max_gen_tokens = int(gen_out.size(1) - input_ids.size(1))
        s_per_token = (t_gen / max(1, max_gen_tokens)) if max_gen_tokens > 0 else 0.0
        logger.info(
            "Batch %d..%d | tok=%.3fs | gen=%.3fs | max_new=%d | s/token≈%.4f",
            i, i + len(batch) - 1, t_tok, t_gen, max_gen_tokens, s_per_token
        )


        # Decode generated tokens only
        for rec_idx, rec in enumerate(batch):
            comp_ids = gen_out[rec_idx, input_lens[rec_idx]:]
            text = tok.decode(comp_ids, skip_special_tokens=True)

            student_out = extract_json_or_text(text)
            if isinstance(student_out, str):
                try:
                    student_out = json.loads(student_out.strip())
                except Exception:
                    pass

            outputs.append(
                {
                    "system_prompt": rec["system_prompt"],
                    "user_prompt": rec["user_prompt"],
                    "student_output": student_out,
                }
            )

        # Free VRAM between batches
        del gen_out, enc, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    write_outputs(output_file, outputs)
    logger.info("Wrote %d rows to %s", len(outputs), output_file)


if __name__ == "__main__":
    main()

