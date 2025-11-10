"""
Student fine-tuning using Unsloth (LoRA) based on labeled data produced by a Teacher LLM.

Input:
  - YAML config (student_fine_tuning_config.yaml) with Unsloth + TRL + saving options
  - JSON or JSONL labeled dataset with records:
        {
          "system_prompt": "...",                 # repeated context/instructions (PT-BR ok)
          "user_prompt": {... or "..."},          # dict or string; we'll JSON-dump if dict
          "teacher_output": "... or {...}"        # preferred: strict JSON string; fallback: text
        }

This script:
  - Loads config
  - Loads & validates dataset
  - Converts to Alpaca format (instruction/input/output)
  - Trains with SFTTrainer
  - Saves LoRA and (optionally) merged / GGUF artifacts

Run:
  python student_fine_tuning.py --config student_fine_tuning_config.yaml

Notes:
  - Keep the Teacher's output as-is; we append EOS correctly.
  - If user_prompt is a dict, we embed as JSON in the Alpaca "input".
  - For sharegpt/chat templates, swap the prompt template if you wish. Here we stick to Alpaca.
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from typing import Optional

import yaml

# Unsloth / Transformers / TRL
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer


# ----------------------------
# Logging
# ----------------------------

def setup_logger(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("student_finetune")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ----------------------------
# Config dataclasses
# ----------------------------

@dataclass
class ModelConfig:
    model_name: str = "unsloth/Mistral-Nemo-Base-2407"
    max_seq_length: int = 2048
    dtype: Optional[str] = None          # "float16", "bfloat16", or None for auto
    load_in_4bit: bool = True
    token: Optional[str] = None          # HF token if model is gated

@dataclass
class LoraConfig:
    r: int = 16
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: Union[bool, str] = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[Any] = None

@dataclass
class TrainArgsConfig:
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: Optional[int] = 60       # set None and use num_train_epochs for full runs
    num_train_epochs: Optional[int] = None
    learning_rate: float = 2e-4
    logging_steps: int = 10
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "outputs"
    report_to: str = "none"             # or "wandb", "tensorboard"
    max_grad_norm: Optional[float] = None  # optional

@dataclass
class TrainerConfig:
    packing: bool = False               # can speed up with many short examples
    dataset_text_field: str = "text"    # internal, do not change
    # You can extend this with TRL SFTTrainer kwargs if needed

@dataclass
class SaveConfig:
    save_lora_dir: str = "lora_model"
    save_merged_16bit: bool = False
    save_merged_4bit: bool = False
    save_gguf: Optional[str] = None     # None or one of: "f16", "q4_k_m", "q8_0", etc.
    push_to_hub: bool = False
    hf_repo: Optional[str] = None       # e.g. "username/my-student-model"
    hf_token: Optional[str] = None      # huggingface token for push_to_hub

@dataclass
class DataConfig:
    labeled_file: str = "training_labeled.json"  # JSON or JSONL
    # Optional: limit dataset for quick tests
    max_examples: Optional[int] = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train_args: TrainArgsConfig = field(default_factory=TrainArgsConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    save: SaveConfig = field(default_factory=SaveConfig)
    data: DataConfig = field(default_factory=DataConfig)
    log_level: str = "INFO"
    log_file: Optional[str] = None



# ----------------------------
# Utilities
# ----------------------------

ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{}\n\n"
    "### Input:\n{}\n\n"
    "### Response:\n{}"
)

def set_global_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Simple dict → dataclasses conversion
    def dc_from_dict(dc_cls, data):
        if data is None:
            return dc_cls()
        return dc_cls(**{k: v for k, v in data.items() if k in dc_cls.__annotations__})

    model = dc_from_dict(ModelConfig, raw.get("model"))
    lora = dc_from_dict(LoraConfig, raw.get("lora"))
    train_args = dc_from_dict(TrainArgsConfig, raw.get("train_args"))
    trainer = dc_from_dict(TrainerConfig, raw.get("trainer"))
    save = dc_from_dict(SaveConfig, raw.get("save"))
    data = dc_from_dict(DataConfig, raw.get("data"))

    cfg = Config(
        model=model,
        lora=lora,
        train_args=train_args,
        trainer=trainer,
        save=save,
        data=data,
        log_level=raw.get("log_level", "INFO"),
        log_file=raw.get("log_file"),
    )
    return cfg


def read_labeled_data(path: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Accepts:
      - JSON array file (list of dicts)
      - JSONL file (one JSON per line)
    Each record must include: system_prompt, user_prompt, teacher_output
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Labeled dataset not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    data: List[Dict[str, Any]] = []

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except Exception as e:
                    logger.error(f"Invalid JSONL at line {i}: {e}")
                    continue
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            try:
                arr = json.load(f)
            except Exception as e:
                raise ValueError(f"Invalid JSON file: {e}")
        if not isinstance(arr, list):
            raise ValueError("JSON must be an array of records.")
        data = arr
    else:
        raise ValueError("Dataset must be .json or .jsonl")

    # Basic validation
    required = {"system_prompt", "user_prompt", "teacher_output"}
    filtered = []
    for idx, rec in enumerate(data):
        if not isinstance(rec, dict):
            logger.warning(f"Skipping non-dict record at index {idx}.")
            continue
        missing = required - set(rec.keys())
        if missing:
            logger.warning(f"Skipping record {idx}: missing keys {missing}")
            continue
        filtered.append(rec)

    logger.info(f"Loaded {len(filtered)} valid records from {path}.")
    return filtered


def to_alpaca_examples(records: List[Dict[str, Any]],
                       max_examples: Optional[int],
                       logger: logging.Logger) -> Dataset:
    """
    Convert Teacher-labeled records into Alpaca-compatible triples:
      instruction = system_prompt
      input       = JSON-dumped user_prompt (if dict) or str(user_prompt)
      output      = teacher_output (as string)
    """
    if max_examples is not None:
        records = records[:max_examples]

    instructions, inputs, outputs = [], [], []

    drop_count = 0
    for i, rec in enumerate(records):
        try:
            instruction = str(rec["system_prompt"])
            user_prompt = rec["user_prompt"]
            if isinstance(user_prompt, (dict, list)):
                user_in = json.dumps(user_prompt, ensure_ascii=False)
            else:
                user_in = str(user_prompt)

            teacher_out = rec["teacher_output"]
            # Keep as given; if dict/list, dump to JSON
            if isinstance(teacher_out, (dict, list)):
                out = json.dumps(teacher_out, ensure_ascii=False)
            else:
                out = str(teacher_out)

            instructions.append(instruction)
            inputs.append(user_in)
            outputs.append(out)

        except Exception as e:
            drop_count += 1
            logger.error(f"Dropping record {i} due to conversion error: {e}")

    if drop_count:
        logger.warning(f"Dropped {drop_count} problematic records during conversion.")

    # Build HF dataset with text field (after formatting with EOS)
    data = {"instruction": instructions, "input": inputs, "output": outputs}
    ds = Dataset.from_dict(data)
    return ds


def build_texts_from_alpaca(ds: Dataset, tokenizer, logger: logging.Logger) -> Dataset:
    """
    Build final 'text' field using the ALPACA_PROMPT and append EOS to each sample.
    """
    EOS_TOKEN = tokenizer.eos_token

    def _format(batch):
        texts = []
        for inst, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
            txt = ALPACA_PROMPT.format(inst, inp, out) + EOS_TOKEN
            texts.append(txt)
        return {"text": texts}

    logger.info("Formatting dataset into Alpaca prompt with EOS...")
    ds = ds.map(_format, batched=True, remove_columns=ds.column_names)
    return ds


# ----------------------------
# Trainer class
# ----------------------------

class StudentFineTuner:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.model = None
        self.tokenizer = None

    def load_model(self):
        m = self.cfg.model
        self.logger.info(f"Loading base model: {m.model_name}")
        dtype = None
        if isinstance(m.dtype, str):
            if m.dtype.lower() in ("float16", "fp16", "half"):
                dtype = torch.float16
            elif m.dtype.lower() in ("bfloat16", "bf16"):
                dtype = torch.bfloat16
            else:
                dtype = None

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=m.model_name,
            max_seq_length=m.max_seq_length,
            dtype=dtype,
            load_in_4bit=m.load_in_4bit,
            token=m.token,  # optional, for gated models
        )

        # Attach LoRA
        l = self.cfg.lora
        self.logger.info("Attaching LoRA adapters...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=l.r,
            target_modules=l.target_modules,
            lora_alpha=l.lora_alpha,
            lora_dropout=l.lora_dropout,
            bias=l.bias,
            use_gradient_checkpointing=l.use_gradient_checkpointing,
            random_state=l.random_state,
            use_rslora=l.use_rslora,
            loftq_config=l.loftq_config,
        )

    def prepare_dataset(self) -> Dataset:
        data_cfg = self.cfg.data
        recs = read_labeled_data(data_cfg.labeled_file, self.logger)
        ds = to_alpaca_examples(recs, data_cfg.max_examples, self.logger)
        ds = build_texts_from_alpaca(ds, self.tokenizer, self.logger)
        self.logger.info(f"Prepared dataset with {len(ds)} samples.")

        # save Alpaca-style dataset for inspection/debugging
        debug_path = "traindata_debug.json"
        self.logger.info(f"Saving debug dataset to {debug_path}")
        os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)
        # Export only a few fields (to keep JSON human-readable)
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(ds.to_dict(), f, ensure_ascii=False, indent=2)

        return ds

    def train(self, ds: Dataset):
        targs = self.cfg.train_args
        tcfg = self.cfg.trainer

        set_global_seed(targs.seed)

        sft_kwargs = dict(
            per_device_train_batch_size=targs.per_device_train_batch_size,
            gradient_accumulation_steps=targs.gradient_accumulation_steps,
            warmup_steps=targs.warmup_steps,
            learning_rate=targs.learning_rate,
            logging_steps=targs.logging_steps,
            optim=targs.optim,
            weight_decay=targs.weight_decay,
            lr_scheduler_type=targs.lr_scheduler_type,
            seed=targs.seed,
            output_dir=targs.output_dir,
            report_to=targs.report_to,
        )

        if targs.max_steps is not None:
            sft_kwargs["max_steps"] = targs.max_steps
        if targs.num_train_epochs is not None:
            sft_kwargs["num_train_epochs"] = targs.num_train_epochs
        if targs.max_grad_norm is not None:
            sft_kwargs["max_grad_norm"] = targs.max_grad_norm

        self.logger.info("Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=ds,
            dataset_text_field=tcfg.dataset_text_field,
            max_seq_length=self.cfg.model.max_seq_length,
            packing=tcfg.packing,
            args=SFTConfig(**sft_kwargs),
        )

        # Show device/memory info if CUDA available
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            self.logger.info(f"GPU: {gpu_props.name} | {round(gpu_props.total_memory/1024**3, 2)} GB total")

        self.logger.info("Starting training...")
        stats = trainer.train()
        self.logger.info(f"Training complete. Runtime (s): {stats.metrics.get('train_runtime')}")
        return trainer, stats

    def save_artifacts(self):
        s = self.cfg.save
        tokenizer = self.tokenizer
        model = self.model

        # Always save LoRA adapters
        self.logger.info(f"Saving LoRA adapters to: {s.save_lora_dir}")
        os.makedirs(s.save_lora_dir, exist_ok=True)
        model.save_pretrained(s.save_lora_dir)
        tokenizer.save_pretrained(s.save_lora_dir)

        # Push LoRA to hub if requested
        if s.push_to_hub and s.hf_repo and s.hf_token:
            self.logger.info(f"Pushing LoRA adapters to Hub: {s.hf_repo}")
            try:
                model.push_to_hub(s.hf_repo, token=s.hf_token)
                tokenizer.push_to_hub(s.hf_repo, token=s.hf_token)
            except Exception as e:
                self.logger.error(f"Failed to push LoRA to Hub: {e}")

        # Optional merges
        # 16-bit merge
        if s.save_merged_16bit:
            self.logger.info("Merging and saving 16-bit model...")
            out_dir = os.path.join("model_merged_16bit")
            try:
                model.save_pretrained_merged(out_dir, tokenizer, save_method="merged_16bit")
            except Exception as e:
                self.logger.error(f"Failed 16-bit merge save: {e}")
            if s.push_to_hub and s.hf_repo and s.hf_token:
                try:
                    model.push_to_hub_merged(s.hf_repo, tokenizer, save_method="merged_16bit", token=s.hf_token)
                except Exception as e:
                    self.logger.error(f"Failed to push 16-bit merged to Hub: {e}")

        # 4-bit merge
        if s.save_merged_4bit:
            self.logger.info("Merging and saving 4-bit model...")
            out_dir = os.path.join("model_merged_4bit")
            try:
                model.save_pretrained_merged(out_dir, tokenizer, save_method="merged_4bit")
            except Exception as e:
                self.logger.error(f"Failed 4-bit merge save: {e}")
            if s.push_to_hub and s.hf_repo and s.hf_token:
                try:
                    model.push_to_hub_merged(s.hf_repo, tokenizer, save_method="merged_4bit", token=s.hf_token)
                except Exception as e:
                    self.logger.error(f"Failed to push 4-bit merged to Hub: {e}")

        # GGUF export (optional)
        if s.save_gguf:
            q = s.save_gguf
            self.logger.info(f"Saving GGUF with quantization '{q}'...")
            try:
                model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method=q if q != "f16" else "f16")
            except Exception as e:
                self.logger.error(f"Failed GGUF save: {e}")
            if s.push_to_hub and s.hf_repo and s.hf_token:
                try:
                    model.push_to_hub_gguf(s.hf_repo, tokenizer, quantization_method=q, token=s.hf_token)
                except Exception as e:
                    self.logger.error(f"Failed to push GGUF to Hub: {e}")

    def run(self):
        self.load_model()
        ds = self.prepare_dataset()
        _, _ = self.train(ds)
        self.save_artifacts()
        self.logger.info("All done!")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Student fine-tuning (Unsloth + LoRA) from Teacher-labeled data.")
    p.add_argument("--config", type=str, required=True, help="Path to student_fine_tuning_config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    logger = setup_logger(cfg.log_level, cfg.log_file)
    tuner = StudentFineTuner(cfg, logger)
    tuner.run()


if __name__ == "__main__":
    main()
