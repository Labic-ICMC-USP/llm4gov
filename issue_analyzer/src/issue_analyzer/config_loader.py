import yaml
from pathlib import Path

def load_config(path: str = "config/llm_config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
