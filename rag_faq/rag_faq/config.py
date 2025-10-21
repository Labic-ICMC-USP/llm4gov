import yaml

def load_config(path="config.yaml"):
    """Load YAML configuration from file"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
