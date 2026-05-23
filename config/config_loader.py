import yaml
import os
from pathlib import Path
from loguru import logger

# Cache variable
_config = None
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def get_config(path):
    global _config
    if _config is not None:
        return _config

    target = Path(path) if path else _CONFIG_PATH
    if not target.exists():
        raise FileNotFoundError(f"Config file not found at {target}")

    with open(target, "r") as f:
        _config = yaml.safe_load(f)

    logger.info(f"Config loaded from {target}")
    return _config


def reload_config(path):
    global _config
    _config = None
    return get_config(path)


def get(section, key, default=None):
    cfg = get_config()
    return cfg.get(section, {}).get(key, default)