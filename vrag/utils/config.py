"""
VRAG Configuration Management

Loads and manages configuration from YAML files with environment variable
substitution support.
"""

import os
import re
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"


def _resolve_env_vars(value: str) -> str:
    """Replace ${ENV_VAR} patterns with environment variable values.
    Returns None for unresolved variables so downstream defaults can trigger."""
    pattern = re.compile(r"\$\{(\w+)\}")
    has_unresolved = False
    def replacer(match):
        nonlocal has_unresolved
        env_var = match.group(1)
        env_val = os.environ.get(env_var)
        if env_val is None:
            has_unresolved = True
            return match.group(0)  # keep placeholder temporarily
        return env_val
    resolved = pattern.sub(replacer, value)
    # If the entire value was a single unresolved var, return None
    if has_unresolved and pattern.fullmatch(value):
        return None
    return resolved


def _resolve_config(config: Any) -> Any:
    """Recursively resolve environment variables in config values."""
    if isinstance(config, dict):
        return {k: _resolve_config(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_resolve_config(v) for v in config]
    elif isinstance(config, str):
        return _resolve_env_vars(config)
    return config


class Config:
    """Configuration container with dot-notation access."""

    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __iter__(self):
        """Allow iteration over config keys (like a dict)."""
        return iter(self.__dict__)

    def __contains__(self, key: str) -> bool:
        """Support 'key in config' checks."""
        return key in self.__dict__

    def keys(self):
        """Return config keys."""
        return self.__dict__.keys()

    def values(self):
        """Return config values."""
        return self.__dict__.values()

    def items(self):
        """Return config key-value pairs."""
        return self.__dict__.items()

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file. If None, uses default.

    Returns:
        Config object with all settings.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults.")
        return Config({})

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    resolved = _resolve_config(raw_config)
    config = Config(resolved)

    logger.info(f"Configuration loaded from {config_path}")
    return config
