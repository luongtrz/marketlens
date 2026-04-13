"""Configuration loader: merges environment variables with YAML config files."""

from pathlib import Path
from typing import TypeVar, Type

import yaml

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return its contents as a dictionary.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed dictionary from the YAML file.
    """
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def load_config(config_class: Type[T], yaml_path: str | Path | None = None) -> T:
    """Load configuration by merging YAML file values with environment variables.

    Environment variables take precedence over YAML values.

    Args:
        config_class: A Pydantic BaseSettings subclass to instantiate.
        yaml_path: Optional path to a YAML config file.

    Returns:
        An instance of config_class populated from env + YAML.
    """
    yaml_values = load_yaml(yaml_path) if yaml_path else {}
    return config_class(**yaml_values)
