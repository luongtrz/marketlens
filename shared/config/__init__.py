"""Configuration loading and base settings."""

from shared.config.base_config import BaseAppConfig
from shared.config.loader import load_config

__all__ = ["BaseAppConfig", "load_config"]
