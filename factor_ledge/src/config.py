"""FactorLedge module configuration."""

from shared.config.base_config import BaseAppConfig


class FactorLedgeConfig(BaseAppConfig):
    """Configuration specific to the FactorLedge module."""

    stockmem_url: str = "http://localhost:8003"

    model_config = {"env_prefix": "FACTORLEDGE_", "extra": "ignore"}
