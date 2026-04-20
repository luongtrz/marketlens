"""AIHub module configuration."""

from shared.config.base_config import BaseAppConfig


class AIHubConfig(BaseAppConfig):
    """Configuration specific to the AIHub module."""

    model_path: str = "models/cryptobert"
    hf_model_path: str = "https://dien2112-cryptobert.hf.space/api/sentiment"
    gpt_api_key: str = ""
    gpt_model: str = "gpt-oss-120b"
    predict_backend: str = "gpt"  # "gpt" | "mock"
    aihub_mock: bool = False  # True = return deterministic fixture responses

    model_config = {"env_prefix": "AIHUB_", "extra": "ignore"}
