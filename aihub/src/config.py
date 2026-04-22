"""AIHub module configuration."""

from shared.config.base_config import BaseAppConfig


class AIHubConfig(BaseAppConfig):
    """Configuration specific to the AIHub module."""

    model_path: str = "models/cryptobert"
    hf_model_path: str = "https://dien2112-cryptobert.hf.space/api/sentiment"

    # LLM backend: "gemini" | "openai"
    llm_backend: str = "gemini"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""  # leave empty to use official OpenAI endpoint

    predict_backend: str = "llm"  # "llm" | "mock"
    aihub_mock: bool = False  # True = return deterministic fixture responses

    model_config = {"env_prefix": "AIHUB_", "extra": "ignore"}
