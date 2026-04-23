"""AIHub module configuration."""

from shared.config.base_config import BaseAppConfig


class AIHubConfig(BaseAppConfig):
    """Configuration specific to the AIHub module."""

    model_path: str = "models/cryptobert"
    hf_model_path: str = "https://dien2112-cryptobert.hf.space/api/sentiment"

    # LLM backend: "gemini" | "openai"
    llm_backend: str = "gemini"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite-preview"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""

    predict_backend: str = "llm"  
    mock_mode: bool = False

    model_config = {
        "env_prefix": "AIHUB_",
        "env_file": "aihub/.env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
