from shared.config.base_config import BaseAppConfig


class AIHubConfig(BaseAppConfig):
    """Configuration specific to the AIHub module.
    """

    model_path: str = "models/cryptobert"
    hf_model_path: str = "https://dien2112-finbert.hf.space/api/sentiment"

    # StockMem similarity search service
    stockmem_url: str = "http://localhost:8003"

    # Default LLM backend: "gemini" | "openai" | "groq"
    llm_backend: str = "gemini"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    predict_llm_backend: str = "gemini"

    # Sentiment uses CryptoBert HTTP API, not an LLM backend
    mock_mode: bool = False

    # Post-signal deterministic guardrails (applied after LLM /predict output)
    post_rule_enabled: bool = True
    post_rule_bull_30d_pct: float = 10.0
    post_rule_bear_30d_pct: float = -10.0
    post_rule_up_3d_pct: float = 3.0
    post_rule_down_3d_pct: float = -3.0
    post_rule_macd_confirm_eps: float = 0.0
    post_rule_hold_override_max_conf: float = 0.72

    # kNN confirmation: suppress directional signal when similar-case avg 7d return contradicts.
    # Set to 0.0 to disable.
    knn_confirm_threshold: float = 1.0
    model_config = {
        "env_prefix": "AIHUB_",
        "env_file": ".env",          # root /marketlens/.env — no aihub/.env
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
