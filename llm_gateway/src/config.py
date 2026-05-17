

from pydantic_settings import BaseSettings


class LLMGatewayConfig(BaseSettings):
    """Runtime settings loaded from root ``.env`` and process environment."""

    opencode_go_api_key: str = ""
    opencode_endpoint: str = "https://opencode.ai/zen/go/v1/chat/completions"
    models_endpoint: str = "https://opencode.ai/zen/go/v1/models"
    default_model: str = "deepseek-v4-flash"
    fallback_models: str = "qwen3.5-plus"
    request_timeout_seconds: float = 15.0
    max_attempts: int = 1
    temperature: float = 0.0
    max_output_tokens: int = 800

    model_config = {
        "env_prefix": "LLM_GATEWAY_",
        "env_file": ".env",          # root /marketlens/.env — no llm_gateway/.env
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def bounded_max_attempts(self) -> int:
        """Keep retries intentional for large backtest batches."""
        return max(1, min(self.max_attempts, 3))

    @property
    def parsed_fallback_models(self) -> list[str]:
        out: list[str] = []
        for raw in (self.fallback_models or "").split(","):
            m = raw.strip()
            if m:
                out.append(m)
        return out
