"""External model API client for GPT-oss-120b predictions."""


class PredictClient:
    """Client for the external GPT-oss-120b model API.

    Args:
        api_key: API key for authentication.
        model_name: Model identifier (default: gpt-oss-120b).
        base_url: External API base URL.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-oss-120b",
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._base_url = base_url

    async def generate(self, prompt: str) -> str:
        """Send a prompt to the external model and return the response.

        Args:
            prompt: The assembled RAG prompt.

        Returns:
            Raw text response from the model.
        """
        raise NotImplementedError
