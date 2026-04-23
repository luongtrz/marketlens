"""External model API client for GPT-oss-120b predictions."""


class PredictClient:
    """Client for the external GPT-oss-120b model API.

    Args:
        model_name: Model identifier (default: gpt-oss-120b).
    """


    def __init__(
        self,
        model_name: str = "",
    ) -> None:
        self._model_name = model_name
        self._model = llm_model_factory().get_model(model_name)

    async def generate(self, prompt: str) -> str:
        return self._model.generate(prompt)
