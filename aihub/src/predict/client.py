"""Prediction client — sends RAG-assembled context to the LLM and parses the result."""

from aihub.src.llm.base import LLMClient
from aihub.src.predict.prompt import PREDICT_PROMPT


class PredictClient:
    """Generates a trading signal by calling the LLM with a RAG context prompt.

    Args:
        llm: Configured LLMClient backend (Gemini or OpenAI).
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def generate(self, rag_context: str) -> dict:  # type: ignore[type-arg]
        """Send the assembled RAG prompt to the LLM and return the parsed JSON dict.

        Args:
            rag_context: Formatted context string from RAGContextBuilder.

        Returns:
            Dict with keys: signal, confidence, explanation, reasoning_steps.
        """
        prompt = PREDICT_PROMPT.format(rag_context=rag_context)
        return await self._llm.generate_json(prompt)
