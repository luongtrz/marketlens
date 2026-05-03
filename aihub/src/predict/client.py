"""Prediction client — sends RAG-assembled context to the LLM and parses the result."""

from aihub.src.llm.base import LLMClient
from aihub.src.predict.prompt import PREDICT_SYSTEM_PROMPT, PREDICT_USER_PROMPT


class PredictClient:
    """Generates a trading signal by calling the LLM with a RAG context prompt.

    Args:
        llm: Configured LLMClient backend (e.g. GroqClient).
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def generate(self, today_record: str, similar_records: str) -> dict:  # type: ignore[type-arg]
        """Send the assembled RAG prompt to the LLM and return the parsed JSON dict.

        Args:
            today_record: Formatted text of the current market situation.
            similar_records: Formatted text of similar historical cases.

        Returns:
            Dict with keys: signal, confidence, explanation, reasoning_steps.
        """
        user_prompt = PREDICT_USER_PROMPT.format(
            today_record=today_record,
            similar_records=similar_records
        )
        
        # We try to use generate_with_reasoning if the client supports it (e.g. GroqClient)
        if hasattr(self._llm, "generate_with_reasoning"):
            resp = await self._llm.generate_with_reasoning(
                prompt=user_prompt, 
                system=PREDICT_SYSTEM_PROMPT
            )
            data = getattr(self._llm, "parse_json", lambda x: getattr(self._llm, "generate_json", lambda y: y))(resp.content)
            if callable(data): 
                # fallback parsing
                import json
                import re
                text = re.sub(r"^```(?:json)?\s*", "", resp.content.strip())
                text = re.sub(r"\s*```$", "", text.strip())
                data = json.loads(text)
            data["reasoning_steps"] = resp.reasoning_steps
            return data
            
        else:
            return await getattr(self._llm, "generate_json")(
                prompt=user_prompt, 
                system=PREDICT_SYSTEM_PROMPT
            )
