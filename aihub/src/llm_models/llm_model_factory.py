from src.predict.llm_models.gpt_oss_120b import GPTOSS120BModel
from src.predict.llm_models.model import LLMModel

class llm_model_factory:
    def get_model(self, model_name: str) -> LLMModel:
        if model_name == "gpt-oss-120b":
            return GPTOSS120BModel()
        else:
            raise ValueError(f"Unknown model name: {model_name}")