import abc

class LLMModel(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        pass
    