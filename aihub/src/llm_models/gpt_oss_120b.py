from src.predict.llm_models.model import LLMModel
from groq import Groq

SYSTEM_PROMT = '''You are a helpful assistant.

'''

class GPTOSS120BModel(LLMModel):
    def __init__(self):
        self._api_key = os.getenv("GROQ_API")
        self._model_name = "openai/gpt-oss-120b"
        self._groq = Groq(api_key=self._api_key)
    
    def generate(self, prompt: str) -> str:
        chat_completion = self._groq.chat.completions.create(
            model=self._model_name,
            temperature = 0.6,
            max_token = 2048,
            messages = [
                {"role":"system", "content":SYSTEM_PROMT},
                {"role":"user", "content":prompt}
            ]
        )
        return self.clean_instruction_tags(chat_completion.choices[0].message.content)

    def generate(self, prompt: str, system_prompt: str) -> str:
        chat_completion = self._groq.chat.completions.create(
            model=self._model_name,
            temperature = 0.6,
            max_token = 2048,
            messages = [
                {"role":"system", "content":system_prompt},
                {"role":"user", "content":prompt}
            ]
        )
        return self.clean_instruction_tags(chat_completion.choices[0].message.content)

    def clean_instruction_tags(self, text):
        if not text:
            return text
        
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<\|.*?\|>', '', text)
        text = re.sub(r'\[/?(INST|SYS|USER|ASSISTANT).*?\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(assistant|user|system):\s*', '', text, flags=re.IGNORECASE)
        return text.strip()