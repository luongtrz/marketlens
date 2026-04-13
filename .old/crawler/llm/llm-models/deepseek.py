import os
from openai import OpenAI

class DeepSeekClient:
    def __init__(self):
        # Retrieve the API key from environment variables
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError("HF_TOKEN environment variable not set")
        
        # Configure the OpenAI client for DeepSeek via HuggingFace
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )
        self.model_name = "deepseek-ai/DeepSeek-V3.2:novita"

    def generate_content(self, prompt):
        """
        Generates content based on the provided prompt.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating content: {e}")
            return None

def ask_deepseek(prompt):
    client = DeepSeekClient()
    return client.generate_content(prompt)

if __name__ == "__main__":
    # For local testing
    # Ensure HF_TOKEN is set in your environment or .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
        
    test_prompt = "What is the capital of France?"
    print(f"Asking DeepSeek: {test_prompt}")
    try:
        response = ask_deepseek(test_prompt)
        print("DeepSeek Response:")
        print(response)
    except Exception as e:
        print(f"Test failed: {e}")
