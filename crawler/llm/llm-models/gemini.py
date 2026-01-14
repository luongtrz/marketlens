import os
import google.generativeai as genai

class GeminiClient:
    def __init__(self):
        # Retrieve the API key from environment variables
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Configure the library with the API key
        genai.configure(api_key=api_key)
        
        # Initialize the model (using gemini-pro as default for better availability)
        self.model = genai.GenerativeModel('gemini-flash-lite-latest')

    def generate_content(self, prompt):
        """
        Generates content based on the provided prompt.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating content: {e}")
            return None

def ask_gemini(prompt):
    client = GeminiClient()
    return client.generate_content(prompt)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    test_prompt = "Write a short poem about the stock market."
    response = ask_gemini(test_prompt)
    print("Gemini Response:")
    print(response)