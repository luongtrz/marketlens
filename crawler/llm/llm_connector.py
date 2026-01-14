import os
import sys
import importlib.util
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """
    Abstract base class for LLM clients.
    All LLM implementations should follow this interface.
    """
    @abstractmethod
    def generate_content(self, prompt: str) -> str:
        """
        Generates content based on the provided prompt.
        
        Args:
            prompt (str): The input prompt for the LLM.
            
        Returns:
            str: The generated content.
        """
        pass

class LLMFactory:
    """
    Factory class to create LLM clients based on the provider.
    """
    @staticmethod
    def create_client(provider: str, **kwargs) -> BaseLLM:
        """
        Creates and returns an LLM client instance.
        
        Args:
            provider (str): The name of the LLM provider (e.g., 'gemini', 'gpt', 'claude').
            **kwargs: Additional arguments to pass to the client constructor.
            
        Returns:
            BaseLLM: An instance of an LLM client.
        """
        provider = provider.lower()
        
        if provider == "gemini":
            return LLMFactory._load_gemini_client()
        elif provider == "deepseek":
            return LLMFactory._load_deepseek_client()
        elif provider == "gpt":
            # Placeholder for GPT implementation
            # return LLMFactory._load_gpt_client(**kwargs)
            raise NotImplementedError("GPT provider not yet implemented")
        elif provider == "claude":
            # Placeholder for Claude implementation
            # return LLMFactory._load_claude_client(**kwargs)
            raise NotImplementedError("Claude provider not yet implemented")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def _load_deepseek_client():
        try:
            # Construct path to deepseek.py in llm-models directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            module_path = os.path.join(current_dir, "llm-models", "deepseek.py")
            
            if not os.path.exists(module_path):
                raise FileNotFoundError(f"DeepSeek model file not found at {module_path}")

            # Load module dynamically
            spec = importlib.util.spec_from_file_location("deepseek_client", module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["deepseek_client"] = module
                spec.loader.exec_module(module)
                
                # Return instance of DeepSeekClient
                return module.DeepSeekClient()
            else:
                raise ImportError(f"Could not load module from {module_path}")
        except Exception as e:
            raise ImportError(f"Failed to load DeepSeek client: {str(e)}")

    @staticmethod
    def _load_gemini_client():
        try:
            # Construct path to gemini.py in llm-models directory
            # Assuming this file is in crawler/llm/ and gemini.py is in crawler/llm/llm-models/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            module_path = os.path.join(current_dir, "llm-models", "gemini.py")
            
            if not os.path.exists(module_path):
                raise FileNotFoundError(f"Gemini model file not found at {module_path}")

            # Load module dynamically
            spec = importlib.util.spec_from_file_location("gemini_client", module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["gemini_client"] = module
                spec.loader.exec_module(module)
                
                # Return instance of GeminiClient
                # Note: GeminiClient implementation currently reads env vars and takes no init args
                return module.GeminiClient()
            else:
                raise ImportError(f"Could not load module from {module_path}")
        except Exception as e:
            raise ImportError(f"Failed to load Gemini client: {str(e)}")

def get_llm_client(provider: str = "gemini"):
    """
    Convenience function to get an LLM client.
    """
    return LLMFactory.create_client(provider)
