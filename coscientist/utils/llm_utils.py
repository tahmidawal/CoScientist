from typing import Dict, List, Any, Optional, Union
import logging
import os
import json
import time
import requests
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """
    
    @abstractmethod
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: int = 500, 
                **kwargs) -> str:
        """
        Generate text from the LLM.
        
        Args:
            prompt (str): Prompt for the LLM
            temperature (float, optional): Temperature parameter. Defaults to 0.7.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 500.
            **kwargs: Additional arguments for the LLM
            
        Returns:
            str: Generated text
        """
        pass

class DummyLLMProvider(LLMProvider):
    """
    Dummy LLM provider for testing.
    """
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: int = 500, 
                **kwargs) -> str:
        """
        Generate text from the dummy LLM.
        
        Args:
            prompt (str): Prompt for the LLM
            temperature (float, optional): Temperature parameter. Defaults to 0.7.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 500.
            **kwargs: Additional arguments for the LLM
            
        Returns:
            str: Generated text
        """
        logger.info("Using dummy LLM provider")
        return f"Dummy LLM response for prompt: {prompt[:30]}..."

class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-0125-preview"):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key (str, optional): OpenAI API key. Defaults to None.
            model (str, optional): Model to use. Defaults to "gpt-4-0125-preview".
        """
        # Use provided API key or try to get from environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
        
        self.model = model
        
        # Import openai if available
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self.openai_available = True
        except ImportError:
            logger.warning("OpenAI package not installed. Install with: pip install openai")
            self.openai_available = False
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: int = 500, 
                **kwargs) -> str:
        """
        Generate text from the OpenAI LLM.
        
        Args:
            prompt (str): Prompt for the LLM
            temperature (float, optional): Temperature parameter. Defaults to 0.7.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 500.
            **kwargs: Additional arguments for the LLM
            
        Returns:
            str: Generated text
        """
        if not self.openai_available:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            return "Error: OpenAI package not installed"
        
        if not self.api_key:
            logger.error("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
            return "Error: No OpenAI API key provided"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI Co-Scientist assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating text from OpenAI: {e}")
            return f"Error: {str(e)}"

class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude LLM provider.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
        """
        Initialize the Anthropic provider.
        
        Args:
            api_key (str, optional): Anthropic API key. Defaults to None.
            model (str, optional): Model to use. Defaults to "claude-3-opus-20240229".
        """
        # Use provided API key or try to get from environment
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            logger.warning("No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable.")
        
        self.model = model
        
        # Import anthropic if available
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.anthropic_available = True
        except ImportError:
            logger.warning("Anthropic package not installed. Install with: pip install anthropic")
            self.anthropic_available = False
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: int = 500, 
                **kwargs) -> str:
        """
        Generate text from the Anthropic Claude LLM.
        
        Args:
            prompt (str): Prompt for the LLM
            temperature (float, optional): Temperature parameter. Defaults to 0.7.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 500.
            **kwargs: Additional arguments for the LLM
            
        Returns:
            str: Generated text
        """
        if not self.anthropic_available:
            logger.error("Anthropic package not installed. Install with: pip install anthropic")
            return "Error: Anthropic package not installed"
        
        if not self.api_key:
            logger.error("No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable.")
            return "Error: No Anthropic API key provided"
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            
            return response.content[0].text
        
        except Exception as e:
            logger.error(f"Error generating text from Anthropic: {e}")
            return f"Error: {str(e)}"

class HuggingFaceProvider(LLMProvider):
    """
    Hugging Face Inference API LLM provider.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        """
        Initialize the Hugging Face provider.
        
        Args:
            api_key (str, optional): Hugging Face API key. Defaults to None.
            model (str, optional): Model to use. Defaults to "mistralai/Mixtral-8x7B-Instruct-v0.1".
        """
        # Use provided API key or try to get from environment
        self.api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY")
        
        if not self.api_key:
            logger.warning("No Hugging Face API key provided. Set HUGGINGFACE_API_KEY environment variable.")
        
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: int = 500, 
                **kwargs) -> str:
        """
        Generate text from the Hugging Face LLM.
        
        Args:
            prompt (str): Prompt for the LLM
            temperature (float, optional): Temperature parameter. Defaults to 0.7.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 500.
            **kwargs: Additional arguments for the LLM
            
        Returns:
            str: Generated text
        """
        if not self.api_key:
            logger.error("No Hugging Face API key provided. Set HUGGINGFACE_API_KEY environment variable.")
            return "Error: No Hugging Face API key provided"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                **kwargs
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            
            return str(result)
        
        except Exception as e:
            logger.error(f"Error generating text from Hugging Face: {e}")
            return f"Error: {str(e)}"

class PraisonAIProvider(LLMProvider):
    """
    PraisonAI provider integration.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "default"):
        """
        Initialize the PraisonAI provider.
        
        Args:
            api_key (str, optional): PraisonAI API key. Defaults to None.
            model (str, optional): Model to use. Defaults to "default".
        """
        # Use provided API key or try to get from environment
        self.api_key = api_key or os.environ.get("PRAISON_API_KEY")
        
        if not self.api_key:
            logger.warning("No PraisonAI API key provided. Set PRAISON_API_KEY environment variable.")
        
        self.model = model
        
        # Import PraisonAI if available
        try:
            import praisonai
            self.praison_available = True
        except ImportError:
            logger.warning("PraisonAI package not installed. Install with: pip install praisonai")
            self.praison_available = False
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: int = 500, 
                **kwargs) -> str:
        """
        Generate text from the PraisonAI LLM.
        
        Args:
            prompt (str): Prompt for the LLM
            temperature (float, optional): Temperature parameter. Defaults to 0.7.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 500.
            **kwargs: Additional arguments for the LLM
            
        Returns:
            str: Generated text
        """
        if not self.praison_available:
            logger.error("PraisonAI package not installed. Install with: pip install praisonai")
            return "Error: PraisonAI package not installed"
        
        if not self.api_key:
            logger.error("No PraisonAI API key provided. Set PRAISON_API_KEY environment variable.")
            return "Error: No PraisonAI API key provided"
        
        try:
            # Since we don't have direct access to PraisonAI's API structure,
            # this is a placeholder. Replace with actual implementation.
            from praisonai import PraisonAI
            
            client = PraisonAI(api_key=self.api_key)
            response = client.generate(
                prompt=prompt,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating text from PraisonAI: {e}")
            return f"Error: {str(e)}"

class GeminiProvider(LLMProvider):
    """
    Google Gemini provider integration.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash-thinking-exp-01-21"):
        """
        Initialize the Google Gemini provider.
        
        Args:
            api_key (str, optional): Google Gemini API key. Defaults to None.
            model (str, optional): Model to use. Defaults to "gemini-2.0-flash-thinking-exp-01-21".
        """
        # Use provided API key or try to get from environment
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        if not self.api_key:
            logger.warning("No Gemini API key provided. Set GEMINI_API_KEY environment variable.")
        
        self.model = model
        
        # Import Google Generative AI if available
        try:
            import google.generativeai as genai
            self.genai = genai
            self.genai_available = True
            # Configure the API
            self.genai.configure(api_key=self.api_key)
        except ImportError:
            logger.warning("Google Generative AI package not installed. Install with: pip install google-generativeai")
            self.genai_available = False
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: int = 500, 
                **kwargs) -> str:
        """
        Generate text from the Google Gemini LLM.
        
        Args:
            prompt (str): Prompt for the LLM
            temperature (float, optional): Temperature parameter. Defaults to 0.7.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 500.
            **kwargs: Additional arguments for the LLM
            
        Returns:
            str: Generated text
        """
        if not self.genai_available:
            logger.error("Google Generative AI package not installed. Install with: pip install google-generativeai")
            return "Error: Google Generative AI package not installed"
        
        if not self.api_key:
            logger.error("No Gemini API key provided. Set GEMINI_API_KEY environment variable.")
            return "Error: No Gemini API key provided"
        
        try:
            # Set up the generation config
            generation_config = {
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 64),
                "max_output_tokens": max_tokens,
                "response_mime_type": "text/plain",
            }
            
            # Create the model
            model = self.genai.GenerativeModel(
                model_name=self.model,
                generation_config=generation_config,
            )
            
            # Start a chat session and send the message
            chat_session = model.start_chat(history=[])
            response = chat_session.send_message(prompt)
            
            return response.text
        
        except Exception as e:
            logger.error(f"Error generating text from Google Gemini: {e}")
            return f"Error: {str(e)}"

def get_llm_provider(provider_name: str = "default", **kwargs) -> LLMProvider:
    """
    Get an LLM provider by name.
    
    Args:
        provider_name (str, optional): Name of the provider. Defaults to "default".
        **kwargs: Additional arguments for the provider
        
    Returns:
        LLMProvider: The provider
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "huggingface": HuggingFaceProvider,
        "praison": PraisonAIProvider,
        "gemini": GeminiProvider,
        "dummy": DummyLLMProvider
    }
    
    # Default provider
    if provider_name == "default":
        # Try to use available providers in order
        for provider in ["openai", "anthropic", "huggingface", "gemini", "praison"]:
            try:
                return providers[provider](**kwargs)
            except Exception:
                continue
        
        # If no provider is available, use dummy
        return DummyLLMProvider()
    
    # Specific provider
    if provider_name in providers:
        return providers[provider_name](**kwargs)
    
    # Unknown provider
    logger.warning(f"Unknown provider: {provider_name}. Using dummy provider.")
    return DummyLLMProvider()

def get_llm_response(prompt: str, 
                    model: str = "default", 
                    provider: str = "default",
                    temperature: float = 0.7,
                    max_tokens: int = 500,
                    **kwargs) -> str:
    """
    Get a response from an LLM.
    
    Args:
        prompt (str): Prompt for the LLM
        model (str, optional): Model to use. Defaults to "default".
        provider (str, optional): Provider to use. Defaults to "default".
        temperature (float, optional): Temperature parameter. Defaults to 0.7.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 500.
        **kwargs: Additional arguments for the LLM
        
    Returns:
        str: Generated text
    """
    # If model is specified with provider prefix (e.g., "openai:gpt-4"),
    # extract the provider and model
    if ":" in model and provider == "default":
        provider, model = model.split(":", 1)
    
    # Get provider
    llm_provider = get_llm_provider(provider, model=model, **kwargs)
    
    # Generate text
    return llm_provider.generate(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    ) 