from typing import Dict, Optional, Literal, Any
import os
import json
from abc import ABC, abstractmethod
import httpx

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Get completion from LLM"""
        pass

class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
    
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content

class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        from ollama import chat
        self.model = model
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        # Use HTTP to localhost:11434 (Ollama) instead of litellm library
        resp = httpx.post(
            "http://localhost:11434/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "response_format": response_format,
            },
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

class LiteLLMGatewayController(BaseLLMController):
    """Controller for LiteLLM proxy gateway.

    Uses the LiteLLM gateway at localhost:4000 for unified model access.
    Model aliases:
    - memory-evolution: Gemini Pro for memory evolution tasks
    - metadata: Gemini Flash for fast metadata extraction
    """

    def __init__(
        self,
        model: str = "memory-evolution",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """Initialize LiteLLM gateway controller.

        Args:
            model: Model alias (memory-evolution, metadata) or full model name
            base_url: LiteLLM gateway URL (default: http://localhost:4000)
            api_key: API key for the gateway (if required)
        """
        self.model = model
        self.base_url = base_url or os.getenv("LITELLM_GATEWAY_URL", "http://localhost:4000")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY", "")

    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        """Get completion from LiteLLM gateway via HTTP."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            # Reasoning models (gemini-2.5-flash) spend thinking tokens inside
            # max_tokens (observed ~600+ per evolution call), so 1000 truncated
            # the JSON mid-output on real prompts and every parse failed.
            "max_tokens": 4000,
        }

        if response_format:
            payload["response_format"] = response_format

        resp = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class LLMController:
    """LLM-based controller for memory metadata generation"""
    def __init__(self,
                 backend: Literal["openai", "ollama", "litellm"] = "openai",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        elif backend == "litellm":
            self.llm = LiteLLMGatewayController(model, base_url, api_key)
        else:
            raise ValueError("Backend must be one of: 'openai', 'ollama', 'litellm'")
            
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)
