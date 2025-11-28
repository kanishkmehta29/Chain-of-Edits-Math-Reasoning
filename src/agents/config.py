"""
Configuration management for Gemini API integration.
"""

import os
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class GeminiConfig:
    """Configuration for Gemini API."""
    api_key: str
    model: str = "gemini-2.0-flash-lite"
    temperature: float = 0.7
    max_output_tokens: int = 1024
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> 'GeminiConfig':
        """
        Load configuration from environment variables.
        
        Args:
            env_file: Path to .env file (optional)
            
        Returns:
            GeminiConfig instance
            
        Raises:
            ValueError: If API key is not found
        """
        # Load from .env file if it exists
        if env_file:
            env_path = Path(env_file)
            if env_path.exists():
                load_dotenv(env_path)
        else:
            # Try to load from default .env in project root
            load_dotenv()
        
        # Get API key (required)
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Please set it in .env file or as environment variable."
            )
        
        # Get optional parameters
        model = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-lite')
        temperature = float(os.getenv('GEMINI_TEMPERATURE', '0.7'))
        max_output_tokens = int(os.getenv('GEMINI_MAX_OUTPUT_TOKENS', '1024'))
        
        return cls(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
    
    @classmethod
    def from_api_key(cls, api_key: str, **kwargs) -> 'GeminiConfig':
        """
        Create configuration from API key with optional parameters.
        
        Args:
            api_key: Gemini API key
            **kwargs: Additional configuration parameters
            
        Returns:
            GeminiConfig instance
        """
        return cls(api_key=api_key, **kwargs)
