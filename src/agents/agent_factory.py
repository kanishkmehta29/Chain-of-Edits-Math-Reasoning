"""
Agent factory for creating CoE agents.
"""

from typing import Union, Optional
from .gemini_agent import GeminiCoEAgent
from .local_llm_agent import LocalLLMAgent
from .config import GeminiConfig
from .local_llm_config import LocalLLMConfig


def create_agent(
    model_type: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    adapter_path: Optional[str] = None,
    **kwargs
) -> Union[GeminiCoEAgent, LocalLLMAgent]:
    """
    Create a Chain-of-Edits agent.
    
    Args:
        model_type: Type of model ("gemini" or "local")
        api_key: Gemini API key (for gemini type)
        model_name: Model name/path (optional)
        adapter_path: Path to fine-tuned adapter (for local type)
        **kwargs: Additional config parameters
        
    Returns:
        Agent instance (GeminiCoEAgent or LocalLLMAgent)
        
    Examples:
        # Gemini agent
        agent = create_agent("gemini", api_key="your_key")
        
        # Local base model
        agent = create_agent("local")
        
        # Local fine-tuned model
        agent = create_agent("local", adapter_path="models/phi3-coe-finetuned")
    """
    if model_type == "gemini":
        if api_key:
            config = GeminiConfig.from_api_key(api_key, model=model_name or "gemini-2.0-flash-lite")
        else:
            config = GeminiConfig.from_env()
            if model_name:
                config.model = model_name
        
        return GeminiCoEAgent(config)
    
    elif model_type == "local":
        if adapter_path:
            config = LocalLLMConfig.from_finetuned(adapter_path, **kwargs)
        else:
            config = LocalLLMConfig(
                model_name_or_path=model_name or "microsoft/Phi-3.5-mini-instruct",
                **kwargs
            )
        
        return LocalLLMAgent(config)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'gemini' or 'local'")


def create_gemini_agent(
    api_key: Optional[str] = None,
    model: str = "gemini-2.0-flash-lite",
    **kwargs
) -> GeminiCoEAgent:
    """Convenience function to create Gemini agent."""
    return create_agent("gemini", api_key=api_key, model_name=model, **kwargs)


def create_local_agent(
    model_path: str = "microsoft/Phi-3.5-mini-instruct",
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = False,
    **kwargs
) -> LocalLLMAgent:
    """Convenience function to create local LLM agent."""
    return create_agent(
        "local",
        model_name=model_path,
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
        **kwargs
    )
