"""
Configuration for local LLM agents.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch


@dataclass
class LocalLLMConfig:
    """Configuration for local language model."""
    
    model_name_or_path: str = "microsoft/Phi-3.5-mini-instruct"
    device: Optional[str] = None  # auto-detect if None
    temperature: float = 0.7
    max_new_tokens: int = 100
    top_p: float = 0.9
    do_sample: bool = True
    
    # Quantization options (4-bit reduces memory by 75%)
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    
    # LoRA adapter path (for fine-tuned models)
    adapter_path: Optional[str] = None
    
    def __post_init__(self):
        """Auto-detect device if not specified."""
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"
    
    @classmethod
    def from_finetuned(cls, adapter_path: str, **kwargs) -> 'LocalLLMConfig':
        """
        Create config for fine-tuned model.
        
        Args:
            adapter_path: Path to LoRA adapter weights
            **kwargs: Additional config parameters
        """
        return cls(adapter_path=adapter_path, **kwargs)
    
    @classmethod
    def lightweight(cls, **kwargs) -> 'LocalLLMConfig':
        """
        Create config optimized for lightweight inference.
        Uses 4-bit quantization to reduce memory usage.
        """
        return cls(
            load_in_4bit=True,
            **kwargs
        )
