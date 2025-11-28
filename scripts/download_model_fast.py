#!/usr/bin/env python3
"""
Download Phi-3.5-mini with progress bar.
"""

import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

print("="*70)
print("Downloading Phi-3.5-mini with Progress Tracking")
print("="*70)

print("\n📥 Starting download...")
print("  Model: microsoft/Phi-3.5-mini-instruct")
print("  Size: ~3.8GB")
print("  This may take 5-10 minutes depending on your connection\n")

from transformers import AutoModelForCausalLM, AutoTokenizer

# Download with progress
print("Downloading model files...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu"  # Just download, don't load to GPU
    )
    
    print("\n✅ Model downloaded successfully!")
    print("  Cached in: ~/.cache/huggingface/hub/")
    
    print("\nDownloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        trust_remote_code=True
    )
    
    print("✅ Tokenizer downloaded!")
    
    # Print model info
    print("\n📊 Model Info:")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count / 1e9:.2f}B")
    print(f"  Model type: {model.config.model_type}")
    
    print("\n🎉 Download complete! Ready to use.")
    print("\nNext steps:")
    print("  python examples/demo_local_solver.py")
    print("  OR")
    print("  python examples/solve_cli.py --model-type local --problem '...' --initial '...' --truth '...'")
    
except KeyboardInterrupt:
    print("\n\n❌ Download cancelled by user")
    sys.exit(1)
except Exception as e:
    print(f"\n\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("  - Check internet connection")
    print("  - Try again (Hugging Face servers may be slow)")
    print("  - Use: huggingface-cli download microsoft/Phi-3.5-mini-instruct")
    sys.exit(1)
