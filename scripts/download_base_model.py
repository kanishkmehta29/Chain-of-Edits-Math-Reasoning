#!/usr/bin/env python3
"""
Download base Phi-3.5-mini model for local use.
"""

import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description='Download Phi-3.5-mini base model')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/phi3-base',
        help='Directory to save model'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='microsoft/Phi-3.5-mini-instruct',
        help='Model name on Hugging Face'
    )
    parser.add_argument(
        '--quantize',
        choices=['4bit', '8bit', 'none'],
        default='none',
        help='Download quantized version'
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Downloading Phi-3.5-mini Base Model")
    print("="*70)
    print(f"\nModel: {args.model_name}")
    print(f"Output: {output_path}")
    print(f"Quantization: {args.quantize}")
    
    # Download model
    print("\n📥 Downloading model...")
    
    if args.quantize == 'none':
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True
        )
    else:
        from transformers import BitsAndBytesConfig
        import torch
        
        if args.quantize == '4bit':
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:  # 8bit
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Download tokenizer
    print("📥 Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    
    # Save
    print(f"\n💾 Saving to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("\n✅ Download complete!")
    print(f"  Model saved to: {output_path}")
    
    # Print model info
    print("\n📊 Model Info:")
    try:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count / 1e9:.2f}B")
    except:
        pass
    
    print("\n💡 Usage:")
    print(f"  LocalLLMConfig(model_name_or_path='{output_path}')")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
