#!/usr/bin/env python3
"""
Fine-tune Phi-3.5-mini on Chain-of-Edits data using LoRA.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def main():
    parser = argparse.ArgumentParser(description='Fine-tune local LLM')
    parser.add_argument('--data', type=str, default='data/training_data.json', help='Path to training data')
    parser.add_argument('--model', type=str, default='microsoft/Phi-3.5-mini-instruct', help='Base model')
    parser.add_argument('--output-dir', type=str, default='models/phi3-coe-finetuned', help='Output directory')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size') # Low batch size for Mac
    parser.add_argument('--grad-accum', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Fine-tuning Phi-3.5-mini with LoRA")
    print("="*70)
    
    # 1. Load Data
    print(f"\n📚 Loading data from {args.data}...")
    with open(args.data, 'r') as f:
        raw_data = json.load(f)
    
    # Convert to HuggingFace Dataset
    # Format: Instruction + Input -> Output
    formatted_data = []
    for item in raw_data:
        # Phi-3 prompt format
        text = f"<|user|>\n{item['instruction']}\n\n{item['input']}<|end|>\n<|assistant|>\n{item['output']}<|end|>"
        formatted_data.append({"text": text})
    
    dataset = Dataset.from_list(formatted_data)
    
    # Split train/val
    dataset = dataset.train_test_split(test_size=0.1)
    print(f"  Train size: {len(dataset['train'])}")
    print(f"  Val size: {len(dataset['test'])}")
    
    # 2. Load Model & Tokenizer
    print(f"\n🤖 Loading model: {args.model}")
    
    # Quantization config for memory efficiency
    # Note: 4-bit quantization might be slow on MPS, trying standard loading first for stability on Mac
    # If OOM occurs, we can enable 4-bit
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")


    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map=None, # Load on CPU first
        torch_dtype=torch.float16
    )
    
    if device != "cpu":
        print(f"  Moving model to {device}...")
        model.to(device)
    
    # Disable cache for training (gradient checkpointing compatibility)
    model.config.use_cache = False


    
    # 3. Setup LoRA
    print("\n🔧 Setting up LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=False, # MPS doesn't support full fp16 training mixed precision well sometimes
        bf16=False,
        use_mps_device=True if device == "mps" else False,
        report_to="none"
    )
    
    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # 6. Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    # 7. Save
    print(f"\n💾 Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n✅ Fine-tuning complete!")

if __name__ == "__main__":
    main()
