# Setup Instructions - Local LLM

## ✅ Python Version Issue RESOLVED!

The local LLM integration is now **fully working** with a conda environment using Python 3.12.

## Quick Setup

### 1. Activate Conda Environment

```bash
conda activate coe_math
```

### 2. Verify Installation

```bash
python -c "import torch; import transformers; print(' PyTorch:', torch.__version__)"
```

Expected output:
```
✓ PyTorch: 2.9.1
✓ Transformers: 4.x.x
✓ Device: MPS (or CUDA/CPU)
```

### 3. Test Local LLM

```bash
# Interactive demo (downloads ~3.8GB model on first run)
python examples/demo_local_solver.py

# Or use CLI
python examples/solve_cli.py \
  --model-type local \
  --problem "Solve for x: 2x + 5 = 13" \
  --initial "x = 4\nAnswer: 4" \
  --truth "x = 4"
```

## Environment Details

- **Conda Env:** `coe_math`
- **Python:** 3.12.12
- **PyTorch:** 2.9.1
- **Device:** Auto-detected (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)

## What's Installed

✅ PyTorch 2.9.1 (with Apple Silicon MPS support)
✅ Transformers (Hugging Face)
✅ PEFT (LoRA fine-tuning)
✅ Datasets
✅ Accelerate
✅ Safetensors
✅ Google Generative AI (Gemini)
✅ All project dependencies

## Training Data

Training data already generated:
- **Location:** `data/training_data.json`
- **Examples:** 1,156
- **Size:** 967 KB

## Usage

### Option 1: Gemini API (Cloud)
```bash
conda activate coe_math
python examples/solve_cli.py --model-type gemini --problem "..." --initial "..." --truth "..."
```

### Option 2: Local Phi-3.5-mini (On-Device)
```bash
conda activate coe_math
python examples/solve_cli.py --model-type local --problem "..." --initial "..." --truth "..."
```

### Option 3: Fine-Tuned Local Model
```bash
# After fine-tuning (see scripts/finetune_local_llm.py)
python examples/solve_cli.py \
  --model-type local \
  --adapter-path models/phi3-coe-finetuned \
  --problem "..." --initial "..." --truth "..."
```

## Memory Optimization

If you encounter memory issues:

```bash
python examples/solve_cli.py \
  --model-type local \
  --load-in-4bit \  # Reduces memory by 75%
  --problem "..." --initial "..." --truth "..."
```

## Old Environment

The old Python 3.13 venv has been backed up to `venv_old_python313` and can be removed:

```bash
rm -rf venv_old_python313
```

## Activation Reminder

Always activate the conda environment before running:
```bash
conda activate coe_math
```

To deactivate:
```bash
conda deactivate
```
