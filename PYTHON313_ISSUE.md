# Python 3.13 Compatibility Note

## Issue

Python 3.13.3 is currently installed in the virtual environment. However, **PyTorch does not yet support Python 3.13**.

## Solutions

### Option 1: Use Python 3.11 or 3.12 (Recommended)

Create a new virtual environment with Python 3.11 or 3.12:

```bash
# Remove current venv
rm -rf venv

# Create new venv with Python 3.11 or 3.12
python3.11 -m venv venv  # or python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
pip install transformers torch peft datasets accelerate safetensors
```

### Option 2: Use Gemini Only (Current Setup)

The Gemini API integration works perfectly without local LLM support:

```bash
python examples/solve_cli.py --model-type gemini --problem "..." --initial "..." --truth "..."
```

### Option 3: Wait for PyTorch 3.13 Support

PyTorch team is working on Python 3.13 support. Check: https://github.com/pytorch/pytorch/issues

## Training Data Generation (Non-PyTorch)

You can still generate training data without PyTorch installed:

```bash
python scripts/generate_training_data.py
```

This uses only the existing `SyntheticDemoGenerator` which doesn't require PyTorch.

## Current Status

✅ Gemini integration - **FULLY WORKING**
✅ Training data generation - **WORKS WITHOUT PYTORCH**  
✅ Local LLM infrastructure - **CODE READY**
⚠️ Local LLM inference - **REQUIRES PYTHON 3.11/3.12**
⚠️ Fine-tuning - **REQUIRES PYTHON 3.11/3.12**

## Recommendation

For now, continue using Gemini API which works perfectly. When you need local LLM:
1. Create new venv with Python 3.11/3.12
2. Install PyTorch and other deps
3. Download Phi-3.5-mini
4. Generate training data
5. Fine-tune model
