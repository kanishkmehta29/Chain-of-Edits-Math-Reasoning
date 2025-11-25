# Quick Start Guide - Gemini Solver

## Get Your API Key

1. Go to https://makersuite.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy the key

## Setup

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env and paste your API key
# GEMINI_API_KEY=paste_your_key_here

# 3. Install new dependencies (if not already done)
pip install google-generativeai python-dotenv
```

## Try It Out

```bash
# Watch Gemini solve a problem
python examples/demo_gemini_solver.py

# Run full evaluation
python examples/demo_gemini_evaluation.py

# Solve custom problem
python examples/solve_cli.py \
  --problem "Solve 2x + 5 = 13" \
  --initial "x = 4\nAnswer: 4" \
  --truth "Subtract 5: 2x = 8\nDivide by 2: x = 4\nAnswer: 4"
```

## What It Does

The Gemini agent:
1. Looks at the corrupted solution
2. Generates an edit command (e.g., "REPL 1 >>>Let x = 5")
3. Applies the edit
4. Checks if correct
5. Repeats until solved

## Files Created

- `src/agents/config.py` - API key management
- `src/agents/gemini_agent.py` - Main solver agent
- `examples/demo_gemini_solver.py` - Interactive demo
- `examples/demo_gemini_evaluation.py` - Performance benchmark
- `examples/solve_cli.py` - Command-line tool
- `.env.example` - Configuration template

See `walkthrough.md` for full details!
