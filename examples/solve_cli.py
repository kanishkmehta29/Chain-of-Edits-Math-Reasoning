#!/usr/bin/env python3
"""
Command-line tool for solving math problems with Gemini.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agents import GeminiConfig, GeminiCoEAgent


def main():
    parser = argparse.ArgumentParser(
        description='Solve math problems using LLM with Chain-of-Edits'
    )
    parser.add_argument(
        '--problem',
        type=str,
        required=True,
        help='Math problem to solve'
    )
    parser.add_argument(
        '--initial',
        type=str,
        required=True,
        help='Initial (corrupted) solution'
    )
    parser.add_argument(
        '--truth',
        type=str,
        help='Expected correct solution (optional, for verification)'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=15,
        help='Maximum edit steps (default: 15)'
    )
    
    # Model selection
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['gemini', 'local'],
        default='gemini',
        help='Type of model to use: gemini (API) or local (downloaded)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model name (for Gemini: gemini-2.0-flash-lite, for Local: microsoft/Phi-3.5-mini-instruct)'
    )
    parser.add_argument(
        '--adapter-path',
        type=str,
        help='Path to fine-tuned LoRA adapter (for local models only)'
    )
    parser.add_argument(
        '--load-in-4bit',
        action='store_true',
        help='Load local model in 4-bit quantization (saves memory)'
    )
    
    # Gemini-specific
    parser.add_argument(
        '--api-key',
        type=str,
        help='Gemini API key (optional, can use .env instead)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    # Create agent based on model type
    try:
        if args.model_type == 'gemini':
            from agents import create_gemini_agent
            
            if args.api_key:
                agent = create_gemini_agent(
                    api_key=args.api_key,
                    model=args.model or 'gemini-2.0-flash-lite'
                )
            else:
                from agents import GeminiConfig, GeminiCoEAgent
                config = GeminiConfig.from_env()
                if args.model:
                    config.model = args.model
                agent = GeminiCoEAgent(config)
            
            model_name = agent.config.model
            
        else:  # local
            from agents import create_local_agent
            
            agent = create_local_agent(
                model_path=args.model or 'microsoft/Phi-3.5-mini-instruct',
                adapter_path=args.adapter_path,
                load_in_4bit=args.load_in_4bit
            )
            
            model_name = agent.config.model_name_or_path
            if args.adapter_path:
                model_name += f" (fine-tuned: {args.adapter_path})"
    
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        if args.model_type == 'gemini':
            print("Please provide API key via --api-key or .env file", file=sys.stderr)
        sys.exit(1)

    
    if not args.quiet:
        print("=" * 70)
        print("LLM Math Solver (Chain-of-Edits)")
        print("=" * 70)
        print(f"\nModel Type: {args.model_type}")
        print(f"Model: {model_name}")
        print()
    
    # Prepare initial state
    initial_lines = args.initial.split('\\n')
    initial_state = '\n'.join(initial_lines)
    initial_feedback = "Please fix this solution"
    
    # Set ground truth
    # If not provided, use a generic placeholder (free-form solving)
    if args.truth:
        ground_truth = args.truth
    else:
        # Extract just the final answer for minimal verification
        ground_truth = "Correct solution with proper steps and final answer"

    
    # Solve
    success, num_steps, edit_history = agent.solve_problem(
        problem=args.problem,
        ground_truth=ground_truth,
        initial_state=initial_state,
        initial_feedback=initial_feedback,
        max_steps=args.max_steps,
        verbose=not args.quiet
    )
    
    # Print summary
    if args.quiet:
        print(f"{'SUCCESS' if success else 'FAILED'},{num_steps},{len(edit_history)}")
    else:
        print()
        print("=" * 70)
        print(f"Result: {'✓ SOLVED' if success else '✗ FAILED'}")
        print(f"Steps: {num_steps}")
        print(f"Edits: {len(edit_history)}")
        print("=" * 70)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
