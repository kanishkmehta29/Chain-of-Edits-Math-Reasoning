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
        description='Solve math problems using Gemini with Chain-of-Edits'
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
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.5-flash',
        help='Gemini model to use (default: gemini-2.5-flash)'
    )
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
    
    # Load configuration
    try:
        if args.api_key:
            config = GeminiConfig.from_api_key(
                api_key=args.api_key,
                model=args.model
            )
        else:
            config = GeminiConfig.from_env()
            # Override model if specified
            if args.model != 'gemini-2.5-flash':
                config.model = args.model
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please provide API key via --api-key or .env file", file=sys.stderr)
        sys.exit(1)
    
    # Create agent
    agent = GeminiCoEAgent(config)
    
    if not args.quiet:
        print("=" * 70)
        print("Gemini Math Solver (Chain-of-Edits)")
        print("=" * 70)
        print(f"\nProblem: {args.problem}")
        print(f"Model: {config.model}")
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
