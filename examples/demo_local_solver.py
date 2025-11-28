#!/usr/bin/env python3
"""
Demo of local LLM solver (without Gemini API).
Shows how to use Phi-3.5-mini or fine-tuned model locally.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agents import create_local_agent
from data import create_example_tasks


def main():
    print("="*70)
    print("CoE Math Reasoning - Local LLM Solver Demo")
    print("="*70)
    
    # Load configuration
    print("\n✓ Loading local model...")
    print("  Model: microsoft/Phi-3.5-mini-instruct")
    print("  Note: First run will download ~3.8GB model")
    print("  Device: Auto-detected (CUDA/MPS/CPU)")
    
    # Create local agent
    # To use fine-tuned model, add: adapter_path="models/phi3-coe-finetuned"
    # To use 4-bit quantization, add: load_in_4bit=True
    agent = create_local_agent(
        model_path="microsoft/Phi-3.5-mini-instruct",
        # adapter_path=None,  # Uncomment and set path to use fine-tuned model
        load_in_4bit=False,    # Set True to reduce memory usage
        # device="cpu"         # Removed to allow MPS (GPU) usage
    )






    
    print("\n✓ Model loaded!")
    
    # Get example task
    tasks = create_example_tasks()
    task = tasks[1]  # Algebra problem
    
    print("\n" + "="*70)
    print("PROBLEM")
    print("="*70)
    print(task.problem_text)
    print(f"\nExpected Solution:\n{task.ground_truth}")
    
    # Create corrupted initial state
    initial_state = "2x = 13\nx = 6.5\nAnswer: 6.5"
    initial_feedback = "Please fix this solution"
    
    print("\n" + "="*70)
    print("INITIAL CORRUPTED STATE")
    print("="*70)
    print(initial_state)
    
    print("\n" + "="*70)
    print("SOLVING WITH LOCAL LLM...")
    print("="*70)
    
    # Solve the problem
    success, num_steps, edit_history = agent.solve_problem(
        problem=task.problem_text,
        ground_truth=task.ground_truth,
        initial_state=initial_state,
        initial_feedback=initial_feedback,
        max_steps=15,
        verbose=True
    )
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Success: {'✓' if success else '✗'}")
    print(f"Steps taken: {num_steps}")
    print(f"Edit commands used: {len(edit_history)}")
    
    if edit_history:
        print("\nEdit History:")
        for i, cmd in enumerate(edit_history, 1):
            print(f"  {i}. {cmd}")
    
    # Print agent stats
    stats = agent.get_stats()
    print(f"\nAgent Statistics:")
    print(f"  Inference calls: {stats['num_requests']}")
    print(f"  Model: {stats['model']}")
    print(f"  Device: {stats['device']}")
    
    print("\n💡 Tips:")
    print("  - Use --load-in-4bit to reduce memory usage (75% less)")
    print("  - Fine-tune for better accuracy (see scripts/finetune_local_llm.py)")
    print("  - Run on GPU for 10x faster inference")


if __name__ == "__main__":
    main()
