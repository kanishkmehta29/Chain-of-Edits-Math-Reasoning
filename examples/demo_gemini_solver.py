import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agents import GeminiConfig, GeminiCoEAgent
from data import create_example_tasks


def main():
    """Demonstrate Gemini solving math problems with Chain-of-Edits."""
    print("=" * 70)
    print("CoE Math Reasoning - Gemini Solver Demo")
    print("=" * 70)
    print()
    
    # Load configuration
    try:
        config = GeminiConfig.from_env()
        print(f"✓ Loaded configuration")
        print(f"  Model: {config.model}")
        print(f"  Temperature: {config.temperature}")
        print()
    except ValueError as e:
        print(f"✗ Configuration Error: {e}")
        print()
        print("Please set up your Gemini API key:")
        print("1. Copy .env.example to .env")
        print("2. Add your API key to .env")
        print("3. Run this script again")
        return
    
    # Create agent
    agent = GeminiCoEAgent(config)
    print("✓ Initialized Gemini agent")
    print()
    
    # Load tasks
    tasks = create_example_tasks()
    task = tasks[1]  # Use the algebra problem
    
    print("=" * 70)
    print("PROBLEM")
    print("=" * 70)
    print(task.problem_text)
    print()
    print("Expected Solution:")
    print(task.ground_truth)
    print()
    
    print("=" * 70)
    print("INITIAL CORRUPTED STATE")
    print("=" * 70)
    
    # Get initial solution lines
    initial_lines = task.initial_solutions[0].split('\n')
    for i, line in enumerate(initial_lines, 1):
        print(f"L {i} {line}")
    print()
    
    # Prepare initial state
    initial_state = '\n'.join(initial_lines)
    initial_feedback = f"Test failed: expected {task.ground_truth}, got incorrect solution"
    
    print("=" * 70)
    print("SOLVING WITH GEMINI...")
    print("=" * 70)
    
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
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Success: {'✓' if success else '✗'}")
    print(f"Steps taken: {num_steps}")
    print(f"Edit commands used: {len(edit_history)}")
    print()
    print("Edit History:")
    for i, cmd in enumerate(edit_history, 1):
        print(f"  {i}. {cmd}")
    print()
    
    # Print stats
    stats = agent.get_stats()
    print("Agent Statistics:")
    print(f"  API Requests: {stats['num_requests']}")
    print(f"  Model: {stats['model']}")
    print()


if __name__ == "__main__":
    main()
