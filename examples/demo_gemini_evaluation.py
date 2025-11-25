import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agents import GeminiConfig, GeminiCoEAgent
from data import create_example_tasks


def main():
    """Evaluate Gemini's performance on multiple tasks."""
    print("=" * 70)
    print("CoE Math Reasoning - Gemini Evaluation")
    print("=" * 70)
    print()
    
    # Load configuration
    try:
        config = GeminiConfig.from_env()
        print(f"✓ Configuration loaded")
        print(f"  Model: {config.model}")
        print()
    except ValueError as e:
        print(f"✗ Error: {e}")
        print("Please configure your API key in .env file")
        return
    
    # Create agent
    agent = GeminiCoEAgent(config)
    
    # Load tasks
    tasks = create_example_tasks()
    print(f"Loaded {len(tasks)} test tasks")
    print()
    
    # Evaluate on all tasks
    results = []
    
    for i, task in enumerate(tasks, 1):
        print("=" * 70)
        print(f"Task {i}/{len(tasks)}: {task.problem_text}")
        print("=" * 70)
        
        initial_lines = task.initial_solutions[0].split('\n')
        initial_state = '\n'.join(initial_lines)
        initial_feedback = f"Test failed: expected {task.ground_truth}"
        
        success, num_steps, edit_history = agent.solve_problem(
            problem=task.problem_text,
            ground_truth=task.ground_truth,
            initial_state=initial_state,
            initial_feedback=initial_feedback,
            max_steps=15,
            verbose=False  # Suppress detailed output
        )
        
        result = {
            'task_id': task.task_id,
            'problem': task.problem_text,
            'success': success,
            'num_steps': num_steps,
            'num_edits': len(edit_history),
            'edits': edit_history
        }
        results.append(result)
        
        status = "✓ SOLVED" if success else "✗ FAILED"
        print(f"{status} in {num_steps} steps ({len(edit_history)} edits)")
        print()
    
    # Print summary
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print()
    
    num_success = sum(1 for r in results if r['success'])
    success_rate = (num_success / len(results)) * 100
    
    avg_steps = sum(r['num_steps'] for r in results if r['success']) / max(num_success, 1)
    avg_edits = sum(r['num_edits'] for r in results if r['success']) / max(num_success, 1)
    
    print(f"Tasks Attempted: {len(results)}")
    print(f"Tasks Solved: {num_success}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Avg Steps (successful): {avg_steps:.1f}")
    print(f"Avg Edits (successful): {avg_edits:.1f}")
    print()
    
    # Per-task breakdown
    print("Detailed Results:")
    print("-" * 70)
    for i, result in enumerate(results, 1):
        status = "✓" if result['success'] else "✗"
        print(f"{i}. {status} {result['problem'][:50]}...")
        print(f"   Steps: {result['num_steps']}, Edits: {result['num_edits']}")
    print()
    
    # Agent stats
    stats = agent.get_stats()
    print("Agent Statistics:")
    print(f"  Total API Requests: {stats['num_requests']}")
    print(f"  Model Used: {stats['model']}")
    print()


if __name__ == "__main__":
    main()
