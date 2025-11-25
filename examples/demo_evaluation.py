import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import create_example_tasks, SyntheticDemoGenerator
from evaluation import EvaluationHarness


def main():
    """Demonstrate evaluation harness."""
    print("=" * 70)
    print("CoE Math Reasoning - Evaluation Demo")
    print("=" * 70)
    print()
    
    # Create tasks
    tasks = create_example_tasks()
    
    # Generate demonstrations
    print("Generating demonstrations...")
    generator = SyntheticDemoGenerator(tasks)
    dataset = generator.generate_dataset(demos_per_task=2)
    print(f"Generated {len(dataset)} demonstrations\\n")
    
    # Evaluate
    print("Evaluating demonstrations...")
    evaluator = EvaluationHarness(tasks)
    stats = evaluator.evaluate_all(dataset)
    
    print(f"\\nEvaluation Results:")
    print(f"  Total: {stats['total_evaluations']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Avg actions per task: {stats['avg_actions_per_task']:.1f}")
    
    # Show some individual results
    print(f"\\nSample individual results:")
    for i, result in enumerate(stats['results'][:3], 1):
        print(f"\\n  Task {i} ({result['task_id']}):")
        print(f"    Success: {result['success']}")
        print(f"    Actions: {result['num_actions']}")
        if result['error']:
            print(f"    Error: {result['error']}")


if __name__ == "__main__":
    main()