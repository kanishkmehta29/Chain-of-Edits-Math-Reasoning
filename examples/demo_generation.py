import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import create_example_tasks, SyntheticDemoGenerator


def main():
    """Demonstrate synthetic demo generation."""
    print("=" * 70)
    print("CoE Math Reasoning - Demo Generation")
    print("=" * 70)
    print()
    
    # Create tasks
    tasks = create_example_tasks()
    
    # Generate demonstrations
    generator = SyntheticDemoGenerator(tasks)
    demo = generator.generate_demo(tasks[0], num_corruptions=3)
    
    print(f"Task: {demo['problem']}")
    print(f"\\nInitial (corrupted) state:")
    print(demo['initial_state'])
    print(f"\\nTarget (correct) state:")
    print(demo['target_state'])
    print(f"\\nRepair actions: {len(demo['actions'])}")
    for i, action in enumerate(demo['actions'], 1):
        print(f"  {i}. {action}")
    
    print()
    print("Full trace with states:")
    for i, (state, action) in enumerate(zip(demo['states'], demo['actions'])):
        print(f"\\n--- State {i} ---")
        print(state)
        print(f"\\nAction: {action}")
    
    # Generate dataset
    print("\\n" + "=" * 70)
    print("Generating full dataset...")
    dataset = generator.generate_dataset(demos_per_task=3)
    print(f"Generated {len(dataset)} demonstrations")
    print(f"Tasks covered: {len(tasks)}")
    print(f"Avg demos per task: {len(dataset) / len(tasks):.1f}")


if __name__ == "__main__":
    main()