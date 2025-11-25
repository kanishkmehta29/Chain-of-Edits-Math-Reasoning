import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core import DSLParser
from data import create_example_tasks
from environment import MathEditEnvironment


def main():
    """Demonstrate basic environment usage."""
    print("=" * 70)
    print("CoE Math Reasoning - Basic Demo")
    print("=" * 70)
    print()
    
    # Create tasks
    tasks = create_example_tasks()
    task = tasks[1]  # 2x + 5 = 13
    
    print(f"Task: {task.problem_text}")
    print(f"Ground truth:\\n{task.ground_truth}")
    print()
    
    # Create environment
    env = MathEditEnvironment(task, task.initial_solutions[0])
    state = env.reset()
    
    print("Initial state:")
    print(env.format_state_for_llm())
    print()
    
    # Apply corrections
    parser = DSLParser()
    corrections = [
        "REPL 1 >>>Subtract 5 from both sides: 2x = 8",
        "REPL 2 >>>Divide by 2: x = 4",
        "REPL 3 >>>Answer: 4"
    ]
    
    print("Applying corrections:")
    for i, cmd in enumerate(corrections, 1):
        print(f"\\nStep {i}: {cmd}")
        action = parser.parse(cmd)
        state, done, error = env.step(action)
        
        if error:
            print(f"Error: {error}")
            break
        
        print(env.format_state_for_llm())
        
        if done:
            print(f"\\n✓ Task solved!" if env.is_solved() else "\\n✗ Task not solved")
            break


if __name__ == "__main__":
    main()