import json
from typing import List
from pathlib import Path

from core import MathTask


def create_example_tasks() -> List[MathTask]:
    """Create a set of example math tasks."""
    tasks = [
        MathTask(
            task_id="task_001",
            problem_text="John has 3 apples and buys 5 more. How many apples does he have?",
            ground_truth="Let initial = 3\\nLet bought = 5\\nAnswer: 8",
            initial_solutions=["Let initial = 3\\nAnswer: 5"]
        ),
        MathTask(
            task_id="task_002",
            problem_text="Solve for x: 2x + 5 = 13",
            ground_truth="Subtract 5: 2x = 8\\nDivide by 2: x = 4\\nAnswer: 4",
            initial_solutions=["2x = 13\\nx = 6.5\\nAnswer: 6.5"]
        ),
        MathTask(
            task_id="task_003",
            problem_text="Simplify: (x + 2)^2",
            ground_truth="Expand: x^2 + 4x + 4\\nAnswer: x^2 + 4x + 4",
            initial_solutions=["Result: x^2 + 4\\nAnswer: x^2 + 4"]
        ),
        MathTask(
            task_id="task_004",
            problem_text="What is 15% of 200?",
            ground_truth="Calculate: 0.15 * 200 = 30\\nAnswer: 30",
            initial_solutions=["Result: 15 * 200\\nAnswer: 3000"]
        ),
        MathTask(
            task_id="task_005",
            problem_text="A rectangle has length 8 and width 5. What is its area?",
            ground_truth="Area = length * width\\nArea = 8 * 5\\nAnswer: 40",
            initial_solutions=["Area = 8 + 5\\nAnswer: 13"]
        ),
    ]
    return tasks


def load_tasks_from_json(filepath: Path) -> List[MathTask]:
    """Load tasks from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [MathTask.from_dict(task_dict) for task_dict in data]


def save_tasks_to_json(tasks: List[MathTask], filepath: Path):
    """Save tasks to a JSON file."""
    data = [task.to_dict() for task in tasks]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)