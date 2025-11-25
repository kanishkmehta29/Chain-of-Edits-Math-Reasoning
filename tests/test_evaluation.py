import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core import MathTask
from evaluation import EvaluationHarness


class TestEvaluationHarness(unittest.TestCase):
    """Test evaluation harness."""
    
    def setUp(self):
        """Set up test tasks."""
        self.task = MathTask(
            task_id="eval_test_001",
            problem_text="Test",
            ground_truth="Answer: 10",
            initial_solutions=["Answer: 5"]
        )
        self.harness = EvaluationHarness([self.task])
    
    def test_evaluate_successful_sequence(self):
        """Test evaluating successful sequence."""
        actions = ["REPL 1 >>>Answer: 10", "EXIT"]
        result = self.harness.evaluate_sequence(
            self.task,
            "Answer: 5",
            actions
        )
        self.assertTrue(result['success'])
        self.assertEqual(result['task_id'], "eval_test_001")
    
    def test_evaluate_failed_sequence(self):
        """Test evaluating failed sequence."""
        actions = ["REPL 1 >>>Answer: 999", "EXIT"]
        result = self.harness.evaluate_sequence(
            self.task,
            "Answer: 5",
            actions
        )
        self.assertFalse(result['success'])
    
    def test_evaluate_all(self):
        """Test batch evaluation."""
        demos = [
            {
                'task_id': 'eval_test_001',
                'initial_state': 'Answer: 5',
                'actions': ['REPL 1 >>>Answer: 10', 'EXIT']
            }
        ]
        stats = self.harness.evaluate_all(demos)
        self.assertEqual(stats['total_evaluations'], 1)
        self.assertIn('success_rate', stats)


if __name__ == '__main__':
    unittest.main()