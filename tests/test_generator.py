import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core import MathTask
from data import SyntheticDemoGenerator


class TestSyntheticDemoGenerator(unittest.TestCase):
    """Test demonstration generator."""
    
    def setUp(self):
        """Set up test tasks."""
        self.tasks = [
            MathTask(
                task_id="gen_test_001",
                problem_text="Test problem",
                ground_truth="Line 1\\nLine 2\\nAnswer: 5",
                initial_solutions=[]
            )
        ]
        self.generator = SyntheticDemoGenerator(self.tasks)
    
    def test_generate_demo(self):
        """Test single demo generation."""
        demo = self.generator.generate_demo(self.tasks[0], num_corruptions=2)
        self.assertIn('task_id', demo)
        self.assertIn('actions', demo)
        self.assertIn('states', demo)
        self.assertGreater(len(demo['actions']), 0)
    
    def test_generate_dataset(self):
        """Test dataset generation."""
        dataset = self.generator.generate_dataset(demos_per_task=3)
        self.assertEqual(len(dataset), 3)
        for demo in dataset:
            self.assertIn('task_id', demo)
            self.assertIn('actions', demo)
    
    def test_introduce_typo(self):
        """Test typo introduction."""
        original = "hello world"
        typo = self.generator._introduce_typo(original)
        # Should be different (usually, though might be same by chance)
        self.assertIsInstance(typo, str)
        self.assertEqual(len(typo), len(original))


if __name__ == '__main__':
    unittest.main()