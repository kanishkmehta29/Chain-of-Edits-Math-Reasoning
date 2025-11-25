import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core import MathTask, DSLParser
from environment import MathEditEnvironment


class TestMathEditEnvironment(unittest.TestCase):
    """Test environment functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.task = MathTask(
            task_id="test_001",
            problem_text="Test problem",
            ground_truth="Answer: 42",
            initial_solutions=["Answer: 0"]
        )
        self.env = MathEditEnvironment(self.task, "Answer: 0")
        self.parser = DSLParser()
    
    def test_reset(self):
        """Test environment reset."""
        state = self.env.reset()
        self.assertIsNotNone(state)
        self.assertFalse(state.solved)
        self.assertEqual(len(state.lines), 1)
    
    def test_addl_command(self):
        """Test ADDL command."""
        self.env.reset()
        action = self.parser.parse("ADDL 1 >>>New line")
        state, done, error = self.env.step(action)
        self.assertEqual(len(state.lines), 2)
        self.assertEqual(state.lines[0], "New line")
        self.assertEqual(error, "")
    
    def test_repl_command(self):
        """Test REPL command."""
        self.env.reset()
        action = self.parser.parse("REPL 1 >>>Answer: 42")
        state, done, error = self.env.step(action)
        self.assertEqual(state.lines[0], "Answer: 42")
        self.assertTrue(state.solved)
    
    def test_dell_command(self):
        """Test DELL command."""
        self.env = MathEditEnvironment(self.task, "Line 1\\nLine 2")
        self.env.reset()
        action = self.parser.parse("DELL 1")
        state, done, error = self.env.step(action)
        self.assertEqual(len(state.lines), 1)
        self.assertEqual(state.lines[0], "Line 2")
    
    def test_invalid_line_number(self):
        """Test invalid line number."""
        self.env.reset()
        action = self.parser.parse("REPL 99 >>>Invalid")
        state, done, error = self.env.step(action)
        self.assertIn("out of range", error)
    
    def test_format_state(self):
        """Test state formatting."""
        self.env.reset()
        formatted = self.env.format_state_for_llm()
        self.assertIn("L 1", formatted)
        self.assertIn("***", formatted)


if __name__ == '__main__':
    unittest.main()