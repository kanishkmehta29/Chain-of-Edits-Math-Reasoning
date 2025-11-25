import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core import DSLParser, EditCommandType, EditAction


class TestDSLParser(unittest.TestCase):
    """Test DSL parser functionality."""
    
    def setUp(self):
        self.parser = DSLParser()
    
    def test_parse_exit(self):
        """Test parsing EXIT command."""
        action = self.parser.parse("EXIT")
        self.assertEqual(action.command_type, EditCommandType.EXIT)
    
    def test_parse_addl(self):
        """Test parsing ADDL command."""
        action = self.parser.parse("ADDL 3 >>>x = 5")
        self.assertEqual(action.command_type, EditCommandType.ADDL)
        self.assertEqual(action.line_number, 3)
        self.assertEqual(action.content, "x = 5")
    
    def test_parse_repl(self):
        """Test parsing REPL command."""
        action = self.parser.parse("REPL 2 >>>y = 10")
        self.assertEqual(action.command_type, EditCommandType.REPL)
        self.assertEqual(action.line_number, 2)
        self.assertEqual(action.content, "y = 10")
    
    def test_parse_dell(self):
        """Test parsing DELL command."""
        action = self.parser.parse("DELL 4")
        self.assertEqual(action.command_type, EditCommandType.DELL)
        self.assertEqual(action.line_number, 4)
    
    def test_parse_repw(self):
        """Test parsing REPW command."""
        action = self.parser.parse("REPW 1 >>>old >>>new")
        self.assertEqual(action.command_type, EditCommandType.REPW)
        self.assertEqual(action.line_number, 1)
        self.assertEqual(action.old_word, "old")
        self.assertEqual(action.new_word, "new")
    
    def test_invalid_command(self):
        """Test parsing invalid command."""
        with self.assertRaises(ValueError):
            self.parser.parse("INVALID")
    
    def test_action_to_string(self):
        """Test converting action back to string."""
        original = "ADDL 3 >>>test line"
        action = self.parser.parse(original)
        reconstructed = action.to_string()
        self.assertEqual(original, reconstructed)


if __name__ == '__main__':
    unittest.main()