from dataclasses import dataclass
from typing import Optional
from enum import Enum

from core import EditAction, EditCommandType


class CorruptionType(Enum):
    """Types of corruptions for generating demonstrations."""
    DELETE_LINE = "delete_line"
    ADD_WRONG_LINE = "add_wrong_line"
    REPLACE_LINE = "replace_line"
    INTRODUCE_TYPO = "introduce_typo"
    SWAP_OPERAND = "swap_operand"


@dataclass
class Corruption:
    """Represents a corruption applied to a solution."""
    corruption_type: CorruptionType
    line_number: int
    original_content: Optional[str] = None
    corrupted_content: Optional[str] = None
    
    def reverse_action(self) -> EditAction:
        """Generate the DSL action that reverses this corruption."""
        if self.corruption_type == CorruptionType.DELETE_LINE:
            # Reverse: Add the line back
            return EditAction(
                command_type=EditCommandType.ADDL,
                line_number=self.line_number,
                content=self.original_content
            )
        
        elif self.corruption_type == CorruptionType.ADD_WRONG_LINE:
            # Reverse: Delete the added line
            return EditAction(
                command_type=EditCommandType.DELL,
                line_number=self.line_number
            )
        
        elif self.corruption_type in [CorruptionType.REPLACE_LINE, 
                                       CorruptionType.INTRODUCE_TYPO,
                                       CorruptionType.SWAP_OPERAND]:
            # Reverse: Replace with original
            return EditAction(
                command_type=EditCommandType.REPL,
                line_number=self.line_number,
                content=self.original_content
            )
        
        raise ValueError(f"Unknown corruption type: {self.corruption_type}")