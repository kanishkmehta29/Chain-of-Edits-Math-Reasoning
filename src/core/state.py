from dataclasses import dataclass
from typing import List


@dataclass
class EditorState:
    """Represents the current state of the math editor."""
    lines: List[str]  # Content lines (no line numbers)
    feedback: str  # Verifier feedback
    solved: bool  # Whether task is solved
    
    def copy(self) -> 'EditorState':
        """Create a copy of this state."""
        return EditorState(
            lines=self.lines.copy(),
            feedback=self.feedback,
            solved=self.solved
        )
    
    def __repr__(self) -> str:
        return f"EditorState(lines={len(self.lines)}, solved={self.solved})"