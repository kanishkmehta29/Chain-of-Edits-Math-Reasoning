from typing import Tuple, Optional

from core import EditAction, EditCommandType, EditorState
from .verifier import MathVerifier


class MathEditEnvironment:
    """
    Interactive environment for editing math solutions with verification.
    """
    
    def __init__(self, task, initial_solution: str):
        """
        Initialize environment.
        
        Args:
            task: MathTask instance
            initial_solution: Initial (incorrect) solution as multi-line string
        """
        self.task = task
        self.initial_solution = initial_solution
        self.verifier = MathVerifier(task)
        self.state = None
        self.history = []
        self.max_turns = 20
        self.current_turn = 0
    
    def reset(self) -> EditorState:
        """Reset environment to initial state."""
        lines = self.initial_solution.strip().split('\n')
        is_solved, feedback = self.verifier.verify_solution(self.initial_solution)
        
        self.state = EditorState(
            lines=lines,
            feedback=feedback,
            solved=is_solved
        )
        self.history = [self.state.copy()]
        self.current_turn = 0
        
        return self.state
    
    def format_state_for_llm(self, state: Optional[EditorState] = None) -> str:
        """
        Format state as it would be shown to an LLM.
        
        Args:
            state: State to format (uses current state if None)
            
        Returns:
            Formatted string with line numbers and feedback
        """
        if state is None:
            state = self.state
        
        lines_str = []
        for i, line in enumerate(state.lines, start=1):
            lines_str.append(f"L {i} {line}")
        
        result = '\n'.join(lines_str)
        result += '\n***\n'
        
        if state.feedback:
            result += state.feedback
        
        result += ';'
        
        return result
    
    def step(self, action: EditAction) -> Tuple[EditorState, bool, str]:
        """
        Apply an edit action and return new state.
        
        Args:
            action: The edit action to apply
            
        Returns:
            (new_state, done, error_message)
        """
        if self.state.solved:
            return self.state, True, ""
        
        if action.command_type == EditCommandType.EXIT:
            return self.state, True, ""
        
        # Validate line number
        if action.line_number is not None:
            if action.line_number < 1 or action.line_number > len(self.state.lines) + 1:
                error_msg = f"Error: Line {action.line_number} out of range (1-{len(self.state.lines)})"
                return self.state, False, error_msg
        
        # Apply edit
        new_lines = self.state.lines.copy()
        
        try:
            if action.command_type == EditCommandType.ADDL:
                # Insert at line_number (1-indexed)
                new_lines.insert(action.line_number - 1, action.content)
            
            elif action.command_type == EditCommandType.REPL:
                # Replace line (1-indexed)
                new_lines[action.line_number - 1] = action.content
            
            elif action.command_type == EditCommandType.DELL:
                # Delete line (1-indexed)
                if len(new_lines) == 0:
                    return self.state, False, "Error: Cannot delete from empty solution"
                del new_lines[action.line_number - 1]
            
            elif action.command_type == EditCommandType.REPW:
                # Replace word in line
                line = new_lines[action.line_number - 1]
                new_lines[action.line_number - 1] = line.replace(action.old_word, action.new_word)
        
        except IndexError:
            error_msg = f"Error: Invalid line number {action.line_number}"
            return self.state, False, error_msg
        
        # Verify new solution
        new_solution = '\n'.join(new_lines)
        is_solved, feedback = self.verifier.verify_solution(new_solution)
        
        # Create new state
        new_state = EditorState(
            lines=new_lines,
            feedback=feedback,
            solved=is_solved
        )
        
        self.state = new_state
        self.history.append(new_state.copy())
        self.current_turn += 1
        
        done = is_solved or self.current_turn >= self.max_turns
        
        return new_state, done, ""
    
    def is_solved(self) -> bool:
        """Check if current state is solved."""
        return self.state.solved if self.state else False
    
    def get_full_trace(self) -> str:
        """Get the full editing trace as a string."""
        return '\n'.join([self.format_state_for_llm(s) for s in self.history])