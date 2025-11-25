import re
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class EditCommandType(Enum):
    """Types of edit commands available in the DSL."""
    ADDL = "ADDL"  # Add line
    REPL = "REPL"  # Replace line
    DELL = "DELL"  # Delete line
    REPW = "REPW"  # Replace word
    EXIT = "EXIT"  # Exit editing


@dataclass
class EditAction:
    """Represents a single DSL edit command."""
    command_type: EditCommandType
    line_number: Optional[int] = None
    content: Optional[str] = None
    old_word: Optional[str] = None
    new_word: Optional[str] = None
    
    def to_string(self) -> str:
        """Convert action to DSL command string."""
        if self.command_type == EditCommandType.EXIT:
            return "EXIT"
        elif self.command_type == EditCommandType.ADDL:
            return f"ADDL {self.line_number} >>>{self.content}"
        elif self.command_type == EditCommandType.REPL:
            return f"REPL {self.line_number} >>>{self.content}"
        elif self.command_type == EditCommandType.DELL:
            return f"DELL {self.line_number}"
        elif self.command_type == EditCommandType.REPW:
            return f"REPW {self.line_number} >>>{self.old_word} >>>{self.new_word}"
        else:
            raise ValueError(f"Unknown command type: {self.command_type}")
    
    def __repr__(self) -> str:
        return f"EditAction({self.to_string()})"


class DSLParser:
    """Parser for DSL commands."""
    
    @staticmethod
    def parse(command_str: str) -> EditAction:
        """
        Parse a DSL command string into an EditAction object.
        
        Args:
            command_str: String containing the DSL command
            
        Returns:
            EditAction object
            
        Raises:
            ValueError: If command syntax is invalid
        """
        command_str = command_str.strip()
        
        # EXIT command
        if command_str == "EXIT":
            return EditAction(command_type=EditCommandType.EXIT)
        
        # Split command and arguments
        parts = command_str.split(None, 1)
        if not parts:
            raise ValueError("Empty command")
        
        cmd = parts[0].upper()
        
        # DELL command: DELL <line_number>
        if cmd == "DELL":
            if len(parts) != 2:
                raise ValueError("DELL requires line number")
            try:
                line_num = int(parts[1])
            except ValueError:
                raise ValueError(f"Invalid line number: {parts[1]}")
            return EditAction(
                command_type=EditCommandType.DELL,
                line_number=line_num
            )
        
        # Commands with >>> separator
        if len(parts) < 2 or ">>>" not in parts[1]:
            raise ValueError(f"Command {cmd} requires >>> separator")
        
        # ADDL command: ADDL <line_number> >>><content>
        if cmd == "ADDL":
            match = re.match(r'(\d+)\s*>>>(.*)' , parts[1])
            if not match:
                raise ValueError("ADDL syntax: ADDL <line_number> >>><content>")
            line_num = int(match.group(1))
            content = match.group(2)
            return EditAction(
                command_type=EditCommandType.ADDL,
                line_number=line_num,
                content=content
            )
        
        # REPL command: REPL <line_number> >>><content>
        if cmd == "REPL":
            match = re.match(r'(\d+)\s*>>>(.*)', parts[1])
            if not match:
                raise ValueError("REPL syntax: REPL <line_number> >>><content>")
            line_num = int(match.group(1))
            content = match.group(2)
            return EditAction(
                command_type=EditCommandType.REPL,
                line_number=line_num,
                content=content
            )
        
        # REPW command: REPW <line_number> >>><old_word> >>><new_word>
        if cmd == "REPW":
            match = re.match(r'(\d+)\s*>>>(.*?)>>>(.*)', parts[1])
            if not match:
                raise ValueError("REPW syntax: REPW <line_number> >>><old> >>><new>")
            line_num = int(match.group(1))
            old_word = match.group(2).strip()
            new_word = match.group(3).strip()
            return EditAction(
                command_type=EditCommandType.REPW,
                line_number=line_num,
                old_word=old_word,
                new_word=new_word
            )
        
        raise ValueError(f"Unknown command: {cmd}")