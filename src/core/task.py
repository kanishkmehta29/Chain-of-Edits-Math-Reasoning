from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class MathTask:
    """Represents a math reasoning task with verification."""
    task_id: str
    problem_text: str
    ground_truth: str  # Canonical answer as string
    initial_solutions: List[str]  # Incorrect or partial solutions
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'task_id': self.task_id,
            'problem_text': self.problem_text,
            'ground_truth': self.ground_truth,
            'initial_solutions': self.initial_solutions,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MathTask':
        """Create from dictionary."""
        return cls(**data)
    
    def __repr__(self) -> str:
        return f"MathTask(id={self.task_id}, problem='{self.problem_text[:50]}...')"