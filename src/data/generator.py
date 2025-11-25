import random
from typing import List, Optional, Dict

from core import MathTask
from environment import MathEditEnvironment
from .corruptions import CorruptionType, Corruption


class SyntheticDemoGenerator:
    """Generates synthetic Chain-of-Edits demonstrations."""
    
    WRONG_LINES = [
        "Let z = 0",
        "Add 100 to both sides",
        "Multiply by -1",
        "Therefore x = 0",
        "The answer is 42",
        "Simplify to get 1",
    ]
    
    def __init__(self, tasks: List[MathTask]):
        self.tasks = tasks
    
    def generate_demo(self, task: MathTask, num_corruptions: int = 3) -> Dict:
        """
        Generate a single synthetic demonstration.
        
        Args:
            task: The task to generate demo for
            num_corruptions: Number of corruptions to apply
            
        Returns:
            Dictionary containing the demonstration trace
        """
        # Start with ground truth solution
        correct_lines = task.ground_truth.split('\\n')
        
        # Apply corruptions
        corruptions = []
        current_lines = correct_lines.copy()
        
        for _ in range(num_corruptions):
            corruption = self._apply_random_corruption(current_lines)
            if corruption:
                corruptions.append(corruption)
        
        # Build CoE trace by reversing corruptions
        trace = {
            'task_id': task.task_id,
            'problem': task.problem_text,
            'initial_state': '\\n'.join(current_lines),
            'target_state': '\\n'.join(correct_lines),
            'actions': [],
            'states': []
        }
        
        # Reverse the corruptions
        env = MathEditEnvironment(task, '\\n'.join(current_lines))
        state = env.reset()
        trace['states'].append(env.format_state_for_llm(state))
        
        for corruption in reversed(corruptions):
            action = corruption.reverse_action()
            trace['actions'].append(action.to_string())
            
            state, done, error = env.step(action)
            trace['states'].append(env.format_state_for_llm(state))
            
            if done and env.is_solved():
                break
        
        # Add EXIT action
        trace['actions'].append('EXIT')
        
        return trace
    
    def _apply_random_corruption(self, lines: List[str]) -> Optional[Corruption]:
        """Apply a random corruption to the lines."""
        if not lines:
            return None
        
        corruption_type = random.choice(list(CorruptionType))
        
        if corruption_type == CorruptionType.DELETE_LINE and len(lines) > 1:
            line_idx = random.randint(0, len(lines) - 1)
            original = lines[line_idx]
            del lines[line_idx]
            return Corruption(
                corruption_type=corruption_type,
                line_number=line_idx + 1,
                original_content=original
            )
        
        elif corruption_type == CorruptionType.ADD_WRONG_LINE:
            line_idx = random.randint(0, len(lines))
            wrong_line = random.choice(self.WRONG_LINES)
            lines.insert(line_idx, wrong_line)
            return Corruption(
                corruption_type=corruption_type,
                line_number=line_idx + 1,
                corrupted_content=wrong_line
            )
        
        elif corruption_type == CorruptionType.REPLACE_LINE:
            line_idx = random.randint(0, len(lines) - 1)
            original = lines[line_idx]
            wrong_line = random.choice(self.WRONG_LINES)
            lines[line_idx] = wrong_line
            return Corruption(
                corruption_type=corruption_type,
                line_number=line_idx + 1,
                original_content=original,
                corrupted_content=wrong_line
            )
        
        elif corruption_type == CorruptionType.INTRODUCE_TYPO:
            line_idx = random.randint(0, len(lines) - 1)
            original = lines[line_idx]
            corrupted = self._introduce_typo(original)
            if corrupted != original:
                lines[line_idx] = corrupted
                return Corruption(
                    corruption_type=corruption_type,
                    line_number=line_idx + 1,
                    original_content=original,
                    corrupted_content=corrupted
                )
        
        return None
    
    def _introduce_typo(self, text: str) -> str:
        """Introduce a typo in text."""
        if len(text) < 3:
            return text
        
        words = text.split()
        if not words:
            return text
        
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx]
        
        if len(word) > 2:
            # Swap two adjacent characters
            char_idx = random.randint(0, len(word) - 2)
            word_list = list(word)
            word_list[char_idx], word_list[char_idx + 1] = word_list[char_idx + 1], word_list[char_idx]
            words[word_idx] = ''.join(word_list)
        
        return ' '.join(words)
    
    def generate_dataset(self, demos_per_task: int = 5) -> List[Dict]:
        """Generate full dataset of demonstrations."""
        dataset = []
        for task in self.tasks:
            for _ in range(demos_per_task):
                num_corruptions = random.randint(2, 5)
                demo = self.generate_demo(task, num_corruptions)
                dataset.append(demo)
        return dataset