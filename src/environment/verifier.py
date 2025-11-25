from typing import Tuple, Optional

try:
    from sympy import sympify, simplify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


class MathVerifier:
    """Verifies mathematical solutions."""
    
    def __init__(self, task):
        """
        Initialize verifier.
        
        Args:
            task: MathTask instance
        """
        self.task = task
    
    def verify_solution(self, solution_text: str) -> Tuple[bool, str]:
        """
        Verify if solution matches ground truth.
        
        Args:
            solution_text: The solution to verify (multi-line string)
            
        Returns:
            (is_correct, feedback_message)
        """
        # Extract the final answer from solution
        final_answer = self._extract_final_answer(solution_text)
        
        if not final_answer:
            return False, "Error: No final answer found. Solution must contain 'Answer:' or 'Result:' line."
        
        # Compare with ground truth
        is_correct = self._compare_answers(final_answer, self.task.ground_truth)
        
        if is_correct:
            return True, ""
        else:
            return False, f"Test failed: expected {self.task.ground_truth}, got {final_answer}"
    
    def _extract_final_answer(self, solution_text: str) -> Optional[str]:
        """Extract the final answer from solution text."""
        lines = solution_text.strip().split('\\n')
        
        # Look for lines starting with Answer: or Result:
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("Answer:") or line.startswith("Result:"):
                answer = line.split(":", 1)[1].strip()
                return answer
        
        # If no explicit answer, take the last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        
        return None
    
    def _compare_answers(self, answer: str, ground_truth: str) -> bool:
        """Compare two mathematical answers for equivalence."""
        # Try exact string match first
        if answer.strip() == ground_truth.strip():
            return True
        
        # Try numeric comparison
        try:
            ans_val = float(answer)
            gt_val = float(ground_truth)
            return abs(ans_val - gt_val) < 1e-6
        except (ValueError, TypeError):
            pass
        
        # Try symbolic comparison if sympy available
        if SYMPY_AVAILABLE:
            try:
                ans_expr = sympify(answer)
                gt_expr = sympify(ground_truth)
                diff = simplify(ans_expr - gt_expr)
                return diff == 0
            except:
                pass
        
        return False