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
        Verify if solution matches ground truth by comparing final answers.
        
        Args:
            solution_text: The solution to verify (multi-line string)
            
        Returns:
            (is_correct, feedback_message)
        """
        # Extract final answers from both solution and ground truth
        solution_answer = self._extract_final_answer(solution_text)
        truth_answer = self._extract_final_answer(self.task.ground_truth)
        
        if not solution_answer:
            return False, "Error: No final answer found. Solution must contain 'Answer:' or end with answer like 'x = 4'"
        
        if not truth_answer:
            # If ground truth has no extractable answer, use it as-is
            truth_answer = self.task.ground_truth.strip()
        
        # Compare the final answers
        is_correct = self._compare_answers(solution_answer, truth_answer)
        
        if is_correct:
            return True, ""
        else:
            return False, f"Test failed: expected answer '{truth_answer}', got '{solution_answer}'"

    
    def _extract_final_answer(self, solution_text: str) -> Optional[str]:
        """Extract the final answer from solution text."""
        lines = solution_text.strip().split('\n')

        
        # Look for lines starting with Answer: or Result:
        for line in reversed(lines):
            line = line.strip()
            if line.startswith(("Answer:", "Result:")):
                answer = line.split(":", 1)[1].strip()
                return answer
        
        # Look for lines that look like variable assignments (x = value, y = value, etc.)
        for line in reversed(lines):
            line = line.strip()
            # Match patterns like "x = 4", "y = 10", "answer = 5", etc.
            if '=' in line and len(line.split('=')) == 2:
                var, val = line.split('=')
                # Check if it looks like a final answer (variable on left, value on right)
                if var.strip() and val.strip():
                    # Prioritize single-letter variables or "answer"
                    var_name = var.strip().lower()
                    if len(var_name) <= 2 or 'answer' in var_name:
                        return line.strip()
        
        # If no explicit answer or variable found, take the last non-empty line
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