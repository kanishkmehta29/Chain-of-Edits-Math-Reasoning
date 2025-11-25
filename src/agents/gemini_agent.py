"""
Gemini Agent for Chain-of-Edits math reasoning.
"""

import google.generativeai as genai
from typing import Optional, Tuple
from .config import GeminiConfig


class GeminiCoEAgent:
    """
    Gemini-powered agent that generates edit commands to solve math problems.
    Uses Chain-of-Edits approach to iteratively fix corrupted solutions.
    """
    
    def __init__(self, config: GeminiConfig):
        """
        Initialize Gemini agent.
        
        Args:
            config: GeminiConfig instance with API key and settings
        """
        self.config = config
        
        # Configure Gemini API
        genai.configure(api_key=config.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=config.model,
            generation_config={
                'temperature': config.temperature,
                'max_output_tokens': config.max_output_tokens,
            }
        )
        
        # Track usage
        self.total_tokens = 0
        self.num_requests = 0
    
    def generate_edit_command(
        self,
        problem: str,
        ground_truth: str,
        current_state: str,
        feedback: str,
    ) -> Optional[str]:
        """
        Generate next edit command to fix the current state.
        
        Args:
            problem: Math problem description
            ground_truth: Expected correct solution
            current_state: Current solution state (with line numbers)
            feedback: Verifier feedback on current state
            
        Returns:
            DSL edit command (e.g., "REPL 1 >>>Let x = 5") or None if failed
        """
        prompt = self._build_prompt(problem, ground_truth, current_state, feedback)
        
        try:
            response = self.model.generate_content(prompt)
            self.num_requests += 1
            
            # Extract command from response
            command = self._extract_command(response.text)
            
            return command
            
        except Exception as e:
            print(f"Error generating edit command: {e}")
            return None
    
    def _build_prompt(
        self,
        problem: str,
        ground_truth: str,
        current_state: str,
        feedback: str
    ) -> str:
        """Build prompt for Gemini to generate edit command."""
        
        prompt = f"""You are an expert math problem solver using a Chain-of-Edits approach. Your task is to fix incorrect math solutions by applying precise edit commands.

**Problem:**
{problem}

**Expected Correct Solution:**
{ground_truth}

**Current Solution State:**
{current_state}

**Verifier Feedback:**
{feedback}

**Available Edit Commands:**
- ADDL <line> >>>content - Add a new line at position
- REPL <line> >>>content - Replace line with new content
- DELL <line> - Delete a line
- REPW <line> >>>old >>>new - Replace word/phrase in line
- EXIT - Exit editing (when solution is correct)

**Instructions:**
1. Analyze the current state and feedback
2. Identify ONE specific error to fix
3. Generate the EXACT DSL command to fix it
4. Output ONLY the command, nothing else

**Your Response (command only):**"""
        
        return prompt
    
    def _extract_command(self, response_text: str) -> str:
        """
        Extract clean DSL command from model response.
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Cleaned DSL command
        """
        # Remove common prefixes/suffixes
        command = response_text.strip()
        
        # Remove markdown code blocks if present
        if command.startswith('```'):
            lines = command.split('\n')
            # Find first non-code-block line
            for line in lines:
                line = line.strip()
                if line and not line.startswith('```'):
                    command = line
                    break
        
        # Remove any explanatory text (take first line that looks like a command)
        lines = command.split('\n')
        for line in lines:
            line = line.strip()
            # Check if line starts with a valid command
            if line.startswith(('ADDL ', 'REPL ', 'DELL ', 'REPW ', 'EXIT')):
                return line
        
        # If no valid command found, return the first non-empty line
        for line in lines:
            if line.strip():
                return line.strip()
        
        return command.strip()
    
    def solve_problem(
        self,
        problem: str,
        ground_truth: str,
        initial_state: str,
        initial_feedback: str,
        max_steps: int = 20,
        verbose: bool = True
    ) -> Tuple[bool, int, list]:
        """
        Attempt to solve a problem using iterative edits.
        
        Args:
            problem: Math problem description
            ground_truth: Expected solution
            initial_state: Starting (corrupted) state
            initial_feedback: Initial verifier feedback
            max_steps: Maximum edit steps allowed
            verbose: Print progress
            
        Returns:
            Tuple of (success, num_steps, edit_history)
        """
        from core import DSLParser
        from environment import MathEditEnvironment
        from core import MathTask
        
        # Create task and environment
        task = MathTask(
            task_id="gemini_solve",
            problem_text=problem,
            ground_truth=ground_truth,
            initial_solutions=[initial_state.split('\n')]
        )
        
        env = MathEditEnvironment(task, initial_state)
        state = env.reset()
        
        parser = DSLParser()
        edit_history = []
        
        current_state_str = env.format_state_for_llm()
        current_feedback = initial_feedback
        
        for step in range(max_steps):
            if verbose:
                print(f"\n{'='*70}")
                print(f"Step {step + 1}/{max_steps}")
                print(f"{'='*70}")
                print(current_state_str)
            
            # Generate edit command
            command_str = self.generate_edit_command(
                problem=problem,
                ground_truth=ground_truth,
                current_state=current_state_str,
                feedback=current_feedback
            )
            
            if not command_str:
                if verbose:
                    print("Failed to generate command")
                break
            
            if verbose:
                print(f"\nGenerated command: {command_str}")
            
            edit_history.append(command_str)
            
            # Check for EXIT
            if command_str.strip().upper() == 'EXIT':
                if verbose:
                    print("Agent signaled EXIT")
                break
            
            # Parse and apply command
            try:
                action = parser.parse(command_str)
                state, done, error = env.step(action)
                
                if error:
                    if verbose:
                        print(f"Error applying command: {error}")
                    # Continue anyway - let agent try to recover
                
                current_state_str = env.format_state_for_llm()
                current_feedback = state.feedback
                
                if done or env.is_solved():
                    if verbose:
                        print(f"\n{'='*70}")
                        print("✓ PROBLEM SOLVED!")
                        print(f"{'='*70}")
                    return True, step + 1, edit_history
                    
            except Exception as e:
                if verbose:
                    print(f"Error parsing/applying command: {e}")
                # Continue to next iteration
        
        if verbose:
            print(f"\n{'='*70}")
            print("✗ Failed to solve within step limit")
            print(f"{'='*70}")
        
        return False, max_steps, edit_history
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            'num_requests': self.num_requests,
            'total_tokens': self.total_tokens,
            'model': self.config.model
        }
