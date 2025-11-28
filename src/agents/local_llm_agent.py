"""
Local LLM agent for Chain-of-Edits math reasoning.
"""

from typing import Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from .local_llm_config import LocalLLMConfig


class LocalLLMAgent:
    """
    Local language model agent for generating edit commands.
    Uses small fine-tuned models like Phi-3.5-mini.
    """
    
    def __init__(self, config: LocalLLMConfig):
        """
        Initialize local LLM agent.
        
        Args:
            config: LocalLLMConfig instance
        """
        self.config = config
        
        print(f"Loading model: {config.model_name_or_path}")
        print(f"Device: {config.device}")
        
        # Setup quantization if requested
        quantization_config = None
        if config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load model (load on CPU first to avoid MPS buffer issues)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            quantization_config=quantization_config,
            device_map=None, # Load on CPU first
            torch_dtype=torch.float16, # Force float16 to save memory (even on CPU)
            trust_remote_code=True

        )
        
        # Move to device if not CPU and not using auto-map
        if config.device and config.device != "cpu" and not quantization_config:
            print(f"Moving model to {config.device}...")
            self.model.to(config.device)

        
        # Load LoRA adapter if specified
        if config.adapter_path:
            print(f"Loading fine-tuned adapter: {config.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, config.adapter_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        # Track usage
        self.num_requests = 0
    
    def generate_edit_command(
        self,
        problem: str,
        ground_truth: str,
        current_state: str,
        feedback: str,
    ) -> Optional[str]:
        """
        Generate next edit command using local LLM.
        
        Args:
            problem: Math problem description
            ground_truth: Expected correct solution
            current_state: Current solution state with line numbers
            feedback: Verifier feedback
            
        Returns:
            DSL edit command or None if failed
        """
        prompt = self._build_prompt(problem, ground_truth, current_state, feedback)
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False # Fix for DynamicCache issue with Phi-3
                )

            
            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            self.num_requests += 1
            
            # Extract command
            command = self._extract_command(generated_text)
            return command
            
        except Exception as e:
            print(f"⚠ Error generating edit command: {e}")
            return None
    
    def _build_prompt(
        self,
        problem: str,
        ground_truth: str,
        current_state: str,
        feedback: str
    ) -> str:
        """Build prompt for local LLM (same format as Gemini)."""
        
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
        """Extract clean DSL command from model response."""
        command = response_text.strip()
        
        # Remove markdown code blocks
        if command.startswith('```'):
            lines = command.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('```'):
                    command = line
                    break
        
        # Find first valid command line
        lines = command.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('ADDL ', 'REPL ', 'DELL ', 'REPW ', 'EXIT')):
                return line
        
        # Return first non-empty line
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
        Solve problem using iterative edits (same interface as GeminiCoEAgent).
        
        Args:
            problem: Math problem description
            ground_truth: Expected solution
            initial_state: Starting (corrupted) state
            initial_feedback: Initial verifier feedback
            max_steps: Maximum edit steps
            verbose: Print progress
            
        Returns:
            (success, num_steps, edit_history)
        """
        from core import DSLParser
        from environment import MathEditEnvironment
        from core import MathTask
        
        # Create task and environment
        task = MathTask(
            task_id="local_llm_solve",
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
        
        if verbose:
            print(f"\n{'='*70}")
            print("✗ Failed to solve within step limit")
            print(f"{'='*70}")
        
        return False, max_steps, edit_history
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            'num_requests': self.num_requests,
            'model': self.config.model_name_or_path,
            'device': self.config.device
        }
