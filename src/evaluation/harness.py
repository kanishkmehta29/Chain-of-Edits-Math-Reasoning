from typing import List, Dict

from core import MathTask, DSLParser, EditCommandType
from environment import MathEditEnvironment


class EvaluationHarness:
    """Evaluate CoE performance on math tasks."""
    
    def __init__(self, tasks: List[MathTask]):
        self.tasks = tasks
        self.results = []
    
    def evaluate_sequence(self, task: MathTask, 
                         initial_solution: str,
                         actions: List[str]) -> Dict:
        """
        Evaluate a sequence of DSL commands.
        
        Args:
            task: The task being solved
            initial_solution: Starting solution
            actions: List of DSL command strings
            
        Returns:
            Dictionary with evaluation results
        """
        env = MathEditEnvironment(task, initial_solution)
        state = env.reset()
        
        result = {
            'task_id': task.task_id,
            'initial_solved': state.solved,
            'num_actions': len(actions),
            'actions_taken': [],
            'success': False,
            'error': None,
            'final_solution': '\\n'.join(state.lines)
        }
        
        parser = DSLParser()
        
        for action_str in actions:
            try:
                action = parser.parse(action_str)
                result['actions_taken'].append(action_str)
                
                if action.command_type == EditCommandType.EXIT:
                    break
                
                state, done, error = env.step(action)
                
                if error:
                    result['error'] = error
                    break
                
                if done:
                    break
                    
            except Exception as e:
                result['error'] = str(e)
                break
        
        result['success'] = env.is_solved()
        result['final_solution'] = '\\n'.join(env.state.lines)
        
        return result
    
    def evaluate_all(self, demonstrations: List[Dict]) -> Dict:
        """
        Evaluate all demonstrations.
        
        Returns:
            Dictionary with aggregate statistics
        """
        self.results = []
        
        for demo in demonstrations:
            task = next((t for t in self.tasks if t.task_id == demo['task_id']), None)
            if not task:
                continue
            
            result = self.evaluate_sequence(
                task,
                demo['initial_state'],
                demo['actions']
            )
            self.results.append(result)
        
        # Compute statistics
        total = len(self.results)
        successful = sum(1 for r in self.results if r['success'])
        avg_actions = sum(r['num_actions'] for r in self.results) / max(total, 1)
        
        stats = {
            'total_evaluations': total,
            'successful': successful,
            'success_rate': successful / max(total, 1),
            'avg_actions_per_task': avg_actions,
            'results': self.results
        }
        
        return stats