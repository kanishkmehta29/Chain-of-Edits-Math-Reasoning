#!/usr/bin/env python3
"""
Generate training data for fine-tuning local LLM.
Uses SyntheticDemoGenerator to create edit sequences.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import create_example_tasks, SyntheticDemoGenerator


def main():
    print("="*70)
    print("Training Data Generation for Local LLM Fine-Tuning")
    print("="*70)
    
    # Create tasks
    print("\n📚 Loading example tasks...")
    tasks = create_example_tasks()
    print(f"  Loaded {len(tasks)} tasks")
    
    # Generate synthetic demonstrations
    print("\n🔧 Generating corrupted solutions and edit sequences...")
    generator = SyntheticDemoGenerator(tasks)
    
    # Generate demonstrations (100 per task)
    demos_per_task = 100
    dataset = generator.generate_dataset(demos_per_task=demos_per_task)
    
    print(f"  Generated {len(dataset)} demonstrations")
    
    # Format for instruction fine-tuning
    print("\n📝 Formatting for instruction fine-tuning...")
    training_data = []
    
    for demo in dataset:
        # Each demo has: task_id, problem, initial_state, target_state, actions, states
        # We create training examples for each edit step
        
        for i, action in enumerate(demo['actions']):
            if action == 'EXIT':
                continue  # Skip EXIT commands in training
            
            # Build instruction prompt
            instruction = """Fix the math solution using one edit command from the Chain-of-Edits DSL."""
            
            # Current state for this step
            current_state = demo['states'][i] if i < len(demo['states']) else demo['initial_state']
            
            # Build input with problem, state
            input_text = f"""**Problem:**
{demo['problem']}

**Expected Correct Solution:**
{demo['target_state']}

**Current Solution State:**
{current_state}

**Available Edit Commands:**
- ADDL <line> >>>content - Add a new line at position
- REPL <line> >>>content - Replace line with new content
- DELL <line> - Delete a line  
- REPW <line> >>>old >>>new - Replace word/phrase in line
- EXIT - Exit editing (when solution is correct)

Generate ONE edit command to fix this:"""
            
            # The output is the edit command for this step
            output = action
            
            training_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "task_id": demo['task_id']
            })

    
    # Save to JSON
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'training_data.json'
    
    print(f"\n💾 Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\n✅ Generated {len(training_data)} training examples")
    print(f"  Saved to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")
    
    # Print statistics
    print("\n📊 Statistics:")
    print(f"  Total examples: {len(training_data)}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Examples per task: ~{len(training_data) // len(tasks)}")
    
    # Print sample
    print("\n📖 Sample Example:")
    sample = training_data[0]
    print(f"  Task: {sample['task_id']}")
    print(f"  Input length: {len(sample['input'])} chars")
    print(f"  Output: {sample['output']}")
    
    print("\n" + "="*70)
    print("✨ Training data generation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
