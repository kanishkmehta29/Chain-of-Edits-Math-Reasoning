# CoE Math Reasoning - Project Overview

A research codebase for symbolic math reasoning using edit-based approaches, inspired by "Replacing thinking with tool usage enables reasoning in small language models".

## Project Concept

This project implements **Chain-of-Edits (CoE)** - a novel approach where language models iteratively edit incorrect solutions rather than generating reasoning from scratch. This enables smaller models to perform complex reasoning tasks by using structured edit operations.

## Architecture

### Directory Structure

```
coe_math_reasoning/
├── src/                    # Source code
│   ├── core/              # Core data structures (Task, DSL, State)
│   ├── environment/       # Editor and Verifier
│   ├── data/              # Data generation and tasks
│   └── evaluation/        # Evaluation harness
├── examples/              # Usage examples
├── tests/                 # Unit tests
├── setup.py
├── requirements.txt
└── README.md
```

## Core Components

### 1. Domain-Specific Language (DSL)

The DSL provides five edit commands for modifying math solutions:

- **ADDL** `<line> >>>content` - Add a new line at the specified position
- **REPL** `<line> >>>content` - Replace an entire line with new content
- **DELL** `<line>` - Delete a line
- **REPW** `<line> >>>old >>>new` - Replace a specific word/phrase in a line
- **EXIT** - Exit the editing session

**Example:**
```
REPL 1 >>>Let initial = 3
ADDL 2 >>>Let bought = 5
DELL 4
EXIT
```

### 2. Core Data Structures

#### MathTask
Represents a math reasoning problem with:
- `task_id`: Unique identifier
- `problem_text`: The math question
- `ground_truth`: Correct solution (canonical answer)
- `initial_solutions`: Corrupted/incorrect starting solutions
- `metadata`: Additional task information

#### EditAction
Encapsulates a single DSL edit command with:
- `command_type`: Type of edit (ADDL, REPL, DELL, REPW, EXIT)
- `line_number`: Target line (if applicable)
- `content`: New content (for ADDL/REPL)
- `old_word`, `new_word`: Word replacement (for REPW)

#### EditorState
Tracks the current state during editing:
- `lines`: Current solution content (list of strings)
- `feedback`: Verifier feedback on correctness
- `solved`: Boolean indicating if task is solved

### 3. MathEditEnvironment

The interactive editing environment that:
1. Maintains the current solution state
2. Applies edit actions to transform the solution
3. Verifies correctness after each edit
4. Provides feedback to guide further edits

**Workflow:**
```
Initial State → Apply Edit → Verify → New State
       ↑                                    ↓
       └────────── Continue Editing ────────┘
```

### 4. SyntheticDemoGenerator

Generates training data by:
1. Starting with correct solutions
2. Applying random corruptions (deletions, typos, wrong lines)
3. Computing the repair action sequence
4. Creating (corrupted_state, action_sequence, target_state) tuples

**Corruption Types:**
- `DELETE_LINE` - Remove a correct line
- `ADD_WRONG_LINE` - Insert an incorrect line
- `REPLACE_LINE` - Replace correct line with wrong content
- `INTRODUCE_TYPO` - Add typos to correct content
- `SWAP_OPERAND` - Swap mathematical operands

### 5. MathVerifier

Validates solution correctness by:
- Comparing final answer against ground truth
- Checking solution structure and format
- Providing actionable feedback

### 6. Gemini Agent Integration

The project now includes **GeminiCoEAgent** - an LLM-powered agent that generates edit commands to solve math problems.

#### Agent Architecture

```
Problem + Current State
         ↓
    Gemini API (with CoE prompt)
         ↓
    Edit Command (DSL)
         ↓
   Apply to Environment
         ↓
    New State + Feedback
         ↓
    (Repeat until solved)
```

#### Prompt Engineering

The agent uses carefully crafted prompts that include:
- **Problem context**: The original math question
- **Expected solution**: Ground truth for reference
- **Current state**: Solution with line numbers (e.g., "L 1 2x = 13")
- **Feedback**: Verifier output indicating errors
- **DSL reference**: Available edit commands
- **Instruction**: Generate ONE specific edit command

**Example Prompt Structure:**
```
You are an expert math problem solver using Chain-of-Edits.

Problem: Solve 2x + 5 = 13
Expected: Subtract 5: 2x = 8\nDivide by 2: x = 4\nAnswer: 4

Current State:
L 1 2x = 13
L 2 x = 6.5
***
Test failed: wrong answer

Available Commands: ADDL, REPL, DELL, REPW, EXIT

Generate ONE command to fix this:
```

#### Response Extraction

The agent intelligently extracts commands from LLM responses by:
1. Removing markdown code blocks
2. Identifying lines starting with valid command keywords
3. Cleaning whitespace and formatting
4. Validating DSL syntax

#### Iterative Solving

The `solve_problem()` method:
1. Initializes environment with corrupted solution
2. Loops up to `max_steps` times:
   - Generates edit command via Gemini
   - Parses and applies command
   - Gets new state and feedback
   - Checks if solved
3. Returns success status, steps taken, and edit history

### 7. EvaluationHarness



Evaluates model performance:
- Runs action sequences on tasks
- Tracks success/failure rates
- Measures edit efficiency
- Computes metrics (accuracy, avg edits per task)

## Research Workflow

### Phase 1: Data Generation

```python
from src.data import create_example_tasks, SyntheticDemoGenerator

tasks = create_example_tasks()
generator = SyntheticDemoGenerator(tasks)
dataset = generator.generate_dataset(demos_per_task=10)
```

**Output:** Training demonstrations showing how to repair corrupted solutions.

### Phase 2: Model Training

Train small language models using:
- Supervised learning on synthetic demonstrations
- Reinforcement learning with verifier feedback
- Behavioral cloning from expert edit sequences

### Phase 3: Evaluation

```python
from src.evaluation import EvaluationHarness

harness = EvaluationHarness(test_tasks)
results = harness.evaluate_model(model, max_steps=20)
```

**Metrics:**
- Task success rate
- Average edits per task
- Verification pass rate

## Key Design Principles

### 1. Structured Editing
Instead of free-form text generation, the model uses a constrained DSL. This reduces the search space and improves reliability.

### 2. Iterative Refinement
Solutions are improved step-by-step, allowing for:
- Error recovery
- Intermediate verification
- Guided exploration

### 3. Explicit Verification
After each edit, the verifier provides:
- Pass/fail signal
- Specific error messages
- Guidance for next edits

### 4. Synthetic Data Generation
By corrupting correct solutions, we can:
- Generate unlimited training data
- Control difficulty levels
- Ensure diverse error patterns

## Example Flow

**Initial (Corrupted) State:**
```
L 1 Lte initial = 3
L 2 Answer: 8
L 3 Multiply by -1
***
Test failed: expected solution
```

**Repair Sequence:**
1. `REPL 1 >>>Let initial = 3` (fix typo)
2. `DELL 3` (remove wrong operation)
3. `ADDL 2 >>>Let bought = 5` (add missing step)
4. `EXIT`

**Final (Correct) State:**
```
L 1 Let initial = 3
L 2 Let bought = 5
L 3 Answer: 8
***
Test passed!
```

## Research Applications

This codebase enables:

1. **Training Data Creation** - Generate large-scale CoE demonstrations
2. **Model Development** - Train small LMs for structured reasoning
3. **Evaluation** - Benchmark reasoning capabilities
4. **Analysis** - Study edit patterns and error recovery strategies
5. **Experimentation** - Test different corruption types, verification strategies, and DSL designs