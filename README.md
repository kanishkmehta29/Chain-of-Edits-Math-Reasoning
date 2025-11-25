# CoE Math Reasoning - LLM-Powered Math Solver

> **🤖 Powered by Gemini AI** - Chain-of-Edits approach for reasoning in language models

## Gemini API Setup

This project uses Google's Gemini API to power the Chain-of-Edits solver.

### 1. Get API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key

### 2. Configure API Key

**Option A: Environment File (Recommended)**
```bash
# Copy the template
cp .env.example .env

# Edit .env and add your API key
# GEMINI_API_KEY=your_actual_api_key_here
```

**Option B: Environment Variable**
```bash
export GEMINI_API_KEY=your_actual_api_key_here
```

**Option C: Command Line Argument**
```bash
python examples/solve_cli.py --api-key your_actual_api_key_here ...
```



### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory**

```bash
cd /path/to/coe_math_reasoning
```

2. **Create and activate virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# Or: venv\Scripts\activate  # On Windows
```

3. **Install dependencies**

```bash
# Install in development mode (recommended)
pip install -e .

# Or install requirements directly
pip install -r requirements.txt
```

## Running Examples

### Gemini-Powered Solver

**Watch Gemini solve math problems using Chain-of-Edits:**

```bash
python examples/demo_gemini_solver.py
```

**What it does:** Uses Gemini API to iteratively generate edit commands that fix corrupted math solutions. Shows the complete solving process step-by-step.

### Gemini Evaluation

**Benchmark Gemini's performance on multiple tasks:**

```bash
python examples/demo_gemini_evaluation.py
```

**What it does:** Runs Gemini on all example tasks and reports success rate, average steps, and detailed results.

### Command-Line Solver

**Solve custom problems from the terminal:**

```bash
python examples/solve_cli.py \
  --problem "What is 5 + 3?" \
  --initial "5 + 3 = 7\nAnswer: 7" \
  --truth "Let sum = 5 + 3\nAnswer: 8"
```

**Options:**
- `--problem`: The math problem to solve
- `--initial`: Corrupted initial solution  
- `--truth`: Expected correct solution
- `--max-steps`: Maximum edit steps (default: 15)
- `--model`: Gemini model (default: gemini-1.5-flash)
- `--api-key`: API key (optional if using .env)

---

### Traditional Demos (Without LLM)



### Basic Environment Demo

Demonstrates interactive editing with DSL commands:

```bash
python examples/demo_basic.py
```

**What it does:** Shows how to apply edit commands (REPL, ADDL, DELL) to fix a corrupted math solution step-by-step.

### Synthetic Demo Generation

Generates training demonstrations by corrupting solutions:

```bash
python examples/demo_generation.py
```

**What it does:** Creates a dataset of corrupted solutions with their repair sequences. Useful for training machine learning models.

### Evaluation Demo

Evaluates model performance on tasks:

```bash
python examples/demo_evaluation.py
```

**What it does:** Runs evaluation harness on example tasks and computes success metrics.

## Running Tests

Run the full test suite:

```bash
python -m unittest discover tests
```

Run specific test modules:

```bash
python -m unittest tests.test_dsl
python -m unittest tests.test_environment
python -m unittest tests.test_data
```

## Basic Usage

### Creating a Task

```python
from src.core import MathTask

task = MathTask(
    task_id="example_1",
    problem_text="What is 3 + 5?",
    ground_truth="Let sum = 3 + 5\nAnswer: 8",
    initial_solutions=["3 + 5 = 7\nAnswer: 7"]
)
```

### Using the Editor Environment

```python
from src.core import DSLParser
from src.environment import MathEditEnvironment

# Create environment
env = MathEditEnvironment(task, task.initial_solutions[0])
state = env.reset()

# Parse and apply edit command
parser = DSLParser()
action = parser.parse("REPL 1 >>>Let sum = 3 + 5")
new_state, done, error = env.step(action)

# Check if solved
if env.is_solved():
    print("Task completed!")
```

### Generating Training Data

```python
from src.data import create_example_tasks, SyntheticDemoGenerator

# Create tasks
tasks = create_example_tasks()

# Generate demonstrations
generator = SyntheticDemoGenerator(tasks)
dataset = generator.generate_dataset(demos_per_task=5)

# Each demo contains:
# - 'problem': The task description
# - 'initial_state': Corrupted solution
# - 'target_state': Correct solution
# - 'actions': List of repair commands
# - 'states': State after each action
```

## DSL Command Reference

| Command | Syntax | Description | Example |
|---------|--------|-------------|---------|
| **ADDL** | `ADDL <line> >>>content` | Add line at position | `ADDL 2 >>>Let x = 5` |
| **REPL** | `REPL <line> >>>content` | Replace entire line | `REPL 1 >>>Answer: 10` |
| **DELL** | `DELL <line>` | Delete line | `DELL 3` |
| **REPW** | `REPW <line> >>>old >>>new` | Replace word in line | `REPW 1 >>>sum >>>total` |
| **EXIT** | `EXIT` | Exit editing session | `EXIT` |

## Project Structure

```
coe_math_reasoning/
├── src/
│   ├── core/              # Task, DSL, and State classes
│   ├── environment/       # Editor and Verifier
│   ├── data/              # Task creation and corruption
│   └── evaluation/        # Evaluation harness
├── examples/              # Demo scripts
├── tests/                 # Unit tests
├── setup.py              # Package setup
└── requirements.txt      # Dependencies
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure:
1. Virtual environment is activated
2. Package is installed with `pip install -e .`
3. You're running from the project root directory

### Test Failures

Some tests may fail due to verification strictness. This is expected - the system prioritizes demonstrating the CoE workflow over perfect accuracy.

## Development

### Adding New Tasks

Edit `src/data/tasks.py`:

```python
def create_example_tasks() -> List[MathTask]:
    tasks = [
        MathTask(
            task_id="new_task",
            problem_text="Your math problem",
            ground_truth="Correct solution",
            initial_solutions=["Incorrect solution"]
        ),
        # ... more tasks
    ]
    return tasks
```

### Adding New Corruption Types

Edit `src/data/corruptions.py` to add new corruption patterns in the `CorruptionType` enum.