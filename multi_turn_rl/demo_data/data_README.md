# Demo Data

This directory contains task definitions and success criteria for the Multi-Turn RL framework.

## Structure

### Tasks Directory (`tasks/`)
Contains JSON files defining different categories of tasks:

- **`programming_tasks.json`** - Basic programming tasks (fibonacci, data analysis, file processing, web data)
- **`math_science_tasks.json`** - Math and science tasks (physics, statistics)
- **`astronomy_tasks.json`** - Astronomy-specific tasks (FITS file analysis)
- **`interactive_tasks.json`** - Game and interactive tasks
- **`output_test_tasks.json`** - Tasks designed to test output capture
- **`multiturn_test_tasks.json`** - Multi-turn pipeline testing tasks

### Success Criteria
- **`success_criteria.py`** - Python implementation of success criteria functions
- **`success_criteria.json`** - JSON definition of success criteria rules

### Task Loader
- **`task_loader.py`** - Utility class for loading and managing tasks from JSON files

## Task Format

Each task JSON file follows this structure:

```json
{
  "category": "category_name",
  "description": "Category description",
  "tasks": [
    {
      "name": "task_name",
      "prompt": "Task prompt for the LLM",
      "completion_criteria": "success criteria text",
      "expected_steps": 3,
      "description": "Brief task description"
    }
  ]
}
```

## Usage

### Loading Tasks

```python
from demo_data.task_loader import TaskLoader

# Create loader
loader = TaskLoader()

# Get all tasks
all_tasks = loader.get_all_tasks()

# Get task by name
fibonacci_task = loader.get_task_by_name("fibonacci_sequence")

# Get tasks by category
programming_tasks = loader.get_tasks_by_category("programming")

# Get simple tasks for demos
simple_tasks = loader.get_simple_tasks()
```

### Backward Compatibility

For backward compatibility, you can still use the old function-based API:

```python
from demo_data.task_loader import get_task_by_name, get_tasks_by_category

task = get_task_by_name("fibonacci_sequence")
tasks = get_tasks_by_category("programming")
```

## Success Criteria

Success criteria are defined in both Python (for complex logic) and JSON (for simple rules). The system automatically maps task names to their corresponding success criteria.

### Criteria Types

- **`function_check`** - Check if specific functions exist and work correctly
- **`variable_check`** - Check if required variables exist
- **`data_shape_check`** - Check data structure shapes
- **`random`** - Random success for testing
- **`step_based_random`** - Success rate increases with progress

## Adding New Tasks

1. **Add to appropriate JSON file** in `tasks/` directory
2. **Define success criteria** in `success_criteria.json` 
3. **Implement complex criteria** in `success_criteria.py` if needed
4. **Test with the TaskLoader** to ensure proper loading

## File Types

- **`.json`** - Task definitions and simple success criteria
- **`.py`** - Complex success criteria logic and task loader utility
- **`.fits`** - Astronomy data files for FITS analysis tasks
- **`.md`** - Documentation files