# Demo Data Organization

This directory contains organized demo input data and configurations for the Astro RL system.

## Structure

```
demo_data/
├── astro1_uv_imaging_telescope.fits    # FITS file for astronomy tasks
├── simple_multistep_tasks.py           # Multi-step task definitions  
├── demo_config.py                      # Demo configurations
├── success_criteria.py                 # Success criterion functions
└── README.md                           # This file
```

## New Function-Based Completion Criteria

The system now uses **programmatic success criterion functions** instead of text parsing for determining task completion. This provides:

- **Robust validation**: Functions check actual variables and execution state
- **Precise control**: Can verify specific conditions are met  
- **Multi-step enforcement**: Won't complete early on generic "success" messages
- **Extensible design**: Easy to add new custom criteria

### How It Works

1. **Function Definition**: Each task has an associated success criterion function in `success_criteria.py`
2. **Execution**: The Ray executor runs the main code, then calls the success function
3. **Context Checking**: Success functions receive the execution context (`locals_dict`) to examine variables
4. **Boolean Result**: Function returns `True` if task objectives met, `False` otherwise

### Example Success Criterion
```python
def fibonacci_success_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if fibonacci functions are defined and working"""
    if 'fibonacci' not in locals_dict or 'fibonacci_memo' not in locals_dict:
        return False
    
    # Test both functions
    fib_func = locals_dict['fibonacci']
    fib_memo_func = locals_dict['fibonacci_memo']
    
    return fib_func(5) == 5 and fib_memo_func(5) == 5
```

## Demo Configurations

Choose from these pre-configured demo types:

| Config | Tasks | Duration | Description |
|--------|-------|----------|-------------|
| `quick` | 2 tasks | ~3 min | Fast demo: fibonacci + data analysis |
| `standard` | 4 tasks | ~8 min | Balanced: programming + astronomy |
| `comprehensive` | 8 tasks | ~15 min | Full demo: all categories |
| `astronomy_focused` | 2 tasks | ~10 min | FITS analysis only |
| `programming_focused` | 4 tasks | ~10 min | Coding challenges only |
| `output_tests` | 5 tasks | ~5 min | Output capture verification tests |
| `multiturn_tests` | 4 tasks | ~8 min | Multi-turn interaction testing |

## Task Categories

### Programming Tasks
- **fibonacci_sequence**: Recursive → optimized with memoization
- **data_analysis_pipeline**: Generate data → stats → visualization  
- **file_processing**: Write CSV → read → filter data
- **web_data_fetch**: HTTP request → parse → save

### Math/Science Tasks  
- **physics_projectile**: Calculate trajectory → plot → find max height
- **statistics_experiment**: Generate datasets → t-test → visualize

### Astronomy Tasks
- **fits_basic_analysis**: Load FITS → info → grayscale plot
- **fits_advanced_analysis**: Load FITS → stats → histogram

### Interactive Tasks
- **number_guessing_game**: Setup → simulate → track results
- **text_adventure**: Rooms → commands → navigation

### Output Test Tasks
- **output_test_basic**: Explicit stdout generation with print statements
- **output_test_calculations**: Verbose stdout with step-by-step calculations
- **output_test_loops**: Continuous stdout output during loop execution
- **output_test_stderr**: Explicit stderr/stdout separation using sys.stderr
- **output_test_multiline**: Multi-line stdout output with ASCII art

### Multi-Turn Test Tasks
- **multiturn_iterative_optimization**: Progressive algorithm development and optimization
- **multiturn_data_exploration**: Sequential data analysis building on previous results  
- **multiturn_debugging_journey**: Step-by-step debugging and enhancement process
- **multiturn_progressive_features**: Incremental feature building across multiple turns
- **multiturn_adaptive_analysis**: Analysis that adapts based on execution feedback
- **multiturn_error_recovery**: Learning error handling through trial and iteration

## Multi-Step Design

Each task is designed to require **2-4 steps with environment feedback**:

1. **Initial Setup**: Basic implementation
2. **Testing/Analysis**: Verify functionality 
3. **Enhancement/Visualization**: Add features or visualize
4. **Completion Verification**: Programmatic success check

The new success criterion functions ensure multi-step execution by only returning `True` when specific objectives are actually achieved, not just when code runs without errors.

## Success Criteria Types

### Real Validation Functions
- **fibonacci_sequence**: Checks both functions exist and return correct value
- **data_analysis_pipeline**: Verifies 100 data points and statistics calculated
- **file_processing**: Confirms filtered data exists
- **fits_basic_analysis**: Validates FITS data loaded with correct shape

### Random Functions (for demo)
- **50/50 random**: `random_success_criterion_50_50`
- **70% success**: `random_success_criterion_70_30` 
- **Progress-based**: `random_success_criterion_step_based` (more variables = higher success rate)
- **Always true/false**: For testing

## Usage

### With SLURM Script
```bash
# Edit run_demo.slurm and change:
export DEMO_CONFIG="standard"  # or "quick", "comprehensive", etc.
sbatch run_demo.slurm
```

### Direct Usage
```bash
# Quick demo
python astro_data_generator.py --config quick --fits-file demo_data/astro1_uv_imaging_telescope.fits

# Category-specific
python astro_data_generator.py --category programming --fits-file demo_data/astro1_uv_imaging_telescope.fits

# Specific tasks
python astro_data_generator.py --tasks fibonacci_sequence fits_basic_analysis --fits-file demo_data/astro1_uv_imaging_telescope.fits
```

## Output Organization

All outputs now go into timestamped run directories:

```
trajectories/
├── run_20250720_202930/
│   ├── demo_results_standard_294506.json    # Main results
│   ├── astronomy_code_snippets_*.py         # Generated code
│   └── trajectory_*.json                    # Individual trajectories
└── run_20250720_201208/
    └── ...
```

The trajectory files now include `success_criterion_met: true/false` fields showing when the programmatic validation succeeded.

## Output Capture Testing

New output test tasks verify that the pipeline correctly captures stdout/stderr:

### Quick Output Test
```bash
# Run output capture verification tests
python astro_data_generator.py --config output_tests --fits-file demo_data/astro1_uv_imaging_telescope.fits

# Or run the dedicated test runner
python demo_data/run_output_tests.py --service-url http://localhost:8002
```

### Output Test Types
1. **Basic Output**: Explicit stdout generation with print() statements
2. **Calculations**: Verbose stdout with detailed step-by-step output  
3. **Loops**: Continuous stdout during loop execution
4. **STDERR/STDOUT**: Explicit stderr/stdout separation using sys.stderr
5. **Multi-line**: Complex multi-line stdout with ASCII patterns

### Expected Output Examples
Each test has corresponding verification examples in `output_verification_examples.py`:
- Reference implementations showing expected output
- Verification functions to check captured vs expected
- Automated validation of output capture correctness

This keeps each demo run's outputs cleanly organized and prevents file conflicts. 