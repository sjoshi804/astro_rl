# Demo Data for Astro RL

This directory contains organized demo input data and configurations for testing the multi-turn code generation and execution system.

## Directory Structure

```
demo_data/
├── README.md                          # This file
├── astro1_uv_imaging_telescope.fits   # Astronomy FITS file for demo
├── simple_multistep_tasks.py          # Multi-step task definitions
├── demo_config.py                     # Configuration settings
├── demo_runner.py                     # Main demo execution script
└── sample_outputs/                    # Example outputs (created after runs)
```

## Demo Tasks Categories

### 1. Programming Tasks
- **Fibonacci Sequence**: Basic recursion → testing → optimization with memoization
- **Data Analysis Pipeline**: Generate data → calculate stats → create visualization
- **File Processing**: Create CSV → read CSV → filter data
- **Web Data Fetch**: HTTP request → parse JSON → save results

### 2. Math/Science Tasks  
- **Physics Projectile**: Calculate trajectory → plot path → find maximum height
- **Statistics Experiment**: Generate datasets → run t-test → visualize results

### 3. Astronomy Tasks
- **FITS Basic Analysis**: Load FITS file → display info → create grayscale plot
- **FITS Advanced Analysis**: Load FITS → calculate statistics → create histogram

### 4. Interactive Tasks
- **Number Guessing Game**: Setup game → simulate guesses → track results
- **Text Adventure**: Create structure → add rooms → implement commands → test navigation

## Demo Configurations

### Quick Demo (`quick`)
- **Tasks**: 2 simple tasks (fibonacci, data analysis)
- **Duration**: ~2-3 minutes
- **Purpose**: Fast verification that system works

### Standard Demo (`standard`)
- **Tasks**: 4 balanced tasks across categories
- **Duration**: ~5-8 minutes  
- **Purpose**: Show multi-step capabilities and variety

### Comprehensive Demo (`comprehensive`)
- **Tasks**: 8 tasks covering all categories
- **Duration**: ~10-15 minutes
- **Purpose**: Full system demonstration

### Specialized Demos
- **Astronomy Focused**: FITS file analysis tasks
- **Programming Focused**: Code development tasks

## Key Features Demonstrated

### Multi-Step Execution
Each task requires 2-4 steps to complete, with the model needing to:
1. **Plan** the approach
2. **Implement** initial code
3. **Debug** based on execution feedback  
4. **Refine** or extend functionality

### Environment Feedback Loop
Tasks are designed so the model must:
- Check if data loaded correctly
- Verify calculations/outputs
- Handle errors and retry
- Build upon previous steps

### Completion Criteria
Each task has specific completion criteria that require:
- Successful code execution
- Specific outputs (plots, files, calculations)
- Verification steps
- Error handling

## Usage

### Quick Start
```bash
cd /path/to/astro_rl
python demo_data/demo_runner.py --config quick
```

### Run Specific Configuration
```bash
python demo_data/demo_runner.py --config standard --save-code
```

### Custom Tasks
```bash
python demo_data/demo_runner.py --tasks fibonacci_sequence file_processing
```

### With Custom Service URLs
```bash
export CODE_EXEC_SERVICE_URL="http://your-service:8002"
python demo_data/demo_runner.py --config comprehensive
```

## Integration with Main System

These demo tasks integrate with:

- **`code_and_exec_service.py`**: Multi-turn code generation and execution
- **`ray_execution_engine.py`**: Persistent execution environment
- **`completion_server.py`**: LLM-based code completion
- **`astro_data_generator.py`**: Task execution and result analysis

## Expected Outcomes

### Learning Behaviors
The RL system should learn to:
- **Plan ahead**: Break complex tasks into steps
- **Debug iteratively**: Use execution feedback to fix errors
- **Build incrementally**: Extend working code rather than rewriting
- **Handle persistence**: Use variables and state across turns

### Success Metrics
- **Completion rate**: Percentage of tasks completed successfully
- **Step efficiency**: Average steps needed per task
- **Error recovery**: Ability to fix errors and continue
- **Code quality**: Correctness and style of generated code

## Troubleshooting

### FITS File Not Found
If astronomy tasks fail, check that `astro1_uv_imaging_telescope.fits` is in this directory.

### Service Connection Errors
Ensure all services are running:
```bash
# Check service health
curl http://localhost:8002/health
curl http://localhost:8000/health
```

### Ray Execution Issues
Clear Ray processes if needed:
```bash
ray stop --force
```

### Task Completion Issues
Check completion criteria in task definitions - they may need adjustment based on your model's output patterns. 