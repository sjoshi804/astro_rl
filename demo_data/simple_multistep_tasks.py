"""
Simple Multi-Step Demo Tasks for RL Training
These tasks are designed to be simple but require multiple steps and environment feedback
"""

# Simple programming tasks that require multiple steps
PROGRAMMING_TASKS = [
    {
        "name": "fibonacci_sequence",
        "prompt": "Create a function to calculate the nth Fibonacci number, test it with n=10, and then optimize it using memoization. Print 'FIBONACCI TEST PASSED' when both functions work correctly and show the optimization benefit.",
        "completion_criteria": "test passed and optimized",
        "expected_steps": 3,
        "description": "Start with basic recursive function, test it, then improve with memoization"
    },
    
    {
        "name": "data_analysis_pipeline", 
        "prompt": "Create a list of 100 random numbers, calculate basic statistics (mean, median, std), and create a histogram visualization. Print 'VISUALIZATION COMPLETE' when the histogram is successfully created.",
        "completion_criteria": "visualization complete",
        "expected_steps": 3,
        "description": "Generate data -> analyze -> visualize"
    },
    
    {
        "name": "file_processing",
        "prompt": "Create a CSV file with sample data (name, age, city), read it back, and filter for people over 25. Print 'FILTERING COMPLETE' when you've successfully filtered and displayed the results.",
        "completion_criteria": "filtering complete",
        "expected_steps": 3,
        "description": "Write CSV -> read CSV -> process data"
    },
    
    {
        "name": "web_data_fetch",
        "prompt": "Make a simple HTTP request to get JSON data, parse it to extract specific fields, and save results to a file. Print 'DATA SAVED' when file is successfully written.",
        "completion_criteria": "data saved",
        "expected_steps": 3,
        "description": "Fetch -> parse -> save"
    }
]

# Math/Science tasks that build on each other
MATH_SCIENCE_TASKS = [
    {
        "name": "physics_projectile",
        "prompt": "Calculate the trajectory of a projectile with initial velocity 50 m/s at 45 degrees, plot the path, and find the maximum height. Print 'MAXIMUM HEIGHT FOUND: X meters' where X is the calculated maximum height.",
        "completion_criteria": "maximum height found",
        "expected_steps": 3,
        "description": "Calculate trajectory -> plot -> find maximum"
    },
    
    {
        "name": "statistics_experiment",
        "prompt": "Generate two datasets from different distributions, perform a t-test to compare them, and visualize the results. Print 'TEST COMPLETE' when statistical test is done and results are visualized.",
        "completion_criteria": "test complete",
        "expected_steps": 3,
        "description": "Generate data -> run test -> visualize"
    }
]

# Astronomy tasks (simplified versions)
ASTRONOMY_TASKS = [
    {
        "name": "fits_basic_analysis",
        "prompt": f"Load the FITS file from demo_data/astro1_uv_imaging_telescope.fits, display basic info, and create a simple grayscale plot. Print 'PLOT CREATED' when the visualization is successfully generated.",
        "completion_criteria": "plot created",
        "expected_steps": 2,
        "description": "Load FITS -> analyze -> visualize"
    },
    
    {
        "name": "fits_advanced_analysis", 
        "prompt": f"Load the FITS file from demo_data/astro1_uv_imaging_telescope.fits, calculate pixel statistics, and create a histogram of intensity values. Print 'HISTOGRAM COMPLETE' when the histogram is successfully created.",
        "completion_criteria": "histogram complete",
        "expected_steps": 3,
        "description": "Load -> calculate stats -> create histogram"
    }
]

# Game/Interactive tasks
INTERACTIVE_TASKS = [
    {
        "name": "number_guessing_game",
        "prompt": "Create a number guessing game where the computer picks a random number 1-100, simulate a few guesses, and track the results. Print 'GAME COMPLETE' when you've implemented and tested the game logic.",
        "completion_criteria": "game complete", 
        "expected_steps": 3,
        "description": "Setup game -> simulate guesses -> track results"
    },
    
    {
        "name": "text_adventure",
        "prompt": "Create a simple text adventure with rooms, items, and basic commands. Implement at least 3 rooms and test navigation. Print 'NAVIGATION TESTED' when room transitions work correctly.",
        "completion_criteria": "navigation tested",
        "expected_steps": 4,
        "description": "Create structure -> add rooms -> add commands -> test"
    }
]

# Test tasks for output capture verification
OUTPUT_TEST_TASKS = [
    {
        "name": "output_test_basic",
        "prompt": "Write Python code that explicitly generates output to STDOUT. Use print() statements to output 'Hello, World!' and the current timestamp. Make sure to print 'OUTPUT TEST COMPLETE' at the end. The goal is to test that our pipeline correctly captures standard output.",
        "completion_criteria": "output test complete",
        "expected_steps": 1,
        "description": "Basic stdout capture test with explicit print statements"
    },
    
    {
        "name": "output_test_calculations",
        "prompt": "Write Python code that generates verbose output to STDOUT showing step-by-step calculations. Calculate factorial of 5 and print each multiplication step (e.g., '1 * 1 = 1', '1 * 2 = 2', etc.). Print the final result and 'CALCULATION COMPLETE'. Use multiple print() statements to generate rich stdout output.",
        "completion_criteria": "calculation complete",
        "expected_steps": 2,
        "description": "Verbose stdout output with step-by-step calculations"
    },
    
    {
        "name": "output_test_loops",
        "prompt": "Write Python code that generates continuous output to STDOUT during loop execution. Create a loop from 1 to 10, printing each number and running total (e.g., 'Number: 1, Running total: 1'). After the loop, print a summary with the final sum. End with 'LOOP TEST FINISHED'. Focus on generating lots of stdout output.",
        "completion_criteria": "loop test finished",
        "expected_steps": 2,
        "description": "Sequential stdout output during loop execution"
    },
    
    {
        "name": "output_test_stderr",
        "prompt": "Write Python code that explicitly generates output to both STDOUT and STDERR. Import sys and use print('message', file=sys.stderr) to write to stderr, and regular print() for stdout. Create a try/except block that catches a division by zero error, prints the error to stderr, then prints success message to stdout. End with 'ERROR HANDLING COMPLETE' to stdout.",
        "completion_criteria": "error handling complete",
        "expected_steps": 2,
        "description": "Test stderr and stdout separation with explicit sys.stderr usage"
    },
    
    {
        "name": "output_test_multiline",
        "prompt": "Write Python code that generates multi-line output to STDOUT. Create a function that prints ASCII art (like a box made of '+', '-', '|' characters) line by line using multiple print() statements. Call the function, then print statistics about the pattern (number of lines, characters, etc.). End with 'PATTERN COMPLETE'. Focus on generating complex multi-line stdout output.",
        "completion_criteria": "pattern complete",
        "expected_steps": 2,
        "description": "Multi-line stdout output with ASCII art patterns"
    }
]

# Multi-turn test tasks for pipeline testing
MULTITURN_TEST_TASKS = [
    {
        "name": "multiturn_iterative_optimization",
        "prompt": "Start by implementing a simple bubble sort algorithm for the list [64, 34, 25, 12, 22, 11, 90]. First, implement basic bubble sort and test it. Then in subsequent steps, add timing measurements, optimize the algorithm, and finally compare performance. Each step should build on the previous execution results.",
        "completion_criteria": "optimization complete",
        "expected_steps": 4,
        "description": "Multi-turn iterative algorithm development and optimization"
    }
    
    # ,
    
    # {
    #     "name": "multiturn_data_exploration",
    #     "prompt": "Begin by creating a dataset of 50 random numbers between 1-100. Analyze the initial dataset to find basic statistics. Then based on those results, create visualizations. Finally, generate a second dataset and compare the two. Each turn should build on findings from previous steps.",
    #     "completion_criteria": "comparison complete",
    #     "expected_steps": 4,
    #     "description": "Multi-turn data analysis building on previous results"
    # },
    
    # {
    #     "name": "multiturn_debugging_journey",
    #     "prompt": "Start with this intentionally buggy code: 'def divide_numbers(a, b): return a / b; result = divide_numbers(10, 0)'. Run it to see the error, then fix the error. After fixing, enhance the function with input validation, then add comprehensive error handling with custom messages.",
    #     "completion_criteria": "debugging complete",
    #     "expected_steps": 4,
    #     "description": "Multi-turn debugging and enhancement process"
    # },
    
    # {
    #     "name": "multiturn_progressive_features",
    #     "prompt": "Build a simple calculator step by step. Start with basic addition function and test it. Then add subtraction in the next turn. Then multiplication in the next turn. Finally, create a complete calculator interface that uses all functions. Each step must build on previous working code.",
    #     "completion_criteria": "calculator complete",
    #     "expected_steps": 4,
    #     "description": "Multi-turn progressive feature development"
    # },
    
    # {
    #     "name": "multiturn_adaptive_analysis",
    #     "prompt": "Create a function to analyze a list of numbers. Start with [1, 2, 3, 4, 5] and implement basic analysis (mean, sum). Run it and observe results. Then based on the output characteristics, add more sophisticated analysis (standard deviation, outliers). Finally, test with a different dataset [100, 2, 50, 1, 200] and adapt the analysis based on the new results.",
    #     "completion_criteria": "adaptive analysis complete",
    #     "expected_steps": 4,
    #     "description": "Multi-turn adaptive data analysis based on execution feedback"
    # },
    
    # {
    #     "name": "multiturn_error_recovery",
    #     "prompt": "Attempt to import a non-existent module 'fake_module' and handle the ImportError. Then try to access a non-existent file and handle the FileNotFoundError. Finally, create a robust function that handles multiple error types based on what you learned from the previous attempts.",
    #     "completion_criteria": "error recovery complete",
    #     "expected_steps": 3,
    #     "description": "Multi-turn error handling and recovery learning"
    # },
    # {
    #     "name": "multiturn_random_success",
    #     "prompt": "Demonstrate a multi-step process: first print 'Step 1 complete', then print 'Step 2 complete', then print 'Step 3 complete'. Each step should be in a separate code cell. The task will randomly succeed with 50% probability on each turn, so it may require multiple turns to complete.",
    #     "completion_criteria": "random success",
    #     "expected_steps": 3,
    #     "description": "Multi-turn test with random 50% success criterion for pipeline testing"
    # }
]

# All tasks combined
ALL_DEMO_TASKS = PROGRAMMING_TASKS + MATH_SCIENCE_TASKS + ASTRONOMY_TASKS + INTERACTIVE_TASKS + OUTPUT_TEST_TASKS + MULTITURN_TEST_TASKS

def get_task_by_name(name: str):
    """Get a specific task by name"""
    for task in ALL_DEMO_TASKS:
        if task["name"] == name:
            return task
    return None

def get_tasks_by_category(category: str):
    """Get tasks by category"""
    categories = {
        "programming": PROGRAMMING_TASKS,
        "math_science": MATH_SCIENCE_TASKS, 
        "astronomy": ASTRONOMY_TASKS,
        "interactive": INTERACTIVE_TASKS,
        "output_tests": OUTPUT_TEST_TASKS,
        "multiturn_tests": MULTITURN_TEST_TASKS
    }
    return categories.get(category, [])

def get_simple_tasks():
    """Get the simplest tasks for quick demos"""
    simple_tasks = [
        get_task_by_name("fibonacci_sequence"),
        get_task_by_name("data_analysis_pipeline"),
        get_task_by_name("fits_basic_analysis")
    ]
    return [task for task in simple_tasks if task is not None]

def get_output_test_tasks():
    """Get tasks specifically designed to test output capture"""
    return OUTPUT_TEST_TASKS

def get_multiturn_test_tasks():
    """Get tasks specifically designed to test multi-turn functionality"""
    return MULTITURN_TEST_TASKS 