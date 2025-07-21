"""
Simple Multi-Step Demo Tasks for RL Training
These tasks are designed to be simple but require multiple steps and environment feedback
"""

# Simple programming tasks that require multiple steps
PROGRAMMING_TASKS = [
    {
        "name": "fibonacci_sequence",
        "prompt": "Create a function to calculate the nth Fibonacci number, test it with n=10, and then optimize it using memoization",
        "completion_criteria": "test passed and optimized",
        "expected_steps": 3,
        "description": "Start with basic recursive function, test it, then improve with memoization"
    },
    
    {
        "name": "data_analysis_pipeline", 
        "prompt": "Create a list of 100 random numbers, calculate basic statistics (mean, median, std), and create a histogram visualization",
        "completion_criteria": "visualization complete",
        "expected_steps": 3,
        "description": "Generate data -> analyze -> visualize"
    },
    
    {
        "name": "file_processing",
        "prompt": "Create a CSV file with sample data (name, age, city), read it back, and filter for people over 25",
        "completion_criteria": "filtering complete",
        "expected_steps": 3,
        "description": "Write CSV -> read CSV -> process data"
    },
    
    {
        "name": "web_data_fetch",
        "prompt": "Make a simple HTTP request to get JSON data, parse it to extract specific fields, and save results to a file",
        "completion_criteria": "data saved",
        "expected_steps": 3,
        "description": "Fetch -> parse -> save"
    }
]

# Math/Science tasks that build on each other
MATH_SCIENCE_TASKS = [
    {
        "name": "physics_projectile",
        "prompt": "Calculate the trajectory of a projectile with initial velocity 50 m/s at 45 degrees, plot the path, and find the maximum height",
        "completion_criteria": "maximum height found",
        "expected_steps": 3,
        "description": "Calculate trajectory -> plot -> find maximum"
    },
    
    {
        "name": "statistics_experiment",
        "prompt": "Generate two datasets from different distributions, perform a t-test to compare them, and visualize the results",
        "completion_criteria": "test complete",
        "expected_steps": 3,
        "description": "Generate data -> run test -> visualize"
    }
]

# Astronomy tasks (simplified versions)
ASTRONOMY_TASKS = [
    {
        "name": "fits_basic_analysis",
        "prompt": f"Load the FITS file from demo_data/astro1_uv_imaging_telescope.fits, display basic info, and create a simple grayscale plot",
        "completion_criteria": "plot created",
        "expected_steps": 2,
        "description": "Load FITS -> analyze -> visualize"
    },
    
    {
        "name": "fits_advanced_analysis", 
        "prompt": f"Load the FITS file from demo_data/astro1_uv_imaging_telescope.fits, calculate pixel statistics, and create a histogram of intensity values",
        "completion_criteria": "histogram complete",
        "expected_steps": 3,
        "description": "Load -> calculate stats -> create histogram"
    }
]

# Game/Interactive tasks
INTERACTIVE_TASKS = [
    {
        "name": "number_guessing_game",
        "prompt": "Create a number guessing game where the computer picks a random number 1-100, simulate a few guesses, and track the results",
        "completion_criteria": "game complete", 
        "expected_steps": 3,
        "description": "Setup game -> simulate guesses -> track results"
    },
    
    {
        "name": "text_adventure",
        "prompt": "Create a simple text adventure with rooms, items, and basic commands. Implement at least 3 rooms and test navigation",
        "completion_criteria": "navigation tested",
        "expected_steps": 4,
        "description": "Create structure -> add rooms -> add commands -> test"
    }
]

# All tasks combined
ALL_DEMO_TASKS = PROGRAMMING_TASKS + MATH_SCIENCE_TASKS + ASTRONOMY_TASKS + INTERACTIVE_TASKS

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
        "interactive": INTERACTIVE_TASKS
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