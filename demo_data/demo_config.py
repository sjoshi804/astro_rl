"""
Demo Configuration Settings
Centralized configuration for demo runs with different complexity levels
"""

import os
from pathlib import Path

# Base paths
DEMO_DATA_DIR = Path(__file__).parent
PROJECT_ROOT = DEMO_DATA_DIR.parent
FITS_FILE_PATH = DEMO_DATA_DIR / "astro1_uv_imaging_telescope.fits"

# Service URLs (default for local development)
DEFAULT_SERVICE_URLS = {
    "code_exec_service": "http://localhost:8002",
    "completion_server": "http://localhost:8000"
}

# Demo configurations with different complexity levels
DEMO_CONFIGS = {
    "quick": {
        "name": "Quick Demo",
        "description": "Fast demo with 1-2 simple tasks",
        "max_turns": 5,
        "timeout_seconds": 60,
        "concurrent": False,
        "tasks": ["fibonacci_sequence", "data_analysis_pipeline"],
        "completion_criteria": "basic functionality working"
    },
    
    "standard": {
        "name": "Standard Demo", 
        "description": "Balanced demo showing multi-step capabilities",
        "max_turns": 8,
        "timeout_seconds": 120,
        "concurrent": True,
        "concurrency": 2,
        "tasks": ["fibonacci_sequence", "data_analysis_pipeline", "fits_basic_analysis", "file_processing"],
        "completion_criteria": "task objectives met"
    },
    
    "comprehensive": {
        "name": "Comprehensive Demo",
        "description": "Full demo showing all capabilities",
        "max_turns": 10,
        "timeout_seconds": 180,
        "concurrent": True,
        "concurrency": 3,
        "tasks": [
            "fibonacci_sequence", "data_analysis_pipeline", "fits_basic_analysis", 
            "physics_projectile", "file_processing", "statistics_experiment",
            "fits_advanced_analysis", "number_guessing_game"
        ],
        "completion_criteria": "all objectives achieved"
    },
    
    "astronomy_focused": {
        "name": "Astronomy Demo",
        "description": "Demo focused on astronomy data processing", 
        "max_turns": 8,
        "timeout_seconds": 150,
        "concurrent": True,
        "concurrency": 2,
        "tasks": ["fits_basic_analysis", "fits_advanced_analysis"],
        "completion_criteria": "visualization complete"
    },
    
    "programming_focused": {
        "name": "Programming Demo",
        "description": "Demo focused on programming tasks",
        "max_turns": 7,
        "timeout_seconds": 90,
        "concurrent": True, 
        "concurrency": 2,
        "tasks": ["fibonacci_sequence", "file_processing", "number_guessing_game", "text_adventure"],
        "completion_criteria": "programming objectives met"
    },
    
    "output_tests": {
        "name": "Output Capture Tests",
        "description": "Tests specifically designed to verify output capture works correctly",
        "max_turns": 5,
        "timeout_seconds": 60,
        "concurrent": True,
        "concurrency": 2,
        "tasks": ["output_test_basic", "output_test_calculations", "output_test_loops", "output_test_stderr", "output_test_multiline"],
        "completion_criteria": "output tests complete"
    },
    
    "multiturn_tests": {
        "name": "Multi-Turn Pipeline Tests",
        "description": "Tests specifically designed to verify multi-turn interaction functionality",
        "max_turns": 6,
        "timeout_seconds": 120,
        "concurrent": True,
        "concurrency": 2,
        "tasks": ["multiturn_iterative_optimization", "multiturn_data_exploration", "multiturn_debugging_journey", "multiturn_progressive_features"],
        "completion_criteria": "multi-turn functionality verified"
    }
}

# Environment variables override defaults
def get_service_urls():
    """Get service URLs from environment or use defaults"""
    return {
        "code_exec_service": os.getenv("CODE_EXEC_SERVICE_URL", DEFAULT_SERVICE_URLS["code_exec_service"]),
        "completion_server": os.getenv("COMPLETION_SERVER_URL", DEFAULT_SERVICE_URLS["completion_server"])
    }

def get_fits_file_path():
    """Get FITS file path, checking if it exists"""
    if FITS_FILE_PATH.exists():
        return str(FITS_FILE_PATH)
    
    # Fallback to old location if it exists
    old_path = PROJECT_ROOT / "astro1_uv_imaging_telescope.fits"
    if old_path.exists():
        return str(old_path)
    
    # Return demo path even if file doesn't exist (for error handling)
    return str(FITS_FILE_PATH)

def get_demo_config(config_name: str):
    """Get a demo configuration by name"""
    if config_name not in DEMO_CONFIGS:
        available = ", ".join(DEMO_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    config = DEMO_CONFIGS[config_name].copy()
    
    # Add runtime settings
    config["service_urls"] = get_service_urls()
    config["fits_file_path"] = get_fits_file_path()
    config["output_dir"] = str(PROJECT_ROOT / "trajectories")
    
    return config

def list_available_configs():
    """List all available demo configurations"""
    configs = []
    for name, config in DEMO_CONFIGS.items():
        configs.append({
            "name": name,
            "title": config["name"],
            "description": config["description"],
            "tasks": len(config["tasks"]),
            "max_turns": config["max_turns"]
        })
    return configs 