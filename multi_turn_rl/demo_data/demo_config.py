"""
Demo Configuration Definitions
Maps demo config names to task sets and parameters
"""

DEMO_CONFIGS = {
    "quick": {
        "description": "Fast 2-task demo (~3 min) - fibonacci, data analysis",
        "tasks": ["fibonacci_sequence", "data_analysis_pipeline"],
        "max_turns": 5,
        "timeout_seconds": 180,
        "concurrent": False,
        "concurrency": 1
    },
    
    "standard": {
        "description": "Balanced 4-task demo (~8 min) - mixed categories",
        "tasks": ["fibonacci_sequence", "data_analysis_pipeline", "file_processing", "fits_basic_analysis"],
        "max_turns": 8,
        "timeout_seconds": 300,
        "concurrent": True,
        "concurrency": 2
    },
    
    "comprehensive": {
        "description": "Full 8-task demo (~15 min) - all categories",
        "tasks": [
            "fibonacci_sequence", "data_analysis_pipeline", "file_processing", "web_data_fetch",
            "quadratic_solver", "prime_number_sieve", "fits_basic_analysis", "stellar_photometry"
        ],
        "max_turns": 10,
        "timeout_seconds": 600,
        "concurrent": True,
        "concurrency": 3
    },
    
    "astronomy_focused": {
        "description": "FITS analysis tasks (~10 min) - astronomy only",
        "tasks": ["fits_basic_analysis", "stellar_photometry"],
        "max_turns": 8,
        "timeout_seconds": 400,
        "concurrent": True,
        "concurrency": 2
    },
    
    "programming_focused": {
        "description": "Programming tasks (~10 min) - coding challenges",
        "tasks": ["fibonacci_sequence", "data_analysis_pipeline", "file_processing", "web_data_fetch"],
        "max_turns": 8,
        "timeout_seconds": 400,
        "concurrent": True,
        "concurrency": 2
    },
    
    "output_tests": {
        "description": "Output capture tests (~5 min) - verify stdout/stderr capture",
        "tasks": ["stdout_test", "stderr_test", "mixed_output_test", "large_output_test", "multiline_output_test"],
        "max_turns": 3,
        "timeout_seconds": 120,
        "concurrent": True,
        "concurrency": 3
    },
    
    "multiturn_tests": {
        "description": "Multi-turn interaction tests (~8 min) - verify multi-step functionality",
        "tasks": [
            "multiturn_iterative_optimization",
            "fibonacci_sequence", 
            "data_analysis_pipeline",
            "fits_basic_analysis"
        ],
        "max_turns": 10,
        "timeout_seconds": 300,
        "concurrent": True,
        "concurrency": 3
    }
}


def get_demo_config(config_name: str) -> dict:
    """Get demo configuration by name"""
    if config_name not in DEMO_CONFIGS:
        raise ValueError(f"Unknown demo config: {config_name}. Available: {list(DEMO_CONFIGS.keys())}")
    return DEMO_CONFIGS[config_name]


def list_available_configs() -> list:
    """List all available demo configuration names"""
    return list(DEMO_CONFIGS.keys())


def get_config_descriptions() -> dict:
    """Get descriptions for all available configs"""
    return {name: config["description"] for name, config in DEMO_CONFIGS.items()}