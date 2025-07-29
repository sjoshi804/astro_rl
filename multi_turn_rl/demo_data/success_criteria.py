"""
Success Criterion Functions for Demo Tasks
These functions check if task objectives have been met by examining the execution context.
For demo purposes, these include both real checks and some random functions.
"""

import random
import numpy as np
from typing import Dict, Any

def fibonacci_success_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if fibonacci functions are defined and working"""
    try:
        # Check if both fibonacci functions are defined
        if 'fibonacci' not in locals_dict or 'fibonacci_memo' not in locals_dict:
            return False
        
        # Test both functions with n=5
        fib_func = locals_dict['fibonacci']
        fib_memo_func = locals_dict['fibonacci_memo']
        
        result1 = fib_func(5)
        result2 = fib_memo_func(5)
        
        # Both should return 5 (5th fibonacci number)
        return result1 == 5 and result2 == 5
    except:
        return False

def data_analysis_success_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if data analysis was completed with histogram"""
    try:
        # Check if random_numbers array exists and has 100 elements
        if 'random_numbers' not in locals_dict:
            return False
        
        data = locals_dict['random_numbers']
        if len(data) != 100:
            return False
            
        # Check if statistics were calculated
        required_vars = ['mean', 'median', 'std']
        return all(var in locals_dict for var in required_vars)
    except:
        return False

def file_processing_success_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if CSV processing was completed"""
    try:
        # Check if filtered data exists
        if 'filtered_data' not in locals_dict:
            return False
        
        filtered_data = locals_dict['filtered_data']
        # Should have some filtered results
        return len(filtered_data) > 0
    except:
        return False

def fits_analysis_success_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if FITS file was loaded and analyzed"""
    try:
        # Check if data and header were loaded
        required_vars = ['data', 'header']
        if not all(var in locals_dict for var in required_vars):
            return False
        
        data = locals_dict['data']
        # Check if data has expected shape (512x512)
        return hasattr(data, 'shape') and data.shape == (512, 512)
    except:
        return False

def physics_projectile_success_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if projectile physics calculation was completed"""
    try:
        # Look for trajectory data or max height calculation
        physics_vars = ['t', 'x', 'y', 'max_height', 'trajectory']
        return any(var in locals_dict for var in physics_vars)
    except:
        return False

# Output test success criteria - these check for specific outputs
def output_test_basic_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if basic output test completed by looking for expected outputs"""
    # This is a simple test - always return True if code executed without error
    # The real test is whether we capture the output properly
    return True

def output_test_calculations_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if factorial calculation was completed"""
    try:
        # Look for factorial result or calculation
        factorial_vars = ['factorial', 'result', 'fact']
        return any(var in locals_dict for var in factorial_vars)
    except:
        return False

def output_test_loops_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if loop test was completed"""
    try:
        # Look for sum calculation or loop variables
        loop_vars = ['sum', 'total', 'i', 'numbers']
        return any(var in locals_dict for var in loop_vars)
    except:
        return False

def output_test_stderr_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if stderr/stdout test was completed"""
    # Look for sys import and error handling
    return 'sys' in locals_dict or any('error' in str(v).lower() for v in locals_dict.values())

def output_test_multiline_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if multiline pattern test was completed"""
    try:
        # Look for pattern-related variables
        pattern_vars = ['pattern', 'lines', 'art', 'box', 'triangle']
        return any(var in locals_dict for var in pattern_vars)
    except:
        return False

# Multi-turn test success criteria - these are designed to require multiple steps
def multiturn_iterative_optimization_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if iterative optimization has progressed"""
    try:
        # Look for sorting function and timing measurements
        optimization_vars = ['bubble_sort', 'sort', 'time', 'performance', 'optimized']
        has_sorting = any(var in locals_dict for var in ['bubble_sort', 'sort_function', 'sort'])
        has_timing = any(var in locals_dict for var in ['time', 'timing', 'elapsed', 'duration'])
        # Return True if we have evidence of progression
        return has_sorting and (has_timing or len(locals_dict) > 5)
    except:
        return False

def multiturn_data_exploration_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if data exploration has progressed through multiple steps"""
    try:
        # Look for dataset creation, analysis, and visualization
        data_vars = ['dataset', 'data', 'numbers', 'random_data']
        analysis_vars = ['mean', 'std', 'statistics', 'analysis']
        viz_vars = ['plot', 'plt', 'matplotlib', 'chart']
        
        has_data = any(var in locals_dict for var in data_vars)
        has_analysis = any(var in locals_dict for var in analysis_vars)
        has_viz = any(var in locals_dict for var in viz_vars)
        
        # Require at least 2 of the 3 components for progression
        return sum([has_data, has_analysis, has_viz]) >= 2
    except:
        return False

def multiturn_debugging_journey_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if debugging process has progressed"""
    try:
        # Look for function definition and error handling
        has_function = any('divide' in str(var).lower() for var in locals_dict.keys())
        has_error_handling = any(var in locals_dict for var in ['try', 'except', 'error'])
        validation_vars = ['validate', 'check', 'validation']
        has_validation = any(var in locals_dict for var in validation_vars)
        
        # Progress if we have function definition and some form of improvement
        return has_function and (has_error_handling or has_validation)
    except:
        return False

def multiturn_progressive_features_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if progressive features have been built"""
    try:
        # Look for calculator functions
        calc_functions = ['add', 'subtract', 'multiply', 'calculator']
        function_count = sum(1 for var in locals_dict.keys() if any(calc in str(var).lower() for calc in calc_functions))
        
        # Progress if we have multiple calculator functions
        return function_count >= 2
    except:
        return False

def multiturn_adaptive_analysis_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if adaptive analysis has evolved"""
    try:
        # Look for analysis function and multiple analysis types
        has_analysis_func = any('analyz' in str(var).lower() for var in locals_dict.keys())
        analysis_types = ['mean', 'std', 'outlier', 'standard_deviation', 'variance']
        analysis_count = sum(1 for var in locals_dict.keys() if any(atype in str(var).lower() for atype in analysis_types))
        
        # Progress if we have analysis function and multiple analysis types
        return has_analysis_func and analysis_count >= 2
    except:
        return False

def multiturn_error_recovery_criterion(locals_dict: Dict[str, Any]) -> bool:
    """Check if error recovery strategies have been developed"""
    try:
        # Look for error handling patterns
        error_vars = ['error', 'exception', 'try', 'except', 'importerror', 'filenotfounderror']
        error_handling_count = sum(1 for var in str(locals_dict).lower() if any(err in var for err in error_vars))
        
        # Progress if we have evidence of error handling development
        return error_handling_count >= 3
    except:
        return False

# Random success criteria for demo purposes
def random_success_criterion_10_90(locals_dict: Dict[str, Any]) -> bool:
    """Random 50/50 success for demo purposes"""
    return random.random() < 0.1

def random_success_criterion_70_30(locals_dict: Dict[str, Any]) -> bool:
    """Random 70% success rate for demo purposes"""
    return random.random() < 0.7

def random_success_criterion_always_true(locals_dict: Dict[str, Any]) -> bool:
    """Always succeeds - for testing"""
    return True

def random_success_criterion_always_false(locals_dict: Dict[str, Any]) -> bool:
    """Never succeeds - for testing"""
    return False

def random_success_criterion_step_based(locals_dict: Dict[str, Any]) -> bool:
    """Success rate increases with more variables defined (simulates progress)"""
    num_vars = len(locals_dict)
    # More variables = higher success probability
    success_rate = min(0.9, num_vars * 0.1)
    return random.random() < success_rate

# Map task names to their success criteria functions
TASK_SUCCESS_CRITERIA = {
    "fibonacci_sequence": random_success_criterion_10_90,
    "data_analysis_pipeline": random_success_criterion_10_90, 
    "file_processing": random_success_criterion_10_90,
    "fits_basic_analysis": random_success_criterion_10_90,
    "fits_advanced_analysis": random_success_criterion_10_90,
    "physics_projectile": random_success_criterion_10_90,
    
    # Output test criteria
    "output_test_basic": random_success_criterion_10_90,
    "output_test_calculations": random_success_criterion_10_90,
    "output_test_loops": random_success_criterion_10_90,
    "output_test_stderr": random_success_criterion_10_90,
    "output_test_multiline": random_success_criterion_10_90,
    
    # Multi-turn test criteria
    "multiturn_iterative_optimization": random_success_criterion_10_90,
    "multiturn_data_exploration": random_success_criterion_10_90,
    "multiturn_debugging_journey": random_success_criterion_10_90,
    "multiturn_progressive_features": random_success_criterion_10_90,
    "multiturn_adaptive_analysis": random_success_criterion_10_90,
    "multiturn_error_recovery": random_success_criterion_10_90,
    
    # All other tasks use 10_90 random criteria
    "web_data_fetch": random_success_criterion_10_90,
    "statistics_experiment": random_success_criterion_10_90,
    "number_guessing_game": random_success_criterion_10_90,
    "text_adventure": random_success_criterion_10_90,
    "multiturn_random_success": random_success_criterion_10_90,
}

def get_success_criterion_for_task(task_name: str):
    """Get the success criterion function for a given task"""
    return TASK_SUCCESS_CRITERIA.get(task_name, random_success_criterion_10_90) 