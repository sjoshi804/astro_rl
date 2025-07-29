"""
Task loader utility for loading tasks from JSON files
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any


class TaskLoader:
    """Utility class for loading and managing task definitions from JSON files"""
    
    def __init__(self, tasks_dir: str = None):
        if tasks_dir is None:
            # Default to tasks directory relative to this file
            tasks_dir = Path(__file__).parent / "tasks"
        self.tasks_dir = Path(tasks_dir)
        self._task_cache = {}
        self._category_cache = {}
    
    def _load_task_file(self, filename: str) -> Dict[str, Any]:
        """Load a single task file"""
        filepath = self.tasks_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Task file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks from all categories"""
        if not self._task_cache:
            all_tasks = []
            
            # Load all JSON files in the tasks directory
            for json_file in self.tasks_dir.glob("*.json"):
                try:
                    task_data = self._load_task_file(json_file.name)
                    tasks = task_data.get("tasks", [])
                    all_tasks.extend(tasks)
                    
                    # Cache by category
                    category = task_data.get("category", json_file.stem)
                    self._category_cache[category] = tasks
                    
                except Exception as e:
                    print(f"Warning: Failed to load task file {json_file}: {e}")
            
            # Cache by name for quick lookup
            for task in all_tasks:
                task_name = task.get("name")
                if task_name:
                    self._task_cache[task_name] = task
        
        return list(self._task_cache.values())
    
    def get_task_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific task by name"""
        self._get_all_tasks()  # Ensure cache is populated
        return self._task_cache.get(name)
    
    def get_tasks_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get tasks by category"""
        self._get_all_tasks()  # Ensure cache is populated
        return self._category_cache.get(category, [])
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all available tasks"""
        return self._get_all_tasks()
    
    def get_simple_tasks(self) -> List[Dict[str, Any]]:
        """Get the simplest tasks for quick demos"""
        simple_task_names = [
            "fibonacci_sequence",
            "data_analysis_pipeline", 
            "fits_basic_analysis"
        ]
        
        simple_tasks = []
        for name in simple_task_names:
            task = self.get_task_by_name(name)
            if task:
                simple_tasks.append(task)
        
        return simple_tasks
    
    def get_output_test_tasks(self) -> List[Dict[str, Any]]:
        """Get tasks specifically designed to test output capture"""
        return self.get_tasks_by_category("output_tests")
    
    def get_multiturn_test_tasks(self) -> List[Dict[str, Any]]:
        """Get tasks specifically designed to test multi-turn functionality"""
        return self.get_tasks_by_category("multiturn_tests")
    
    def get_available_categories(self) -> List[str]:
        """Get list of all available task categories"""
        self._get_all_tasks()  # Ensure cache is populated
        return list(self._category_cache.keys())
    
    def get_task_count_by_category(self) -> Dict[str, int]:
        """Get count of tasks in each category"""
        self._get_all_tasks()  # Ensure cache is populated
        return {category: len(tasks) for category, tasks in self._category_cache.items()}
    
    def search_tasks(self, query: str) -> List[Dict[str, Any]]:
        """Search tasks by name, description, or prompt content"""
        query_lower = query.lower()
        matching_tasks = []
        
        for task in self.get_all_tasks():
            # Search in name, description, and prompt
            searchable_text = " ".join([
                task.get("name", ""),
                task.get("description", ""),
                task.get("prompt", "")
            ]).lower()
            
            if query_lower in searchable_text:
                matching_tasks.append(task)
        
        return matching_tasks


# Create default instance for backward compatibility
_default_loader = TaskLoader()

# Backward compatibility functions
def get_task_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Get a specific task by name"""
    return _default_loader.get_task_by_name(name)

def get_tasks_by_category(category: str) -> List[Dict[str, Any]]:
    """Get tasks by category"""
    return _default_loader.get_tasks_by_category(category)

def get_simple_tasks() -> List[Dict[str, Any]]:
    """Get the simplest tasks for quick demos"""
    return _default_loader.get_simple_tasks()

def get_output_test_tasks() -> List[Dict[str, Any]]:
    """Get tasks specifically designed to test output capture"""
    return _default_loader.get_output_test_tasks()

def get_multiturn_test_tasks() -> List[Dict[str, Any]]:
    """Get tasks specifically designed to test multi-turn functionality"""
    return _default_loader.get_multiturn_test_tasks()

def get_all_tasks() -> List[Dict[str, Any]]:
    """Get all available tasks"""
    return _default_loader.get_all_tasks()


# Alias for backward compatibility (replacing ALL_DEMO_TASKS)
ALL_DEMO_TASKS = get_all_tasks()