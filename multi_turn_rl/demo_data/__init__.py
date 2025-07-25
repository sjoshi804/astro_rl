"""
Demo data package for Multi-Turn RL
Contains task definitions and success criteria
"""

from .task_loader import TaskLoader, get_task_by_name, get_tasks_by_category
from .success_criteria import get_success_criterion_for_task

__all__ = [
    "TaskLoader",
    "get_task_by_name", 
    "get_tasks_by_category",
    "get_success_criterion_for_task"
]