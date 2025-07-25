"""
Data models for the multi-turn RL framework
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime


class Turn(BaseModel):
    step: int
    prompt: str
    code: str
    execution_output: str
    execution_success: bool
    timestamp: datetime
    success_criterion_met: Optional[bool] = False


class Trajectory(BaseModel):
    trajectory_id: str
    turns: List[Turn]
    final_reward: float
    termination_reason: str  # "completion_criteria_met", "max_steps", "execution_error"
    total_steps: int
    created_at: datetime


class TrajectoryRequest(BaseModel):
    initial_prompts: List[str]
    max_turns: int = 10
    completion_criteria: Optional[str] = None  # Custom completion condition
    task_names: Optional[List[str]] = None  # Optional task names for success criteria


class BatchTrajectoryRequest(BaseModel):
    requests: List[TrajectoryRequest]


class CodeGenerationRequest(BaseModel):
    prompt: str
    num_completions: int = 4
    temperature: float = 0.8
    max_tokens: int = 512


class CodeExecutionRequest(BaseModel):
    code: str
    timeout: int = 10
    language: str = "python"