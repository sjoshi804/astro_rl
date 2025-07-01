"""
Code Generation & Execution Service
Orchestrates multi-turn code generation and execution for RL training
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
import asyncio
import json
import json
import os
from pathlib import Path
import logging
import argparse
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration variables (set via command line args)
COMPLETION_SERVER_URL = None
EXEC_ENGINE_URL = None
MAX_TURNS = None
TIMEOUT_SECONDS = None
TRAJECTORY_OUTPUT_DIR = None
TRAJECTORY_RUN_DIR: Optional[Path] = None  # created on first save

app = FastAPI(title="Code Generation & Execution Service")

# Data Models
class CodeGenerationRequest(BaseModel):
    prompt: str
    num_completions: int = 4
    temperature: float = 0.8
    max_tokens: int = 512

class CodeExecutionRequest(BaseModel):
    code: str
    timeout: int = 10
    language: str = "python"

class Turn(BaseModel):
    step: int
    prompt: str
    code: str
    execution_output: str
    execution_success: bool
    timestamp: datetime

class Trajectory(BaseModel):
    trajectory_id: str
    turns: List[Turn]
    final_reward: float
    termination_reason: str  # "completion_criteria_met", "max_steps", "execution_error"
    total_steps: int
    created_at: datetime

class TrajectoryRequest(BaseModel):
    initial_prompts: List[str]
    max_turns: int = MAX_TURNS
    completion_criteria: Optional[str] = None  # Custom completion condition

class BatchTrajectoryRequest(BaseModel):
    requests: List[TrajectoryRequest]

# In-memory storage for active trajectories
active_trajectories: Dict[str, Dict] = {}

# HTTP client for external services
http_client = httpx.AsyncClient(timeout=TIMEOUT_SECONDS)

# Utility Functions
def save_trajectory_to_file(trajectory: Trajectory) -> str:
    """Save trajectory to a JSON file and return the file path"""
    if not TRAJECTORY_OUTPUT_DIR:
        return ""
    
    try:
        base_dir = Path(TRAJECTORY_OUTPUT_DIR)
        # Lazily create a run-specific subdirectory once per service run
        if TRAJECTORY_RUN_DIR is None:
            run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            TRAJECTORY_RUN_DIR = base_dir / f"run_{run_ts}"
            TRAJECTORY_RUN_DIR.mkdir(parents=True, exist_ok=True)

        # Create filename with trajectory's creation timestamp and id
        ts = trajectory.created_at.strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_{ts}_{trajectory.trajectory_id}.json"
        filepath = TRAJECTORY_RUN_DIR / filename
        
        # Convert trajectory to JSON-serializable format
        trajectory_data = {
            "trajectory_id": trajectory.trajectory_id,
            "total_steps": trajectory.total_steps,
            "final_reward": trajectory.final_reward,
            "termination_reason": trajectory.termination_reason,
            "created_at": trajectory.created_at.isoformat(),
            "turns": [
                {
                    "step": turn.step,
                    "prompt": turn.prompt,
                    "code": turn.code,
                    "execution_output": turn.execution_output,
                    "execution_success": turn.execution_success,
                    "timestamp": turn.timestamp.isoformat()
                }
                for turn in trajectory.turns
            ]
        }
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved trajectory {trajectory.trajectory_id} to {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Failed to save trajectory {trajectory.trajectory_id}: {e}")
        return ""

def meets_completion_criteria(trajectory: List[Turn], criteria: Optional[str]) -> bool:
    """Check if trajectory meets completion criteria"""
    if not criteria or not trajectory:
        return False
    
    # Check if execution engine reported natural completion
    last_turn = trajectory[-1]
    if "success" in last_turn.execution_output.lower():
        return True
    
    # Simple heuristic: check if last execution output contains success indicators
    last_output = last_turn.execution_output.lower()
    success_indicators = ["test passed", "success", "completed", "correct", "done"]
    
    return any(indicator in last_output for indicator in success_indicators)

def calculate_reward(trajectory: List[Turn], termination_reason: str) -> float:
    """Calculate reward for a completed trajectory"""
    base_reward = 0.0
    
    # Completion bonus based on termination reason
    if termination_reason == "completion_criteria_met":
        base_reward += 1.0
    elif termination_reason == "max_steps":
        base_reward += 0.3
    else:  # execution_error, generation_error, etc.
        base_reward += 0.1
    
    # Execution success bonus
    if trajectory:
        successful_executions = sum(1 for turn in trajectory if turn.execution_success)
        execution_rate = successful_executions / len(trajectory)
        base_reward += execution_rate * 0.5
        
        # Progress bonus (more successful turns = more interaction)
        progress_bonus = min(len(trajectory) / MAX_TURNS, 1.0) * 0.3
        base_reward += progress_bonus
        
        # Bonus for trajectories that don't crash immediately
        if len(trajectory) > 1:
            base_reward += 0.2
    
    return min(base_reward, 2.0)  # Cap at 2.0

# API Endpoints

@app.post("/generate_trajectories")
async def generate_trajectories(request: BatchTrajectoryRequest) -> List[Trajectory]:
    """
    Main endpoint for generating RL training trajectories
    Used by the Policy Training Module (TRL)
    """
    trajectories = []
    
    for traj_request in request.requests:
        for initial_prompt in traj_request.initial_prompts:
            try:
                trajectory = await generate_single_trajectory(
                    initial_prompt=initial_prompt,
                    max_turns=traj_request.max_turns,
                    completion_criteria=traj_request.completion_criteria
                )
                trajectories.append(trajectory)
            except Exception as e:
                logger.error(f"Failed to generate trajectory for prompt '{initial_prompt}': {e}")
                # Create failed trajectory
                failed_trajectory = Trajectory(
                    trajectory_id=str(uuid.uuid4()),
                    turns=[],
                    final_reward=0.0,
                    termination_reason="generation_error",
                    total_steps=0,
                    created_at=datetime.now()
                )
                trajectories.append(failed_trajectory)
    
    return trajectories

async def generate_single_trajectory(
    initial_prompt: str, 
    max_turns: int = MAX_TURNS,
    completion_criteria: Optional[str] = None
) -> Trajectory:
    """Generate a single multi-turn trajectory"""
    
    trajectory_id = str(uuid.uuid4())
    turns = []
    current_instruction = initial_prompt
    
    logger.info(f"Starting trajectory {trajectory_id} with prompt: {initial_prompt}")
    
    # Create a new execution instance for this trajectory
    instance_id = await create_execution_instance()
    if not instance_id:
        logger.error(f"Failed to create execution instance for trajectory {trajectory_id}")
        return Trajectory(
            trajectory_id=trajectory_id,
            turns=[],
            final_reward=0.0,
            termination_reason="execution_instance_error",
            total_steps=0,
            created_at=datetime.now()
        )
    
    logger.info(f"Created execution instance {instance_id} for trajectory {trajectory_id}")
    
    try:
        for step in range(1, max_turns + 1):
            try:
                # Build trajectory object for completion server
                trajectory_for_completion = {
                    "turns": [
                        {
                            "step": turn.step,
                            "prompt": turn.prompt,
                            "code": turn.code,
                            "execution_output": turn.execution_output,
                            "execution_success": turn.execution_success
                        }
                        for turn in turns
                    ]
                }
                
                # Step 1: Generate code completions using trajectory context
                response = await http_client.post(
                    f"{COMPLETION_SERVER_URL}/generate",
                    json={
                        "trajectory": trajectory_for_completion,
                        "instruction": current_instruction,
                        "n": 4,
                        "temperature": 0.8,
                        "max_tokens": 512,
                        "stop": ["```", "\nUser:", "\nExecution:"]
                    }
                )
                response.raise_for_status()
                completion_result = response.json()
                code_completions = completion_result.get("completions", [])
                
                # Select best completion (for now, just take first)
                selected_code = code_completions[0] if code_completions else ""
                
                if not selected_code.strip():
                    logger.warning(f"Empty code generation at step {step}")
                    break
                
                # Step 2: Execute code in the persistent instance
                execution_result = await execute_in_instance(instance_id, selected_code)
                
                # Step 3: Create turn record
                turn = Turn(
                    step=step,
                    prompt=current_instruction,
                    code=selected_code,
                    execution_output=execution_result["output"],
                    execution_success=execution_result["success"],
                    timestamp=datetime.now()
                )
                turns.append(turn)
                
                # Step 4: Check if execution completed naturally
                if execution_result.get("completed", False):
                    logger.info(f"Trajectory {trajectory_id} completed naturally at step {step}")
                    break
                
                # Step 5: Check if execution should not continue (crash or max steps in engine)
                if not execution_result.get("should_continue", True):
                    logger.info(f"Trajectory {trajectory_id} terminated by execution engine at step {step}")
                    break
                
                # Step 6: Check custom completion criteria
                if meets_completion_criteria(turns, completion_criteria):
                    logger.info(f"Trajectory {trajectory_id} met custom completion criteria at step {step}")
                    break
                
                # Step 7: Update instruction for next turn
                current_instruction = "Continue with the next code snippet to achieve the goal."
                
            except Exception as e:
                logger.error(f"Error in trajectory {trajectory_id} at step {step}: {e}")
                # Add error turn and break
                error_turn = Turn(
                    step=step,
                    prompt=current_instruction,
                    code="# Error occurred during generation",
                    execution_output=f"Error: {str(e)}",
                    execution_success=False,
                    timestamp=datetime.now()
                )
                turns.append(error_turn)
                break
    
    finally:
        # Always attempt to cleanup the execution instance
        try:
            await http_client.delete(f"{EXEC_ENGINE_URL}/instances/{instance_id}")
            logger.info(f"Cleaned up execution instance {instance_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup execution instance {instance_id}: {e}")
    
    # Determine termination reason based on execution engine state
    termination_reason = "generation_error"
    if turns:
        last_turn = turns[-1]
        if len(turns) >= max_turns:
            termination_reason = "max_steps"
        elif meets_completion_criteria(turns, completion_criteria):
            termination_reason = "completion_criteria_met"
        elif last_turn.execution_success and "success" in last_turn.execution_output.lower():
            termination_reason = "completion_criteria_met"
        elif not last_turn.execution_success:
            termination_reason = "execution_error"
        else:
            termination_reason = "max_steps"
    
    # Calculate final reward
    final_reward = calculate_reward(turns, termination_reason)
    
    trajectory = Trajectory(
        trajectory_id=trajectory_id,
        turns=turns,
        final_reward=final_reward,
        termination_reason=termination_reason,
        total_steps=len(turns),
        created_at=datetime.now()
    )
    
    logger.info(f"Completed trajectory {trajectory_id}: {len(turns)} turns, reward={final_reward}")
    
    # Save trajectory to file
    if TRAJECTORY_OUTPUT_DIR:
        filepath = save_trajectory_to_file(trajectory)
        if filepath:
            logger.info(f"Trajectory saved to: {filepath}")
    
    return trajectory

async def call_completion_server(prompt: str, num_completions: int = 4) -> List[str]:
    """Call the completion server with trajectory-aware format"""
    try:
        # Build a simple trajectory from the current prompt
        # For first turn, create empty trajectory
        trajectory = {
            "turns": []
        }
        
        # If this looks like a continuation prompt, try to parse previous context
        if "Code:" in prompt and "Output:" in prompt:
            # This is a continuation - extract the last interaction
            parts = prompt.split("Next:")
            if len(parts) > 1:
                history = parts[0]
                # Simple parsing - in real implementation you might want more sophisticated parsing
                turns = []
                if "Code:" in history and "Output:" in history:
                    # Extract the most recent turn for context
                    code_match = history.split("Code:")[-1].split("Output:")[0].strip()
                    output_match = history.split("Output:")[-1].strip()
                    turns.append({
                        "step": 1,
                        "prompt": "Previous request",
                        "code": code_match,
                        "execution_output": output_match,
                        "execution_success": "error" not in output_match.lower()
                    })
                trajectory["turns"] = turns
        
        response = await http_client.post(
            f"{COMPLETION_SERVER_URL}/generate",
            json={
                "trajectory": trajectory,
                "instruction": prompt if not trajectory["turns"] else "Generate next code snippet to achieve the goal",
                "n": num_completions,
                "temperature": 0.8,
                "max_tokens": 512,
                "stop": ["```", "\n\n#", "\nUser:", "\nExecution:"]
            }
        )
        response.raise_for_status()
        result = response.json()
        return result.get("completions", [])
    except Exception as e:
        logger.error(f"Completion server error: {e}")
        return []

async def create_execution_instance() -> Optional[str]:
    """Create a new execution instance"""
    try:
        response = await http_client.post(f"{EXEC_ENGINE_URL}/start_execution", json={})
        response.raise_for_status()
        result = response.json()
        return result.get("instance_id")
    except Exception as e:
        logger.error(f"Failed to create execution instance: {e}")
        return None

async def execute_in_instance(instance_id: str, text: str) -> Dict[str, Any]:
    """Execute code in a specific execution instance"""
    try:
        response = await http_client.post(
            f"{EXEC_ENGINE_URL}/execute",
            json={
                "instance_id": instance_id,
                "text": text
            }
        )
        response.raise_for_status()
        result = response.json()
        
        # Convert execution engine response to our expected format
        return {
            "output": result.get("output", ""),
            "success": result.get("state") not in ["crash", "max_steps_exceeded"],
            "state": result.get("state", "unknown"),
            "step_count": result.get("step_count", 0),
            "completed": result.get("state") == "success",
            "should_continue": result.get("state") == "running"
        }
    except Exception as e:
        logger.error(f"Execution engine error: {e}")
        return {
            "output": f"Execution failed: {str(e)}",
            "success": False,
            "state": "crash",
            "step_count": 0,
            "completed": False,
            "should_continue": False
        }

# Health check and utility endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    trajectory_count = 0
    if TRAJECTORY_OUTPUT_DIR and os.path.exists(TRAJECTORY_OUTPUT_DIR):
        trajectory_count = len([f for f in os.listdir(TRAJECTORY_OUTPUT_DIR) if f.endswith('.json')])
    
    return {
        "active_trajectories": len(active_trajectories),
        "completion_server": COMPLETION_SERVER_URL,
        "execution_engine": EXEC_ENGINE_URL,
        "trajectory_output_dir": TRAJECTORY_OUTPUT_DIR,
        "saved_trajectories": trajectory_count
    }

# Configuration endpoints

@app.get("/config")
async def get_config():
    """Get current service configuration"""
    return {
        "max_turns": MAX_TURNS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "completion_server_url": COMPLETION_SERVER_URL,
        "exec_engine_url": EXEC_ENGINE_URL,
        "trajectory_output_dir": TRAJECTORY_OUTPUT_DIR
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Code Generation & Execution Service")
    parser.add_argument("--completion-server-url", default="http://localhost:8000",
                       help="URL of the completion server (VLLM)")
    parser.add_argument("--exec-engine-url", default="http://localhost:8001", 
                       help="URL of the execution engine")
    parser.add_argument("--max-turns", type=int, default=10,
                       help="Maximum number of turns per trajectory")
    parser.add_argument("--timeout", type=int, default=30,
                       help="HTTP timeout in seconds")
    parser.add_argument("--trajectory-output-dir", default="./trajectories",
                       help="Directory to save trajectory JSON files")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host to bind the service")
    parser.add_argument("--port", type=int, default=8002,
                       help="Port to bind the service")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set global configuration
    COMPLETION_SERVER_URL = args.completion_server_url
    EXEC_ENGINE_URL = args.exec_engine_url
    MAX_TURNS = args.max_turns
    TIMEOUT_SECONDS = args.timeout
    TRAJECTORY_OUTPUT_DIR = args.trajectory_output_dir
    
    logger.info(f"Starting Code Generation & Execution Service")
    logger.info(f"Completion Server: {COMPLETION_SERVER_URL}")
    logger.info(f"Execution Engine: {EXEC_ENGINE_URL}")
    logger.info(f"Max Turns: {MAX_TURNS}")
    logger.info(f"Timeout: {TIMEOUT_SECONDS}s")
    logger.info(f"Trajectory Output Directory: {TRAJECTORY_OUTPUT_DIR}")
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

"""
EXPECTED API CONTRACTS FOR EXTERNAL SERVICES:

1. COMPLETION SERVER at http://localhost:8000
   POST /generate
   Request: {
       "trajectory": {
           "turns": List[{
               "step": int,
               "prompt": str,
               "code": str,
               "execution_output": str,
               "execution_success": bool
           }]
       },
       "instruction": str,
       "n": int,              # number of completions
       "temperature": float,
       "max_tokens": int,
       "stop": List[str]      # stop tokens
   }
   Response: {
       "completions": List[str],
       "server_used": str
   }

   POST /update_model_params
   Request: {
       "model_path": str
   }
   Response: {
       "status": "completed",
       "updated_servers": int,
       "total_servers": int,
       "results": Dict[str, bool],
       "success": bool
   }

2. EXECUTION ENGINE at http://localhost:8001
   POST /start_execution
   Request: {}
   Response: {
       "instance_id": str
   }
   
   POST /execute
   Request: {
       "instance_id": str,
       "text": str
   }
   Response: {
       "output": str,         # execution output or error message
       "state": str,          # "running", "success", "crash", "max_steps_exceeded"
       "step_count": int      # number of steps executed in this instance
   }
   
   DELETE /instances/{instance_id}
   Response: {
       "status": "cleaned_up",
       "instance_id": str
   }

USAGE BY POLICY TRAINING MODULE:
POST /generate_trajectories
{
    "requests": [
        {
            "initial_prompts": ["Write a function to sort a list", "Create a binary search"],
            "max_turns": 5,
            "completion_criteria": "test passed"
        }
    ]
}

Response: List of Trajectory objects with multi-turn code execution sessions
Each trajectory maintains a persistent execution environment across all turns
and uses trajectory-aware prompting for better context understanding.
"""