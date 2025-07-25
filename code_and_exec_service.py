"""
Code Generation & Execution Service
Orchestrates multi-turn code generation and execution for RL training
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Callable
import httpx
import asyncio
import json
import os
from pathlib import Path
import logging
import argparse
import uuid
from datetime import datetime
import re
import ast
import xml.sax.saxutils as saxutils

FORMAT_INSTRUCTIONS = """
Generate python code to solve the task.
Return the code in the following format in markdown format code-blocks.
"""

# Import ray execution engine
from ray_execution_engine import start_instance, execute_code, cleanup_instance, cleanup_all_instances

# Import success criteria functions
try:
    from demo_data.success_criteria import get_success_criterion_for_task
    SUCCESS_CRITERIA_AVAILABLE = True
except ImportError:
    SUCCESS_CRITERIA_AVAILABLE = False
    def get_success_criterion_for_task(task_name: str):
        return None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration variables (set via command line args)
COMPLETION_SERVER_URL = None
MAX_TURNS = None
TIMEOUT_SECONDS = None
TRAJECTORY_OUTPUT_DIR = None
TRAJECTORY_RUN_DIR: Optional[Path] = None  # created on first save
PROMPTS_JSONL_PATH: Optional[Path] = None  # JSONL file for logging prompts

# Ray execution engine configuration
RAY_TIMEOUT_PER_STEP = 30.0
RAY_NUM_CPUS = 1
RAY_NUM_GPUS = 0

app = FastAPI(title="Code Generation & Execution Service")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up Ray instances on service shutdown"""
    logger.info("Shutting down service, cleaning up Ray instances...")
    try:
        cleanup_all_instances()
        logger.info("Successfully cleaned up all Ray instances")
    except Exception as e:
        logger.error(f"Error during Ray cleanup: {e}")

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
    max_turns: int = MAX_TURNS
    completion_criteria: Optional[str] = None  # Custom completion condition

class BatchTrajectoryRequest(BaseModel):
    requests: List[TrajectoryRequest]

# In-memory storage for active trajectories
active_trajectories: Dict[str, Dict] = {}

# HTTP client for external services
http_client = httpx.AsyncClient(timeout=TIMEOUT_SECONDS)

def log_prompt_to_jsonl(request_data: dict):
    """Log the exact prompt request to prompts.jsonl"""
    if not PROMPTS_JSONL_PATH:
        return
    
    try:
        # Add timestamp to the request
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            **request_data
        }
        
        # Ensure parent directory exists
        PROMPTS_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to JSONL file
        with open(PROMPTS_JSONL_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
    except Exception as e:
        logger.error(f"Failed to log prompt to JSONL: {e}")

# Utility Functions
def extract_python_code(text: str) -> str:
    """Extract Python code from a text response that may contain multiple code blocks and explanations."""
    logger.info(f"Extracting Python code from text: {text}...")
    logger.info(f"Text length: {len(text)}")
    
    # Find all code blocks between ```python ... ```
    code_block_pattern = r'```(?:python)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    valid_blocks = []
    for code in matches:
        code = code.strip()
        # Remove any leading comments that are instructions
        lines = code.split('\n')
        clean_lines = []
        for line in lines:
            if line.strip().startswith('python'):
                continue
            clean_lines.append(line)
        code = '\n'.join(clean_lines).strip()
        if is_valid_python_code(code):
            valid_blocks.append(code)
    
    if valid_blocks:
        # Concatenate all valid code blocks with newlines
        return '\n\n'.join(valid_blocks)
    else:
        logger.warning(f"Could not extract valid Python code from VLLM output: {text}...")
        return "# Unable to extract valid Python code from response\npass"

def is_valid_python_code(code: str) -> bool:
    """Check if the given string is valid Python code"""
    if not code or not code.strip():
        return False
    
    try:
        # Try to parse the code as Python AST
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False

def save_trajectory_to_file(trajectory: Trajectory) -> str:
    """Save trajectory to a JSON file and return the file path"""
    global TRAJECTORY_RUN_DIR  # Ensure we update the global variable
    if not TRAJECTORY_OUTPUT_DIR:
        return ""
    
    try:
        # Resolve base directory (handles relative paths and ~)
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
    """Check if trajectory meets completion criteria based on success criterion function results"""
    if not trajectory:
        return False
    
    last_turn = trajectory[-1]
    
    # Check if the success criterion function returned True
    # This is determined by the Ray execution engine after running the criterion function
    if hasattr(last_turn, 'success_criterion_met'):
        return last_turn.success_criterion_met
    
    # Fallback to checking execution output for success criterion result
    last_output = last_turn.execution_output.lower()
    if "success criterion met: true" in last_output:
        return True
    elif "success criterion met: false" in last_output:
        return False
    
    # Legacy fallback - if no criterion function was used, fall back to text matching
    # This ensures backward compatibility
    if criteria:
        criteria_keywords = [
            "test passed", "optimized", "visualization complete", "plot created", 
            "histogram complete", "filtering complete", "data saved", 
            "maximum height found", "game complete", "navigation tested"
        ]
        
        criteria_lower = criteria.lower()
        for keyword in criteria_keywords:
            if keyword in criteria_lower and keyword in last_output:
                return True
    
    return False

def calculate_reward(trajectory: List[Turn], termination_reason: str) -> float:
    """Binary reward: 1 if completion criteria met, else 0"""
    return 1.0 if termination_reason == "completion_criteria_met" else 0.0

# API Endpoints

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    """
    OpenAI-compatible chat completions endpoint
    Forwards requests to completion server
    """
    try:
        response = await http_client.post(
            f"{COMPLETION_SERVER_URL}/v1/chat/completions",
            json=request
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Chat completions error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat completions failed: {str(e)}")

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
    # Force max_turns to 10 always
    max_turns = 10
    
    trajectory_id = str(uuid.uuid4())
    turns = []
    current_instruction = initial_prompt
    
    logger.info(f"Starting trajectory {trajectory_id} with prompt: {initial_prompt}")
    
    # Create a new execution instance for this trajectory
    instance_id = create_execution_instance()
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
                # Build chat messages with XML tags for each turn
                messages = []
                # Optionally add a system prompt
                messages.append({"role": "system", "content": "You are a helpful AI code assistant. Respond with Python code in markdown code blocks."})
                for turn in turns:
                    turn_content = f"<turn>\n<prompt>{saxutils.escape(turn.prompt)}</prompt>\n<code>{saxutils.escape(turn.code)}</code>\n<output>{saxutils.escape(turn.execution_output)}</output>\n<success>{turn.execution_success}</success>\n</turn>"
                    messages.append({"role": "user", "content": turn_content})
                # Add the current instruction as the next user message
                current_content = f"<turn>\n<prompt>{saxutils.escape(current_instruction)}</prompt>\n</turn>\n{FORMAT_INSTRUCTIONS}"
                messages.append({"role": "user", "content": current_content})
                
                # Prepare the request data for vLLM
                request_data = {
                    "messages": messages,
                    "n": 1,
                    "temperature": 0.8,
                    "max_tokens": 512
                }
                
                # Log the exact request to prompts.jsonl
                log_prompt_to_jsonl(request_data)
                
                # Call the chat endpoint
                response = await http_client.post(
                    f"{COMPLETION_SERVER_URL}/v1/chat/completions",
                    json=request_data
                )
                response.raise_for_status()
                completion_result = response.json()
                logger.info(f"Chat completion result: {completion_result}")
                choices = completion_result.get("choices", [])
                if not choices or "message" not in choices[0] or "content" not in choices[0]["message"]:
                    logger.warning(f"Empty code generation at step {step}")
                    break
                raw_code = choices[0]["message"]["content"]
                # Extract just the Python code from the response
                selected_code = extract_python_code(raw_code)
                if not selected_code.strip():
                    logger.warning(f"No Python code found in completion at step {step}")
                    # Fall back to using the raw code
                    selected_code = raw_code
                
                # Step 2: Get success criterion function for this task
                task_name = None
                if step == 1:  # Extract task name from initial instruction
                    if isinstance(initial_prompt, dict) and "name" in initial_prompt:
                        task_name = initial_prompt["name"]
                    elif "fibonacci" in str(initial_prompt).lower():
                        task_name = "fibonacci_sequence"
                    elif "data analysis" in str(initial_prompt).lower() or "histogram" in str(initial_prompt).lower():
                        task_name = "data_analysis_pipeline"
                    elif "csv" in str(initial_prompt).lower() or "filter" in str(initial_prompt).lower():
                        task_name = "file_processing"
                    elif "fits" in str(initial_prompt).lower():
                        task_name = "fits_basic_analysis"
                
                success_criterion_func = None
                if SUCCESS_CRITERIA_AVAILABLE and task_name:
                    success_criterion_func = get_success_criterion_for_task(task_name)
                    logger.info(f"Using success criterion function for task: {task_name}")
                
                # Step 3: Execute code in the persistent instance
                logger.info(f"🚀 SERVICE -> Calling Ray executor with code ({len(selected_code)} chars) for trajectory {trajectory_id}, step {step}")
                logger.debug(f"Code to execute: {selected_code[:200]}{'...' if len(selected_code) > 200 else ''}")
                logger.info(f"🔍 SERVICE -> instance_id='{instance_id}' (type: {type(instance_id)}, len: {len(instance_id) if isinstance(instance_id, str) else 'N/A'})")
                logger.info(f"🔍 SERVICE -> selected_code first 50 chars: '{selected_code[:50]}'")
                execution_result = execute_in_instance(instance_id, selected_code, success_criterion_func)
                logger.info(f"✅ SERVICE <- Ray executor returned: state={execution_result.get('state', 'unknown')}, success={execution_result.get('success', False)}")
                
                # Step 4: Create turn record
                turn = Turn(
                    step=step,
                    prompt=current_instruction,
                    code=selected_code,
                    execution_output=execution_result["output"],
                    execution_success=execution_result["success"],
                    timestamp=datetime.now()
                )
                
                # Add success criterion result to turn object
                turn.success_criterion_met = execution_result.get("success_criterion_met", False)
                turns.append(turn)
                
                # Step 5: Check custom completion criteria (now includes success criterion function results)
                if meets_completion_criteria(turns, completion_criteria):
                    logger.info(f"Trajectory {trajectory_id} met custom completion criteria at step {step}")
                    break
                
                # Step 8: Update instruction for next turn
                current_instruction = "Continue with the next code snippet to achieve the goal and fix any observed errors."
                
            except Exception as e:
                logger.error(f"Error in trajectory {trajectory_id} at step {step}: {e}")
                # Add error turn and continue (do not break)
                error_turn = Turn(
                    step=step,
                    prompt=current_instruction,
                    code="# Error occurred during generation",
                    execution_output=f"Error: {str(e)}",
                    execution_success=False,
                    timestamp=datetime.now()
                )
                turns.append(error_turn)
                # Continue to next turn instead of breaking
                continue
    
    finally:
        # Always attempt to cleanup the execution instance
        try:
            cleanup_success = cleanup_instance(instance_id)
            if cleanup_success:
                logger.info(f"Cleaned up Ray execution instance {instance_id}")
            else:
                logger.warning(f"Failed to cleanup Ray execution instance {instance_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup Ray execution instance {instance_id}: {e}")
    
    # Determine termination reason based on execution engine state
    termination_reason = "generation_error"
    if turns:
        last_turn = turns[-1]
        if len(turns) >= max_turns:
            termination_reason = "max_steps"
        elif meets_completion_criteria(turns, completion_criteria):
            termination_reason = "completion_criteria_met"
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
        
        # Always append markdown wrapping instruction
        response = await http_client.post(
            f"{COMPLETION_SERVER_URL}/v1/chat/completions",
            json={
                "messages": trajectory,
                "n": num_completions,
                "temperature": 0.8,
                "max_tokens": 512
            }
        )
        response.raise_for_status()
        result = response.json()
        return result.get("completions", [])
    except Exception as e:
        logger.error(f"Completion server error: {e}")
        return []

def create_execution_instance() -> Optional[str]:
    """Create a new execution instance using Ray"""
    try:
        instance_id = start_instance(
            timeout_in_secs=RAY_TIMEOUT_PER_STEP,
            num_cpus=RAY_NUM_CPUS,
            num_gpus=RAY_NUM_GPUS
        )
        logger.info(f"Created Ray execution instance: {instance_id}")
        return instance_id
    except Exception as e:
        logger.error(f"Failed to create Ray execution instance: {e}")
        return None

def execute_in_instance(instance_id: str, text: str, success_criterion: Optional[Callable] = None) -> Dict[str, Any]:
    """Execute code in a specific Ray execution instance"""
    try:
        logger.info(f"📡 EXEC_IN_INSTANCE -> Calling Ray execute_code() with instance_id={instance_id}")
        logger.debug(f"Code to execute: {text[:100]}{'...' if len(text) > 100 else ''}")
        # Execute code using Ray execution engine
        result = execute_code(instance_id, text, success_criterion=success_criterion)
        logger.info(f"📡 EXEC_IN_INSTANCE <- Ray execute_code() returned: {result.get('state', 'unknown')}")
        logger.info(f"📡 EXEC_IN_INSTANCE <- Ray execute_code() returned: {result.get('execution_output', '')}")
        
        # Convert Ray execution engine response to our expected format
        state = result.get("state", "unknown")
        execution_output = result.get("execution_output", "")
        return {
            "output": execution_output,
            "success": state == "success",
            "state": state,
            "step_count": 0,  # Ray engine doesn't track step count this way
            "completed": state == "success",
            "should_continue": state not in ["crashed", "max_steps_exceeded"],
            "success_criterion_met": result.get("success", False)
        }
    except Exception as e:
        logger.error(f"Ray execution engine error: {e}")
        return {
            "output": f"Execution failed: {str(e)}",
            "success": False,
            "state": "crashed",
            "step_count": 0,
            "completed": False,
            "should_continue": False,
            "success_criterion_met": False
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
        "ray_execution_engine": "embedded",
        "ray_timeout_per_step": RAY_TIMEOUT_PER_STEP,
        "ray_num_cpus": RAY_NUM_CPUS,
        "ray_num_gpus": RAY_NUM_GPUS,
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
        "ray_timeout_per_step": RAY_TIMEOUT_PER_STEP,
        "ray_num_cpus": RAY_NUM_CPUS,
        "ray_num_gpus": RAY_NUM_GPUS,
        "trajectory_output_dir": TRAJECTORY_OUTPUT_DIR
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Code Generation & Execution Service")
    parser.add_argument("--completion-server-url", default="http://localhost:8000",
                       help="URL of the completion server (VLLM)")
    parser.add_argument("--max-turns", type=int, default=10,
                       help="Maximum number of turns per trajectory")
    parser.add_argument("--timeout", type=int, default=30,
                       help="HTTP timeout in seconds")
    parser.add_argument("--ray-timeout-per-step", type=float, default=30.0,
                       help="Timeout per execution step in Ray engine")
    parser.add_argument("--ray-num-cpus", type=int, default=1,
                       help="Number of CPUs per Ray actor")
    parser.add_argument("--ray-num-gpus", type=int, default=0,
                       help="Number of GPUs per Ray actor")
    parser.add_argument("--trajectory-output-dir", default="/work/10450/sjoshi804/vista/astro_rl/trajectories",
                       help="Directory to save trajectory JSON files")
    parser.add_argument("--prompts-jsonl-path", default="/work/10450/sjoshi804/vista/astro_rl/prompts.jsonl",
                       help="Path to save prompts JSONL file")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host to bind the service")
    parser.add_argument("--port", type=int, default=8002,
                       help="Port to bind the service")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set global configuration
    COMPLETION_SERVER_URL = args.completion_server_url
    MAX_TURNS = args.max_turns
    TIMEOUT_SECONDS = args.timeout
    RAY_TIMEOUT_PER_STEP = args.ray_timeout_per_step
    RAY_NUM_CPUS = args.ray_num_cpus
    RAY_NUM_GPUS = args.ray_num_gpus
    TRAJECTORY_OUTPUT_DIR = args.trajectory_output_dir
    PROMPTS_JSONL_PATH = Path(args.prompts_jsonl_path)
    
    logger.info(f"Starting Code Generation & Execution Service with Ray")
    logger.info(f"Completion Server: {COMPLETION_SERVER_URL}")
    logger.info(f"Ray Execution Engine: Embedded")
    logger.info(f"Ray Timeout per Step: {RAY_TIMEOUT_PER_STEP}s")
    logger.info(f"Ray CPUs per Actor: {RAY_NUM_CPUS}")
    logger.info(f"Ray GPUs per Actor: {RAY_NUM_GPUS}")
    logger.info(f"Max Turns: {MAX_TURNS}")
    logger.info(f"Timeout: {TIMEOUT_SECONDS}s")
    logger.info(f"Trajectory Output Directory: {TRAJECTORY_OUTPUT_DIR}")
    logger.info(f"Prompts JSONL Path: {PROMPTS_JSONL_PATH}")
    
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
   
   NOTE: For multi-turn trajectories:
   - Step 1: trajectory.turns = [], instruction = initial prompt
   - Step 2+: trajectory.turns = [], instruction = concatenated history + next prompt
     where concatenated history format is:
     "Step X:\nPrompt: <prompt>\nCode:\n<code>\nExecution Output: <output>\n============\n\n"
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

2. RAY EXECUTION ENGINE (Embedded)
   Direct function calls to ray_execution_engine module:
   
   start_instance(timeout_in_secs, num_cpus, num_gpus) -> str
   Returns: instance_id
   
   exec(instance_id, code, success_criterion) -> Dict[str, Any]
   Returns: {
       "state": str,           # "completed", "crashed", "max_steps_exceeded"
       "execution_output": str # execution output or error message
   }
   
   cleanup_instance(instance_id) -> bool
   Returns: success status of cleanup

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