"""
Orchestrator class for multi-turn code generation and execution
"""

import uuid
import json
import os
import re
import ast
import xml.sax.saxutils as saxutils
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
import logging

from .models import Turn, Trajectory, TrajectoryRequest, BatchTrajectoryRequest
from .execution_engine import RayExecutionEngine

# Configure logging
logger = logging.getLogger(__name__)

FORMAT_INSTRUCTIONS = """
Generate python code to solve the task.
Return the code in markdown format code-blocks.
Do not include any other text, only valid python code.
"""


class Orchestrator:
    """
    Orchestrator for multi-turn RL trajectory generation
    Manages code generation, execution, and trajectory tracking
    """
    
    def __init__(
        self,
        completion_load_balancer,
        max_turns: int = 10,
        timeout_seconds: int = 30,
        trajectory_output_dir: Optional[str] = None,
        prompts_jsonl_path: Optional[str] = None,
        ray_timeout_per_step: float = 30.0,
        ray_num_cpus: int = 1,
        ray_num_gpus: int = 0,
        debug_mode: bool = True
    ):
        self.completion_load_balancer = completion_load_balancer
        self.max_turns = max_turns
        self.timeout_seconds = timeout_seconds
        self.trajectory_output_dir = Path(trajectory_output_dir) if trajectory_output_dir else None
        self.prompts_jsonl_path = Path(prompts_jsonl_path) if prompts_jsonl_path else None
        self.trajectory_run_dir: Optional[Path] = None
        
        # Ray configuration
        self.ray_timeout_per_step = ray_timeout_per_step
        self.ray_num_cpus = ray_num_cpus
        self.ray_num_gpus = ray_num_gpus
        
        # Debug configuration
        self.debug_mode = debug_mode
        
        # Initialize execution engine
        self.execution_engine = RayExecutionEngine()
        
        # Trajectory tracking - maps trajectory_id to instance_id
        self.active_trajectories: Dict[str, str] = {}
        # Maps instance_id to list of turns
        self.trajectory_turns: Dict[str, List[Turn]] = {}
        
        # Success criteria function support
        self.success_criteria_available = False
        self.get_success_criterion_for_task = None
        try:
            from .demo_data.success_criteria import get_success_criterion_for_task
            self.get_success_criterion_for_task = get_success_criterion_for_task
            self.success_criteria_available = True
        except ImportError:
            pass
    
    def log_prompt_to_jsonl(self, request_data: dict):
        """Log the exact prompt request to prompts.jsonl and prompts.txt"""
        if not self.prompts_jsonl_path:
            return
        
        try:
            # Add timestamp to the request
            timestamp = datetime.now()
            log_entry = {
                "timestamp": timestamp.isoformat(),
                **request_data
            }
            
            # Ensure parent directory exists
            self.prompts_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Append to JSONL file
            with open(self.prompts_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            # Also write to TXT file with timestamp if debug mode is enabled
            if self.debug_mode:
                txt_path = self.prompts_jsonl_path.with_suffix('.txt')
                with open(txt_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"TIMESTAMP: {timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
                    f.write(f"{'='*80}\n\n")
                    
                    # Write messages in a readable format
                    messages = request_data.get('messages', [])
                    for i, message in enumerate(messages):
                        role = message.get('role', 'unknown')
                        content = message.get('content', '')
                        f.write(f"Message {i+1} ({role.upper()}):\n")
                        f.write(f"{'-'*40}\n")
                        f.write(f"{content}\n")
                        f.write(f"{'-'*40}\n\n")
                    
                    # Write other request parameters
                    for key, value in request_data.items():
                        if key != 'messages':
                            f.write(f"{key}: {value}\n")
                    
                    f.write(f"\n{'='*80}\n\n")
                
        except Exception as e:
            logger.error(f"Failed to log prompt to JSONL/TXT: {e}")
    
    def format_trajectory_for_chat(self, original_goal: str, trajectory_turns: List[Turn]) -> str:
        """
        Format the trajectory history for chat completion requests.
        
        Structure:
        <goal>original task prompt</goal>
        <turn>model's code</turn>
        <output>stdout content</output>
        <error>stderr content</error>  (only if stderr exists)
        ... (repeat for all turns)
        
        Then add instruction to generate next code based on output/errors.
        """
        formatted_content = f"<goal>\n{original_goal}\n</goal>\n\n"
        
        for turn in trajectory_turns:
            # Add the turn with code
            formatted_content += f"<turn>\n{turn.code}\n</turn>\n"
            
            # Add execution output if present
            if turn.execution_output and turn.execution_output.strip():
                formatted_content += f"{turn.execution_output}\n"
            
            formatted_content += "\n"
        
        # Add instruction for next turn
        formatted_content += (
            "Based on the goal and the output/errors from previous turns, "
            "generate the next turn of code to achieve the goal. "
            "Use the output and errors from the previous step to fix any issues "
            "and continue progressing toward the goal.\n\n"
            f"{FORMAT_INSTRUCTIONS}"
        )
        
        return formatted_content
    
    def extract_python_code(self, text: str) -> str:
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
            if self.is_valid_python_code(code):
                valid_blocks.append(code)
        
        if valid_blocks:
            # Concatenate all valid code blocks with newlines
            return '\n\n'.join(valid_blocks)
        else:
            logger.warning(f"Could not extract valid Python code from VLLM output: {text}...")
            return "# Unable to extract valid Python code from response\npass"
    
    def is_valid_python_code(self, code: str) -> bool:
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
    
    def save_trajectory_to_file(self, trajectory: Trajectory) -> str:
        """Save trajectory to a JSON file and return the file path"""
        if not self.trajectory_output_dir:
            return ""
        
        try:
            # Resolve base directory (handles relative paths and ~)
            base_dir = self.trajectory_output_dir
            # Lazily create a run-specific subdirectory once per service run
            if self.trajectory_run_dir is None:
                run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.trajectory_run_dir = base_dir / f"run_{run_ts}"
                self.trajectory_run_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with trajectory's creation timestamp and id
            ts = trajectory.created_at.strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{ts}_{trajectory.trajectory_id}.json"
            filepath = self.trajectory_run_dir / filename
            
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
                        "success_criteria_met": getattr(turn, 'success_criterion_met', False),
                        "raw_model_output": getattr(turn, 'raw_model_output', None),
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
    
    def meets_completion_criteria(self, trajectory: List[Turn], criteria: Optional[str]) -> bool:
        """Check if trajectory meets completion criteria based on success criterion function results"""
        if not trajectory:
            return False
        
        last_turn = trajectory[-1]
        
        # Check if the success criterion function returned True
        # This is determined by the Ray execution engine after running the criterion function
        # Only use this if a success criterion function was actually provided to the execution engine
        if hasattr(last_turn, 'success_criterion_met') and last_turn.success_criterion_met:
            return True
        
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
            
            # Debug logging
            logger.debug(f"Checking completion criteria: '{criteria_lower}' in '{last_output[:200]}...'")
            
            # First try exact match for the criteria string
            if criteria_lower in last_output:
                logger.info(f"Completion criteria met: '{criteria_lower}' found in output")
                return True
                
            # Then try keyword matching
            for keyword in criteria_keywords:
                if keyword in criteria_lower and keyword in last_output:
                    logger.info(f"Completion criteria met via keyword: '{keyword}'")
                    return True
            
            logger.debug(f"Completion criteria not met. Looking for: '{criteria_lower}'")
            logger.debug(f"Output: '{last_output}')")
        
        return False
    
    def calculate_reward(self, trajectory: List[Turn], termination_reason: str) -> float:
        """Binary reward: 1 if completion criteria met, else 0"""
        return 1.0 if termination_reason == "completion_criteria_met" else 0.0
    
    async def generate_trajectories(self, request: BatchTrajectoryRequest) -> List[Trajectory]:
        """
        Main method for generating RL training trajectories
        Used by the Policy Training Module (TRL)
        """
        trajectories = []
        
        for traj_request in request.requests:
            for i, initial_prompt in enumerate(traj_request.initial_prompts):
                try:
                    # Get task name if provided
                    task_name = None
                    if traj_request.task_names and i < len(traj_request.task_names):
                        task_name = traj_request.task_names[i]
                    
                    trajectory = await self.generate_single_trajectory(
                        initial_prompt=initial_prompt,
                        max_turns=traj_request.max_turns,
                        completion_criteria=traj_request.completion_criteria,
                        task_name=task_name
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
        self,
        initial_prompt: str, 
        max_turns: int = None,
        completion_criteria: Optional[str] = None,
        task_name: Optional[str] = None
    ) -> Trajectory:
        """Generate a single multi-turn trajectory"""
        if max_turns is None:
            max_turns = self.max_turns
        
        trajectory_id = str(uuid.uuid4())
        current_instruction = initial_prompt
        
        logger.info(f"Starting trajectory {trajectory_id} with prompt: {initial_prompt}")
        logger.info(f"Task name for trajectory {trajectory_id}: {task_name or 'inferred from prompt'}")
        
        # Create a new execution instance for this trajectory
        instance_id = self.execution_engine.start_instance(
            timeout_in_secs=self.ray_timeout_per_step,
            num_cpus=self.ray_num_cpus,
            num_gpus=self.ray_num_gpus
        )
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
        
        # Map trajectory to instance and initialize turns for this instance
        self.active_trajectories[trajectory_id] = instance_id
        self.trajectory_turns[instance_id] = []
        
        logger.info(f"Created execution instance {instance_id} for trajectory {trajectory_id}")
        logger.info(f"Trajectory {trajectory_id} mapped to instance {instance_id}")
        
        try:
            for step in range(1, max_turns + 1):
                try:
                    # Build chat messages using the clean formatting function
                    messages = []
                    # Add system prompt
                    messages.append({"role": "system", "content": "You are a helpful AI code assistant. Respond with Python code in markdown code blocks."})
                    
                    # Get turns for this specific instance
                    instance_turns = self.trajectory_turns.get(instance_id, [])
                    
                    # For the first turn, use the initial prompt as the goal
                    if step == 1:
                        formatted_content = self.format_trajectory_for_chat(initial_prompt, instance_turns)
                    else:
                        # For subsequent turns, continue with the same goal and updated trajectory
                        formatted_content = self.format_trajectory_for_chat(initial_prompt, instance_turns)
                    
                    messages.append({"role": "user", "content": formatted_content})
                    
                    # Prepare the request data for vLLM
                    request_data = {
                        "messages": messages,
                        "n": 1,
                        "temperature": 0.8,
                        "max_tokens": 512
                    }
                    
                    # Log the exact request to prompts.jsonl
                    self.log_prompt_to_jsonl(request_data)
                    
                    # Call the completion load balancer directly
                    completion_result = await self.completion_load_balancer.chat_completions(request_data)
                    
                    logger.info(f"Chat completion result: {completion_result}")
                    choices = completion_result.get("choices", [])
                    if not choices or "message" not in choices[0] or "content" not in choices[0]["message"]:
                        logger.warning(f"Empty code generation at step {step}")
                        break
                    raw_code = choices[0]["message"]["content"]
                    
                    # Extract just the Python code from the response
                    selected_code = self.extract_python_code(raw_code)
                    if not selected_code.strip():
                        logger.warning(f"No Python code found in completion at step {step}")
                        # Fall back to using the raw code
                        selected_code = raw_code
                    
                    # Get success criterion function for this task
                    current_task_name = task_name  # Use provided task name
                    if step == 1 and not current_task_name:  # Extract task name from initial instruction only if not provided
                        if isinstance(initial_prompt, dict) and "name" in initial_prompt:
                            current_task_name = initial_prompt["name"]
                        elif "fibonacci" in str(initial_prompt).lower():
                            current_task_name = "fibonacci_sequence"
                        elif "data analysis" in str(initial_prompt).lower() or "histogram" in str(initial_prompt).lower():
                            current_task_name = "data_analysis_pipeline"
                        elif "csv" in str(initial_prompt).lower() or "filter" in str(initial_prompt).lower():
                            current_task_name = "file_processing"
                        elif "fits" in str(initial_prompt).lower():
                            current_task_name = "fits_basic_analysis"
                    
                    success_criterion_func = None
                    if self.success_criteria_available and current_task_name:
                        success_criterion_func = self.get_success_criterion_for_task(current_task_name)
                        logger.info(f"Using success criterion function for task: {current_task_name}")
                    
                    # Execute code in the persistent instance
                    logger.info(f"🚀 ORCHESTRATOR -> Calling Ray executor with code ({len(selected_code)} chars) for trajectory {trajectory_id}, step {step}, task: {current_task_name or 'unknown'}")
                    logger.debug(f"Code to execute: {selected_code[:200]}{'...' if len(selected_code) > 200 else ''}")
                    logger.info(f"🔍 ORCHESTRATOR -> instance_id='{instance_id}' (type: {type(instance_id)}, len: {len(instance_id) if isinstance(instance_id, str) else 'N/A'})")
                    logger.info(f"🔍 ORCHESTRATOR -> selected_code first 50 chars: '{selected_code[:50]}'")
                    
                    execution_result = self.execution_engine.execute_code(instance_id, selected_code, success_criterion_func)
                    logger.info(f"✅ ORCHESTRATOR <- Ray executor returned: state={execution_result.get('state', 'unknown')}, success={execution_result.get('success', False)}")
                    
                    # Create turn record
                    turn = Turn(
                        step=step,
                        prompt=current_instruction,
                        code=selected_code,
                        execution_output=execution_result["execution_output"],
                        execution_success=execution_result.get("state") == "success",
                        timestamp=datetime.now()
                    )
                    
                    # Store the raw model output for debugging/analysis
                    turn.raw_model_output = raw_code
                    
                    # Add success criterion result to turn object for completion checking
                    turn.success_criterion_met = execution_result.get("success_criterion_met", False)
                    
                    # Store turn in instance-specific list
                    self.trajectory_turns[instance_id].append(turn)
                    
                    # Check custom completion criteria (now includes success criterion function results)
                    instance_turns = self.trajectory_turns.get(instance_id, [])
                    if self.meets_completion_criteria(instance_turns, completion_criteria):
                        logger.info(f"Trajectory {trajectory_id} met custom completion criteria at step {step}")
                        break
                    
                    # Update instruction for next turn
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
                    self.trajectory_turns[instance_id].append(error_turn)
                    # Continue to next turn instead of breaking
                    continue
        
        finally:
            # Always attempt to cleanup the execution instance
            try:
                cleanup_success = self.execution_engine.cleanup_instance(instance_id)
                if cleanup_success:
                    logger.info(f"Cleaned up Ray execution instance {instance_id}")
                else:
                    logger.warning(f"Failed to cleanup Ray execution instance {instance_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup Ray execution instance {instance_id}: {e}")
        
        # Get final turns for this instance BEFORE cleanup
        final_turns = self.trajectory_turns.get(instance_id, [])
        
        # Clean up trajectory tracking
        if trajectory_id in self.active_trajectories:
            del self.active_trajectories[trajectory_id]
        if instance_id in self.trajectory_turns:
            del self.trajectory_turns[instance_id]
        
        # Determine termination reason based on execution engine state
        termination_reason = "generation_error"
        if final_turns:
            if len(final_turns) >= max_turns:
                termination_reason = "max_steps"
            elif self.meets_completion_criteria(final_turns, completion_criteria):
                termination_reason = "completion_criteria_met"
            else:
                termination_reason = "max_steps"
        
        # Calculate final reward
        final_reward = self.calculate_reward(final_turns, termination_reason)
        
        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            turns=final_turns,
            final_reward=final_reward,
            termination_reason=termination_reason,
            total_steps=len(final_turns),
            created_at=datetime.now()
        )
        
        logger.info(f"Completed trajectory {trajectory_id}: {len(final_turns)} turns, reward={final_reward}, instance={instance_id}")
        
        # Save trajectory to file
        if self.trajectory_output_dir:
            filepath = self.save_trajectory_to_file(trajectory)
            if filepath:
                logger.info(f"Trajectory saved to: {filepath}")
        
        return trajectory
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        trajectory_count = 0
        if self.trajectory_output_dir and os.path.exists(self.trajectory_output_dir):
            trajectory_count = len([f for f in os.listdir(self.trajectory_output_dir) if f.endswith('.json')])
        
        return {
            "active_trajectories": len(self.active_trajectories),
            "active_instances": len(self.trajectory_turns),
            "trajectory_to_instance_mapping": self.active_trajectories,
            "ray_execution_engine": "embedded",
            "ray_timeout_per_step": self.ray_timeout_per_step,
            "ray_num_cpus": self.ray_num_cpus,
            "ray_num_gpus": self.ray_num_gpus,
            "trajectory_output_dir": str(self.trajectory_output_dir) if self.trajectory_output_dir else None,
            "saved_trajectories": trajectory_count,
            "debug_mode": self.debug_mode
        }
    
    def shutdown(self):
        """Clean up resources on shutdown"""
        logger.info("Shutting down orchestrator, cleaning up Ray instances...")
        try:
            self.execution_engine.cleanup_all_instances()
            logger.info("Successfully cleaned up all Ray instances")
        except Exception as e:
            logger.error(f"Error during Ray cleanup: {e}")