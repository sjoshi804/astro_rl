import ray
import uuid
import time
import io
import sys
from typing import *
from contextlib import redirect_stdout, redirect_stderr


def get_result(
    x,
    *,
    wait: float = 1.0,  ## 1000 ms
) -> Optional[Any]:
    if isinstance(x, ray.ObjectRef):
        from ray.exceptions import GetTimeoutError

        while True:
            try:
                return ray.get(x, timeout=wait)
            except GetTimeoutError:
                pass
    return x


@ray.remote
class RayCodeExecutor:
    def __init__(self, id: str, timeout_in_secs: float = 30.0):
        self.id = id
        self.locals = {}
        self.current_turn = 0
        self.timeout = timeout_in_secs
        self.max_steps = 100  # Configurable max steps
        
    def execute(self, code: str, success_criterion: Optional[Callable] = None, *args, **kwargs) -> Dict[str, Any]:
        """Execute code and return structured response with state and output."""
        start_time = time.time()
        
        # Check if max steps exceeded
        if self.current_turn >= self.max_steps:
            return {
                "state": "max_steps_exceeded",
                "execution_output": f"Maximum steps ({self.max_steps}) exceeded"
            }
        
        try:
            self.current_turn += 1
            
            # Add any kwargs to locals
            for var_name, value in kwargs.items():
                self.locals[var_name] = value
            
            # Capture stdout and stderr
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Check for timeout during execution
                if time.time() - start_time > self.timeout:
                    return {
                        "state": "crashed",
                        "execution_output": "Execution timed out"
                    }
                
                # Execute the code
                exec_result = exec(code, globals(), self.locals)
                
                # Call success criterion if provided
                if success_criterion is not None:
                    success_criterion()
            
            # Capture any output
            stdout_content = stdout_buffer.getvalue()
            stderr_content = stderr_buffer.getvalue()
            
            execution_output = ""
            if stdout_content:
                execution_output += f"STDOUT:\n{stdout_content}"
            if stderr_content:
                execution_output += f"STDERR:\n{stderr_content}"
            if not stdout_content and not stderr_content:
                execution_output = "Code executed successfully (no output)"
            
            return {
                "state": "completed",
                "execution_output": execution_output
            }
            
        except Exception as e:
            return {
                "state": "crashed", 
                "execution_output": f"Error: {str(e)}"
            }


ACTOR_POOL: Dict[str, Any] = {}


def start_instance(timeout_in_secs: float = 30.0, num_cpus: int = 1, num_gpus: int = 0) -> str:
    """Start a new RayCodeExecutor instance with specified resources."""
    # Initialize Ray if not already initialized
    try:
        ray.init(
            address="auto",
            ignore_reinit_error=True,
        )
    except Exception as e:
        print(f"Ray initialization warning: {e}")
    
    actor_id = str(uuid.uuid4())
    
    # Create actor with resource allocation
    actor_options = {"num_cpus": num_cpus}
    if num_gpus > 0:
        actor_options["num_gpus"] = num_gpus
    
    # Apply resource options to the remote actor
    RemoteExecutor = RayCodeExecutor.options(**actor_options)
    ACTOR_POOL[actor_id] = RemoteExecutor.remote(actor_id, timeout_in_secs)
    
    return actor_id


def exec(actor_id: str, code: str, success_criterion: Optional[Callable] = None) -> Dict[str, Any]:
    """Execute code in a specific RayCodeExecutor instance."""
    if actor_id not in ACTOR_POOL:
        return {
            "state": "crashed",
            "execution_output": f"Actor with ID {actor_id} does not exist."
        }
    
    actor = ACTOR_POOL[actor_id]
    
    try:
        # Set state to running
        result = get_result(actor.execute.remote(code, success_criterion))
        return result
    except Exception as e:
        return {
            "state": "crashed",
            "execution_output": f"Execution failed: {str(e)}"
        }


def cleanup_instance(actor_id: str) -> bool:
    """Clean up a specific RayCodeExecutor instance."""
    if actor_id in ACTOR_POOL:
        try:
            ray.kill(ACTOR_POOL[actor_id])
            del ACTOR_POOL[actor_id]
            return True
        except Exception as e:
            print(f"Error cleaning up actor {actor_id}: {e}")
            return False
    return False


def cleanup_all_instances():
    """Clean up all RayCodeExecutor instances."""
    for actor_id in list(ACTOR_POOL.keys()):
        cleanup_instance(actor_id)