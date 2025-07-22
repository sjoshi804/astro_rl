import ray
import uuid
import time
import sys
import traceback
from typing import *
from io import StringIO


def get_result(
    x,
    *,
    wait: float = 1.0,
) -> Optional[Any]:
    """Get result from Ray ObjectRef with timeout handling."""
    if isinstance(x, ray.ObjectRef):
        try:
            return ray.get(x, timeout=wait)
        except ray.exceptions.GetTimeoutError:
            # For longer operations, we might want to wait longer or handle differently
            return ray.get(x, timeout=30.0)  # Extended timeout for complex operations
        except Exception as e:
            print(f"Error getting result: {e}", file=sys.stderr)
            raise
    return x

@ray.remote
class RayCodeExecutor:
    def __init__(self, actor_id: str, timeout_in_secs: float = 30.0):
        self.actor_id = actor_id
        self.locals = {}
        self.globals = {}
        self.current_turn = 0
        self.timeout = timeout_in_secs
        self.max_steps = 100
        
        # Initialize with basic imports and utilities
        self.globals.update({
            '__builtins__': __builtins__,
            'print': print,
            'len': len,
            'range': range,
            'list': list,
            'dict': dict,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
        })
        
        print(f"🚀 RayCodeExecutor {actor_id} initialized", file=sys.stderr)
        
    def execute(self, code: str, success_criterion: Optional[str] = None) -> Dict[str, Any]:
        """Execute code and return structured response with state and output."""
        start_time = time.time()
        
        # Create fresh string buffers for this execution
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        
        # Save original stdout/stderr 
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            # Redirect output
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer
            
            # Execute the code in the persistent namespace
            # Use both globals and locals for proper scoping
            exec(code, self.globals, self.locals)
            
            # Update globals with any new definitions from locals
            self.globals.update({k: v for k, v in locals().items() 
                               if not k.startswith('_')})
        
            
        except Exception as e:
            # Capture the full traceback
            error_traceback = traceback.format_exc()
            stderr_buffer.write(f"Execution Error: {error_traceback}")
            
        finally:
            # Always restore stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
        # Get captured output
        stdout_content = stdout_buffer.getvalue()
        stderr_content = stderr_buffer.getvalue()
        
        # Combine outputs
        execution_output = ""
        if stdout_content:
            execution_output += stdout_content
        if stderr_content:
            if execution_output:
                execution_output += "\n" + stderr_content
            else:
                execution_output = stderr_content
                
        # Check success criterion if provided
        success = False
        success_message = ""
        
        if success_criterion:
            try:
                # Execute the success criterion in the same namespace
                success_result = eval(success_criterion, self.globals, self.locals)
                success = bool(success_result)
                success_message = f"Success criterion '{success_criterion}' evaluated to: {success_result}"
            except Exception as e:
                success = False
                success_message = f"Success criterion failed: {str(e)}"
        
        # Determine state
        has_error = "Error:" in stderr_content or "Traceback" in stderr_content
        state = "crashed" if has_error else ("success" if success or not success_criterion else "running")
        
        self.current_turn += 1
        execution_time = time.time() - start_time
        
        result = {
            "state": state,
            "execution_output": execution_output.strip(),
            "execution_time": execution_time,
            "turn": self.current_turn,
            "success": success,
            "success_message": success_message,
            "actor_id": self.actor_id,
            "variables": list(self.locals.keys())  # Available variables
        }
        
        print(f"🔍 Actor {self.actor_id} executed turn {self.current_turn}, state: {state}", file=sys.stderr)
        return result
    
    def get_variable(self, var_name: str) -> Any:
        """Get the value of a variable from the executor's namespace."""
        if var_name in self.locals:
            return self.locals[var_name]
        elif var_name in self.globals:
            return self.globals[var_name]
        else:
            raise NameError(f"Variable '{var_name}' not found")
    
    def get_namespace_info(self) -> Dict[str, Any]:
        """Get information about the current namespace."""
        return {
            "actor_id": self.actor_id,
            "turn": self.current_turn,
            "local_variables": list(self.locals.keys()),
            "local_variable_types": {k: type(v).__name__ for k, v in self.locals.items()},
        }


# Global actor pool
ACTOR_POOL: Dict[str, ray.ObjectRef] = {}


def ensure_ray_initialized():
    """Ensure Ray is initialized."""
    if not ray.is_initialized():
        try:
            # Try to connect to existing cluster first
            ray.init(address="auto", ignore_reinit_error=True)
            print("🔗 Connected to existing Ray cluster", file=sys.stderr)
        except:
            # Start local Ray instance
            ray.init(ignore_reinit_error=True)
            print("🚀 Started local Ray instance", file=sys.stderr)


def start_instance(timeout_in_secs: float = 30.0, num_cpus: int = 1, num_gpus: int = 0) -> str:
    """Start a new RayCodeExecutor instance with specified resources."""
    ensure_ray_initialized()
    
    actor_id = str(uuid.uuid4())
    
    # Create actor with resource allocation
    actor_options = {"num_cpus": num_cpus}
    if num_gpus > 0:
        actor_options["num_gpus"] = num_gpus
    
    # Apply resource options to the remote actor
    RemoteExecutor = RayCodeExecutor.options(**actor_options)
    actor_ref = RemoteExecutor.remote(actor_id, timeout_in_secs)
    
    # Store in global pool
    ACTOR_POOL[actor_id] = actor_ref
    
    print(f"🔍 RAY_ENGINE start_instance() created actor_id={actor_id}", file=sys.stderr)
    print(f"🔍 RAY_ENGINE ACTOR_POOL now has {len(ACTOR_POOL)} actors", file=sys.stderr)
    
    return actor_id


def execute_code(actor_id: str, code: str, success_criterion: Optional[str] = None) -> Dict[str, Any]:
    """Execute code in a specific RayCodeExecutor instance."""
    print(f"🔍 RAY_ENGINE exec() called with actor_id={actor_id}", file=sys.stderr)
    
    if actor_id not in ACTOR_POOL:
        print(f"❌ RAY_ENGINE Actor {actor_id} not found in pool", file=sys.stderr)
        print(f"🔍 Available actors: {list(ACTOR_POOL.keys())}", file=sys.stderr)
        return {
            "state": "crashed",
            "execution_output": f"Actor with ID {actor_id} does not exist in pool.",
            "actor_id": actor_id,
            "available_actors": list(ACTOR_POOL.keys())
        }

    actor = ACTOR_POOL[actor_id]
    
    try:
        # Execute code on the actor and get result
        result_ref = actor.execute.remote(code, success_criterion)
        result = get_result(result_ref)
        
        print(f"🔍 RAY_ENGINE exec() completed for actor_id={actor_id}, state={result.get('state', 'unknown')}", file=sys.stderr)
        return result
        
    except Exception as e:
        error_msg = f"Execution failed: {str(e)}"
        print(f"❌ RAY_ENGINE execution failed for actor {actor_id}: {e}", file=sys.stderr)
        print(f"❌ Full traceback: {traceback.format_exc()}", file=sys.stderr)
        
        return {
            "state": "crashed",
            "execution_output": error_msg,
            "actor_id": actor_id,
            "error": str(e)
        }


def get_variable(actor_id: str, var_name: str) -> Any:
    """Get a variable value from a specific actor."""
    if actor_id not in ACTOR_POOL:
        raise ValueError(f"Actor {actor_id} not found")
    
    actor = ACTOR_POOL[actor_id]
    try:
        result_ref = actor.get_variable.remote(var_name)
        return get_result(result_ref)
    except Exception as e:
        raise RuntimeError(f"Failed to get variable '{var_name}' from actor {actor_id}: {e}")


def get_namespace_info(actor_id: str) -> Dict[str, Any]:
    """Get namespace information from a specific actor."""
    if actor_id not in ACTOR_POOL:
        return {"error": f"Actor {actor_id} not found"}
    
    actor = ACTOR_POOL[actor_id]
    try:
        result_ref = actor.get_namespace_info.remote()
        return get_result(result_ref)
    except Exception as e:
        return {"error": f"Failed to get namespace info: {e}"}


def cleanup_instance(actor_id: str) -> bool:
    """Clean up a specific RayCodeExecutor instance."""
    if actor_id in ACTOR_POOL:
        try:
            actor = ACTOR_POOL[actor_id]
            ray.kill(actor)
            del ACTOR_POOL[actor_id]
            print(f"🔍 RAY_ENGINE cleaned up actor_id={actor_id}", file=sys.stderr)
            return True
        except Exception as e:
            print(f"❌ RAY_ENGINE Error cleaning up actor {actor_id}: {e}", file=sys.stderr)
            return False
    else:
        print(f"⚠️ RAY_ENGINE Actor {actor_id} not found for cleanup", file=sys.stderr)
        return False


def cleanup_all_instances():
    """Clean up all RayCodeExecutor instances."""
    actor_ids = list(ACTOR_POOL.keys())
    print(f"🧹 RAY_ENGINE Cleaning up {len(actor_ids)} actors", file=sys.stderr)
    
    for actor_id in actor_ids:
        cleanup_instance(actor_id)
    
    print("🧹 RAY_ENGINE All actors cleaned up", file=sys.stderr)


def list_instances() -> List[str]:
    """List all active actor instances."""
    return list(ACTOR_POOL.keys())


def is_instance_alive(actor_id: str) -> bool:
    """Check if an actor instance is still alive."""
    if actor_id not in ACTOR_POOL:
        return False
    
    try:
        actor = ACTOR_POOL[actor_id]
        # Try to get namespace info as a health check
        result_ref = actor.get_namespace_info.remote()
        get_result(result_ref, wait=5.0)  # Short timeout for health check
        return True
    except Exception as e:
        print(f"⚠️ Actor {actor_id} appears to be dead: {e}", file=sys.stderr)
        # Clean up the dead actor
        if actor_id in ACTOR_POOL:
            del ACTOR_POOL[actor_id]
        return False


# Example usage and testing functions
def example_usage():
    """Example of how to use the Ray code executor."""
    
    # Start an instance
    actor_id = start_instance()
    print(f"Started actor: {actor_id}")
    
    # Execute some code
    result1 = execute_code(actor_id, """
x = 42
y = [1, 2, 3, 4, 5]
print(f"x = {x}")
print(f"y = {y}")
""")
    print("Result 1:", result1)
    
    # Execute more code that uses previous variables
    result2 = execute_code(actor_id, """
z = x * 2
print(f"z = x * 2 = {z}")
total = sum(y)
print(f"sum(y) = {total}")
""", success_criterion="z == 84 and total == 15")
    print("Result 2:", result2)
    
    # Get namespace info
    info = get_namespace_info(actor_id)
    print("Namespace info:", info)
    
    # Get a specific variable
    try:
        x_value = get_variable(actor_id, "x")
        print(f"Value of x: {x_value}")
    except Exception as e:
        print(f"Error getting variable: {e}")
    
    # Clean up
    cleanup_instance(actor_id)


if __name__ == "__main__":
    example_usage()