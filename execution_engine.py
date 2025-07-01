"""
Code Execution Engine
Manages Python execution environments for multi-turn code interaction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
import re
import asyncio
import subprocess
import tempfile
import os
import random
import logging
import argparse
from datetime import datetime
import signal
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
TIMEOUT_PER_STEP = None
MAX_STEPS = None
PROB_COMPLETION = None

app = FastAPI(title="Code Execution Engine")

# Data Models
class StartExecutionRequest(BaseModel):
    """Request to start a new execution instance"""
    pass

class ExecuteCommandRequest(BaseModel):
    """Request to execute code in an existing instance"""
    instance_id: str
    text: str

class ExecutionResponse(BaseModel):
    """Response from code execution"""
    output: str
    state: str  # "running", "success", "crash", "max_steps_exceeded"
    step_count: int

class ExecutionInstance:
    """Manages a single Python execution environment"""
    
    def __init__(self, instance_id: str, timeout_per_step: int, max_steps: int, prob_completion: float):
        self.instance_id = instance_id
        self.timeout_per_step = timeout_per_step
        self.max_steps = max_steps
        self.prob_completion = prob_completion
        self.step_count = 0
        self.process = None
        self.created_at = datetime.now()
        self.temp_dir = tempfile.mkdtemp(prefix=f"exec_{instance_id}_")
        logger.info(f"Created execution instance {instance_id} with temp dir {self.temp_dir}")
    
    async def start(self):
        """Start the Python process"""
        try:
            # Start Python in interactive mode with unbuffered output
            self.process = await asyncio.create_subprocess_exec(
                'python', '-u', '-i',
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.temp_dir
            )
            
            # Send initial setup commands
            setup_commands = [
                "import sys",
                "import os", 
                "import traceback",
                f"os.chdir('{self.temp_dir}')",
                "print('EXEC_ENGINE_READY')"
            ]
            
            for cmd in setup_commands:
                self.process.stdin.write(f"{cmd}\n".encode())
            await self.process.stdin.drain()
            
            # Wait for ready signal
            await self._read_until_ready()
            logger.info(f"Instance {self.instance_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start instance {self.instance_id}: {e}")
            await self.cleanup()
            return False
    
    async def _read_until_ready(self):
        """Read output until we see the ready signal"""
        output_buffer = ""
        try:
            while True:
                line = await asyncio.wait_for(
                    self.process.stdout.readline(), 
                    timeout=5.0
                )
                if not line:
                    break
                line_str = line.decode().strip()
                output_buffer += line_str + "\n"
                if "EXEC_ENGINE_READY" in line_str:
                    break
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for ready signal from {self.instance_id}")
    
    async def execute_code(self, text: str) -> ExecutionResponse:
        """Execute code extracted from text"""
        if not self.process or self.process.returncode is not None:
            return ExecutionResponse(
                output="ERROR: Execution instance not running",
                state="crash",
                step_count=self.step_count
            )
        
        # Check if max steps exceeded
        if self.step_count >= self.max_steps:
            return ExecutionResponse(
                output="Maximum steps exceeded",
                state="max_steps_exceeded", 
                step_count=self.step_count
            )
        
        # Extract code from text
        code = self._extract_code(text)
        if not code:
            return ExecutionResponse(
                output="ERROR: No parseable code found in input",
                state="running",
                step_count=self.step_count
            )
        
        self.step_count += 1
        logger.info(f"Executing step {self.step_count} in instance {self.instance_id}")
        
        try:
            # Execute the code with timeout
            output = await asyncio.wait_for(
                self._execute_code_in_process(code),
                timeout=self.timeout_per_step
            )
            
            # Check completion criteria
            completion_status = self._check_completion()
            
            if completion_status:
                state = "success"
            elif self.step_count >= self.max_steps:
                state = "max_steps_exceeded"
            else:
                state = "running"
            
            return ExecutionResponse(
                output=output,
                state=state,
                step_count=self.step_count
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Execution timeout in instance {self.instance_id}")
            return ExecutionResponse(
                output="ERROR: Execution timeout",
                state="running",
                step_count=self.step_count
            )
        except Exception as e:
            logger.error(f"Execution error in instance {self.instance_id}: {e}")
            return ExecutionResponse(
                output=f"ERROR: {str(e)}",
                state="crash",
                step_count=self.step_count
            )
    
    def _extract_code(self, text: str) -> str:
        """Extract code from text using simple heuristics"""
        # Try to find code blocks with ``` (both python and generic)
        code_block_pattern = r'```(?:python)?\s*\n?(.*?)(?:\n```|$)'
        matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Return the first code block found, cleaned up
            code = matches[0].strip()
            # Remove any trailing ``` that might have been captured
            if code.endswith('```'):
                code = code[:-3].strip()
            return code
        
        # Try to find inline code with single backticks
        inline_code_pattern = r'`([^`]+)`'
        inline_matches = re.findall(inline_code_pattern, text)
        
        if inline_matches:
            # Join multiple inline code snippets
            return '\n'.join(match.strip() for match in inline_matches)
        
        # If no code blocks found, try to identify if the entire text looks like code
        if self._looks_like_code(text):
            return text.strip()
        
        return ""
    
    def _looks_like_code(self, text: str) -> bool:
        """Simple heuristic to determine if text looks like Python code"""
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ',
            'try:', 'except:', 'print(', 'return ', '= ', '+= ', '-= ',
            'lambda ', 'with ', 'assert ', 'yield ', 'async ', 'await '
        ]
        
        text_lower = text.lower().strip()
        # Check if text contains multiple code indicators or looks structured
        indicator_count = sum(1 for indicator in code_indicators if indicator in text_lower)
        
        # Also check for Python-like patterns
        has_indentation = any(line.startswith('    ') or line.startswith('\t') for line in text.split('\n'))
        has_colon_newline = ':' in text and '\n' in text
        
        return indicator_count >= 2 or (indicator_count >= 1 and (has_indentation or has_colon_newline))
    
    async def _execute_code_in_process(self, code: str) -> str:
        """Execute code in the Python process and capture output"""
        # Add a unique marker to identify end of output
        marker = f"EXEC_END_{uuid.uuid4().hex[:8]}"
        
        # Prepare code with error handling and output marker
        wrapped_code = f"""
try:
{self._indent_code(code)}
except Exception as e:
    print(f"Exception: {{e}}")
    traceback.print_exc()
print("{marker}")
"""
        
        # Send code to process
        self.process.stdin.write(wrapped_code.encode())
        await self.process.stdin.drain()
        
        # Read output until we see our marker
        output_lines = []
        while True:
            line = await self.process.stdout.readline()
            if not line:
                break
            
            line_str = line.decode().strip()
            if marker in line_str:
                break
            
            if line_str:  # Skip empty lines
                output_lines.append(line_str)
        
        return '\n'.join(output_lines) if output_lines else "No output"
    
    def _indent_code(self, code: str) -> str:
        """Indent code for wrapping in try-except block"""
        return '\n'.join(f"    {line}" for line in code.splitlines())
    
    def _check_completion(self) -> bool:
        """Check if execution should complete (random for now)"""
        return random.random() < self.prob_completion
    
    async def cleanup(self):
        """Clean up the execution instance"""
        logger.info(f"Cleaning up instance {self.instance_id}")
        
        if self.process and self.process.returncode is None:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        
        # Clean up temp directory
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to clean up temp dir {self.temp_dir}: {e}")

class ExecutionManager:
    """Manages multiple execution instances"""
    
    def __init__(self, timeout_per_step: int, max_steps: int, prob_completion: float):
        self.timeout_per_step = timeout_per_step
        self.max_steps = max_steps
        self.prob_completion = prob_completion
        self.instances: Dict[str, ExecutionInstance] = {}
    
    async def create_instance(self) -> str:
        """Create a new execution instance"""
        instance_id = str(uuid.uuid4())
        
        instance = ExecutionInstance(
            instance_id=instance_id,
            timeout_per_step=self.timeout_per_step,
            max_steps=self.max_steps,
            prob_completion=self.prob_completion
        )
        
        if await instance.start():
            self.instances[instance_id] = instance
            logger.info(f"Created and started instance {instance_id}")
            return instance_id
        else:
            raise Exception(f"Failed to start execution instance {instance_id}")
    
    async def execute_in_instance(self, instance_id: str, text: str) -> ExecutionResponse:
        """Execute code in a specific instance"""
        if instance_id not in self.instances:
            raise HTTPException(status_code=404, detail="Execution instance not found")
        
        instance = self.instances[instance_id]
        response = await instance.execute_code(text)
        
        # Clean up instance if it's finished or crashed
        if response.state in ["success", "crash", "max_steps_exceeded"]:
            await self._cleanup_instance(instance_id)
        
        return response
    
    async def _cleanup_instance(self, instance_id: str):
        """Clean up and remove an instance"""
        if instance_id in self.instances:
            instance = self.instances.pop(instance_id)
            await instance.cleanup()
            logger.info(f"Cleaned up and removed instance {instance_id}")
    
    async def cleanup_all(self):
        """Clean up all instances"""
        for instance_id in list(self.instances.keys()):
            await self._cleanup_instance(instance_id)

# Global execution manager
execution_manager: Optional[ExecutionManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global execution_manager
    execution_manager = ExecutionManager(
        timeout_per_step=TIMEOUT_PER_STEP,
        max_steps=MAX_STEPS,
        prob_completion=PROB_COMPLETION
    )
    logger.info("Execution manager initialized")
    yield
    await execution_manager.cleanup_all()
    logger.info("All execution instances cleaned up")

# Update app with lifespan
app = FastAPI(title="Code Execution Engine", lifespan=lifespan)

# API Endpoints

@app.post("/start_execution")
async def start_execution(request: StartExecutionRequest = StartExecutionRequest()) -> Dict[str, str]:
    """Start a new execution instance"""
    try:
        instance_id = await execution_manager.create_instance()
        return {"instance_id": instance_id}
    except Exception as e:
        logger.error(f"Failed to create execution instance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute")
async def execute_command(request: ExecuteCommandRequest) -> ExecutionResponse:
    """Execute code in an existing instance"""
    return await execution_manager.execute_in_instance(request.instance_id, request.text)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_instances": len(execution_manager.instances) if execution_manager else 0,
        "timestamp": datetime.now()
    }

@app.get("/stats")
async def get_stats():
    """Get execution engine statistics"""
    if not execution_manager:
        return {"error": "Execution manager not initialized"}
    
    return {
        "active_instances": len(execution_manager.instances),
        "timeout_per_step": TIMEOUT_PER_STEP,
        "max_steps": MAX_STEPS,
        "prob_completion": PROB_COMPLETION,
        "instances": {
            instance_id: {
                "step_count": instance.step_count,
                "created_at": instance.created_at.isoformat()
            }
            for instance_id, instance in execution_manager.instances.items()
        }
    }

@app.delete("/instances/{instance_id}")
async def cleanup_instance(instance_id: str):
    """Manually cleanup a specific instance"""
    if instance_id not in execution_manager.instances:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    await execution_manager._cleanup_instance(instance_id)
    return {"status": "cleaned_up", "instance_id": instance_id}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Code Execution Engine")
    parser.add_argument("--timeout-per-step", type=int, default=10,
                       help="Timeout per execution step in seconds")
    parser.add_argument("--max-steps", type=int, default=20,
                       help="Maximum steps per execution instance")
    parser.add_argument("--prob-completion", type=float, default=0.1,
                       help="Probability of completion criteria being met (0.0-1.0)")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host to bind the service")
    parser.add_argument("--port", type=int, default=8001,
                       help="Port to bind the service")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set global configuration
    TIMEOUT_PER_STEP = args.timeout_per_step
    MAX_STEPS = args.max_steps
    PROB_COMPLETION = args.prob_completion
    
    logger.info(f"Starting Code Execution Engine")
    logger.info(f"Timeout per step: {TIMEOUT_PER_STEP}s")
    logger.info(f"Max steps: {MAX_STEPS}")
    logger.info(f"Completion probability: {PROB_COMPLETION}")
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)