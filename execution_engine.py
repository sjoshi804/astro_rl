"""
Code Execution Engine
Manages Python execution environments for multi-turn code interaction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Tuple
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
CONDA_ENV = None

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
    
    def __init__(self, instance_id: str, timeout_per_step: int, max_steps: int, prob_completion: float, conda_env: Optional[str] = None):
        self.instance_id = instance_id
        self.timeout_per_step = timeout_per_step
        self.max_steps = max_steps
        self.prob_completion = prob_completion
        self.conda_env = conda_env
        self.step_count = 0
        self.process = None
        self.created_at = datetime.now()
        self.temp_dir = tempfile.mkdtemp(prefix=f"exec_{instance_id}_")
        logger.info(f"Created execution instance {instance_id} with temp dir {self.temp_dir}, conda env: {conda_env}")
    
    async def _get_conda_python_path(self) -> str:
        """Get the Python executable path for the conda environment"""
        try:
            # Run conda info to get the Python path for the environment
            result = await asyncio.create_subprocess_exec(
                'conda', 'info', '--envs', '--json',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                import json
                env_info = json.loads(stdout.decode())
                envs = env_info.get('envs', [])
                
                # Find the environment path
                for env_path in envs:
                    if env_path.endswith(f'/envs/{self.conda_env}') or env_path.endswith(f'/{self.conda_env}'):
                        python_path = os.path.join(env_path, 'bin', 'python')
                        if os.path.exists(python_path):
                            return python_path
                        # Fallback for Windows
                        python_path = os.path.join(env_path, 'python.exe')
                        if os.path.exists(python_path):
                            return python_path
            
            # Fallback: use conda run to get python path
            result = await asyncio.create_subprocess_exec(
                'conda', 'run', '-n', self.conda_env, 'which', 'python',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                return stdout.decode().strip()
                
        except Exception as e:
            logger.warning(f"Failed to get conda python path: {e}")
        
        # Ultimate fallback
        return 'python'

    async def start(self):
        """Start the Python process"""
        try:
            if self.conda_env:
                # Get the Python executable from the conda environment
                python_executable = await self._get_conda_python_path()
                logger.info(f"Using Python executable: {python_executable} for environment: {self.conda_env}")
                
                # Start Python directly with the environment's executable
                self.process = await asyncio.create_subprocess_exec(
                    python_executable, '-u', '-i',
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=self.temp_dir
                )
            else:
                # Use default Python directly
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
                f"os.chdir('{self.temp_dir}')"
            ]
            
            # Add conda environment verification if using conda
            if self.conda_env:
                setup_commands.extend([
                    f"print(f'Python executable: {{sys.executable}}')",
                    f"print(f'Conda environment: {self.conda_env}')",
                    "try:",
                    "    import conda",
                    "    print('Conda available in environment')",
                    "except ImportError:",
                    "    print('Conda not available (expected for some environments)')",
                ])
            
            setup_commands.append("print('EXEC_ENGINE_READY')")
            
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
        code, added_imports = self._extract_code(text)
        if not code:
            logger.warning(f"No parseable code found in input text: {text[:200]}...")
            return ExecutionResponse(
                output="ERROR: No parseable code found in input",
                state="running",
                step_count=self.step_count
            )
        
        self.step_count += 1
        logger.info(f"Executing step {self.step_count} in instance {self.instance_id}")
        logger.debug(f"Extracted code for execution:\n{code}")
        
        try:
            # Execute the code with timeout
            output = await asyncio.wait_for(
                self._execute_code_in_process(code),
                timeout=self.timeout_per_step
            )
            
            # Prepend import information if imports were added
            if added_imports:
                import_msg = f"Added suggested imports: {', '.join(added_imports)}\n\n"
                output = import_msg + output
            
            # Check completion criteria
            completion_status = self._check_completion()
            
            if completion_status:
                state = "success"
            elif self.step_count >= self.max_steps:
                logger.info(f"Step {self.step_count} completed with state: max_steps_exceeded")
                state = "max_steps_exceeded"
            else:
                state = "running"
            
            logger.info(f"Step {self.step_count} completed with state: {state}")
            
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
    
    def _extract_code(self, text: str) -> Tuple[str, List[str]]:
        """Extract code from text using improved heuristics"""
        added_imports = []
        
        # First, try to extract code blocks with ``` markers
        code_blocks = self._extract_code_blocks(text)
        if code_blocks:
            # Combine all valid code blocks
            valid_blocks = []
            for block in code_blocks:
                cleaned = self._clean_code_block(block)
                if cleaned and len(cleaned.strip()) > 10:  # More lenient than _is_valid_python_code
                    # Quick syntax check - if it has basic code patterns, include it
                    if self._has_code_patterns(cleaned):
                        valid_blocks.append(cleaned)
            
            if valid_blocks:
                combined = '\n\n'.join(valid_blocks)
                # Add missing imports to the combined code
                combined, added_imports = self._add_missing_imports(combined)
                return combined, added_imports
        
        # If no code blocks found, try to extract from the entire text
        cleaned_text = self._clean_and_filter_text(text)
        if cleaned_text and len(cleaned_text.strip()) > 10:
            # Add missing imports and return
            cleaned_text, added_imports = self._add_missing_imports(cleaned_text)
            return cleaned_text, added_imports
        
        return "", []
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract all code blocks marked with ``` from text"""
        # Pattern to match code blocks with optional language specification
        pattern = r'```(?:python|py)?\s*\n?(.*?)(?:\n```|```|$)'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        return matches
    
    def _clean_code_block(self, code: str) -> str:
        """Clean up a code block by removing common artifacts"""
        # Remove trailing ``` if present
        if code.endswith('```'):
            code = code[:-3]
        
        # Split into lines for processing
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines at the beginning, but keep them in the middle/end for structure
            if not line and not cleaned_lines:
                continue
                
            # Skip lines that look like markdown or explanatory text
            if self._is_explanatory_line(line):
                continue
                
            # Skip lines with common artifacts
            if any(artifact in line.lower() for artifact in [
                'this code snippet', 'the code above', 'step 1:', 'step 2:',
                'created question:', 'created answer:', '### ', '# step',
                'execution ✓:', 'indentationerror:', 'syntaxerror:'
            ]):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _is_explanatory_line(self, line: str) -> bool:
        """Check if a line is explanatory text rather than code"""
        line_lower = line.lower().strip()
        
        # Skip lines that are clearly explanatory
        explanatory_patterns = [
            r'^#+\s',  # Markdown headers
            r'^to\s+\w+',  # "To calculate...", "To modify..."
            r'^this\s+code',  # "This code snippet..."
            r'^how\s+would',  # Questions
            r'^you\s+can',  # Instructions
            r'^\*\s',  # Bullet points
            r'^-\s',   # Dash bullet points
            r'execution\s+✓',  # Execution markers
            r'^\d+\.\s',  # Numbered lists
        ]
        
        for pattern in explanatory_patterns:
            if re.match(pattern, line_lower):
                return True
        
        # Skip very short lines that are likely headers or separators
        if len(line.strip()) < 3:
            return False
            
        # Check if line has no Python syntax at all
        python_indicators = ['=', '(', ')', '[', ']', '.', 'import', 'def', 'if', 'for', 'print']
        if not any(indicator in line for indicator in python_indicators):
            # But allow comments
            if not line.strip().startswith('#'):
                return True
        
        return False
    
    def _clean_and_filter_text(self, text: str) -> str:
        """Clean text and filter out non-code content"""
        lines = text.split('\n')
        code_lines = []
        
        in_code_section = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines at the start
            if not line_stripped and not code_lines:
                continue
            
            # Skip explanatory sections
            if self._is_explanatory_line(line_stripped):
                in_code_section = False
                continue
            
            # Detect start of code sections
            if self._looks_like_code_start(line_stripped):
                in_code_section = True
                code_lines.append(line)
                continue
            
            # If we're in a code section, keep the line
            if in_code_section or self._looks_like_code_line(line_stripped):
                code_lines.append(line)
                in_code_section = True
            
        return '\n'.join(code_lines).strip()
    
    def _looks_like_code_start(self, line: str) -> bool:
        """Check if line looks like the start of a code section"""
        line_lower = line.lower()
        code_starters = [
            'import ', 'from ', 'def ', 'class ', 'try:', 'if ', 'for ', 'while ',
            '# import', '# define', '# load', '# create', '# apply'
        ]
        return any(line_lower.startswith(starter) for starter in code_starters)
    
    def _looks_like_code_line(self, line: str) -> bool:
        """Check if a single line looks like code"""
        if not line:
            return False
            
        # Comments are code
        if line.startswith('#'):
            return True
            
        # Lines with assignments, function calls, etc.
        code_patterns = [
            r'=\s+',  # Assignments
            r'\w+\(',  # Function calls
            r'\w+\.\w+',  # Method calls
            r'^\s*(if|for|while|try|except|with|def|class)\s',  # Control structures
            r'print\s*\(',  # Print statements
            r'plt\.',  # Matplotlib calls
            r'np\.',   # NumPy calls
        ]
        
        return any(re.search(pattern, line) for pattern in code_patterns)
    
    def _is_valid_python_code(self, code: str) -> bool:
        """Check if the extracted text is valid Python code"""
        if not code.strip():
            return False
        
        # Add missing imports if we detect their usage
        code = self._add_missing_imports(code)
        
        # Try to parse the code to check syntax
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            # If there's a syntax error, try to fix common issues
            fixed_code = self._fix_common_syntax_issues(code)
            if fixed_code != code:
                try:
                    compile(fixed_code, '<string>', 'exec')
                    return True
                except SyntaxError:
                    pass
            return False
        except Exception:
            # Other compilation errors might still be valid code
            return True
    
    def _add_missing_imports(self, code: str) -> Tuple[str, List[str]]:
        """Add commonly missing imports based on code content"""
        added_imports = []
        
        # Check for astropy usage
        if 'sigma_clip(' in code and 'from astropy.stats import' in code:
            # Add sigma_clip to existing astropy.stats import
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if 'from astropy.stats import' in line and 'sigma_clip' not in line:
                    if 'sigma_clipped_stats' in line:
                        lines[i] = line.replace('sigma_clipped_stats', 'sigma_clipped_stats, sigma_clip')
                    else:
                        lines[i] = line + ', sigma_clip'
                    break
            code = '\n'.join(lines)
        elif 'sigma_clip(' in code and 'from astropy.stats import' not in code:
            added_imports.append('from astropy.stats import sigma_clip')
        
        # Check for other common missing imports
        if 'plt.' in code and 'import matplotlib.pyplot' not in code:
            added_imports.append('import matplotlib.pyplot as plt')
        
        if 'np.' in code and 'import numpy' not in code:
            added_imports.append('import numpy as np')
            
        if added_imports:
            return '\n'.join(added_imports) + '\n\n' + code, added_imports
        
        return code, added_imports
    
    def _fix_common_syntax_issues(self, code: str) -> str:
        """Fix common syntax issues in extracted code"""
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Remove lines with obvious syntax errors that are artifacts
            if 'IndentationError:' in line or 'SyntaxError:' in line:
                continue
            if line.strip().startswith('^^^^'):
                continue
            if 'File "<stdin>"' in line:
                continue
            
            # Fix incomplete try-except blocks
            if line.strip() == 'except Exception as e:' and not any('try:' in prev_line for prev_line in fixed_lines[-5:]):
                continue
                
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    async def _execute_code_in_process(self, code: str) -> str:
        """Execute code in the Python process and capture output"""
        # Add a unique marker to identify end of output
        marker = f"EXEC_END_{uuid.uuid4().hex[:8]}"
        
        # Clean and dedent the code to avoid indentation issues
        cleaned_code = self._clean_code_for_execution(code)
        
        # Instead of wrapping in try-except (which causes indentation issues),
        # send the code directly and handle errors separately
        full_code = f"{cleaned_code}\nprint('{marker}')\n"
        
        # Send code to process
        self.process.stdin.write(full_code.encode())
        await self.process.stdin.drain()
        
        # Read output until we see our marker
        output_lines = []
        error_detected = False
        
        while True:
            line = await self.process.stdout.readline()
            if not line:
                break
            
            line_str = line.decode().strip()
            if marker in line_str:
                break
            
            # Check for errors
            if any(error_indicator in line_str for error_indicator in [
                'SyntaxError:', 'IndentationError:', 'NameError:', 'TypeError:', 
                'ValueError:', 'AttributeError:', 'ImportError:', 'Traceback'
            ]):
                error_detected = True
            
            if line_str:  # Skip empty lines
                output_lines.append(line_str)
        
        result = '\n'.join(output_lines) if output_lines else "No output"
        
        # If errors were detected but no specific error output, add a note
        if error_detected and result == "No output":
            result = "Code execution encountered an error (see above for details)"
        
        return result
    
    def _indent_code(self, code: str) -> str:
        """Indent code for wrapping in try-except block"""
        return '\n'.join(f"    {line}" for line in code.splitlines())
    
    def _clean_code_for_execution(self, code: str) -> str:
        """Clean and prepare code for execution in interactive shell"""
        import textwrap
        
        # Remove any existing indentation to avoid conflicts
        # This handles code that was already indented
        cleaned_code = textwrap.dedent(code)
        
        # Split into lines and filter out problematic lines
        lines = cleaned_code.split('\n')
        clean_lines = []
        
        for line in lines:
            # Skip lines that are execution artifacts or error messages
            if any(artifact in line for artifact in [
                'IndentationError:', 'SyntaxError:', 'File "<stdin>"', 
                '^^^^', '>>>', 'Traceback (most recent call last):',
                'Exception:', 'Error:'
            ]):
                continue
            
            # Skip lines that start with shell prompts
            if line.strip().startswith('>>> ') or line.strip().startswith('... '):
                # Extract the actual code after the prompt
                if '>>> ' in line:
                    line = line.split('>>> ', 1)[1]
                elif '... ' in line:
                    line = line.split('... ', 1)[1]
            
            # Keep the line
            clean_lines.append(line)
        
        # Rejoin and ensure it ends with a newline
        result = '\n'.join(clean_lines).strip()
        
        # If code is empty after cleaning, return empty string
        if not result:
            return ""
        
        # For multi-line code blocks, ensure proper execution by adding double newline at end
        # This helps the interactive shell know when a block is complete
        if any(line.rstrip().endswith(':') for line in result.split('\n')):
            return result + '\n\n'
        else:
            return result + '\n' if not result.endswith('\n') else result
    
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

    def _has_code_patterns(self, code: str) -> bool:
        """Check if code has basic Python patterns (more lenient than full validation)"""
        if not code.strip():
            return False
        
        # Look for common Python patterns
        patterns = [
            r'import\s+\w+',  # imports
            r'from\s+\w+\s+import',  # from imports
            r'\w+\s*=\s*\w+',  # assignments
            r'\w+\(\w*.*?\)',  # function calls
            r'if\s+\w+',  # if statements
            r'for\s+\w+\s+in',  # for loops
            r'def\s+\w+\(',  # function definitions
            r'print\s*\(',  # print statements
            r'#\s*\w+',  # comments
        ]
        
        code_lower = code.lower()
        pattern_count = sum(1 for pattern in patterns if re.search(pattern, code_lower))
        
        # If we have multiple patterns or this looks like structured code, accept it
        return pattern_count >= 2 or self._looks_like_structured_code(code)
    
    def _looks_like_structured_code(self, code: str) -> bool:
        """Check if code has structural patterns that indicate it's Python code"""
        lines = code.split('\n')
        
        # Count lines that look like code
        code_like_lines = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Count lines with code indicators
            if (line.startswith('#') or  # comments
                '=' in line or  # assignments
                line.endswith(':') or  # blocks
                line.startswith('import ') or
                line.startswith('from ') or
                'print(' in line or
                any(op in line for op in ['(', ')', '[', ']', '.'])):  # operators/syntax
                code_like_lines += 1
        
        # If more than half the non-empty lines look like code, accept it
        non_empty_lines = len([l for l in lines if l.strip()])
        return non_empty_lines > 0 and (code_like_lines / non_empty_lines) >= 0.5

class ExecutionManager:
    """Manages multiple execution instances"""
    
    def __init__(self, timeout_per_step: int, max_steps: int, prob_completion: float, conda_env: Optional[str] = None):
        self.timeout_per_step = timeout_per_step
        self.max_steps = max_steps
        self.prob_completion = prob_completion
        self.conda_env = conda_env
        self.instances: Dict[str, ExecutionInstance] = {}
    
    async def create_instance(self) -> str:
        """Create a new execution instance"""
        instance_id = str(uuid.uuid4())
        
        instance = ExecutionInstance(
            instance_id=instance_id,
            timeout_per_step=self.timeout_per_step,
            max_steps=self.max_steps,
            prob_completion=self.prob_completion,
            conda_env=self.conda_env
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
        prob_completion=PROB_COMPLETION,
        conda_env=CONDA_ENV
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
        "conda_env": CONDA_ENV,
        "instances": {
            instance_id: {
                "step_count": instance.step_count,
                "created_at": instance.created_at.isoformat(),
                "conda_env": instance.conda_env
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
    parser.add_argument("--max-steps", type=int, default=10,
                       help="Maximum steps per execution instance")
    parser.add_argument("--prob-completion", type=float, default=0.1,
                       help="Probability of completion criteria being met (0.0-1.0)")
    parser.add_argument("--conda-env", type=str, default=None,
                       help="Conda environment name to use for execution")
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
    CONDA_ENV = args.conda_env
    
    logger.info(f"Starting Code Execution Engine")
    logger.info(f"Timeout per step: {TIMEOUT_PER_STEP}s")
    logger.info(f"Max steps: {MAX_STEPS}")
    logger.info(f"Completion probability: {PROB_COMPLETION}")
    logger.info(f"Conda environment: {CONDA_ENV or 'default'}")
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)