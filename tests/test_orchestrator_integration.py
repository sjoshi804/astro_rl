"""
Integration tests for the Orchestrator class with mocked execution engine and completion server
Tests realistic scenarios similar to how astro_data_generator.py uses the orchestrator
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from multi_turn_rl.orchestrator import Orchestrator
from multi_turn_rl.models import (
    TrajectoryRequest, 
    BatchTrajectoryRequest, 
    Turn, 
    Trajectory
)


class MockCompletionLoadBalancer:
    """Mock completion load balancer that simulates code generation responses"""
    
    def __init__(self):
        self.request_count = 0
        self.responses = []
        
    async def chat_completions(self, request):
        """Mock chat completions that return realistic code responses"""
        self.request_count += 1
        
        # Extract the prompt from the request
        messages = request.get("messages", [])
        if messages:
            last_message = messages[-1].get("content", "")
            
            # Generate different responses based on the prompt content
            if "header" in last_message.lower():
                code_response = self._generate_header_analysis_code_response(last_message)
            elif "statistic" in last_message.lower():
                code_response = self._generate_statistics_code_response(last_message)
            elif "histogram" in last_message.lower():
                code_response = self._generate_histogram_code_response(last_message)
            elif "fits" in last_message.lower() or "astronomy" in last_message.lower():
                code_response = self._generate_astronomy_code_response(last_message)
            elif "error" in last_message.lower():
                code_response = self._generate_error_fix_response(last_message)
            else:
                code_response = self._generate_generic_code_response(last_message)
        else:
            code_response = self._generate_generic_code_response("")
            
        response = {
            "id": f"mock-completion-{self.request_count}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": code_response
                },
                "finish_reason": "stop"
            }]
        }
        
        self.responses.append(response)
        return response
    
    def _generate_astronomy_code_response(self, prompt):
        """Generate realistic astronomy-related code"""
        return '''```python
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# Load the FITS file
with fits.open('/path/to/fits/file.fits') as hdul:
    # Get the primary HDU
    primary_hdu = hdul[0]
    
    # Display basic information
    print(f"Dimensions: {primary_hdu.data.shape}")
    print(f"Data type: {primary_hdu.data.dtype}")
    print(f"Number of header keys: {len(primary_hdu.header)}")
    
    # Show some key header information
    print("Key header information:")
    for key in ['OBJECT', 'TELESCOP', 'INSTRUME', 'NAXIS1', 'NAXIS2']:
        if key in primary_hdu.header:
            print(f"{key}: {primary_hdu.header[key]}")
    
    # Create a simple visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(primary_hdu.data, cmap='gray', origin='lower')
    plt.colorbar(label='Pixel Value')
    plt.title('FITS Image Data')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.show()
    
    print("FITS file analysis complete!")
```'''
    
    def _generate_error_fix_response(self, prompt):
        """Generate code that fixes common errors"""
        return '''```python
try:
    # Fixed version with proper error handling
    from astropy.io import fits
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # Check if file exists first
    fits_file = '/path/to/fits/file.fits'
    if not os.path.exists(fits_file):
        print(f"Warning: FITS file not found at {fits_file}")
        print("Creating a mock dataset for demonstration...")
        
        # Create mock data
        mock_data = np.random.random((512, 512)) * 1000
        plt.figure(figsize=(8, 6))
        plt.imshow(mock_data, cmap='viridis')
        plt.colorbar()
        plt.title('Mock Astronomy Data')
        plt.show()
    else:
        with fits.open(fits_file) as hdul:
            data = hdul[0].data
            plt.imshow(data, cmap='gray')
            plt.colorbar()
            plt.show()
            
    print("Error handled successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    print("Proceeding with alternative approach...")
```'''
    
    def _generate_header_analysis_code_response(self, prompt):
        """Generate header analysis code response"""
        return '''```python
from astropy.io import fits

# Load and analyze FITS header
with fits.open('/path/to/fits/file.fits') as hdul:
    header = hdul[0].header
    
    print("FITS Header Analysis:")
    for key in ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'OBJECT', 'TELESCOP', 'INSTRUME']:
        if key in header:
            print(f"{key:<8} = {str(header[key]):<20} / {header.comments[key]}")
    
    print("\\nHeader analysis complete")
```'''

    def _generate_statistics_code_response(self, prompt):
        """Generate statistics code response"""
        return '''```python
import numpy as np
from astropy.io import fits

# Load FITS data and compute statistics
with fits.open('/path/to/fits/file.fits') as hdul:
    data = hdul[0].data
    
    print("Statistical Summary:")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Median: {np.median(data):.2f}")
    print(f"Standard Deviation: {np.std(data):.2f}")
    print(f"Min: {np.min(data):.1f}")
    print(f"Max: {np.max(data):.1f}")
    print(f"Total pixels: {data.size}")
    
    print("\\nStatistics complete")
```'''

    def _generate_histogram_code_response(self, prompt):
        """Generate histogram code response"""
        return '''```python
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Load FITS data and create histogram
with fits.open('/path/to/fits/file.fits') as hdul:
    data = hdul[0].data
    
    print("Histogram Analysis:")
    bins = np.linspace(data.min(), data.max(), 11)
    counts, bin_edges = np.histogram(data, bins=bins)
    
    print(f"Bin edges: {bin_edges.astype(int).tolist()}")
    print(f"Counts: {counts.tolist()}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(data.flatten(), bins=50, alpha=0.7)
    plt.title('Pixel Value Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
    
    print("Created histogram plot with 10 bins")
    print("Pixel value distribution analyzed")
    print("\\nHistogram complete")
```'''

    def _generate_generic_code_response(self, prompt):
        """Generate generic Python code response"""
        return '''```python
import numpy as np
import matplotlib.pyplot as plt

# Generic task solution
def solve_task():
    """Solve the given task"""
    print("Starting task execution...")
    
    # Simulate some computation
    data = np.random.random(100)
    result = np.mean(data)
    
    print(f"Computed result: {result:.4f}")
    
    # Create a simple visualization
    plt.figure(figsize=(8, 5))
    plt.plot(data)
    plt.title('Task Results')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()
    
    return result

# Execute the task
result = solve_task()
print(f"Task completed with result: {result}")
```'''


class MockExecutionEngine:
    """Mock execution engine that simulates RayExecutionEngine interface"""
    
    def __init__(self):
        self.execution_count = 0
        self.should_fail = False
        self.failure_step = None
        self.actor_pool = {}
        self.actor_counter = 0
        
    def set_failure_mode(self, should_fail=True, failure_step=None):
        """Configure the mock to simulate execution failures"""
        self.should_fail = should_fail
        self.failure_step = failure_step
        
    def start_instance(self, timeout_in_secs: float = 30.0, num_cpus: int = 1, num_gpus: int = 0) -> str:
        """Mock start_instance to create a new actor"""
        self.actor_counter += 1
        actor_id = f"mock-actor-{self.actor_counter}"
        self.actor_pool[actor_id] = {
            "timeout": timeout_in_secs,
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
            "turn": 0
        }
        return actor_id
        
    def execute_code(self, actor_id: str, code: str, success_criterion: str = None):
        """Mock code execution with realistic outputs matching Ray interface"""
        self.execution_count += 1
        
        if actor_id not in self.actor_pool:
            return {
                "state": "crashed",
                "execution_output": f"Actor with ID {actor_id} does not exist in pool.",
                "actor_id": actor_id,
                "available_actors": list(self.actor_pool.keys())
            }
        
        self.actor_pool[actor_id]["turn"] += 1
        turn = self.actor_pool[actor_id]["turn"]
        
        # Simulate execution failure if configured
        if self.should_fail and (self.failure_step is None or self.execution_count == self.failure_step):
            return {
                "state": "crashed",
                "execution_output": "ModuleNotFoundError: No module named 'astropy'",
                "execution_time": 0.1,
                "turn": turn,
                "success": False,
                "success_message": "",
                "actor_id": actor_id,
                "variables": []
            }
        
        # Simulate successful execution based on code content
        if "header" in code.lower():
            output = self._generate_header_analysis_output()
        elif "statistic" in code.lower():
            output = self._generate_statistics_output()
        elif "histogram" in code.lower():
            output = self._generate_histogram_output()
        elif "fits" in code.lower():
            output = self._generate_astronomy_execution_output()
        elif "error" in code.lower() or "exception" in code.lower():
            output = self._generate_error_handling_output()
        else:
            output = self._generate_generic_execution_output()
        
        # Check success criterion
        success = False
        success_message = ""
        if success_criterion:
            # Simple mock success criterion check - ensure both are strings
            criterion_str = str(success_criterion).lower()
            output_str = str(output).lower()
            success = criterion_str in output_str
            success_message = f"Success criterion '{success_criterion}' evaluated to: {success}"
            
        result = {
            "state": "success",  # Always return success for simplicity in tests
            "execution_output": str(output),  # Ensure it's a string
            "execution_time": 1.5,
            "turn": turn,
            "success": success,
            "success_message": str(success_message),  # Ensure it's a string
            "actor_id": str(actor_id),  # Ensure it's a string
            "variables": ["data", "result", "plt"]
        }
        
        # Add success_criterion_met field if there was a success criterion
        if success_criterion:
            result["success_criterion_met"] = success
            
        return result
    
    def cleanup_instance(self, actor_id: str):
        """Mock cleanup_instance method"""
        if actor_id in self.actor_pool:
            del self.actor_pool[actor_id]
            return True
        return False
    
    def _generate_astronomy_execution_output(self):
        """Generate realistic astronomy execution output"""
        return str("""Dimensions: (512, 512)
Data type: float32
Number of header keys: 45

Key header information:
OBJECT: NGC1234
TELESCOP: HST
INSTRUME: ACS
NAXIS1: 512
NAXIS2: 512

FITS file analysis complete""")
    
    def _generate_error_handling_output(self):
        """Generate output showing error was handled"""
        return str("""Warning: FITS file not found at /path/to/fits/file.fits
Creating a mock dataset for demonstration...
Error handled successfully!""")
    
    def _generate_header_analysis_output(self):
        """Generate header analysis specific output"""
        return str("""FITS Header Analysis:
SIMPLE  =                    T / file does conform to FITS standard             
BITPIX  =                  -32 / number of bits per data pixel                  
NAXIS   =                    2 / number of data axes                            
NAXIS1  =                  512 / length of data axis 1                         
NAXIS2  =                  512 / length of data axis 2                         
OBJECT  = 'NGC1234 '           / Name of the object                            
TELESCOP= 'HST     '           / Telescope                                     
INSTRUME= 'ACS     '           / Instrument                                    

Header analysis complete""")
    
    def _generate_statistics_output(self):
        """Generate statistics analysis output"""
        return str("""Statistical Summary:
Mean: 1245.67
Median: 1198.23
Standard Deviation: 892.45
Min: 0.0
Max: 65535.0
Total pixels: 262144

Statistics complete""")
    
    def _generate_histogram_output(self):
        """Generate histogram analysis output"""
        return str("""Histogram Analysis:
Bin edges: [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
Counts: [15234, 23456, 34567, 45678, 56789, 34567, 23456, 12345, 6789, 3456]
Created histogram plot with 10 bins
Pixel value distribution analyzed

Histogram complete""")
    
    def _generate_generic_execution_output(self):
        """Generate generic execution output"""
        return str("""Starting task execution...
Computed result: 0.4823
Task completed with result: 0.4823""")


class TestOrchestratorIntegration:
    """Integration tests for Orchestrator with mocked dependencies"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_completion_lb(self):
        """Create mock completion load balancer"""
        return MockCompletionLoadBalancer()
    
    @pytest.fixture
    def mock_execution_engine(self):
        """Create mock execution engine"""
        return MockExecutionEngine()
    
    @pytest.fixture
    def orchestrator(self, mock_completion_lb, mock_execution_engine, temp_dir):
        """Create orchestrator with mocked dependencies"""
        with patch('multi_turn_rl.orchestrator.RayExecutionEngine') as mock_engine_class:
            mock_engine_class.return_value = mock_execution_engine
            
            orchestrator = Orchestrator(
                completion_load_balancer=mock_completion_lb,
                max_turns=8,
                timeout_seconds=30,
                trajectory_output_dir=str(temp_dir / "trajectories"),
                prompts_jsonl_path=str(temp_dir / "prompts.jsonl"),
                ray_timeout_per_step=30.0,
                ray_num_cpus=1,
                ray_num_gpus=0
            )
            yield orchestrator
    
    @pytest.mark.asyncio
    async def test_single_astronomy_task_success(self, orchestrator, mock_completion_lb, mock_execution_engine):
        """Test successful single astronomy task execution"""
        # Create a request similar to what astro_data_generator.py would send
        request = BatchTrajectoryRequest(
            requests=[
                TrajectoryRequest(
                    initial_prompts=[
                        "Load the FITS file and display basic information about the image (dimensions, data type, header keys)"
                    ],
                    max_turns=8,
                    completion_criteria="Header analysis complete",
                    task_names=["astronomy_basic_info"]
                )
            ]
        )
        
        # Execute the request
        trajectories = await orchestrator.generate_trajectories(request)
        
        # Verify results
        assert len(trajectories) == 1
        trajectory = trajectories[0]
        
        assert trajectory.trajectory_id is not None
        assert len(trajectory.turns) >= 1
        assert trajectory.total_steps >= 1
        
        # With our current mock setup, it should detect completion criteria on first turn
        # since "Header analysis complete" is in the mock output
        assert trajectory.termination_reason == "completion_criteria_met"
        assert trajectory.final_reward > 0  # Should have positive reward for success
        
        # Verify the first turn contains astronomy-related code
        first_turn = trajectory.turns[0]
        assert "fits" in first_turn.code.lower() or "astropy" in first_turn.code.lower()
        assert first_turn.execution_success == True
        assert "Header analysis complete" in first_turn.execution_output
        
        # Verify completion load balancer was called
        assert mock_completion_lb.request_count >= 1
        
        # Verify execution engine was called
        assert mock_execution_engine.execution_count >= 1
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation_with_error_recovery(self, orchestrator, mock_completion_lb, mock_execution_engine):
        """Test multi-turn conversation where first attempt fails and gets recovered"""
        # Configure mock to fail on first execution, succeed on second
        mock_execution_engine.set_failure_mode(should_fail=True, failure_step=1)
        
        request = BatchTrajectoryRequest(
            requests=[
                TrajectoryRequest(
                    initial_prompts=[
                        "Create a visualization of the FITS file data with proper scaling"
                    ],
                    max_turns=5,
                    completion_criteria="visualization complete"
                )
            ]
        )
        
        # Execute the request
        trajectories = await orchestrator.generate_trajectories(request)
        
        # Verify results
        assert len(trajectories) == 1
        trajectory = trajectories[0]
        
        # Should have multiple turns due to error recovery
        assert len(trajectory.turns) >= 2
        
        # First turn should fail
        first_turn = trajectory.turns[0]
        assert first_turn.execution_success == False
        assert "ModuleNotFoundError" in first_turn.execution_output
        
        # Second turn should address the error (if it exists)
        if len(trajectory.turns) > 1:
            second_turn = trajectory.turns[1]
            # The mock will succeed on subsequent calls
            # The code should attempt to handle the error
            assert "error" in second_turn.prompt.lower() or "fix" in second_turn.prompt.lower()
    
    @pytest.mark.asyncio
    async def test_batch_request_multiple_tasks(self, orchestrator, mock_completion_lb, mock_execution_engine):
        """Test batch request with multiple different tasks"""
        request = BatchTrajectoryRequest(
            requests=[
                TrajectoryRequest(
                    initial_prompts=["Load and analyze FITS file header information"],
                    max_turns=3,
                    completion_criteria="header analysis complete",
                    task_names=["header_analysis"]
                ),
                TrajectoryRequest(
                    initial_prompts=["Create a statistical summary of the image data"],
                    max_turns=3,
                    completion_criteria="statistics complete",
                    task_names=["data_statistics"]
                ),
                TrajectoryRequest(
                    initial_prompts=["Generate a simple histogram of pixel values"],
                    max_turns=3,
                    completion_criteria="histogram complete",
                    task_names=["histogram_viz"]
                )
            ]
        )
        
        # Execute the batch request
        trajectories = await orchestrator.generate_trajectories(request)
        
        # Verify results
        assert len(trajectories) == 3
        
        for i, trajectory in enumerate(trajectories):
            assert trajectory.trajectory_id is not None
            assert len(trajectory.turns) >= 1
            assert trajectory.total_steps >= 1
            
            # All should succeed with our mock setup
            assert trajectory.final_reward > 0
            
            # Each trajectory should have astronomy-related content
            first_turn = trajectory.turns[0]
            assert first_turn.execution_success == True
        
        # Verify that multiple completion requests were made
        assert mock_completion_lb.request_count >= 3
        assert mock_execution_engine.execution_count >= 3
    
    @pytest.mark.asyncio 
    async def test_max_turns_termination(self, orchestrator, mock_completion_lb, mock_execution_engine):
        """Test that orchestrator respects max_turns limit"""
        # Configure execution to always succeed but never meet completion criteria
        request = BatchTrajectoryRequest(
            requests=[
                TrajectoryRequest(
                    initial_prompts=["Create complex multi-step analysis"],
                    max_turns=2,  # Very low limit
                    completion_criteria="impossible_criteria_never_met"
                )
            ]
        )
        
        trajectories = await orchestrator.generate_trajectories(request)
        
        assert len(trajectories) == 1
        trajectory = trajectories[0]
        
        # Should hit max turns limit
        assert len(trajectory.turns) == 2
        assert trajectory.total_steps == 2
        assert trajectory.termination_reason == "max_steps"
    
    @pytest.mark.asyncio
    async def test_completion_criteria_detection(self, orchestrator, mock_completion_lb, mock_execution_engine):
        """Test that completion criteria are properly detected"""
        
        # Override the mock execution engine to return the specific completion message
        def custom_execute_code(actor_id, code, success_criterion=None):
            return {
                "state": "success",
                "execution_output": "<output>\nDimensions: (512, 512)\nData type: float32\nNumber of header keys: 45\nFITS file analysis complete!\n</output>",
                "execution_time": 1.0,
                "turn": 1,
                "success_criterion_met": False,  # Random default
                "success_message": "Success criterion met: False (random default)",
                "actor_id": actor_id,
                "variables": ["data", "result"]
            }
        
        mock_execution_engine.execute_code = custom_execute_code
        
        request = BatchTrajectoryRequest(
            requests=[
                TrajectoryRequest(
                    initial_prompts=["Analyze FITS file and print completion message"],
                    max_turns=5,
                    completion_criteria="FITS file analysis complete"  # This appears in our mock output
                )
            ]
        )
        
        trajectories = await orchestrator.generate_trajectories(request)
        
        assert len(trajectories) == 1
        trajectory = trajectories[0]
        
        # Should complete successfully when criteria are met
        assert trajectory.termination_reason == "completion_criteria_met"
        assert trajectory.final_reward > 0
        
        # Should find the completion criteria in the output
        final_turn = trajectory.turns[-1]
        assert "FITS file analysis complete" in final_turn.execution_output
    
    @pytest.mark.asyncio
    async def test_file_output_generation(self, orchestrator, temp_dir):
        """Test that orchestrator generates proper output files"""
        request = BatchTrajectoryRequest(
            requests=[
                TrajectoryRequest(
                    initial_prompts=["Simple task for file output testing"],
                    max_turns=2,
                    completion_criteria="task complete"
                )
            ]
        )
        
        trajectories = await orchestrator.generate_trajectories(request)
        
        # The orchestrator creates a run-specific subdirectory, so we need to find it
        trajectory_base_dir = temp_dir / "trajectories"
        if trajectory_base_dir.exists():
            # Look for run subdirectories
            run_dirs = [d for d in trajectory_base_dir.iterdir() if d.is_dir()]
            if run_dirs:
                # Check the most recent run directory
                run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
                trajectory_files = list(run_dir.glob("*.json"))
                assert len(trajectory_files) >= 1
                
                # Verify file contents
                with open(trajectory_files[0]) as f:
                    trajectory_data = json.load(f)
                    assert "trajectory_id" in trajectory_data
                    assert "turns" in trajectory_data
                    assert "final_reward" in trajectory_data
        
        # Verify prompts JSONL file is created
        prompts_file = temp_dir / "prompts.jsonl"
        assert prompts_file.exists()
    
    @pytest.mark.asyncio
    async def test_orchestrator_stats(self, orchestrator, mock_completion_lb, mock_execution_engine):
        """Test that orchestrator maintains proper statistics"""
        # Run a few tasks to generate stats
        request = BatchTrajectoryRequest(
            requests=[
                TrajectoryRequest(
                    initial_prompts=["Task 1"],
                    max_turns=2
                ),
                TrajectoryRequest(
                    initial_prompts=["Task 2"], 
                    max_turns=2
                )
            ]
        )
        
        await orchestrator.generate_trajectories(request)
        
        # Get stats
        stats = orchestrator.get_stats()
        
        assert "active_trajectories" in stats
        assert "active_instances" in stats
        assert "ray_execution_engine" in stats
        assert "ray_num_cpus" in stats
        
        # Note: stats reflect the current state, not historical counts
        assert stats["ray_execution_engine"] == "embedded"
        assert stats["ray_num_cpus"] == 1


    @pytest.mark.asyncio
    async def test_unparseable_code_generation(self, orchestrator, mock_completion_lb, mock_execution_engine):
        """Test orchestrator behavior when LLM generates unparseable code"""
        
        # Configure mock to return invalid/unparseable code
        class InvalidCodeMockLB:
            def __init__(self):
                self.request_count = 0
                
            async def chat_completions(self, request):
                self.request_count += 1
                return {
                    "id": f"mock-completion-{self.request_count}",
                    "object": "chat.completion", 
                    "created": 1234567890,
                    "model": "mock-model",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Here's some invalid code without proper markdown:\n\nprint('hello world'\nthis is not valid python syntax!!!\n\n```\nprint('unclosed string\nif True\n    no colon and bad indentation\nfor x in 1,2,3\n    print(x\n```"
                        },
                        "finish_reason": "stop"
                    }]
                }
        
        # Replace the completion load balancer
        orchestrator.completion_load_balancer = InvalidCodeMockLB()
        
        request = BatchTrajectoryRequest(
            requests=[
                TrajectoryRequest(
                    initial_prompts=["Generate some code"],
                    max_turns=3,
                    completion_criteria="very specific completion criteria that will never be found in fallback code"
                )
            ]
        )
        
        trajectories = await orchestrator.generate_trajectories(request)
        
        assert len(trajectories) == 1
        trajectory = trajectories[0]
        
        # Should still create turns, even with invalid code
        assert len(trajectory.turns) >= 1
        
        # Should hit max turns since completion criteria won't be met
        assert trajectory.termination_reason == "max_steps"
        assert trajectory.total_steps == 3
        
        # The generated code should be the fallback when extraction fails
        first_turn = trajectory.turns[0]
        assert "# Unable to extract valid Python code from response" in first_turn.code
        assert first_turn.execution_success == True  # The fallback code should execute successfully
    
    @pytest.mark.asyncio
    async def test_max_turns_termination(self, orchestrator, mock_completion_lb, mock_execution_engine):
        """Test that orchestrator correctly terminates when max turns is reached"""
        
        # Configure mock to never meet completion criteria
        class NeverCompleteMockLB:
            def __init__(self):
                self.request_count = 0
                
            async def chat_completions(self, request):
                self.request_count += 1
                return {
                    "id": f"mock-completion-{self.request_count}",
                    "object": "chat.completion",
                    "created": 1234567890, 
                    "model": "mock-model",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f'''```python
print("Attempt {self.request_count}")
print("Still working on the task...")
print("Not done yet!")
```'''
                        },
                        "finish_reason": "stop"
                    }]
                }
        
        orchestrator.completion_load_balancer = NeverCompleteMockLB()
        
        # Also need to update the execution engine to return the correct output
        original_execute_code = mock_execution_engine.execute_code
        
        def custom_execute_code(actor_id, code, success_criterion=None):
            # Execute the actual code that was generated
            if "print(" in code:
                # Extract print statements and simulate their output
                lines = code.split('\n')
                output_lines = []
                for line in lines:
                    if 'print(' in line:
                        # Extract the printed content
                        start = line.find('"') + 1
                        end = line.rfind('"')
                        if start > 0 and end > start:
                            output_lines.append(line[start:end])
                
                output = '\n'.join(output_lines)
            else:
                output = "Code executed successfully"
                
            return {
                "state": "success",
                "execution_output": output,
                "execution_time": 1.5,
                "turn": mock_execution_engine.actor_pool[actor_id]["turn"] + 1,
                "success": False,
                "success_message": "",
                "actor_id": actor_id,
                "variables": []
            }
        
        mock_execution_engine.execute_code = custom_execute_code
        
        # Set a very low max_turns to test termination
        request = BatchTrajectoryRequest(
            requests=[
                TrajectoryRequest(
                    initial_prompts=["Complete an impossible task"],
                    max_turns=2,  # Very low limit
                    completion_criteria="impossible completion criteria that will never be met"
                )
            ]
        )
        
        trajectories = await orchestrator.generate_trajectories(request)
        
        assert len(trajectories) == 1
        trajectory = trajectories[0]
        
        # Should hit exactly max_turns
        assert len(trajectory.turns) == 2
        assert trajectory.total_steps == 2
        assert trajectory.termination_reason == "max_steps"
        assert trajectory.final_reward == 0.0  # No reward for incomplete task
        
        # Verify all turns executed successfully
        for turn in trajectory.turns:
            assert turn.execution_success == True
            assert f"Attempt {turn.step}" in turn.execution_output
    
    @pytest.mark.asyncio
    async def test_concurrent_trajectory_management(self, orchestrator, mock_completion_lb, mock_execution_engine):
        """Test orchestrator managing multiple concurrent trajectories"""
        
        # Track which trajectories are being processed
        trajectory_ids = set()
        original_execute_code = mock_execution_engine.execute_code
        
        def tracking_execute_code(actor_id, code, success_criterion=None):
            """Wrapper to track which trajectories are being processed"""
            # Extract trajectory info from the code or actor_id
            trajectory_ids.add(actor_id)
            return original_execute_code(actor_id, code, success_criterion)
        
        mock_execution_engine.execute_code = tracking_execute_code
        
        # Create a large batch request with different task types
        request = BatchTrajectoryRequest(
            requests=[
                TrajectoryRequest(
                    initial_prompts=["Analyze FITS header data"],
                    max_turns=2,
                    completion_criteria="Header analysis complete",
                    task_names=["header_task_1"]
                ),
                TrajectoryRequest(
                    initial_prompts=["Create statistical summary"],
                    max_turns=2, 
                    completion_criteria="Statistics complete",
                    task_names=["stats_task_1"]
                ),
                TrajectoryRequest(
                    initial_prompts=["Generate histogram visualization"],
                    max_turns=2,
                    completion_criteria="Histogram complete", 
                    task_names=["hist_task_1"]
                ),
                TrajectoryRequest(
                    initial_prompts=["Process FITS header information"],
                    max_turns=2,
                    completion_criteria="Header analysis complete",
                    task_names=["header_task_2"]
                ),
                TrajectoryRequest(
                    initial_prompts=["Compute data statistics"],
                    max_turns=2,
                    completion_criteria="Statistics complete",
                    task_names=["stats_task_2"]
                )
            ]
        )
        
        trajectories = await orchestrator.generate_trajectories(request)
        
        # Verify all trajectories were processed
        assert len(trajectories) == 5
        
        # Verify each trajectory has unique ID and proper completion
        trajectory_ids_result = set()
        task_types = []
        
        for trajectory in trajectories:
            assert trajectory.trajectory_id is not None
            assert trajectory.trajectory_id not in trajectory_ids_result
            trajectory_ids_result.add(trajectory.trajectory_id)
            
            # Should complete successfully with appropriate output
            assert trajectory.termination_reason == "completion_criteria_met"
            assert trajectory.final_reward > 0
            assert len(trajectory.turns) >= 1
            
            # Determine task type from output
            output = trajectory.turns[0].execution_output
            if "Header analysis complete" in output:
                task_types.append("header")
            elif "Statistics complete" in output:
                task_types.append("statistics") 
            elif "Histogram complete" in output:
                task_types.append("histogram")
        
        # Should have processed multiple different task types
        assert len(set(task_types)) >= 2
        
        # Should have used multiple execution instances (one per trajectory)
        assert len(trajectory_ids) == 5
        
        # Verify completion load balancer handled multiple requests
        assert mock_completion_lb.request_count >= 5
    
    @pytest.mark.asyncio
    async def test_execution_failures_and_recovery(self, orchestrator, mock_completion_lb, mock_execution_engine):
        """Test orchestrator behavior when code execution fails"""
        
        # Configure mock to fail on first execution, succeed on subsequent
        original_execute_code = mock_execution_engine.execute_code
        call_count = 0
        
        def failing_execute_code(actor_id, code, success_criterion=None):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First call fails
                return {
                    "state": "crashed",
                    "execution_output": "ImportError: No module named 'astropy'",
                    "execution_time": 0.1,
                    "turn": 1,
                    "success": False,
                    "success_message": "",
                    "actor_id": actor_id,
                    "variables": []
                }
            else:
                # Subsequent calls succeed
                return original_execute_code(actor_id, code, success_criterion)
        
        mock_execution_engine.execute_code = failing_execute_code
        
        request = BatchTrajectoryRequest(
            requests=[
                TrajectoryRequest(
                    initial_prompts=["Load FITS file with error recovery"],
                    max_turns=3,
                    completion_criteria="FITS file analysis complete"
                )
            ]
        )
        
        trajectories = await orchestrator.generate_trajectories(request)
        
        assert len(trajectories) == 1
        trajectory = trajectories[0]
        
        # Should have multiple turns due to initial failure
        assert len(trajectory.turns) >= 2
        
        # First turn should show execution failure
        first_turn = trajectory.turns[0]
        assert first_turn.execution_success == False
        assert "ImportError" in first_turn.execution_output
        
        # Subsequent turns should succeed (if they exist)
        if len(trajectory.turns) > 1:
            for turn in trajectory.turns[1:]:
                assert turn.execution_success == True
    
    @pytest.mark.asyncio
    async def test_empty_output_edge_cases(self, orchestrator, mock_completion_lb, mock_execution_engine):
        """Test orchestrator behavior with empty or minimal outputs"""
        
        # Configure mock to return minimal responses
        class MinimalMockLB:
            def __init__(self):
                self.request_count = 0
                
            async def chat_completions(self, request):
                self.request_count += 1
                return {
                    "id": f"mock-completion-{self.request_count}",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "mock-model", 
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "```python\npass\n```"  # Minimal valid code
                        },
                        "finish_reason": "stop"
                    }]
                }
        
        # Configure execution engine to return empty output
        original_execute_code = mock_execution_engine.execute_code
        
        def empty_output_execute_code(actor_id, code, success_criterion=None):
            result = original_execute_code(actor_id, code, success_criterion)
            result["execution_output"] = ""  # Empty output
            return result
        
        mock_execution_engine.execute_code = empty_output_execute_code
        orchestrator.completion_load_balancer = MinimalMockLB()
        
        request = BatchTrajectoryRequest(
            requests=[
                TrajectoryRequest(
                    initial_prompts=["Do minimal task"],
                    max_turns=2,
                    completion_criteria="task complete"
                )
            ]
        )
        
        trajectories = await orchestrator.generate_trajectories(request)
        
        assert len(trajectories) == 1
        trajectory = trajectories[0]
        
        # Should handle empty outputs gracefully
        assert len(trajectory.turns) >= 1
        assert trajectory.trajectory_id is not None
        
        # Should hit max_turns since completion criteria won't be met with empty output
        assert trajectory.termination_reason == "max_steps"
        
        # Turns should have empty execution output
        for turn in trajectory.turns:
            assert turn.execution_output == ""
            assert turn.code == "pass"  # Should extract the minimal code
            assert turn.execution_success == True


    @pytest.mark.asyncio
    async def test_format_trajectory_for_chat(self, orchestrator):
        """Test the format_trajectory_for_chat function"""
        from datetime import datetime
        
        # Create sample turns with different execution outputs
        turns = [
            Turn(
                step=1,
                prompt="Initial prompt",
                code="print('Hello World')",
                execution_output="<output>\nHello World\n</output>",
                execution_success=True,
                timestamp=datetime.now()
            ),
            Turn(
                step=2,
                prompt="Next step",
                code="print('Step 2')\nraise ValueError('Test error')",
                execution_output="<output>\nStep 2\n</output>\n<error>\nValueError: Test error\n</error>",
                execution_success=False,
                timestamp=datetime.now()
            )
        ]
        
        # Set success_criterion_met on turns
        turns[0].success_criterion_met = False
        turns[1].success_criterion_met = True
        
        goal = "Create a simple Python program that demonstrates error handling"
        formatted_content = orchestrator.format_trajectory_for_chat(goal, turns)
        
        # Verify structure
        assert "<goal>" in formatted_content
        assert goal in formatted_content
        assert "</goal>" in formatted_content
        
        # Verify first turn
        assert "<turn>" in formatted_content
        assert "print('Hello World')" in formatted_content
        assert "</turn>" in formatted_content
        assert "<output>\nHello World\n</output>" in formatted_content
        
        # Verify second turn with error
        assert "print('Step 2')" in formatted_content
        assert "<error>\nValueError: Test error\n</error>" in formatted_content
        
        # Verify instruction
        assert "Based on the goal and the output/errors from previous turns" in formatted_content
        assert "generate the next turn of code" in formatted_content
        assert "Generate python code to solve the task" in formatted_content
    
    @pytest.mark.asyncio
    async def test_format_trajectory_for_chat_empty_turns(self, orchestrator):
        """Test format_trajectory_for_chat with no turns"""
        goal = "Test empty trajectory"
        formatted_content = orchestrator.format_trajectory_for_chat(goal, [])
        
        assert "<goal>" in formatted_content
        assert goal in formatted_content
        assert "</goal>" in formatted_content
        assert "Based on the goal and the output/errors from previous turns" in formatted_content
        assert "Generate python code to solve the task" in formatted_content
    
    @pytest.mark.asyncio
    async def test_trajectory_saving_with_success_criteria_met(self, orchestrator, temp_dir):
        """Test that trajectory saving includes success_criteria_met field"""
        
        # Update the mock execution engine to return success_criterion_met
        def mock_execute_with_criteria(actor_id, code, success_criterion=None):
            return {
                "state": "success",
                "execution_output": "<output>\nTask completed successfully\n</output>",
                "execution_time": 1.0,
                "turn": 1,
                "success_criterion_met": True,  # This should be saved
                "success_message": "Success criterion met: True",
                "actor_id": actor_id,
                "variables": ["result"]
            }
        
        orchestrator.execution_engine.execute_code = mock_execute_with_criteria
        
        request = BatchTrajectoryRequest(
            requests=[
                TrajectoryRequest(
                    initial_prompts=["Test task for success criteria"],
                    max_turns=1,
                    completion_criteria="task completed"
                )
            ]
        )
        
        trajectories = await orchestrator.generate_trajectories(request)
        
        assert len(trajectories) == 1
        trajectory = trajectories[0]
        
        # Verify the turn has success_criterion_met
        assert len(trajectory.turns) == 1
        turn = trajectory.turns[0]
        assert hasattr(turn, 'success_criterion_met')
        assert turn.success_criterion_met == True
        assert turn.execution_success == True  # Based on "success" state
        
        # Verify trajectory file is saved with success_criteria_met field
        trajectory_base_dir = temp_dir / "trajectories"
        if trajectory_base_dir.exists():
            run_dirs = [d for d in trajectory_base_dir.iterdir() if d.is_dir()]
            if run_dirs:
                run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
                trajectory_files = list(run_dir.glob("*.json"))
                assert len(trajectory_files) >= 1
                
                # Load and verify JSON structure
                with open(trajectory_files[0]) as f:
                    trajectory_data = json.load(f)
                    assert "turns" in trajectory_data
                    assert len(trajectory_data["turns"]) == 1
                    
                    turn_data = trajectory_data["turns"][0]
                    assert "execution_success" in turn_data
                    assert "success_criteria_met" in turn_data
                    assert turn_data["execution_success"] == True
                    assert turn_data["success_criteria_met"] == True
                    assert "execution_output" in turn_data
                    assert "<output>" in turn_data["execution_output"]
    
    @pytest.mark.asyncio
    async def test_execution_output_xml_formatting(self, orchestrator):
        """Test that execution output is properly formatted with XML tags"""
        
        # Store the original execution engine method
        original_execute_code = orchestrator.execution_engine.execute_code
        
        # Test 1: Successful execution with stdout only
        def mock_execute_success(actor_id, code, success_criterion=None):
            return {
                "state": "success",
                "execution_output": "<output>\nHello from Python!\nCalculation result: 42\n</output>",
                "execution_time": 1.0,
                "turn": 1,
                "success_criterion_met": True,
                "success_message": "Success criterion met: True (random default)",
                "actor_id": actor_id,
                "variables": ["result"]
            }
        
        orchestrator.execution_engine.execute_code = mock_execute_success
        
        request_success = BatchTrajectoryRequest(
            requests=[
                TrajectoryRequest(
                    initial_prompts=["Create a simple print statement"],
                    max_turns=1,
                    completion_criteria="calculation result"
                )
            ]
        )
        
        trajectories = await orchestrator.generate_trajectories(request_success)
        trajectory = trajectories[0]
        turn = trajectory.turns[0]
        
        assert "<output>" in turn.execution_output
        assert "</output>" in turn.execution_output
        assert "Hello from Python!" in turn.execution_output
        assert "Calculation result: 42" in turn.execution_output
        assert turn.execution_success == True
        
        # Test 2: Execution with both stdout and stderr
        def mock_execute_error(actor_id, code, success_criterion=None):
            return {
                "state": "crashed",
                "execution_output": "<output>\nSome output before error\n</output>\n<error>\nTraceback (most recent call last):\n  File \"<string>\", line 2, in <module>\nNameError: name 'undefined_var' is not defined\n</error>",
                "execution_time": 0.5,
                "turn": 1,
                "success_criterion_met": False,
                "success_message": "Success criterion met: False (random default)",
                "actor_id": actor_id,
                "variables": []
            }
        
        orchestrator.execution_engine.execute_code = mock_execute_error
        
        request_error = BatchTrajectoryRequest(
            requests=[
                TrajectoryRequest(
                    initial_prompts=["Create code with an error"],
                    max_turns=1,
                    completion_criteria="error handled"
                )
            ]
        )
        
        trajectories_error = await orchestrator.generate_trajectories(request_error)
        trajectory_error = trajectories_error[0]
        turn_error = trajectory_error.turns[0]
        
        assert "<output>" in turn_error.execution_output
        assert "</output>" in turn_error.execution_output
        assert "<error>" in turn_error.execution_output
        assert "</error>" in turn_error.execution_output
        assert "Some output before error" in turn_error.execution_output
        assert "NameError" in turn_error.execution_output
        assert turn_error.execution_success == False  # Based on "crashed" state
        
        # Restore original method
        orchestrator.execution_engine.execute_code = original_execute_code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])