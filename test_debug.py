#!/usr/bin/env python3
"""
Debug Unit Tests for VLLM Pipeline Issues
Tests each component independently to isolate failure points
"""

import asyncio
import json
import traceback
from typing import Dict, Any, List
import sys
import os

# Add current directory to path so we can import modules
sys.path.append('.')

# Test imports
try:
    from code_and_exec_service import extract_python_code, is_valid_python_code
    from completion_server import CompletionServerManager, Trajectory, Turn
    from ray_execution_engine import start_instance, execute_code, cleanup_instance
    import httpx
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to run this from the project root directory")
    sys.exit(1)

class TestResults:
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
    
    def add_test(self, name: str, passed: bool, details: str = ""):
        self.tests.append({
            "name": name,
            "passed": passed,
            "details": details
        })
        if passed:
            self.passed += 1
            print(f"✅ {name}")
        else:
            self.failed += 1
            print(f"❌ {name}: {details}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY: {self.passed}/{total} passed, {self.failed}/{total} failed")
        print(f"{'='*60}")

# Global test results
results = TestResults()

def test_code_extraction():
    """Test the Python code extraction logic"""
    
    # Test 1: Valid Python code in markdown blocks
    valid_markdown = """
Here's the solution:

```python
print("Hello, World!")
x = 5 + 3
print(f"Result: {x}")
```

This should work.
"""
    extracted = extract_python_code(valid_markdown)
    expected = 'print("Hello, World!")\nx = 5 + 3\nprint(f"Result: {x}")'
    results.add_test(
        "Code extraction from markdown", 
        extracted.strip() == expected.strip(),
        f"Expected: {repr(expected)}, Got: {repr(extracted)}"
    )
    
    # Test 2: Natural language that should be rejected
    natural_language = "This Python code accomplishes the task by:"
    extracted = extract_python_code(natural_language)
    results.add_test(
        "Natural language rejection",
        "pass" in extracted or "Unable to extract" in extracted,
        f"Should fallback to safe code, got: {repr(extracted)}"
    )
    
    # Test 3: Mixed content (natural language + code)
    mixed_content = """
When this code is run, it should produce the following output:

```python
for i in range(5):
    print(f"Number: {i}")
```

The code loops through numbers.
"""
    extracted = extract_python_code(mixed_content)
    results.add_test(
        "Mixed content extraction",
        "for i in range" in extracted and "When this code" not in extracted,
        f"Should extract only Python code, got: {repr(extracted)}"
    )

def test_python_validation():
    """Test the Python code validation"""
    
    # Test valid Python
    valid_codes = [
        "print('hello')",
        "x = 5\nprint(x)",
        "def foo():\n    return 42",
        "# Just a comment\npass"
    ]
    
    for i, code in enumerate(valid_codes):
        is_valid = is_valid_python_code(code)
        results.add_test(
            f"Valid Python #{i+1}",
            is_valid,
            f"Should be valid: {repr(code)}"
        )
    
    # Test invalid Python
    invalid_codes = [
        "When this code is run",
        "This Python code accomplishes",
        "def foo(\n    # Incomplete function",
        "x = 5 +",
        ""
    ]
    
    for i, code in enumerate(invalid_codes):
        is_valid = is_valid_python_code(code)
        results.add_test(
            f"Invalid Python #{i+1}",
            not is_valid,
            f"Should be invalid: {repr(code)}"
        )

def test_prompt_building():
    """Test the completion server prompt building"""
    
    # Test empty trajectory (first step)
    empty_trajectory = Trajectory(turns=[])
    instruction = "Write a Python function that prints hello world"
    
    manager = CompletionServerManager([])  # Empty hostnames for testing
    prompt = manager._build_prompt_from_trajectory(empty_trajectory, instruction)
    
    # Check if prompt structure is correct
    has_instruction = instruction in prompt
    has_code_directive = "Python code only" in prompt
    has_code_block_start = "```python" in prompt
    
    results.add_test(
        "Prompt building - empty trajectory",
        has_instruction and has_code_directive and has_code_block_start,
        f"Missing elements in prompt: {repr(prompt[:200])}"
    )
    
    # Test trajectory with history
    turn1 = Turn(
        step=1,
        prompt="Previous task",
        code="print('hello')",
        execution_output="hello",
        execution_success=True
    )
    trajectory_with_history = Trajectory(turns=[turn1])
    
    prompt_with_history = manager._build_prompt_from_trajectory(trajectory_with_history, instruction)
    
    has_history = "Previous task" in prompt_with_history
    has_previous_code = "print('hello')" in prompt_with_history
    
    results.add_test(
        "Prompt building - with history",
        has_history and has_previous_code,
        f"Missing history elements in prompt: {repr(prompt_with_history[:200])}"
    )

async def test_vllm_wrapper_direct():
    """Test the VLLM wrapper directly with known inputs"""
    
    # Test if VLLM wrapper is responding
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test health endpoint
            health_response = await client.get("http://localhost:8200/health")
            vllm_healthy = health_response.status_code == 200
            
            results.add_test(
                "VLLM wrapper health check",
                vllm_healthy,
                f"Health check failed: {health_response.status_code}"
            )
            
            if vllm_healthy:
                # Test simple generation
                test_request = {
                    "prompt": "Write Python code:\n```python\n# Print hello world:",
                    "max_tokens": 50,
                    "temperature": 0.1,  # Low temperature for consistent output
                    "n": 1
                }
                
                gen_response = await client.post(
                    "http://localhost:8200/generate",
                    json=test_request
                )
                
                if gen_response.status_code == 200:
                    result = gen_response.json()
                    choices = result.get("choices", [])
                    has_output = len(choices) > 0 and choices[0].get("text", "").strip()
                    
                    results.add_test(
                        "VLLM direct generation",
                        has_output,
                        f"No valid output from VLLM: {result}"
                    )
                    
                    if has_output:
                        generated_text = choices[0]["text"]
                        print(f"  Generated: {repr(generated_text[:100])}")
                else:
                    results.add_test(
                        "VLLM direct generation",
                        False,
                        f"Generation failed: {gen_response.status_code} - {gen_response.text}"
                    )
                
    except Exception as e:
        results.add_test(
            "VLLM wrapper connection",
            False,
            f"Cannot connect to VLLM wrapper: {e}"
        )

async def test_completion_server():
    """Test the completion server"""
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test completion server health
            health_response = await client.get("http://localhost:8000/health")
            server_healthy = health_response.status_code == 200
            
            results.add_test(
                "Completion server health",
                server_healthy,
                f"Health check failed: {health_response.status_code}"
            )
            
            if server_healthy:
                # Test generation request
                test_request = {
                    "trajectory": {"turns": []},
                    "instruction": "Write a simple Python print statement",
                    "n": 1,
                    "temperature": 0.1,
                    "max_tokens": 50
                }
                
                gen_response = await client.post(
                    "http://localhost:8000/generate",
                    json=test_request
                )
                
                if gen_response.status_code == 200:
                    result = gen_response.json()
                    completions = result.get("completions", [])
                    has_completions = len(completions) > 0
                    
                    results.add_test(
                        "Completion server generation",
                        has_completions,
                        f"No completions returned: {result}"
                    )
                    
                    if has_completions:
                        completion_text = completions[0]
                        print(f"  Completion: {repr(completion_text[:100])}")
                else:
                    results.add_test(
                        "Completion server generation",
                        False,
                        f"Generation failed: {gen_response.status_code} - {gen_response.text}"
                    )
                
    except Exception as e:
        results.add_test(
            "Completion server connection",
            False,
            f"Cannot connect to completion server: {e}"
        )

def test_ray_execution():
    """Test Ray execution engine"""
    
    try:
        # Create test instance
        instance_id = start_instance(timeout_in_secs=10.0)
        
        if instance_id:
            results.add_test(
                "Ray instance creation",
                True,
                f"Created instance: {instance_id}"
            )
            
            # Test valid Python code
            valid_code = "x = 5 + 3\nprint(f'Result: {x}')"
            result = execute_code(instance_id, valid_code)
            
            valid_execution = result.get("state") == "success"
            results.add_test(
                "Ray valid code execution",
                valid_execution,
                f"Execution result: {result}"
            )
            
            # Test invalid Python code (natural language)
            invalid_code = "When this code is run, it should produce the following output:"
            result = execute_code(instance_id, invalid_code)
            
            invalid_execution = result.get("state") == "crashed" and "SyntaxError" in result.get("execution_output", "")
            results.add_test(
                "Ray invalid code handling",
                invalid_execution,
                f"Should detect syntax error: {result}"
            )
            
            # Cleanup
            cleanup_success = cleanup_instance(instance_id)
            results.add_test(
                "Ray instance cleanup",
                cleanup_success,
                f"Cleanup result: {cleanup_success}"
            )
        else:
            results.add_test(
                "Ray instance creation",
                False,
                "Failed to create Ray instance"
            )
            
    except Exception as e:
        results.add_test(
            "Ray execution test",
            False,
            f"Ray test failed: {e}\n{traceback.format_exc()}"
        )

async def test_full_pipeline():
    """Test the full pipeline end-to-end"""
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test the code execution service
            pipeline_request = {
                "requests": [
                    {
                        "initial_prompts": ["Write a Python function that prints 'Hello, World!'"],
                        "max_turns": 1,
                        "completion_criteria": None
                    }
                ]
            }
            
            response = await client.post(
                "http://localhost:8002/generate_trajectories",
                json=pipeline_request
            )
            
            if response.status_code == 200:
                trajectories = response.json()
                has_trajectory = len(trajectories) > 0
                
                results.add_test(
                    "Full pipeline execution",
                    has_trajectory,
                    f"No trajectories returned: {trajectories}"
                )
                
                if has_trajectory:
                    trajectory = trajectories[0]
                    turns = trajectory.get("turns", [])
                    
                    if turns:
                        first_turn = turns[0]
                        code = first_turn.get("code", "")
                        execution_output = first_turn.get("execution_output", "")
                        success = first_turn.get("execution_success", False)
                        
                        print(f"  Generated code: {repr(code[:100])}")
                        print(f"  Execution output: {repr(execution_output[:100])}")
                        print(f"  Success: {success}")
                        
                        # Check if we got actual Python code vs natural language
                        is_python = is_valid_python_code(code)
                        results.add_test(
                            "Pipeline generates valid Python",
                            is_python,
                            f"Generated invalid Python: {repr(code)}"
                        )
            else:
                results.add_test(
                    "Full pipeline execution",
                    False,
                    f"Pipeline request failed: {response.status_code} - {response.text}"
                )
                
    except Exception as e:
        results.add_test(
            "Full pipeline test",
            False,
            f"Pipeline test failed: {e}\n{traceback.format_exc()}"
        )

async def main():
    """Run all tests"""
    print("🧪 Starting Debug Unit Tests")
    print("="*60)
    
    # Test 1: Code extraction and validation (offline)
    print("\n🔍 Testing Code Extraction & Validation...")
    test_code_extraction()
    test_python_validation()
    
    # Test 2: Prompt building (offline)
    print("\n🔍 Testing Prompt Building...")
    test_prompt_building()
    
    # Test 3: Ray execution (requires Ray cluster)
    print("\n🔍 Testing Ray Execution...")
    test_ray_execution()
    
    # Test 4: VLLM wrapper (requires VLLM service)
    print("\n🔍 Testing VLLM Wrapper...")
    await test_vllm_wrapper_direct()
    
    # Test 5: Completion server (requires completion service)
    print("\n🔍 Testing Completion Server...")
    await test_completion_server()
    
    # Test 6: Full pipeline (requires all services)
    print("\n🔍 Testing Full Pipeline...")
    await test_full_pipeline()
    
    # Show results
    results.summary()

if __name__ == "__main__":
    asyncio.run(main()) 