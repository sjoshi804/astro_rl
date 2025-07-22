#!/usr/bin/env python3
"""
Manual Test Script to Debug VLLM Issues
Quick tests for specific failure scenarios
"""

import asyncio
import httpx
import json
import sys

async def test_vllm_directly():
    """Test VLLM wrapper with the exact prompt structure that's failing"""
    
    print("🔍 Testing VLLM wrapper directly...")
    
    # Test 1: Empty prompt (what might cause empty outputs)
    print("\n1. Testing empty prompt:")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post("http://localhost:8200/generate", json={
                "prompt": "",
                "max_tokens": 10,
                "temperature": 0.7,
                "n": 1
            })
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   Result: {result}")
            else:
                print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Exception: {e}")
    
    # Test 2: Very long prompt (what might cause processing issues)
    print("\n2. Testing very long prompt:")
    long_prompt = "Write Python code:\n```python\n# " + "Very long instruction " * 100
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post("http://localhost:8200/generate", json={
                "prompt": long_prompt,
                "max_tokens": 50,
                "temperature": 0.7,
                "n": 1
            })
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                choices = result.get("choices", [])
                print(f"   Choices count: {len(choices)}")
                if choices:
                    print(f"   First choice: {repr(choices[0].get('text', '')[:100])}")
            else:
                print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Exception: {e}")
    
    # Test 3: Our improved prompt structure
    print("\n3. Testing improved prompt structure:")
    improved_prompt = """Write a Python function that prints 'Hello, World!'

Respond with Python code only. No explanations or comments outside the code.

```python
# Python code to solve the task:"""
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post("http://localhost:8200/generate", json={
                "prompt": improved_prompt,
                "max_tokens": 100,
                "temperature": 0.1,  # Low temperature for consistent output
                "n": 1,
                "stop": ["```"]  # Stop at code block end
            })
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                choices = result.get("choices", [])
                print(f"   Choices count: {len(choices)}")
                if choices:
                    generated = choices[0].get("text", "")
                    print(f"   Generated: {repr(generated)}")
                    
                    # Test our code extraction on this
                    sys.path.append('.')
                    from code_and_exec_service import extract_python_code, is_valid_python_code
                    extracted = extract_python_code(generated)
                    is_valid = is_valid_python_code(extracted)
                    print(f"   Extracted: {repr(extracted)}")
                    print(f"   Valid Python: {is_valid}")
            else:
                print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Exception: {e}")

async def test_completion_server_directly():
    """Test completion server with same requests that are failing"""
    
    print("\n🔍 Testing Completion Server directly...")
    
    # Test with the exact request structure from code_and_exec_service
    test_request = {
        "trajectory": {"turns": []},
        "instruction": "Task: Write Python code that generates continuous output to STDOUT during loop execution. Create a loop from 1 to 10, printing each number and running total (e.g., 'Number: 1, Running total: 1'). After the loop, print a summary with the final sum. End with 'LOOP TEST FINISHED'. Focus on generating lots of stdout output.\n\nGenerate Python code to accomplish this task. Make sure to:\n1. Include appropriate imports\n2. Add proper error handling\n3. Include informative print statements\n4. Create clear outputs when applicable\n5. Test your implementation",
        "n": 4,
        "temperature": 0.8,
        "max_tokens": 512,
        "stop": ["```", "\nUser:", "\nExecution:"]
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post("http://localhost:8000/generate", json=test_request)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                completions = result.get("completions", [])
                print(f"Completions count: {len(completions)}")
                
                for i, completion in enumerate(completions):
                    print(f"\nCompletion {i+1}:")
                    print(f"  Text: {repr(completion[:200])}")
                    
                    # Test our extraction logic
                    sys.path.append('.')
                    from code_and_exec_service import extract_python_code, is_valid_python_code
                    extracted = extract_python_code(completion)
                    is_valid = is_valid_python_code(extracted)
                    print(f"  Extracted: {repr(extracted[:100])}")
                    print(f"  Valid: {is_valid}")
            else:
                print(f"Error: {response.text}")
                
    except Exception as e:
        print(f"Exception: {e}")

async def test_ray_execution():
    """Test Ray execution with problematic code"""
    
    print("\n🔍 Testing Ray Execution...")
    
    try:
        sys.path.append('.')
        from ray_execution_engine import start_instance, execute_code, cleanup_instance
        
        # Create instance
        instance_id = start_instance(timeout_in_secs=10.0)
        if not instance_id:
            print("Failed to create Ray instance")
            return
        
        print(f"Created instance: {instance_id}")
        
        # Test cases that have been failing
        test_cases = [
            "When this code is run, it should produce the following output:",
            "This Python code accomplishes the task by:",
            "# Unable to extract valid Python code from response\npass",
            "print('Hello, World!')",
            ""
        ]
        
        for i, code in enumerate(test_cases):
            print(f"\nTest {i+1}: {repr(code[:50])}")
            result = execute_code(instance_id, code)
            print(f"  State: {result.get('state')}")
            print(f"  Output: {repr(result.get('execution_output', '')[:100])}")
            print(f"  Success: {result.get('success')}")
        
        # Cleanup
        cleanup_success = cleanup_instance(instance_id)
        print(f"\nCleanup success: {cleanup_success}")
        
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

def test_prompt_building():
    """Test our improved prompt building"""
    
    print("\n🔍 Testing Prompt Building...")
    
    try:
        sys.path.append('.')
        from completion_server import CompletionServerManager, Trajectory
        
        manager = CompletionServerManager([])
        empty_trajectory = Trajectory(turns=[])
        
        instruction = "Write a Python function that prints 'Hello, World!'"
        prompt = manager._build_prompt_from_trajectory(empty_trajectory, instruction)
        
        print("Generated prompt:")
        print("-" * 40)
        print(prompt)
        print("-" * 40)
        
        # Check what we expect to see
        checks = [
            ("Has instruction", instruction in prompt),
            ("Has code directive", "Python code only" in prompt),
            ("Has code block", "```python" in prompt),
            ("Has helpful comment", "# Python code" in prompt)
        ]
        
        for check_name, check_result in checks:
            status = "✅" if check_result else "❌"
            print(f"{status} {check_name}")
            
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run manual tests"""
    print("🧪 Manual Debug Tests")
    print("=" * 50)
    
    # Test prompt building (offline)
    test_prompt_building()
    
    # Test Ray execution (requires Ray)
    await test_ray_execution()
    
    # Test VLLM wrapper (requires VLLM service)
    await test_vllm_directly()
    
    # Test completion server (requires completion service)
    await test_completion_server_directly()
    
    print("\n✅ Manual tests completed")

if __name__ == "__main__":
    asyncio.run(main()) 