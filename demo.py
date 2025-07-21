#!/usr/bin/env python3
"""
Demo script for Ray Execution Engine
Shows how to use the ray_execution_engine for code execution
"""

import time
import json
from ray_execution_engine import start_instance, execute_code, cleanup_instance, cleanup_all_instances

def main():
    print("=== Ray Execution Engine Demo ===")
    print()
    
    # Start a new Ray execution instance
    print("1. Starting Ray execution instance...")
    instance_id = start_instance(
        timeout_in_secs=30.0,
        num_cpus=1,
        num_gpus=0
    )
    
    if not instance_id:
        print("❌ Failed to start instance")
        return
    
    print(f"✅ Started instance: {instance_id}")
    print()
    
    # Test cases
    test_cases = [
        {
            "name": "Basic arithmetic",
            "code": "print(2 + 3)"
        },
        {
            "name": "Variable assignment and use",
            "code": "x = 42\nprint(f'The answer is {x}')"
        },
        {
            "name": "Import and use library",
            "code": "import math\nprint(f'Pi is approximately {math.pi:.2f}')"
        },
        {
            "name": "Check persistent variables",
            "code": "print(f'x from previous execution: {x}')"
        },
        {
            "name": "Error handling test",
            "code": "print(undefined_variable)"
        },
        {
            "name": "Recovery after error", 
            "code": "print('Execution continues after error')"
        },
        {
            "name": "List operations",
            "code": "numbers = [1, 2, 3, 4, 5]\nprint(f'Sum: {sum(numbers)}')\nprint(f'Max: {max(numbers)}')"
        }
    ]
    
    # Execute test cases
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case['name']}")
        print(f"   Code: {test_case['code']}")
        
        try:
            result = execute_code(instance_id, test_case['code'])
            
            print(f"   State: {result['state']}")
            print(f"   Output: {result['execution_output']}")
            
            if result['state'] == 'completed':
                print("   ✅ Success")
            else:
                print("   ⚠️  Non-successful state")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        print()
        time.sleep(0.5)  # Small delay for readability
    
    # Cleanup
    print("Cleaning up...")
    success = cleanup_instance(instance_id)
    if success:
        print("✅ Instance cleaned up successfully")
    else:
        print("⚠️  Instance cleanup had issues")
    
    print()
    print("=== Demo Complete ===")

def test_multiple_instances():
    """Test multiple concurrent instances"""
    print("=== Testing Multiple Instances ===")
    print()
    
    # Start multiple instances
    instances = []
    for i in range(3):
        print(f"Starting instance {i+1}...")
        instance_id = start_instance(timeout_in_secs=30.0, num_cpus=1, num_gpus=0)
        if instance_id:
            instances.append(instance_id)
            print(f"✅ Started: {instance_id}")
        else:
            print("❌ Failed to start instance")
    
    print(f"\nStarted {len(instances)} instances")
    print()
    
    # Execute different code in each instance
    for i, instance_id in enumerate(instances):
        code = f"instance_number = {i+1}\nprint(f'This is instance {{instance_number}}')"
        print(f"Executing in instance {i+1}: {instance_id}")
        result = execute_code(instance_id, code)
        print(f"Result: {result['execution_output']}")
        print()
    
    # Verify isolation - each instance should have its own variables
    for i, instance_id in enumerate(instances):
        code = "print(f'Instance {instance_number} still remembers its number')"
        result = execute_code(instance_id, code)
        print(f"Instance {i+1} memory test: {result['execution_output']}")
    
    # Cleanup all instances
    print("\nCleaning up all instances...")
    cleanup_all_instances()
    print("✅ All instances cleaned up")
    print()

if __name__ == "__main__":
    # Run basic demo
    main()
    
    print()
    
    # Run multiple instances demo
    test_multiple_instances() 