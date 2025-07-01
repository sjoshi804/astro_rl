#!/usr/bin/env python3
"""
Test script to verify improved code extraction works with the execution system.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from execution_engine import ExecutionInstance

async def test_execution_with_mixed_content():
    """Test execution with mixed content that previously caused issues"""
    
    # Create a real execution instance
    instance = ExecutionInstance("test", 30, 10, 0.1)
    
    try:
        # Start the instance
        if not await instance.start():
            print("Failed to start execution instance")
            return
        
        print("Testing execution with problematic mixed content...")
        
        # This is the exact content from the trajectory that was causing issues
        problematic_text = '''

# Apply sigma clipping to the data
mu, med, sig = sigma_clipped_stats(data, sigma=3.0, maxiters=5)

# Print the calculated statistics
print(f"Mean: {mu}")
print(f"Median: {med}")
print(f"Standard Deviation: {sig}")

# Check if sigma clipping changed the data
data_clipped = sigma_clip(data, sigma=3.0, maxiters=5, copy=True)
if np.array_equal(data, data_clipped):
    print("Sigma clipping did not change the data.")
else:
    # Display the sigma-clipped image
    plt.figure(figsize=(8, 8))
    plt.imshow(data_clipped, norm=norm, origin='lower')
    plt.colorbar(label='Intensity')
    plt.title('Astro1 UV Imaging Telescope Image (Sigma Clipped)')
    plt.show()
```
This code snippet calculates the mean, median, and standard deviation of the image data after applying sigma clipping. It then checks if sigma clipping changed the data. If it did, it displays the sigma-clipped image.
'''
        
        # Execute the problematic text
        response = await instance.execute_code(problematic_text)
        
        print(f"Execution completed!")
        print(f"State: {response.state}")
        print(f"Step count: {response.step_count}")
        print(f"Output: {response.output}")
        
        if "ERROR: No parseable code found" in response.output:
            print("❌ Code extraction failed")
        elif "NameError" in response.output or "ImportError" in response.output:
            print("⚠️  Code extracted but missing dependencies (expected)")
        elif "IndentationError" in response.output or "SyntaxError" in response.output:
            print("❌ Syntax errors in extracted code")
        else:
            print("✅ Code extraction successful!")
            
        # Test with simpler case that should work
        print("\nTesting with simple working code...")
        simple_text = '''
```python
# Simple test
import math
result = math.sqrt(16)
print(f"Square root of 16 is: {result}")
```
This will calculate the square root.
'''
        
        response2 = await instance.execute_code(simple_text)
        print(f"Simple test - State: {response2.state}, Output: {response2.output}")
        
    finally:
        await instance.cleanup()

if __name__ == "__main__":
    asyncio.run(test_execution_with_mixed_content()) 