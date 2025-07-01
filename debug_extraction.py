#!/usr/bin/env python3
"""
Debug script to see exactly what code is being extracted.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from execution_engine import ExecutionInstance

def debug_extraction():
    """Debug the code extraction to see exactly what's being extracted"""
    
    instance = ExecutionInstance("debug", 10, 20, 0.1)
    
    # Test the same problematic text
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
    
    print("Original text:")
    print("=" * 80)
    print(repr(problematic_text))
    print("=" * 80)
    print()
    
    extracted = instance._extract_code(problematic_text)
    
    print("Extracted code:")
    print("=" * 80)
    print(repr(extracted))
    print("=" * 80)
    print()
    
    print("Extracted code (readable):")
    print("=" * 80)
    print(extracted)
    print("=" * 80)
    print()
    
    # Test syntax checking
    try:
        compile(extracted, '<string>', 'exec')
        print("✅ Code compiles successfully!")
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        print(f"Error on line {e.lineno}: {e.text}")
    
    # Test the simple case too
    simple_text = '''
```python
# Simple test
import math
result = math.sqrt(16)
print(f"Square root of 16 is: {result}")
```
This will calculate the square root.
'''
    
    print("\n" + "="*80)
    print("SIMPLE CASE:")
    print("="*80)
    
    simple_extracted = instance._extract_code(simple_text)
    print("Simple extracted code:")
    print(repr(simple_extracted))
    print()
    print(simple_extracted)
    
    try:
        compile(simple_extracted, '<string>', 'exec')
        print("✅ Simple code compiles successfully!")
    except SyntaxError as e:
        print(f"❌ Simple syntax error: {e}")

if __name__ == "__main__":
    debug_extraction() 