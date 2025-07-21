"""
Output Verification Examples
These are reference implementations showing what output we expect our pipeline to capture.
Use these to manually verify that the captured output matches expectations.
"""

# Example 1: Basic Output Test
BASIC_OUTPUT_EXAMPLE = '''
import datetime

print("Hello, World!")
current_time = datetime.datetime.now()
print(f"Current date and time: {current_time}")
print("OUTPUT TEST COMPLETE")
'''

EXPECTED_BASIC_OUTPUT = '''
Hello, World!
Current date and time: [timestamp]
OUTPUT TEST COMPLETE
'''

# Example 2: Calculation Output Test  
CALCULATION_OUTPUT_EXAMPLE = '''
def factorial(n):
    result = 1
    print(f"Calculating factorial of {n}:")
    for i in range(1, n + 1):
        result *= i
        print(f"  Step {i}: {result}")
    return result

fact_5 = factorial(5)
print(f"Final result: {fact_5}")
print("CALCULATION COMPLETE")
'''

EXPECTED_CALCULATION_OUTPUT = '''
Calculating factorial of 5:
  Step 1: 1
  Step 2: 2
  Step 3: 6
  Step 4: 24
  Step 5: 120
Final result: 120
CALCULATION COMPLETE
'''

# Example 3: Loop Output Test
LOOP_OUTPUT_EXAMPLE = '''
numbers = []
total = 0

print("Counting from 1 to 10:")
for i in range(1, 11):
    numbers.append(i)
    total += i
    print(f"Number: {i}, Running total: {total}")

print(f"\\nSummary:")
print(f"Numbers: {numbers}")
print(f"Sum of all numbers: {total}")
print("LOOP TEST FINISHED")
'''

EXPECTED_LOOP_OUTPUT = '''
Counting from 1 to 10:
Number: 1, Running total: 1
Number: 2, Running total: 3
Number: 3, Running total: 6
Number: 4, Running total: 10
Number: 5, Running total: 15
Number: 6, Running total: 21
Number: 7, Running total: 28
Number: 8, Running total: 36
Number: 9, Running total: 45
Number: 10, Running total: 55

Summary:
Numbers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Sum of all numbers: 55
LOOP TEST FINISHED
'''

# Example 4: STDERR/STDOUT Test
STDERR_STDOUT_EXAMPLE = '''
import sys

print("Testing stdout and stderr output...")

# Generate output to stdout
print("This message goes to STDOUT", file=sys.stdout)

# Generate output to stderr  
print("This warning goes to STDERR", file=sys.stderr)

try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"ERROR: {e}", file=sys.stderr)
    print("Error was handled gracefully!")
    result = "undefined"

print(f"Result after error handling: {result}")
print("ERROR HANDLING COMPLETE")
'''

EXPECTED_STDERR_OUTPUT = '''
STDOUT:
Testing stdout and stderr output...
This message goes to STDOUT
Error was handled gracefully!
Result after error handling: undefined
ERROR HANDLING COMPLETE

STDERR:
This warning goes to STDERR
ERROR: division by zero
'''

# Example 5: Multi-line Pattern Test
MULTILINE_PATTERN_EXAMPLE = '''
def draw_box(width, height):
    """Draw a simple ASCII box"""
    pattern = []
    
    # Top border
    top = "+" + "-" * (width - 2) + "+"
    pattern.append(top)
    print(top)
    
    # Middle rows
    for i in range(height - 2):
        middle = "|" + " " * (width - 2) + "|"
        pattern.append(middle)
        print(middle)
    
    # Bottom border
    bottom = "+" + "-" * (width - 2) + "+"
    pattern.append(bottom)
    print(bottom)
    
    return pattern

print("Drawing a 8x5 box:")
box_pattern = draw_box(8, 5)

print(f"\\nPattern statistics:")
print(f"Lines: {len(box_pattern)}")
print(f"Total characters: {sum(len(line) for line in box_pattern)}")
print("PATTERN COMPLETE")
'''

EXPECTED_MULTILINE_OUTPUT = '''
Drawing a 8x5 box:
+------+
|      |
|      |
|      |
+------+

Pattern statistics:
Lines: 5
Total characters: 40
PATTERN COMPLETE
'''

# Verification functions
def verify_output_capture(captured_output: str, expected_pattern: str, test_name: str) -> bool:
    """
    Verify that captured output matches expected patterns.
    Returns True if output looks correct, False otherwise.
    """
    captured_lines = captured_output.strip().split('\n')
    expected_lines = expected_pattern.strip().split('\n')
    
    print(f"\\n=== Verifying {test_name} ===")
    print(f"Captured {len(captured_lines)} lines, expected ~{len(expected_lines)} lines")
    
    # Basic checks
    if len(captured_lines) == 0:
        print("❌ No output captured!")
        return False
    
    # Check for key phrases in the last few lines
    last_lines = ' '.join(captured_lines[-3:]).lower()
    
    if test_name == "basic_output":
        return "hello, world" in last_lines and "output test complete" in last_lines
    elif test_name == "calculation":
        return "calculation complete" in last_lines and "factorial" in captured_output.lower()
    elif test_name == "loop":
        return "loop test finished" in last_lines and "sum" in captured_output.lower()
    elif test_name == "stderr_stdout":
        return "error handling complete" in last_lines and ("stderr" in captured_output.lower() or "error:" in captured_output.lower())
    elif test_name == "multiline":
        return "pattern complete" in last_lines and ("+" in captured_output or "|" in captured_output)
    
    return True

# Test data for automated verification
OUTPUT_TEST_CASES = {
    "output_test_basic": {
        "example_code": BASIC_OUTPUT_EXAMPLE,
        "expected_output": EXPECTED_BASIC_OUTPUT,
        "verification_func": lambda output: verify_output_capture(output, EXPECTED_BASIC_OUTPUT, "basic_output")
    },
    "output_test_calculations": {
        "example_code": CALCULATION_OUTPUT_EXAMPLE,
        "expected_output": EXPECTED_CALCULATION_OUTPUT,
        "verification_func": lambda output: verify_output_capture(output, EXPECTED_CALCULATION_OUTPUT, "calculation")
    },
    "output_test_loops": {
        "example_code": LOOP_OUTPUT_EXAMPLE,
        "expected_output": EXPECTED_LOOP_OUTPUT,
        "verification_func": lambda output: verify_output_capture(output, EXPECTED_LOOP_OUTPUT, "loop")
    },
    "output_test_stderr": {
        "example_code": STDERR_STDOUT_EXAMPLE,
        "expected_output": EXPECTED_STDERR_OUTPUT,
        "verification_func": lambda output: verify_output_capture(output, EXPECTED_STDERR_OUTPUT, "stderr_stdout")
    },
    "output_test_multiline": {
        "example_code": MULTILINE_PATTERN_EXAMPLE,
        "expected_output": EXPECTED_MULTILINE_OUTPUT,
        "verification_func": lambda output: verify_output_capture(output, EXPECTED_MULTILINE_OUTPUT, "multiline")
    }
} 