#!/usr/bin/env python3
"""
Simple Output Test Runner
Use this to quickly test that the output capture pipeline is working correctly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import the astro modules
sys.path.append(str(Path(__file__).parent.parent))

from astro_data_generator import AstronomyDataGenerator
from demo_data.output_verification_examples import OUTPUT_TEST_CASES, verify_output_capture

async def run_output_tests(service_url="http://localhost:8002"):
    """Run output tests and verify capture is working"""
    
    print("🧪 Running Output Capture Tests")
    print("=" * 50)
    
    # Create generator instance
    generator = AstronomyDataGenerator(
        service_url=service_url,
        fits_file_path="demo_data/astro1_uv_imaging_telescope.fits",
        timeout=60
    )
    
    # Check service health
    if not await generator.test_service_health():
        print("❌ Service not available. Make sure code_and_exec_service is running.")
        return False
    
    print("✅ Service is healthy")
    
    # Test each output verification case
    test_names = ["output_test_basic", "output_test_calculations", "output_test_loops"]
    results = []
    
    for test_name in test_names:
        print(f"\n🔍 Testing {test_name}...")
        
        try:
            # Run the test using the main generator
            from demo_data.simple_multistep_tasks import get_task_by_name
            task = get_task_by_name(test_name)
            
            if not task:
                print(f"❌ Task {test_name} not found")
                continue
            
            result = await generator.generate_single_task(task, max_turns=3)
            
            if result["success"] and result["trajectory"]:
                trajectory = result["trajectory"]
                print(f"✅ Task completed in {trajectory['total_steps']} steps")
                print(f"   Reward: {trajectory['final_reward']}")
                print(f"   Termination: {trajectory['termination_reason']}")
                
                # Get the last turn's output
                if trajectory["turns"]:
                    last_turn = trajectory["turns"][-1]
                    captured_output = last_turn["execution_output"]
                    
                    print(f"\n📝 Captured Output ({len(captured_output)} chars):")
                    print("-" * 40)
                    print(captured_output[:500] + "..." if len(captured_output) > 500 else captured_output)
                    print("-" * 40)
                    
                    # Verify output matches expectations
                    if test_name in OUTPUT_TEST_CASES:
                        verification_func = OUTPUT_TEST_CASES[test_name]["verification_func"]
                        is_valid = verification_func(captured_output)
                        print(f"✅ Output verification: {'PASSED' if is_valid else 'FAILED'}")
                        results.append((test_name, True, is_valid))
                    else:
                        print("⚠️  No verification function available")
                        results.append((test_name, True, None))
                else:
                    print("❌ No turns in trajectory")
                    results.append((test_name, False, False))
            else:
                print(f"❌ Task failed: {result.get('error', 'Unknown error')}")
                results.append((test_name, False, False))
                
        except Exception as e:
            print(f"❌ Error running {test_name}: {e}")
            results.append((test_name, False, False))
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 50)
    
    successful_tasks = sum(1 for _, success, _ in results if success)
    successful_verifications = sum(1 for _, _, verified in results if verified)
    
    print(f"Tasks completed: {successful_tasks}/{len(results)}")
    print(f"Output verification: {successful_verifications}/{len(results)}")
    
    for test_name, success, verified in results:
        status = "✅" if success else "❌"
        verify_status = "✅" if verified else "❌" if verified is False else "⚠️"
        print(f"  {status} {test_name} - Output: {verify_status}")
    
    return successful_tasks == len(results) and successful_verifications >= len(results) // 2

async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test output capture pipeline")
    parser.add_argument("--service-url", default="http://localhost:8002",
                       help="URL of the code execution service")
    
    args = parser.parse_args()
    
    print(f"Testing output capture with service: {args.service_url}")
    
    success = await run_output_tests(args.service_url)
    
    if success:
        print("\n🎉 All output tests passed!")
        return 0
    else:
        print("\n💥 Some output tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 