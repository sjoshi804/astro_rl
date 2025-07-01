"""
Astronomy Data Visualization Generator
Generates tasks for exploring and visualizing FITS files using the Code Generation & Execution Service
"""

import asyncio
import httpx
import json
import time
import argparse
from typing import List, Dict, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Astronomy data visualization tasks for FITS files
ASTRONOMY_TASKS = [
    "Load the FITS file and display basic information about the image (dimensions, data type, header keys)",
    
#    "Create a simple grayscale visualization of the UV image with proper axis labels and a colorbar",
    
#    "Generate a histogram of pixel intensity values to understand the brightness distribution in the UV image"
    
#     "Apply different color maps (viridis, plasma, hot) to the image and display them side by side for comparison",
    
#     "Find and highlight the brightest regions in the image by creating a binary mask for pixels above the 95th percentile",
    
#     "Calculate and display basic statistics (mean, median, standard deviation) of pixel values in different regions of the image",
    
#     "Create a contour plot overlay on the original image to show intensity levels and structure",
    
#     "Implement a simple background subtraction by subtracting the median value and show before/after images",
    
#     "Create a radial profile plot showing how brightness varies with distance from the image center",
    
#     "Generate a 3D surface plot of a central region (e.g., 100x100 pixels) to visualize the intensity landscape"
# 
]

class AstronomyDataGenerator:
    """Generates astronomy data visualization tasks for the Code Generation & Execution Service"""
    
    def __init__(self, service_url: str, fits_file_path: str, timeout: int = 900):
        self.service_url = service_url.rstrip('/')
        self.fits_file_path = fits_file_path
        self.timeout = timeout
        self.results = []
        self.start_time = None
        
    def _format_task_with_context(self, task: str) -> str:
        """Format task with FITS file context and common imports"""
        context = f"""
You are working with an astronomy FITS file located at: {self.fits_file_path}

This is an Astro1 Ultraviolet Imaging Telescope image with dimensions 512 x 512 pixels.

Common imports you might need:
- from astropy.io import fits
- import numpy as np
- import matplotlib.pyplot as plt
- from astropy.visualization import ZScaleInterval, ImageNormalize
- from astropy.stats import sigma_clipped_stats

Task: {task}

Generate Python code to accomplish this task. Make sure to:
1. Handle the FITS file properly
2. Include appropriate error handling
3. Create clear, labeled visualizations when applicable
4. Add informative print statements for any calculated values
"""
        return context.strip()
    
    async def test_service_health(self) -> bool:
        """Test if the service is healthy and responding"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.service_url}/health")
                if response.status_code == 200:
                    logger.info("✓ Service health check passed")
                    return True
                else:
                    logger.error(f"✗ Service health check failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"✗ Failed to connect to service: {e}")
            return False
    
    async def get_service_config(self) -> Dict[str, Any]:
        """Get service configuration"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.service_url}/config")
                if response.status_code == 200:
                    config = response.json()
                    logger.info(f"Service config: {config}")
                    return config
                else:
                    logger.warning(f"Could not get service config: {response.status_code}")
                    return {}
        except Exception as e:
            logger.warning(f"Failed to get service config: {e}")
            return {}
    
    async def generate_single_task(self, task: str, max_turns: int = 8) -> Dict[str, Any]:
        """Generate code for a single astronomy task"""
        test_start = time.time()
        formatted_task = self._format_task_with_context(task)
        
        logger.info(f"Generating code for: {task[:60]}...")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                request_data = {
                    "requests": [
                        {
                            "initial_prompts": [formatted_task],
                            "max_turns": max_turns,
                            "completion_criteria": "visualization complete"
                        }
                    ]
                }
                
                response = await client.post(
                    f"{self.service_url}/generate_trajectories",
                    json=request_data
                )
                
                test_duration = time.time() - test_start
                
                if response.status_code == 200:
                    trajectories = response.json()
                    trajectory = trajectories[0] if trajectories else None
                    
                    result = {
                        "task": task,
                        "formatted_prompt": formatted_task,
                        "success": True,
                        "duration_seconds": test_duration,
                        "trajectory": trajectory,
                        "error": None
                    }
                    
                    if trajectory:
                        logger.info(f"✓ Completed in {test_duration:.2f}s - "
                                  f"{trajectory['total_steps']} steps, "
                                  f"reward: {trajectory['final_reward']:.3f}, "
                                  f"termination: {trajectory['termination_reason']}")
                        
                        # Log the final code generated
                        if trajectory['turns']:
                            last_turn = trajectory['turns'][-1]
                            logger.debug(f"Final code: {last_turn['code'][:100]}...")
                    else:
                        logger.warning("✗ No trajectory returned")
                        result["success"] = False
                        result["error"] = "No trajectory in response"
                    
                    return result
                else:
                    logger.error(f"✗ Request failed: {response.status_code} - {response.text}")
                    return {
                        "task": task,
                        "formatted_prompt": formatted_task,
                        "success": False,
                        "duration_seconds": test_duration,
                        "trajectory": None,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    }
                    
        except asyncio.TimeoutError:
            logger.error(f"✗ Request timed out after {self.timeout}s")
            return {
                "task": task,
                "formatted_prompt": formatted_task,
                "success": False,
                "duration_seconds": self.timeout,
                "trajectory": None,
                "error": "Request timeout"
            }
        except Exception as e:
            test_duration = time.time() - test_start
            logger.error(f"✗ Request failed: {e}")
            return {
                "task": task,
                "formatted_prompt": formatted_task,
                "success": False,
                "duration_seconds": test_duration,
                "trajectory": None,
                "error": str(e)
            }
    
    async def generate_batch_sequential(self, tasks: List[str], max_turns: int = 8) -> List[Dict[str, Any]]:
        """Generate code for tasks sequentially"""
        logger.info(f"Processing {len(tasks)} astronomy tasks sequentially...")
        results = []
        
        for i, task in enumerate(tasks, 1):
            logger.info(f"\n--- Task {i}/{len(tasks)} ---")
            result = await self.generate_single_task(task, max_turns)
            results.append(result)
            
            # Small delay between requests to be nice to the service
            await asyncio.sleep(2)
        
        return results
    
    async def generate_batch_concurrent(self, tasks: List[str], max_turns: int = 8, concurrency: int = 2) -> List[Dict[str, Any]]:
        """Generate code for tasks concurrently with limited concurrency"""
        logger.info(f"Processing {len(tasks)} astronomy tasks with concurrency {concurrency}...")
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def generate_with_semaphore(task):
            async with semaphore:
                return await self.generate_single_task(task, max_turns)
        
        tasks_with_semaphore = [generate_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*tasks_with_semaphore, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "task": tasks[i],
                    "formatted_prompt": "",
                    "success": False,
                    "duration_seconds": 0,
                    "trajectory": None,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze generation results and create summary"""
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r["success"])
        failed_tasks = total_tasks - successful_tasks
        
        durations = [r["duration_seconds"] for r in results if r["success"]]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Analyze trajectories
        trajectories = [r["trajectory"] for r in results if r["trajectory"]]
        
        if trajectories:
            avg_steps = sum(t["total_steps"] for t in trajectories) / len(trajectories)
            avg_reward = sum(t["final_reward"] for t in trajectories) / len(trajectories)
            
            termination_reasons = {}
            for t in trajectories:
                reason = t["termination_reason"]
                termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
                
            # Analyze code complexity (rough estimate)
            total_code_lines = 0
            total_successful_executions = 0
            for t in trajectories:
                for turn in t["turns"]:
                    if turn["code"]:
                        total_code_lines += len(turn["code"].split('\n'))
                    if turn["execution_success"]:
                        total_successful_executions += 1
            
            avg_code_lines = total_code_lines / len(trajectories) if trajectories else 0
            total_exec_steps = sum(t["total_steps"] for t in trajectories)
            execution_success_rate = (total_successful_executions / total_exec_steps) if total_exec_steps else 0
        else:
            avg_steps = 0
            avg_reward = 0
            termination_reasons = {}
            avg_code_lines = 0
            execution_success_rate = 0
        
        # Error analysis
        errors = {}
        for r in results:
            if not r["success"] and r["error"]:
                error_type = r["error"].split(":")[0] if ":" in r["error"] else r["error"]
                errors[error_type] = errors.get(error_type, 0) + 1
        
        analysis = {
            "summary": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks,
                "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
                "average_duration_seconds": avg_duration,
                "total_duration_seconds": sum(r["duration_seconds"] for r in results)
            },
            "trajectory_analysis": {
                "total_trajectories": len(trajectories),
                "average_steps": avg_steps,
                "average_reward": avg_reward,
                "average_code_lines": avg_code_lines,
                "execution_success_rate": execution_success_rate,
                "termination_reasons": termination_reasons
            },
            "error_analysis": errors,
            "astronomy_tasks": [r["task"] for r in results],
            "detailed_results": results
        }
        
        return analysis
    
    def print_summary(self, analysis: Dict[str, Any]):
        """Print a formatted summary of generation results"""
        summary = analysis["summary"]
        traj_analysis = analysis["trajectory_analysis"]
        
        print("\n" + "="*70)
        print("ASTRONOMY DATA GENERATION RESULTS")
        print("="*70)
        
        print(f"Total Tasks:              {summary['total_tasks']}")
        print(f"Successful:               {summary['successful_tasks']}")
        print(f"Failed:                   {summary['failed_tasks']}")
        print(f"Success Rate:             {summary['success_rate']:.1%}")
        print(f"Average Duration:         {summary['average_duration_seconds']:.2f}s")
        print(f"Total Duration:           {summary['total_duration_seconds']:.2f}s")
        
        print(f"\nTrajectory Analysis:")
        print(f"Total Trajectories:       {traj_analysis['total_trajectories']}")
        print(f"Average Steps:            {traj_analysis['average_steps']:.1f}")
        print(f"Average Reward:           {traj_analysis['average_reward']:.3f}")
        print(f"Average Code Lines:       {traj_analysis['average_code_lines']:.1f}")
        print(f"Execution Success Rate:   {traj_analysis['execution_success_rate']:.1%}")
        
        if traj_analysis['termination_reasons']:
            print(f"\nTermination Reasons:")
            for reason, count in traj_analysis['termination_reasons'].items():
                print(f"  {reason}: {count}")
        
        if analysis['error_analysis']:
            print(f"\nError Types:")
            for error, count in analysis['error_analysis'].items():
                print(f"  {error}: {count}")
        
        print("="*70)
    
    def save_results(self, analysis: Dict[str, Any], filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"astronomy_generation_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def extract_final_code_snippets(self, results: List[Dict[str, Any]], output_file: str = None):
        """Extract and save final working code snippets"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"astronomy_code_snippets_{timestamp}.py"
        
        try:
            with open(output_file, 'w') as f:
                f.write("# Generated Astronomy Data Visualization Code\n")
                f.write(f"# Generated on: {datetime.now().isoformat()}\n")
                f.write(f"# FITS file: {self.fits_file_path}\n\n")
                
                for i, result in enumerate(results, 1):
                    if result["success"] and result["trajectory"]:
                        f.write(f"# Task {i}: {result['task']}\n")
                        f.write("# " + "="*60 + "\n\n")
                        
                        trajectory = result["trajectory"]
                        for turn in trajectory["turns"]:
                            if turn["code"] and turn["execution_success"]:
                                f.write(f"# Step {turn['step']} - Working Code:\n")
                                f.write(turn["code"])
                                f.write("\n\n")
                                f.write(f"# Execution Output: {turn['execution_output'][:200]}...\n\n")
                        
                        f.write("\n" + "#"*60 + "\n\n")
            
            logger.info(f"Code snippets saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save code snippets: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Generate Astronomy Data Visualization Code")
    parser.add_argument("--service-url", default="http://localhost:8002",
                       help="URL of the code generation service")
    parser.add_argument("--fits-file", required=True,
                       help="Path to the FITS file (Astro1 UV Imaging Telescope data)")
    parser.add_argument("--max-turns", type=int, default=8,
                       help="Maximum turns per trajectory")
    parser.add_argument("--timeout", type=int, default=900,
                       help="Request timeout in seconds (longer for complex visualizations)")
    parser.add_argument("--concurrent", action="store_true",
                       help="Run tasks concurrently instead of sequentially")
    parser.add_argument("--concurrency", type=int, default=2,
                       help="Number of concurrent requests (if --concurrent)")
    parser.add_argument("--tasks", nargs="*",
                       help="Custom tasks to process (overrides default astronomy tasks)")
    parser.add_argument("--output-file",
                       help="File to save results (default: auto-generated)")
    parser.add_argument("--save-code", action="store_true",
                       help="Save extracted working code snippets to a Python file")
    
    args = parser.parse_args()
    
    # Use custom tasks or default astronomy dataset
    tasks = args.tasks if args.tasks else ASTRONOMY_TASKS
    
    logger.info(f"Starting astronomy data generation with {len(tasks)} tasks")
    logger.info(f"Service URL: {args.service_url}")
    logger.info(f"FITS file: {args.fits_file}")
    logger.info(f"Max turns: {args.max_turns}")
    logger.info(f"Timeout: {args.timeout}s")
    
    generator = AstronomyDataGenerator(args.service_url, args.fits_file, args.timeout)
    
    # Health check
    if not await generator.test_service_health():
        logger.error("Service health check failed. Exiting.")
        return
    
    # Get service config
    config = await generator.get_service_config()
    
    # Display the tasks to be processed
    logger.info("\nTasks to be processed:")
    for i, task in enumerate(tasks, 1):
        logger.info(f"{i:2d}. {task}")
    
    # Run generation
    start_time = time.time()
    
    if args.concurrent:
        results = await generator.generate_batch_concurrent(tasks, args.max_turns, args.concurrency)
    else:
        results = await generator.generate_batch_sequential(tasks, args.max_turns)
    
    total_time = time.time() - start_time
    logger.info(f"\nAll tasks completed in {total_time:.2f}s")
    
    # Analyze and display results
    analysis = generator.analyze_results(results)
    analysis["generation_config"] = {
        "service_url": args.service_url,
        "fits_file": args.fits_file,
        "max_turns": args.max_turns,
        "timeout": args.timeout,
        "concurrent": args.concurrent,
        "concurrency": args.concurrency if args.concurrent else 1,
        "total_generation_time": total_time,
        "service_config": config
    }
    
    generator.print_summary(analysis)
    
    # Save results
    generator.save_results(analysis, args.output_file)
    
    # Save working code snippets if requested
    if args.save_code:
        generator.extract_final_code_snippets(results)
    
    # Show some example successful results
    successful_results = [r for r in results if r["success"] and r["trajectory"]]
    if successful_results:
        print(f"\nExample successful trajectory:")
        example = successful_results[0]
        print(f"Task: {example['task']}")
        traj = example["trajectory"]
        print(f"Steps: {traj['total_steps']}, Reward: {traj['final_reward']:.3f}")
        
        # Show the final working code
        if traj["turns"]:
            last_turn = traj["turns"][-1]
            if last_turn["execution_success"]:
                print(f"Final working code snippet:")
                print("-" * 40)
                print(last_turn["code"][:300] + "..." if len(last_turn["code"]) > 300 else last_turn["code"])
                print("-" * 40)
                print(f"Output: {last_turn['execution_output'][:150]}...")
    
    # Summary of what was generated
    successful_count = sum(1 for r in results if r["success"])
    print(f"\n🎯 Successfully generated code for {successful_count}/{len(tasks)} astronomy visualization tasks")
    
    if successful_count > 0:
        print("✨ Generated code can be used for:")
        print("   - FITS file exploration and analysis")
        print("   - UV telescope image visualization")
        print("   - Astronomical data processing workflows")
        print("   - Educational astronomy data science examples")


if __name__ == "__main__":
    asyncio.run(main())