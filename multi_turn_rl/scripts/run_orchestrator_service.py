#!/usr/bin/env python3
"""
Orchestrator Service Script
Runs the combined orchestrator service that includes both the orchestrator for managing
trajectories and the completion load balancer for distributing requests to vLLM servers.
"""

import argparse
import logging
import asyncio
import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_turn_rl.orchestrator import Orchestrator
from multi_turn_rl.completion_load_balancer import CompletionLoadBalancer
from multi_turn_rl.main import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multi-Turn RL Orchestrator Service")
    
    # Run configuration
    parser.add_argument("--run-id", required=True,
                       help="Unique run identifier (e.g., SLURM job ID)")
    
    # Service configuration
    parser.add_argument("--host", default="0.0.0.0", help="Host for the service")
    parser.add_argument("--port", type=int, default=8080, help="Port for the service")
    
    # vLLM completion servers
    parser.add_argument("--vllm-servers", nargs="+", required=True,
                       help="List of vLLM server URLs (e.g., http://localhost:8000)")
    
    # Orchestrator configuration
    parser.add_argument("--max-turns", type=int, default=10,
                       help="Maximum number of turns per trajectory")
    parser.add_argument("--timeout-seconds", type=int, default=30,
                       help="Timeout for code execution")
    
    # Ray configuration
    parser.add_argument("--ray-timeout-per-step", type=float, default=30.0,
                       help="Ray timeout per execution step")
    parser.add_argument("--ray-num-cpus", type=int, default=1,
                       help="Number of CPUs per Ray execution")
    parser.add_argument("--ray-num-gpus", type=int, default=0,
                       help="Number of GPUs per Ray execution")
    
    # Load balancer configuration
    parser.add_argument("--load-balancer-strategy", choices=["round_robin", "random"], 
                       default="round_robin", help="Load balancing strategy")
    parser.add_argument("--health-check-interval", type=int, default=30,
                       help="Health check interval in seconds")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retries for failed requests")
    
    return parser.parse_args()


def create_orchestrator_service(args) -> FastAPI:
    """Create the orchestrator service with configured components"""
    
    # Create run-specific directories
    run_trajectory_dir = Path("trajectories") / args.run_id
    run_logs_dir = Path("logs") / args.run_id
    
    # Ensure directories exist
    run_trajectory_dir.mkdir(parents=True, exist_ok=True)
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up run-specific paths
    trajectory_output_dir = str(run_trajectory_dir)
    prompts_jsonl_path = str(run_trajectory_dir / "prompts.jsonl")
    
    logger.info(f"Run ID: {args.run_id}")
    logger.info(f"Trajectory output directory: {trajectory_output_dir}")
    logger.info(f"Prompts JSONL path: {prompts_jsonl_path}")
    logger.info(f"Logs directory: {run_logs_dir}")
    
    # Extract hostnames from full URLs for completion load balancer
    vllm_hostnames = []
    for url in args.vllm_servers:
        if url.startswith('http://'):
            hostname = url[7:]  # Remove 'http://' prefix
        elif url.startswith('https://'):
            hostname = url[8:]  # Remove 'https://' prefix  
        else:
            hostname = url  # Assume it's already just hostname:port
        vllm_hostnames.append(hostname)
    
    logger.info(f"VLLM server URLs: {args.vllm_servers}")
    logger.info(f"VLLM hostnames for load balancer: {vllm_hostnames}")
    
    # Create completion load balancer
    completion_load_balancer = CompletionLoadBalancer(
        vllm_hostnames=vllm_hostnames,
        health_check_interval=args.health_check_interval,
        request_timeout=120
    )
    
    # Create orchestrator
    orchestrator = Orchestrator(
        completion_load_balancer=completion_load_balancer,
        max_turns=args.max_turns,
        timeout_seconds=args.timeout_seconds,
        trajectory_output_dir=trajectory_output_dir,
        prompts_jsonl_path=prompts_jsonl_path,
        ray_timeout_per_step=args.ray_timeout_per_step,
        ray_num_cpus=args.ray_num_cpus,
        ray_num_gpus=args.ray_num_gpus
    )
    
    # Create FastAPI app
    app = create_app(orchestrator, completion_load_balancer)
    
    return app


def main():
    """Main entry point"""
    args = parse_args()
    
    logger.info("Starting Multi-Turn RL Orchestrator Service")
    logger.info(f"Run ID: {args.run_id}")
    logger.info(f"Service will run at {args.host}:{args.port}")
    logger.info(f"vLLM servers: {args.vllm_servers}")
    logger.info(f"Max turns: {args.max_turns}")
    logger.info(f"Ray configuration: CPUs={args.ray_num_cpus}, GPUs={args.ray_num_gpus}")
    
    # Create the service
    app = create_orchestrator_service(args)
    
    # Run the service
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()