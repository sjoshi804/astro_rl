"""
Main script for running the Multi-Turn RL service
Combines Orchestrator and CompletionLoadBalancer with direct function calls
"""

import asyncio
import argparse
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from typing import List

from .orchestrator import Orchestrator
from .completion_load_balancer import CompletionLoadBalancer
from .models import BatchTrajectoryRequest, Trajectory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
orchestrator: Orchestrator = None
completion_load_balancer: CompletionLoadBalancer = None

app = FastAPI(title="Multi-Turn RL Service")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global orchestrator, completion_load_balancer
    
    logger.info("Starting Multi-Turn RL Service...")
    
    # Initialize completion load balancer
    if not await completion_load_balancer.initialize():
        raise RuntimeError("Failed to initialize completion load balancer - no healthy VLLM servers")
    
    logger.info("Multi-Turn RL Service startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up services on shutdown"""
    global orchestrator, completion_load_balancer
    
    logger.info("Shutting down Multi-Turn RL Service...")
    
    if orchestrator:
        orchestrator.shutdown()
    
    if completion_load_balancer:
        await completion_load_balancer.shutdown()
    
    logger.info("Multi-Turn RL Service shutdown complete")


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    """
    OpenAI-compatible chat completions endpoint
    Forwards requests to completion load balancer
    """
    if not completion_load_balancer:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        return await completion_load_balancer.chat_completions(request)
    except Exception as e:
        logger.error(f"Chat completions error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat completions failed: {str(e)}")


@app.post("/generate_trajectories")
async def generate_trajectories(request: BatchTrajectoryRequest) -> List[Trajectory]:
    """
    Main endpoint for generating RL training trajectories
    Used by the Policy Training Module (TRL)
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        return await orchestrator.generate_trajectories(request)
    except Exception as e:
        logger.error(f"Trajectory generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Trajectory generation failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check for the service"""
    if not orchestrator or not completion_load_balancer:
        return {"status": "starting"}
    
    lb_health = completion_load_balancer.get_health_status()
    orchestrator_stats = orchestrator.get_stats()
    
    return {
        "status": "healthy" if lb_health["status"] == "healthy" else "unhealthy",
        "completion_load_balancer": lb_health,
        "orchestrator": orchestrator_stats
    }


@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    if not orchestrator or not completion_load_balancer:
        return {"error": "Services not initialized"}
    
    return {
        "completion_load_balancer": completion_load_balancer.get_stats(),
        "orchestrator": orchestrator.get_stats()
    }


@app.get("/models")
async def list_models():
    """List models currently loaded on each server"""
    if not completion_load_balancer:
        raise HTTPException(status_code=503, detail="Load balancer not initialized")
    
    return completion_load_balancer.list_models()


@app.post("/update_model_params")
async def update_model_params(request: dict):
    """Update model parameters on all VLLM servers"""
    if not completion_load_balancer:
        raise HTTPException(status_code=503, detail="Load balancer not initialized")
    
    model_path = request.get("model_path")
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path is required")
    
    results = await completion_load_balancer.update_all_models(model_path)
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    return {
        "status": "completed",
        "updated_servers": success_count,
        "total_servers": total_count,
        "results": results,
        "success": success_count == total_count,
        "model_path": model_path
    }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multi-Turn RL Service")
    
    # VLLM server configuration
    parser.add_argument("--vllm-servers", nargs="+", required=True,
                       help="List of VLLM server hostnames (e.g., localhost:8000 localhost:8001)")
    
    # Orchestrator configuration
    parser.add_argument("--max-turns", type=int, default=10,
                       help="Maximum number of turns per trajectory")
    parser.add_argument("--timeout", type=int, default=30,
                       help="HTTP timeout in seconds")
    parser.add_argument("--ray-timeout-per-step", type=float, default=30.0,
                       help="Timeout per execution step in Ray engine")
    parser.add_argument("--ray-num-cpus", type=int, default=1,
                       help="Number of CPUs per Ray actor")
    parser.add_argument("--ray-num-gpus", type=int, default=0,
                       help="Number of GPUs per Ray actor")
    parser.add_argument("--trajectory-output-dir", default="/work/10450/sjoshi804/vista/astro_rl/trajectories",
                       help="Directory to save trajectory JSON files")
    parser.add_argument("--prompts-jsonl-path", default="/work/10450/sjoshi804/vista/astro_rl/prompts.jsonl",
                       help="Path to save prompts JSONL file")
    
    # Load balancer configuration
    parser.add_argument("--health-check-interval", type=int, default=10,
                       help="Health check interval in seconds")
    parser.add_argument("--request-timeout", type=int, default=300,
                       help="Request timeout in seconds")
    
    # Service configuration
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host to bind the service")
    parser.add_argument("--port", type=int, default=8002,
                       help="Port to bind the service")
    
    return parser.parse_args()


async def create_services(args):
    """Create and configure the service instances"""
    global orchestrator, completion_load_balancer
    
    # Create completion load balancer
    completion_load_balancer = CompletionLoadBalancer(
        vllm_hostnames=args.vllm_servers,
        health_check_interval=args.health_check_interval,
        request_timeout=args.request_timeout
    )
    
    # Create orchestrator with direct reference to load balancer
    orchestrator = Orchestrator(
        completion_load_balancer=completion_load_balancer,
        max_turns=args.max_turns,
        timeout_seconds=args.timeout,
        trajectory_output_dir=args.trajectory_output_dir,
        prompts_jsonl_path=args.prompts_jsonl_path,
        ray_timeout_per_step=args.ray_timeout_per_step,
        ray_num_cpus=args.ray_num_cpus,
        ray_num_gpus=args.ray_num_gpus
    )
    
    logger.info("Services created successfully")


def main():
    """Main entry point for console script"""
    args = parse_args()
    
    logger.info(f"Starting Multi-Turn RL Service")
    logger.info(f"VLLM Servers: {args.vllm_servers}")
    logger.info(f"Max Turns: {args.max_turns}")
    logger.info(f"Timeout: {args.timeout}s")
    logger.info(f"Ray Timeout per Step: {args.ray_timeout_per_step}s")
    logger.info(f"Ray CPUs per Actor: {args.ray_num_cpus}")
    logger.info(f"Ray GPUs per Actor: {args.ray_num_gpus}")
    logger.info(f"Trajectory Output Directory: {args.trajectory_output_dir}")
    logger.info(f"Prompts JSONL Path: {args.prompts_jsonl_path}")
    logger.info(f"Health Check Interval: {args.health_check_interval}s")
    logger.info(f"Request Timeout: {args.request_timeout}s")
    
    # Create services
    asyncio.run(create_services(args))
    
    # Run the FastAPI application
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()