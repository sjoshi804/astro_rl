#!/usr/bin/env python3
"""
vLLM Wrapper Script
Starts a vLLM wrapper server with the specified model and configuration.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_turn_rl.vllm_wrapper import VLLMWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="vLLM Wrapper Server")
    
    # Run configuration
    parser.add_argument("--run-id", required=True,
                       help="Unique run identifier (e.g., SLURM job ID)")
    parser.add_argument("--service-id", required=True,
                       help="Service identifier (e.g., vllm-node1, vllm-node2)")
    
    # Required arguments
    parser.add_argument("--model-path", required=True,
                       help="Path to the model to load")
    parser.add_argument("--port", type=int, required=True,
                       help="Port for the wrapper service")
    
    # Optional arguments
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host for the wrapper service")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Tensor parallel size for vLLM")
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Create run-specific logs directory
    run_logs_dir = Path("logs") / args.run_id
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting vLLM Wrapper Server")
    logger.info(f"Run ID: {args.run_id}")
    logger.info(f"Service ID: {args.service_id}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Server address: {args.host}:{args.port}")
    logger.info(f"Tensor parallel size: {args.tensor_parallel_size}")
    logger.info(f"Logs directory: {run_logs_dir}")
    
    # Create vLLM wrapper
    try:
        wrapper = VLLMWrapper(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size
        )
        
        # Run the server
        wrapper.run(host=args.host, port=args.port)
        
    except KeyboardInterrupt:
        logger.info("vLLM wrapper stopped by user")
    except Exception as e:
        logger.error(f"vLLM wrapper error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()