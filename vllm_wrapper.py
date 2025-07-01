"""
VLLM Server Wrapper
Wraps the VLLM OpenAI API server with model updating capability
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import httpx
import asyncio
import subprocess
import signal
import os
import logging
import argparse
import time
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
VLLM_MODEL_PATH = None
TENSOR_PARALLEL_SIZE = 1
VLLM_HOST = "127.0.0.1"
VLLM_PORT = None
WRAPPER_PORT = None

class UpdateModelRequest(BaseModel):
    model_path: str

class VLLMProcess:
    """Manages the VLLM server process"""
    
    def __init__(self, model_path: str, tensor_parallel_size: int, host: str, port: int):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.host = host
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{host}:{port}"
    
    async def start(self) -> bool:
        """Start the VLLM server process"""
        if self.process and self.process.poll() is None:
            logger.warning("VLLM process is already running")
            return True
        
        logger.info(f"Starting VLLM server with model: {self.model_path}")
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--host", self.host,
            "--port", str(self.port)
        ]
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group for clean shutdown
            )
            
            logger.info(f"VLLM process started with PID: {self.process.pid}")
            
            # Wait for server to be ready
            if await self._wait_for_ready():
                logger.info("VLLM server is ready")
                return True
            else:
                logger.error("VLLM server failed to start properly")
                await self.stop()
                return False
                
        except Exception as e:
            logger.error(f"Failed to start VLLM server: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the VLLM server process"""
        if not self.process:
            return True
        
        logger.info("Stopping VLLM server")
        
        try:
            # Send SIGTERM to the process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            
            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process_end()),
                    timeout=30.0
                )
                logger.info("VLLM server stopped gracefully")
                return True
            except asyncio.TimeoutError:
                logger.warning("VLLM server didn't stop gracefully, forcing kill")
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                await self._wait_for_process_end()
                logger.info("VLLM server force killed")
                return True
                
        except Exception as e:
            logger.error(f"Error stopping VLLM server: {e}")
            return False
        finally:
            self.process = None
    
    async def restart_with_new_model(self, model_path: str) -> bool:
        """Restart VLLM server with a new model"""
        logger.info(f"Restarting VLLM server with new model: {model_path}")
        
        # Stop current process
        if not await self.stop():
            logger.error("Failed to stop current VLLM server")
            return False
        
        # Update model path
        self.model_path = model_path
        
        # Start with new model
        return await self.start()
    
    async def _wait_for_ready(self, timeout: int = 300) -> bool:
        """Wait for VLLM server to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self.base_url}/health")
                    if response.status_code == 200:
                        return True
            except Exception:
                pass
            
            # Check if process is still running
            if self.process and self.process.poll() is not None:
                logger.error("VLLM process terminated unexpectedly")
                return False
            
            await asyncio.sleep(2)
        
        logger.error(f"VLLM server did not become ready within {timeout} seconds")
        return False
    
    async def _wait_for_process_end(self):
        """Wait for the process to end"""
        while self.process and self.process.poll() is None:
            await asyncio.sleep(0.1)
    
    def is_running(self) -> bool:
        """Check if VLLM process is running"""
        return self.process is not None and self.process.poll() is None
    
    async def proxy_request(self, request: Request) -> Dict[str, Any]:
        """Proxy a request to the VLLM server"""
        if not self.is_running():
            raise HTTPException(status_code=503, detail="VLLM server is not running")
        
        # Build target URL (no legacy translation, vllm 0.8.6 supports chat endpoint natively)
        path = request.url.path
        query = str(request.url.query) if request.url.query else ""
        target_url = f"{self.base_url}{path}"
        if query:
            target_url += f"?{query}"

        body = await request.body()

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=dict(request.headers),
                content=body
            )

        return {
            "status_code": response.status_code,
            "content": response.content,
            "headers": dict(response.headers)
        }

# Global VLLM process manager
vllm_process: Optional[VLLMProcess] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global vllm_process
    
    vllm_process = VLLMProcess(
        model_path=VLLM_MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        host=VLLM_HOST,
        port=VLLM_PORT
    )
    
    # Start VLLM server
    if await vllm_process.start():
        logger.info("VLLM wrapper started successfully")
    else:
        logger.error("Failed to start VLLM server")
        raise RuntimeError("Failed to start VLLM server")
    
    yield
    
    # Cleanup on shutdown
    if vllm_process:
        await vllm_process.stop()
    logger.info("VLLM wrapper shutdown complete")

app = FastAPI(title="VLLM Server Wrapper", lifespan=lifespan)

# API Endpoints

@app.post("/update_model_params")
async def update_model_params(request: UpdateModelRequest):
    """Update the model parameters by restarting with new model"""
    if not vllm_process:
        raise HTTPException(status_code=500, detail="VLLM process not initialized")
    
    logger.info(f"Updating model to: {request.model_path}")
    
    success = await vllm_process.restart_with_new_model(request.model_path)
    
    if success:
        return {
            "status": "success",
            "message": f"Model updated to {request.model_path}",
            "model_path": request.model_path
        }
    else:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to update model to {request.model_path}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not vllm_process or not vllm_process.is_running():
        raise HTTPException(status_code=503, detail="VLLM server is not running")
    
    # Also check if VLLM server is responding
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{vllm_process.base_url}/health")
            if response.status_code == 200:
                return {"status": "healthy", "vllm_status": "running"}
            else:
                raise HTTPException(status_code=503, detail="VLLM server is not responding")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"VLLM server health check failed: {e}")

@app.get("/wrapper_status")
async def wrapper_status():
    """Get wrapper and VLLM server status"""
    if not vllm_process:
        return {"wrapper_status": "not_initialized"}
    
    return {
        "wrapper_status": "running",
        "vllm_status": "running" if vllm_process.is_running() else "stopped",
        "model_path": vllm_process.model_path,
        "vllm_url": vllm_process.base_url,
        "process_pid": vllm_process.process.pid if vllm_process.process else None
    }

# Proxy all other requests to VLLM server
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_vllm(request: Request, path: str):
    """Proxy all other requests to the VLLM server"""
    if not vllm_process:
        raise HTTPException(status_code=503, detail="VLLM process not initialized")
    
    try:
        result = await vllm_process.proxy_request(request)
        
        from fastapi import Response
        return Response(
            content=result["content"],
            status_code=result["status_code"],
            headers=dict(result["headers"])
        )
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to VLLM server timed out")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot connect to VLLM server")
    except Exception as e:
        logger.error(f"Error proxying request: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="VLLM Server Wrapper")
    parser.add_argument("--model-path", required=True,
                       help="Path to the model to load")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Tensor parallel size for VLLM")
    parser.add_argument("--vllm-host", default="127.0.0.1",
                       help="Host for the VLLM server")
    parser.add_argument("--vllm-port", type=int, required=True,
                       help="Port for the VLLM server")
    parser.add_argument("--wrapper-host", default="0.0.0.0",
                       help="Host for the wrapper service")
    parser.add_argument("--wrapper-port", type=int, required=True,
                       help="Port for the wrapper service")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set global configuration
    VLLM_MODEL_PATH = args.model_path
    TENSOR_PARALLEL_SIZE = args.tensor_parallel_size
    VLLM_HOST = args.vllm_host
    VLLM_PORT = args.vllm_port
    WRAPPER_PORT = args.wrapper_port
    
    logger.info(f"Starting VLLM Wrapper")
    logger.info(f"Model path: {VLLM_MODEL_PATH}")
    logger.info(f"VLLM server: {VLLM_HOST}:{VLLM_PORT}")
    logger.info(f"Wrapper port: {WRAPPER_PORT}")
    logger.info(f"Tensor parallel size: {TENSOR_PARALLEL_SIZE}")
    
    import uvicorn
    uvicorn.run(app, host=args.wrapper_host, port=WRAPPER_PORT)