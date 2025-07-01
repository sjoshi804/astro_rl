"""
Completion Server
Load balances requests across multiple VLLM servers with model updating capability
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import httpx
import asyncio
import logging
import argparse
from datetime import datetime
import itertools
from contextlib import asynccontextmanager
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
VLLM_HOSTNAMES = []
HEALTH_CHECK_INTERVAL = 10  # seconds
REQUEST_TIMEOUT = 300  # seconds (increased for large generations)

# Data Models
class Turn(BaseModel):
    step: int
    prompt: str
    code: str
    execution_output: str
    execution_success: bool

class Trajectory(BaseModel):
    turns: List[Turn]

class GenerateCompletionRequest(BaseModel):
    trajectory: Trajectory
    instruction: str = "Generate next code snippet to achieve the goal"
    n: int = 4
    temperature: float = 0.8
    max_tokens: int = 512
    stop: Optional[List[str]] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class GenerateCompletionResponse(BaseModel):
    completions: List[str]
    server_used: str
    model_used: Optional[str] = None

class UpdateModelRequest(BaseModel):
    model_path: str

class VLLMServer:
    """Represents a single VLLM server instance"""
    
    def __init__(self, hostname: str):
        self.hostname = hostname
        self.is_healthy = False
        self.is_updating = False
        self.last_health_check = None
        self.model_path = None
        self.base_url = f"http://{hostname}"
    
    async def health_check(self) -> bool:
        """Check if the VLLM server is healthy"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    self.is_healthy = True
                    health_data = response.json()
                    self.model_path = health_data.get("model")
                else:
                    self.is_healthy = False
                
                self.last_health_check = datetime.now()
                return self.is_healthy
        except Exception as e:
            logger.debug(f"Health check failed for {self.hostname}: {e}")
            self.is_healthy = False
            self.last_health_check = datetime.now()
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get detailed status from VLLM server"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/status")
                if response.status_code == 200:
                    return response.json()
                return {}
        except Exception:
            return {}
    
    async def generate_completion(self, prompt: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Generate completion using this VLLM server"""
        if not self.is_healthy or self.is_updating:
            raise Exception(f"Server {self.hostname} is not available")
        
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                # Use the direct generate endpoint from our VLLM wrapper
                response = await client.post(
                    f"{self.base_url}/generate",
                    json={
                        "prompt": prompt,
                        "n": kwargs.get("n", 1),
                        "temperature": kwargs.get("temperature", 0.8),
                        "max_tokens": kwargs.get("max_tokens", 512),
                        "top_p": kwargs.get("top_p", 1.0),
                        "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                        "presence_penalty": kwargs.get("presence_penalty", 0.0),
                        "stop": kwargs.get("stop"),
                        "stream": False  # Always false for this use case
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract completions from response
                if isinstance(prompt, str):
                    # Single prompt response
                    completions = [choice["text"] for choice in result.get("choices", [])]
                    return {
                        "completions": completions,
                        "model": result.get("model")
                    }
                else:
                    # Multiple prompts response
                    all_completions = []
                    for res in result.get("results", []):
                        completions = [choice["text"] for choice in res.get("choices", [])]
                        all_completions.extend(completions)
                    return {
                        "completions": all_completions,
                        "model": result.get("results", [{}])[0].get("model") if result.get("results") else None
                    }
                
        except httpx.TimeoutException:
            logger.error(f"Request timeout for {self.hostname}")
            self.is_healthy = False
            raise Exception(f"Request to {self.hostname} timed out")
        except Exception as e:
            logger.error(f"Completion request failed for {self.hostname}: {e}")
            # Mark as unhealthy on failure
            self.is_healthy = False
            raise
    
    async def update_model(self, model_path: str) -> bool:
        """Update the model on this VLLM server"""
        logger.info(f"Starting model update for {self.hostname} with path: {model_path}")
        self.is_updating = True
        
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:  # 10 minute timeout for model updates
                response = await client.post(
                    f"{self.base_url}/update_model_params",
                    json={"model_path": model_path}
                )
                response.raise_for_status()
                result = response.json()
                
                if result.get("status") == "success":
                    logger.info(f"Model update initiated for {self.hostname}")
                else:
                    logger.error(f"Model update failed for {self.hostname}: {result}")
                    self.is_updating = False
                    return False
            
            # Wait for server to come back online with new model
            logger.info(f"Waiting for {self.hostname} to come back online with new model...")
            for attempt in range(60):  # Wait up to 5 minutes
                await asyncio.sleep(5)
                if await self.health_check():
                    # Verify the model was actually updated
                    status = await self.get_status()
                    if status.get("model_path") == model_path:
                        logger.info(f"Server {self.hostname} successfully updated to {model_path}")
                        self.is_updating = False
                        return True
            
            logger.error(f"Server {self.hostname} did not come back online with new model")
            self.is_updating = False
            return False
            
        except Exception as e:
            logger.error(f"Model update failed for {self.hostname}: {e}")
            self.is_updating = False
            return False

class CompletionServerManager:
    """Manages multiple VLLM servers with load balancing"""
    
    def __init__(self, hostnames: List[str]):
        self.servers = [VLLMServer(hostname) for hostname in hostnames]
        self.round_robin_iterator = itertools.cycle(self.servers)
        self.health_check_task = None
        self.request_counter = 0
        self.server_request_counts = {server.hostname: 0 for server in self.servers}
    
    async def start_health_monitoring(self):
        """Start periodic health checking"""
        self.health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop_health_monitoring(self):
        """Stop health monitoring"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
    
    async def _health_check_loop(self):
        """Periodically check health of all servers"""
        while True:
            try:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
                health_tasks = [server.health_check() for server in self.servers]
                await asyncio.gather(*health_tasks, return_exceptions=True)
                
                healthy_count = sum(1 for server in self.servers if server.is_healthy and not server.is_updating)
                logger.debug(f"Health check: {healthy_count}/{len(self.servers)} servers available")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    def get_available_server(self) -> Optional[VLLMServer]:
        """Get next available server using round-robin"""
        available_servers = [s for s in self.servers if s.is_healthy and not s.is_updating]
        
        if not available_servers:
            return None
        
        # Simple round-robin among available servers
        for _ in range(len(self.servers)):
            server = next(self.round_robin_iterator)
            if server.is_healthy and not server.is_updating:
                return server
        
        return None
    
    def _build_prompt_from_trajectory(self, trajectory: Trajectory, instruction: str) -> str:
        """Build a prompt from trajectory"""
        if not trajectory.turns:
            return f"{instruction}\n\nGenerate Python code to solve this problem:\n```python"
        
        prompt_parts = [instruction, "\n\nConversation history:"]
        
        for turn in trajectory.turns:
            prompt_parts.append(f"\nStep {turn.step}:")
            prompt_parts.append(f"Request: {turn.prompt}")
            if turn.code.strip():
                prompt_parts.append(f"Code:\n```python\n{turn.code}\n```")
            if turn.execution_output.strip():
                status = "✓" if turn.execution_success else "✗"
                prompt_parts.append(f"Execution {status}: {turn.execution_output}")
        
        prompt_parts.append(f"\nNext step - Generate the next code snippet:")
        prompt_parts.append("```python")
        
        return "\n".join(prompt_parts)
    
    async def generate_completion(self, request: GenerateCompletionRequest) -> GenerateCompletionResponse:
        """Generate completion using available servers"""
        server = self.get_available_server()
        if not server:
            raise HTTPException(status_code=503, detail="No available VLLM servers")
        
        # Build prompt from trajectory
        prompt = self._build_prompt_from_trajectory(request.trajectory, request.instruction)
        
        try:
            # Track request count
            self.request_counter += 1
            self.server_request_counts[server.hostname] += 1
            
            result = await server.generate_completion(
                prompt=prompt,
                n=request.n,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop or ["```", "\nStep", "\nRequest:", "\nExecution"]
            )
            
            return GenerateCompletionResponse(
                completions=result["completions"],
                server_used=server.hostname,
                model_used=result.get("model")
            )
            
        except Exception as e:
            logger.error(f"Completion generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Completion generation failed: {str(e)}")
    
    async def update_all_models(self, model_path: str) -> Dict[str, bool]:
        """Update model on all servers sequentially to maintain availability"""
        results = {}
        
        # Check if we have enough servers to maintain availability
        available_count = sum(1 for s in self.servers if s.is_healthy and not s.is_updating)
        if available_count <= 1 and len(self.servers) > 1:
            logger.warning("Only one server available, updating all servers may cause downtime")
        
        for server in self.servers:
            logger.info(f"Updating model on server {server.hostname}")
            success = await server.update_model(model_path)
            results[server.hostname] = success
            
            if not success:
                logger.error(f"Failed to update model on {server.hostname}")
            else:
                logger.info(f"Successfully updated model on {server.hostname}")
            
            # Small delay between updates to avoid overwhelming the system
            await asyncio.sleep(2)
        
        return results

# Global manager
completion_manager: Optional[CompletionServerManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global completion_manager
    completion_manager = CompletionServerManager(VLLM_HOSTNAMES)
    await completion_manager.start_health_monitoring()
    
    # Initial health check
    health_tasks = [server.health_check() for server in completion_manager.servers]
    await asyncio.gather(*health_tasks, return_exceptions=True)
    
    healthy_count = sum(1 for s in completion_manager.servers if s.is_healthy)
    logger.info(f"Completion server started with {healthy_count}/{len(VLLM_HOSTNAMES)} healthy VLLM servers")
    
    yield
    
    await completion_manager.stop_health_monitoring()
    logger.info("Completion server shutdown complete")

app = FastAPI(title="Completion Server", lifespan=lifespan)

# API Endpoints

@app.post("/generate", response_model=GenerateCompletionResponse)
async def generate_completion(request: GenerateCompletionRequest):
    """Generate code completions from trajectory"""
    if not completion_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return await completion_manager.generate_completion(request)

@app.post("/update_model_params")
async def update_model_params(request: UpdateModelRequest):
    """Update model parameters on all VLLM servers"""
    if not completion_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    results = await completion_manager.update_all_models(request.model_path)
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    return {
        "status": "completed",
        "updated_servers": success_count,
        "total_servers": total_count,
        "results": results,
        "success": success_count == total_count,
        "model_path": request.model_path
    }

@app.get("/health")
async def health_check():
    """Health check for the completion server"""
    if not completion_manager:
        return {"status": "starting"}
    
    server_status = {}
    for server in completion_manager.servers:
        status_info = {
            "healthy": server.is_healthy,
            "updating": server.is_updating,
            "last_check": server.last_health_check.isoformat() if server.last_health_check else None,
            "model": server.model_path
        }
        server_status[server.hostname] = status_info
    
    available_count = sum(1 for server in completion_manager.servers 
                         if server.is_healthy and not server.is_updating)
    
    return {
        "status": "healthy" if available_count > 0 else "unhealthy",
        "available_servers": available_count,
        "total_servers": len(completion_manager.servers),
        "servers": server_status
    }

@app.get("/stats")
async def get_stats():
    """Get completion server statistics"""
    if not completion_manager:
        return {"error": "Manager not initialized"}
    
    # Get detailed status for each server
    server_details = []
    for server in completion_manager.servers:
        detail = {
            "hostname": server.hostname,
            "healthy": server.is_healthy,
            "updating": server.is_updating,
            "last_health_check": server.last_health_check.isoformat() if server.last_health_check else None,
            "model": server.model_path,
            "request_count": completion_manager.server_request_counts.get(server.hostname, 0)
        }
        server_details.append(detail)
    
    return {
        "total_servers": len(completion_manager.servers),
        "healthy_servers": sum(1 for s in completion_manager.servers if s.is_healthy),
        "updating_servers": sum(1 for s in completion_manager.servers if s.is_updating),
        "available_servers": sum(1 for s in completion_manager.servers 
                                if s.is_healthy and not s.is_updating),
        "total_requests": completion_manager.request_counter,
        "server_details": server_details
    }

@app.get("/models")
async def list_models():
    """List models currently loaded on each server"""
    if not completion_manager:
        return {"error": "Manager not initialized"}
    
    models = {}
    for server in completion_manager.servers:
        if server.is_healthy and server.model_path:
            models[server.hostname] = server.model_path
    
    return {
        "models": models,
        "unique_models": list(set(models.values()))
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Completion Server with VLLM Load Balancing")
    parser.add_argument("--vllm-servers", nargs="+", required=True,
                       help="List of VLLM server hostnames (e.g., localhost:8000 localhost:8001)")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host to bind the service")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind the service")
    parser.add_argument("--health-check-interval", type=int, default=10,
                       help="Health check interval in seconds")
    parser.add_argument("--request-timeout", type=int, default=300,
                       help="Request timeout in seconds")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set global configuration
    VLLM_HOSTNAMES = args.vllm_servers
    HEALTH_CHECK_INTERVAL = args.health_check_interval
    REQUEST_TIMEOUT = args.request_timeout
    
    logger.info(f"Starting Completion Server")
    logger.info(f"VLLM Servers: {VLLM_HOSTNAMES}")
    logger.info(f"Health check interval: {HEALTH_CHECK_INTERVAL}s")
    logger.info(f"Request timeout: {REQUEST_TIMEOUT}s")
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)