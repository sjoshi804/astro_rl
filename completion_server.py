"""
Completion Server
Load balances requests across multiple VLLM servers with model updating capability
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
import asyncio
import logging
import argparse
from datetime import datetime
import itertools
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
VLLM_HOSTNAMES = []
HEALTH_CHECK_INTERVAL = 10  # seconds
REQUEST_TIMEOUT = 60  # seconds

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

class GenerateCompletionResponse(BaseModel):
    completions: List[str]
    server_used: str

class UpdateModelRequest(BaseModel):
    model_path: str

class VLLMServer:
    """Represents a single VLLM server instance"""
    
    def __init__(self, hostname: str):
        self.hostname = hostname
        self.is_healthy = False
        self.is_updating = False
        self.last_health_check = None
        self.base_url = f"http://{hostname}"
    
    async def health_check(self) -> bool:
        """Check if the VLLM server is healthy"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                self.is_healthy = response.status_code == 200
                self.last_health_check = datetime.now()
                return self.is_healthy
        except Exception as e:
            logger.debug(f"Health check failed for {self.hostname}: {e}")
            self.is_healthy = False
            self.last_health_check = datetime.now()
            return False
    
    async def generate_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion using this VLLM server"""
        if not self.is_healthy or self.is_updating:
            raise Exception(f"Server {self.hostname} is not available")
        
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                # Newer VLLM API: use chat completions only
                chat_messages = [
                    {"role": "user", "content": prompt}
                ]
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": "current",
                        "messages": chat_messages,
                        "n": kwargs.get("n", 1),
                        "temperature": kwargs.get("temperature", 0.8),
                        "max_tokens": kwargs.get("max_tokens", 512),
                        "stop": kwargs.get("stop", [])
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract completions from chat response
                completions = [choice["message"]["content"] for choice in result.get("choices", [])]
                return {"completions": completions}
                
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
            async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for model updates
                response = await client.post(
                    f"{self.base_url}/update_model_params",
                    json={"model_path": model_path}
                )
                response.raise_for_status()
            
            # Wait for server to come back online
            logger.info(f"Waiting for {self.hostname} to come back online...")
            for attempt in range(60):  # Wait up to 5 minutes
                await asyncio.sleep(5)
                if await self.health_check():
                    logger.info(f"Server {self.hostname} is back online after model update")
                    self.is_updating = False
                    return True
            
            logger.error(f"Server {self.hostname} did not come back online after model update")
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
        """Build a conversation-style prompt from trajectory"""
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
            result = await server.generate_completion(
                prompt=prompt,
                n=request.n,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop or ["```", "\nStep", "\nRequest:", "\nExecution"]
            )
            
            return GenerateCompletionResponse(
                completions=result["completions"],
                server_used=server.hostname
            )
            
        except Exception as e:
            logger.error(f"Completion generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Completion generation failed: {str(e)}")
    
    async def update_all_models(self, model_path: str) -> Dict[str, bool]:
        """Update model on all servers sequentially"""
        results = {}
        
        for server in self.servers:
            logger.info(f"Updating model on server {server.hostname}")
            success = await server.update_model(model_path)
            results[server.hostname] = success
            
            if not success:
                logger.error(f"Failed to update model on {server.hostname}")
            else:
                logger.info(f"Successfully updated model on {server.hostname}")
        
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
    
    logger.info(f"Completion server started with {len(VLLM_HOSTNAMES)} VLLM servers")
    yield
    
    await completion_manager.stop_health_monitoring()
    logger.info("Completion server shutdown complete")

app = FastAPI(title="Completion Server", lifespan=lifespan)

# API Endpoints

@app.post("/generate", response_model=GenerateCompletionResponse)
async def generate_completion(request: GenerateCompletionRequest):
    """Generate code completions from trajectory"""
    return await completion_manager.generate_completion(request)

@app.post("/update_model_params")
async def update_model_params(request: UpdateModelRequest):
    """Update model parameters on all VLLM servers"""
    results = await completion_manager.update_all_models(request.model_path)
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    return {
        "status": "completed",
        "updated_servers": success_count,
        "total_servers": total_count,
        "results": results,
        "success": success_count == total_count
    }

@app.get("/health")
async def health_check():
    """Health check for the completion server"""
    if not completion_manager:
        return {"status": "starting"}
    
    server_status = {
        server.hostname: {
            "healthy": server.is_healthy,
            "updating": server.is_updating,
            "last_check": server.last_health_check.isoformat() if server.last_health_check else None
        }
        for server in completion_manager.servers
    }
    
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
    
    return {
        "total_servers": len(completion_manager.servers),
        "healthy_servers": sum(1 for s in completion_manager.servers if s.is_healthy),
        "updating_servers": sum(1 for s in completion_manager.servers if s.is_updating),
        "available_servers": sum(1 for s in completion_manager.servers 
                                if s.is_healthy and not s.is_updating),
        "server_details": [
            {
                "hostname": server.hostname,
                "healthy": server.is_healthy,
                "updating": server.is_updating,
                "last_health_check": server.last_health_check.isoformat() if server.last_health_check else None
            }
            for server in completion_manager.servers
        ]
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
    parser.add_argument("--request-timeout", type=int, default=60,
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