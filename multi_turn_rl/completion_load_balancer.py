"""
Completion Load Balancer for distributing requests across multiple VLLM servers
"""

import httpx
import asyncio
import logging
import itertools
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


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
        logger.info(f"🔍 COMPLETION_LOAD_BALANCER <- VLLMServer.generate_completion() called with prompt type: {type(prompt)}")
        if not self.is_healthy or self.is_updating:
            raise Exception(f"Server {self.hostname} is not available")
        
        try:
            async with httpx.AsyncClient(timeout=kwargs.get('request_timeout', 300)) as client:
                # Only support chat messages format
                if isinstance(prompt, list):
                    logger.info(f"🔍 COMPLETION_LOAD_BALANCER <- VLLMServer.generate_completion() using /v1/chat/completions")
                    response = await client.post(
                        f"{self.base_url}/v1/chat/completions",
                        json={
                            "messages": prompt,
                            "n": kwargs.get("n", 1),
                            "temperature": kwargs.get("temperature", 0.8),
                            "max_tokens": kwargs.get("max_tokens", 512),
                            "top_p": kwargs.get("top_p", 1.0),
                            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                            "presence_penalty": kwargs.get("presence_penalty", 0.0),
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    logger.info(f"🔍 COMPLETION_LOAD_BALANCER <- VLLMServer.generate_completion() response: {result}")
                    completions = [choice["message"]["content"] for choice in result.get("choices", [])]
                    return {
                        "completions": completions,
                        "model": result.get("model"),
                        "raw_response": result
                    }
                else:
                    raise Exception("Only chat messages format is supported")
                    
        except httpx.TimeoutException:
            logger.error(f"Request timeout for {self.hostname}")
            self.is_healthy = False
            raise Exception(f"Request to {self.hostname} timed out")
        except Exception as e:
            logger.error(f"Completion request failed for {self.hostname}: {e}")
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


class CompletionLoadBalancer:
    """Manages multiple VLLM servers with load balancing"""
    
    def __init__(
        self, 
        vllm_hostnames: List[str],
        health_check_interval: int = 10,
        request_timeout: int = 300
    ):
        self.servers = [VLLMServer(hostname) for hostname in vllm_hostnames]
        self.round_robin_iterator = itertools.cycle(self.servers)
        self.health_check_task = None
        self.request_counter = 0
        self.server_request_counts = {server.hostname: 0 for server in self.servers}
        self.health_check_interval = health_check_interval
        self.request_timeout = request_timeout
        
        logger.info(f"Initialized CompletionLoadBalancer with {len(self.servers)} servers")
    
    async def start_health_monitoring(self):
        """Start periodic health checking"""
        logger.info("Starting health monitoring")
        self.health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop_health_monitoring(self):
        """Stop health monitoring"""
        logger.info("Stopping health monitoring")
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
                await asyncio.sleep(self.health_check_interval)
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
    
    async def chat_completions(self, request: dict) -> dict:
        """
        OpenAI-compatible chat completions endpoint
        Forwards to VLLM servers
        """
        server = self.get_available_server()
        if not server:
            raise Exception("No available VLLM servers")
        
        try:
            messages = request.get("messages", [])
            result = await server.generate_completion(
                prompt=messages,
                n=request.get("n", 1),
                temperature=request.get("temperature", 0.8),
                max_tokens=request.get("max_tokens", 512),
                top_p=request.get("top_p", 1.0),
                frequency_penalty=request.get("frequency_penalty", 0.0),
                presence_penalty=request.get("presence_penalty", 0.0),
                request_timeout=self.request_timeout
            )
            
            # Update request tracking
            self.request_counter += 1
            self.server_request_counts[server.hostname] += 1
            
            # Return OpenAI-compatible response format
            choices = []
            for i, completion in enumerate(result["completions"]):
                choices.append({
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": completion
                    },
                    "finish_reason": "stop"
                })
            
            return {
                "choices": choices,
                "model": result.get("model", "unknown"),
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
        except Exception as e:
            logger.error(f"Chat completions failed: {e}")
            raise Exception(f"Chat completions failed: {str(e)}")
    
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
    
    async def initialize(self):
        """Initialize the load balancer and perform initial health checks"""
        logger.info("Initializing CompletionLoadBalancer...")
        
        # Start health monitoring
        await self.start_health_monitoring()
        
        # Initial health check
        health_tasks = [server.health_check() for server in self.servers]
        await asyncio.gather(*health_tasks, return_exceptions=True)
        
        healthy_count = sum(1 for s in self.servers if s.is_healthy)
        logger.info(f"CompletionLoadBalancer initialized with {healthy_count}/{len(self.servers)} healthy VLLM servers")
        
        return healthy_count > 0
    
    async def shutdown(self):
        """Shutdown the load balancer"""
        logger.info("Shutting down CompletionLoadBalancer...")
        await self.stop_health_monitoring()
        logger.info("CompletionLoadBalancer shutdown complete")
    
    def get_health_status(self) -> dict:
        """Get health status for the completion load balancer"""
        server_status = {}
        for server in self.servers:
            status_info = {
                "healthy": server.is_healthy,
                "updating": server.is_updating,
                "last_check": server.last_health_check.isoformat() if server.last_health_check else None,
                "model": server.model_path
            }
            server_status[server.hostname] = status_info
        
        available_count = sum(1 for server in self.servers 
                             if server.is_healthy and not server.is_updating)
        
        return {
            "status": "healthy" if available_count > 0 else "unhealthy",
            "available_servers": available_count,
            "total_servers": len(self.servers),
            "servers": server_status
        }
    
    def get_stats(self) -> dict:
        """Get completion load balancer statistics"""
        # Get detailed status for each server
        server_details = []
        for server in self.servers:
            detail = {
                "hostname": server.hostname,
                "healthy": server.is_healthy,
                "updating": server.is_updating,
                "last_health_check": server.last_health_check.isoformat() if server.last_health_check else None,
                "model": server.model_path,
                "request_count": self.server_request_counts.get(server.hostname, 0)
            }
            server_details.append(detail)
        
        return {
            "total_servers": len(self.servers),
            "healthy_servers": sum(1 for s in self.servers if s.is_healthy),
            "updating_servers": sum(1 for s in self.servers if s.is_updating),
            "available_servers": sum(1 for s in self.servers 
                                    if s.is_healthy and not s.is_updating),
            "total_requests": self.request_counter,
            "server_details": server_details
        }
    
    def list_models(self) -> dict:
        """List models currently loaded on each server"""
        models = {}
        for server in self.servers:
            if server.is_healthy and server.model_path:
                models[server.hostname] = server.model_path
        
        return {
            "models": models,
            "unique_models": list(set(models.values()))
        }