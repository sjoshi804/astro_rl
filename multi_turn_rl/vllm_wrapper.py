"""
VLLM Wrapper class for multi-turn RL
Object-oriented wrapper for VLLM with direct LLM() integration and model updating capability
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import logging
from typing import Optional, Dict, Any, List, Union
from contextlib import asynccontextmanager
import time
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
import torch

# Configure logging
logger = logging.getLogger(__name__)


class UpdateModelRequest(BaseModel):
    model_path: str


class GenerateRequest(BaseModel):
    prompt: Union[str, List[str]]
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int = -1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    n: int = 1
    logprobs: Optional[int] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int = -1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    n: int = 1
    logprobs: Optional[int] = None


class VLLMWrapper:
    """
    Object-oriented VLLM wrapper for multi-turn RL
    Manages the VLLM LLM instance and provides FastAPI endpoints
    """
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.llm: Optional[LLM] = None
        self._lock = asyncio.Lock()
        self.app = None
        self._setup_app()
    
    def _setup_app(self):
        """Setup FastAPI application with endpoints"""
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Application lifespan management"""
            # Initialize VLLM
            if await self.initialize():
                logger.info("VLLM wrapper started successfully")
            else:
                logger.error("Failed to initialize VLLM")
                raise RuntimeError("Failed to initialize VLLM")
            
            yield
            
            # Cleanup on shutdown
            await self.shutdown()
            logger.info("VLLM wrapper shutdown complete")
        
        self.app = FastAPI(title="VLLM Server Wrapper", lifespan=lifespan)
        
        # Add endpoints
        self.app.post("/generate")(self.generate_endpoint)
        self.app.post("/v1/chat/completions")(self.chat_completions_endpoint)
        self.app.post("/v1/completions")(self.completions_endpoint)
        self.app.post("/update_model_params")(self.update_model_params_endpoint)
        self.app.get("/health")(self.health_check)
        self.app.get("/v1/models")(self.list_models)
        self.app.get("/status")(self.status)
    
    async def initialize(self) -> bool:
        """Initialize the VLLM LLM instance"""
        try:
            logger.info(f"Initializing VLLM with model: {self.model_path}")
            
            # Initialize LLM with the specified configuration
            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True,
                dtype="auto",
                gpu_memory_utilization=0.95,
            )
            
            logger.info("VLLM LLM initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize VLLM: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the VLLM instance"""
        if self.llm:
            logger.info("Shutting down VLLM")
            # Clean up GPU memory
            del self.llm
            self.llm = None
            torch.cuda.empty_cache()
    
    async def update_model(self, model_path: str) -> bool:
        """Update to a new model"""
        async with self._lock:
            logger.info(f"Updating model from {self.model_path} to {model_path}")
            
            # Shutdown current model
            await self.shutdown()
            
            # Update model path
            self.model_path = model_path
            
            # Initialize with new model
            return await self.initialize()
    
    async def generate(self, prompt: Union[str, List[str]], sampling_params: SamplingParams) -> List[RequestOutput]:
        """Generate completions for the given prompt(s)"""
        if not self.llm:
            raise RuntimeError("VLLM not initialized")
        
        # Run generation in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.llm.generate,
            prompt,
            sampling_params
        )
    
    def format_chat_prompt(self, messages: List[ChatMessage]) -> str:
        """Format chat messages into a prompt string"""
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"System: {message.content}\n"
            elif message.role == "user":
                prompt += f"User: {message.content}\n"
            elif message.role == "assistant":
                prompt += f"Assistant: {message.content}\n"
        
        # Add the assistant prompt prefix
        prompt += "Assistant: "
        return prompt
    
    # FastAPI endpoint methods
    
    async def generate_endpoint(self, request: GenerateRequest):
        """Generate completions for the given prompt(s)"""
        if not self.llm:
            raise HTTPException(status_code=500, detail="VLLM not initialized")
        
        # Validate request prompt
        if not request.prompt or (isinstance(request.prompt, str) and not request.prompt.strip()):
            raise HTTPException(status_code=400, detail="Empty prompt provided")
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            n=request.n,
            logprobs=request.logprobs,
        )
        
        try:
            # Generate completions
            outputs = await self.generate(request.prompt, sampling_params)
            
            # Validate outputs is not empty
            if not outputs:
                logger.error("VLLM generated empty outputs list")
                raise HTTPException(status_code=500, detail="Generation failed: empty outputs")
            
            # Format response
            if isinstance(request.prompt, str):
                # Single prompt
                output = outputs[0]
                return {
                    "id": f"gen-{int(time.time() * 1000)}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": self.model_path,
                    "choices": [
                        {
                            "text": completion.text,
                            "index": i,
                            "logprobs": completion.logprobs if request.logprobs else None,
                            "finish_reason": completion.finish_reason,
                        }
                        for i, completion in enumerate(output.outputs)
                    ],
                    "usage": {
                        "prompt_tokens": len(output.prompt_token_ids),
                        "completion_tokens": sum(len(c.token_ids) for c in output.outputs),
                        "total_tokens": len(output.prompt_token_ids) + sum(len(c.token_ids) for c in output.outputs),
                    }
                }
            else:
                # Multiple prompts
                results = []
                for output in outputs:
                    results.append({
                        "id": f"gen-{int(time.time() * 1000)}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": self.model_path,
                        "choices": [
                            {
                                "text": completion.text,
                                "index": i,
                                "logprobs": completion.logprobs if request.logprobs else None,
                                "finish_reason": completion.finish_reason,
                            }
                            for i, completion in enumerate(output.outputs)
                        ],
                        "usage": {
                            "prompt_tokens": len(output.prompt_token_ids),
                            "completion_tokens": sum(len(c.token_ids) for c in output.outputs),
                            "total_tokens": len(output.prompt_token_ids) + sum(len(c.token_ids) for c in output.outputs),
                        }
                    })
                return {"results": results}
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    async def chat_completions_endpoint(self, request: ChatCompletionRequest):
        """OpenAI-compatible chat completions endpoint"""
        if not self.llm:
            raise HTTPException(status_code=500, detail="VLLM not initialized")
        
        # Convert chat messages to prompt
        prompt = self.format_chat_prompt(request.messages)
        
        # Validate prompt is not empty
        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="Empty prompt generated from messages")
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            n=request.n,
            logprobs=request.logprobs,
        )
        
        try:
            # Generate completions
            outputs = await self.generate(prompt, sampling_params)
            
            # Validate outputs is not empty
            if not outputs:
                logger.error("VLLM generated empty outputs list for chat completion")
                raise HTTPException(status_code=500, detail="Chat completion failed: empty outputs")
            
            output = outputs[0]
            
            # Format response in OpenAI format
            response = {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model or self.model_path,
                "choices": [
                    {
                        "index": i,
                        "message": {
                            "role": "assistant",
                            "content": completion.text,
                        },
                        "finish_reason": completion.finish_reason,
                    }
                    for i, completion in enumerate(output.outputs)
                ],
                "usage": {
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": sum(len(c.token_ids) for c in output.outputs),
                    "total_tokens": len(output.prompt_token_ids) + sum(len(c.token_ids) for c in output.outputs),
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

    async def completions_endpoint(self, request: GenerateRequest):
        """OpenAI-compatible completions endpoint"""
        return await self.generate_endpoint(request)

    async def update_model_params_endpoint(self, request: UpdateModelRequest):
        """Update the model parameters by loading a new model"""
        if not self.llm:
            raise HTTPException(status_code=500, detail="VLLM not initialized")
        
        logger.info(f"Updating model to: {request.model_path}")
        
        success = await self.update_model(request.model_path)
        
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

    async def health_check(self):
        """Health check endpoint (returns 200 only if healthy)"""
        if self.llm:
            return {
                "status": "healthy",
                "model": self.model_path,
                "tensor_parallel_size": self.tensor_parallel_size
            }
        else:
            raise HTTPException(status_code=503, detail="VLLM not initialized or not healthy")

    async def list_models(self):
        """List available models (OpenAI-compatible)"""
        if not self.llm:
            raise HTTPException(status_code=503, detail="VLLM not initialized")
        
        return {
            "data": [
                {
                    "id": self.model_path,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "vllm",
                }
            ]
        }

    async def status(self):
        """Get detailed status information"""
        return {
            "status": "running" if self.llm else "not_initialized",
            "model_path": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "llm_initialized": self.llm is not None,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the VLLM wrapper server"""
        import uvicorn
        logger.info(f"Starting VLLM Wrapper at {host}:{port}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Tensor parallel size: {self.tensor_parallel_size}")
        uvicorn.run(self.app, host=host, port=port)