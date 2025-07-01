"""
VLLM Server Wrapper
Wraps VLLM with direct LLM() integration and model updating capability
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import asyncio
import logging
import argparse
from typing import Optional, Dict, Any, List, Union
from contextlib import asynccontextmanager
import json
from datetime import datetime
import time
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
VLLM_MODEL_PATH = None
TENSOR_PARALLEL_SIZE = 1
WRAPPER_PORT = None

class UpdateModelRequest(BaseModel):
    model_path: str

from typing import Union, List, Optional
from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: Union[str, List[str]]
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    logprobs: Optional[int] = None


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1

class VLLMManager:
    """Manages the VLLM LLM instance"""
    
    def __init__(self, model_path: str, tensor_parallel_size: int):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.llm: Optional[LLM] = None
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> bool:
        """Initialize the VLLM LLM instance"""
        try:
            logger.info(f"Initializing VLLM with model: {self.model_path}")
            
            # Initialize LLM with the specified configuration
            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True,  # Enable for custom models
                dtype="auto",  # Auto-detect dtype
                gpu_memory_utilization=0.95,  # Use most of GPU memory
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
        # This is a simple format - you may want to customize based on your model
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

# Global VLLM manager
vllm_manager: Optional[VLLMManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global vllm_manager
    
    vllm_manager = VLLMManager(
        model_path=VLLM_MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE
    )
    
    # Initialize VLLM
    if await vllm_manager.initialize():
        logger.info("VLLM wrapper started successfully")
    else:
        logger.error("Failed to initialize VLLM")
        raise RuntimeError("Failed to initialize VLLM")
    
    yield
    
    # Cleanup on shutdown
    if vllm_manager:
        await vllm_manager.shutdown()
    logger.info("VLLM wrapper shutdown complete")

app = FastAPI(title="VLLM Server Wrapper", lifespan=lifespan)

# API Endpoints

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate completions for the given prompt(s)"""
    if not vllm_manager:
        raise HTTPException(status_code=500, detail="VLLM not initialized")
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        logprobs=request.logprobs,
    )
    
    try:
        # Generate completions
        outputs = await vllm_manager.generate(request.prompt, sampling_params)
        
        # Format response
        if isinstance(request.prompt, str):
            # Single prompt
            output = outputs[0]
            return {
                "id": f"gen-{int(time.time() * 1000)}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": vllm_manager.model_path,
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
                    "model": vllm_manager.model_path,
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

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    if not vllm_manager:
        raise HTTPException(status_code=500, detail="VLLM not initialized")
    
    # Convert chat messages to prompt
    prompt = vllm_manager.format_chat_prompt(request.messages)
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        stop=request.stop,
        n=request.n,
        seed=request.seed,
    )
    
    try:
        # Generate completions
        outputs = await vllm_manager.generate(prompt, sampling_params)
        output = outputs[0]
        
        # Format response in OpenAI format
        response = {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model or vllm_manager.model_path,
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

@app.post("/v1/completions")
async def completions(request: GenerateRequest):
    """OpenAI-compatible completions endpoint"""
    # This just wraps the generate endpoint with OpenAI response format
    return await generate(request)

@app.post("/update_model_params")
async def update_model_params(request: UpdateModelRequest):
    """Update the model parameters by loading a new model"""
    if not vllm_manager:
        raise HTTPException(status_code=500, detail="VLLM not initialized")
    
    logger.info(f"Updating model to: {request.model_path}")
    
    success = await vllm_manager.update_model(request.model_path)
    
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
    """Health check endpoint (returns 200 only if healthy)"""
    if vllm_manager and vllm_manager.llm:
        return {
            "status": "healthy",
            "model": vllm_manager.model_path,
            "tensor_parallel_size": vllm_manager.tensor_parallel_size
        }
    else:
        raise HTTPException(status_code=503, detail="VLLM not initialized or not healthy")

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    if not vllm_manager:
        raise HTTPException(status_code=503, detail="VLLM not initialized")
    
    return {
        "data": [
            {
                "id": vllm_manager.model_path,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllm",
            }
        ]
    }

@app.get("/status")
async def status():
    """Get detailed status information"""
    if not vllm_manager:
        return {"status": "not_initialized"}
    
    return {
        "status": "running",
        "model_path": vllm_manager.model_path,
        "tensor_parallel_size": vllm_manager.tensor_parallel_size,
        "llm_initialized": vllm_manager.llm is not None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="VLLM Server Wrapper")
    parser.add_argument("--model-path", required=True,
                       help="Path to the model to load")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Tensor parallel size for VLLM")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host for the wrapper service")
    parser.add_argument("--port", type=int, required=True,
                       help="Port for the wrapper service")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set global configuration
    VLLM_MODEL_PATH = args.model_path
    TENSOR_PARALLEL_SIZE = args.tensor_parallel_size
    WRAPPER_PORT = args.port
    
    logger.info(f"Starting VLLM Wrapper")
    logger.info(f"Model path: {VLLM_MODEL_PATH}")
    logger.info(f"Wrapper address: {args.host}:{WRAPPER_PORT}")
    logger.info(f"Tensor parallel size: {TENSOR_PARALLEL_SIZE}")
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=WRAPPER_PORT)