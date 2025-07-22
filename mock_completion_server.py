import logging
from fastapi import FastAPI, Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock_completion_server")

app = FastAPI()

@app.post("/generate", response_model=GenerateCompletionResponse)
async def generate_completion(request: GenerateCompletionRequest, raw_request: Request):
    body = await raw_request.body()
    logger.info(f"/generate called with: {body.decode('utf-8')}")
    return GenerateCompletionResponse(
        completions=[f'print("I\'m a mock server")' for _ in range(request.n)],
        server_used="mock",
        model_used="mock-model"
    )

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    body = await raw_request.body()
    logger.info(f"/v1/chat/completions called with: {body.decode('utf-8')}")
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 0,
        "model": request.model or "mock-model",
        "choices": [
            {
                "index": i,
                "message": {
                    "role": "assistant",
                    "content": 'print("I\'m a mock server")',
                },
                "finish_reason": "stop",
            }
            for i in range(request.n)
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }
    } 