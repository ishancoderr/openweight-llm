"""FastAPI service: Qwen3-8B with and without the LoRA adapter, behind an X-API-Key check."""

import os
import secrets
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.llm import LoadedModel, generate, load_models

API_KEY = os.environ.get("API_KEY", "")

# Filled once at startup by the lifespan below: endpoint name -> loaded model.
MODELS: dict[str, LoadedModel] = {}


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    model: Literal["qwen8B", "lora-finetune"]
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODELS.update(load_models())
    yield
    MODELS.clear()


app = FastAPI(title="openweight-llm", lifespan=lifespan)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Return nothing if the X-API-Key header matches API_KEY, otherwise raise 401."""
    # Runs as a route dependency, so a wrong key is rejected before the body is validated.
    if not API_KEY or x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def run_chat(model_name: str, request: ChatRequest) -> ChatResponse:
    """Return the chosen model's answer to the request's messages."""
    messages = [message.model_dump() for message in request.messages]
    try:
        result = generate(MODELS[model_name], messages, request.max_tokens, request.temperature)
    except ValueError as error:
        # llama-cpp raises ValueError when prompt + max_tokens exceed CTX_SIZE.
        raise HTTPException(status_code=400, detail=str(error))
    return ChatResponse(model=model_name, **result)


# Each route is registered with and without the trailing slash, so neither form gets a redirect.
@app.post("/openWeightllm/qwen8B", dependencies=[Depends(require_api_key)])
@app.post("/openWeightllm/qwen8B/", dependencies=[Depends(require_api_key)], include_in_schema=False)
def chat_qwen8b(request: ChatRequest) -> ChatResponse:
    """Return Qwen3-8B's answer (no adapter)."""
    return run_chat("qwen8B", request)


@app.post("/openWeightllm/qwen8B/lora-finetune", dependencies=[Depends(require_api_key)])
@app.post("/openWeightllm/qwen8B/lora-finetune/", dependencies=[Depends(require_api_key)], include_in_schema=False)
def chat_lora_finetune(request: ChatRequest) -> ChatResponse:
    """Return the answer of Qwen3-8B with the fine-tuned LoRA adapter."""
    return run_chat("lora-finetune", request)


@app.get("/health")
@app.get("/health/", include_in_schema=False)
def health() -> dict:
    """Return the service status and the names of the loaded models."""
    return {"status": "ok", "models_loaded": list(MODELS)}
