"""One call_llm() for the agents' pipeline/: Qwen3-8B or Qwen3-8B + LoRA on the openweight-llm server.

Copy this file into Agent-001/pipeline/ and Agent-002/pipeline/ and pick the model in .env:

    LLM_PROVIDER=lora-finetune   # POST QWEN_HOST/openWeightllm/qwen8B/lora-finetune (default)
    LLM_PROVIDER=qwen8B          # POST QWEN_HOST/openWeightllm/qwen8B
    QWEN_HOST=http://<ec2-public-ip>
    QWEN_API_KEY=<API_KEY from the server's .env>
"""

import os
import time

import httpx

QWEN_ENDPOINTS = {
    "qwen8B": "/openWeightllm/qwen8B",
    "lora-finetune": "/openWeightllm/qwen8B/lora-finetune",
}
QWEN_TIMEOUT_SECONDS = 300


def call_llm(messages: list[dict], max_tokens: int = 512) -> dict:
    """Return {model, content, prompt_tokens, completion_tokens, total_tokens, latency_ms} from LLM_PROVIDER.

    latency_ms is the round trip measured here (network + waiting + generation).
    """
    # Read on every call, so an evaluation script can switch models between runs.
    provider = os.environ.get("LLM_PROVIDER", "lora-finetune")
    if provider not in QWEN_ENDPOINTS:
        raise ValueError(f"LLM_PROVIDER must be qwen8B or lora-finetune, not {provider!r}")

    started = time.perf_counter()
    response = httpx.post(
        os.environ["QWEN_HOST"].rstrip("/") + QWEN_ENDPOINTS[provider],
        headers={"X-API-Key": os.environ["QWEN_API_KEY"]},
        json={"messages": messages, "max_tokens": max_tokens, "temperature": 0.0},
        timeout=QWEN_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    answer = response.json()
    return {
        "model": answer["model"],
        "content": answer["content"],
        "prompt_tokens": answer["prompt_tokens"],
        "completion_tokens": answer["completion_tokens"],
        "total_tokens": answer["total_tokens"],
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }
