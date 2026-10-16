"""One call_llm() for the agents' pipeline/: GPT-4o mini, Qwen3-8B, or Qwen3-8B + LoRA.

Copy this file into Agent-001/pipeline/ and Agent-002/pipeline/ and pick the model in .env:

    LLM_PROVIDER=openai          # gpt-4o-mini via the OpenAI SDK (needs OPENAI_API_KEY)
    LLM_PROVIDER=qwen8B          # POST QWEN_HOST/openWeightllm/qwen8B
    LLM_PROVIDER=lora-finetune   # POST QWEN_HOST/openWeightllm/qwen8B/lora-finetune
    QWEN_HOST=http://<ec2-public-ip>
    QWEN_API_KEY=<API_KEY from the server's .env>
"""

import os
import time

import httpx

OPENAI_MODEL = "gpt-4o-mini"
QWEN_ENDPOINTS = {
    "qwen8B": "/openWeightllm/qwen8B",
    "lora-finetune": "/openWeightllm/qwen8B/lora-finetune",
}
QWEN_TIMEOUT_SECONDS = 300


def call_openai(messages: list[dict], max_tokens: int) -> dict:
    """Return gpt-4o-mini's answer and token counts."""
    # Imported here so the Qwen providers work without the openai package installed.
    from openai import OpenAI

    completion = OpenAI().chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return {
        "model": OPENAI_MODEL,
        "content": completion.choices[0].message.content or "",
        "prompt_tokens": completion.usage.prompt_tokens,
        "completion_tokens": completion.usage.completion_tokens,
        "total_tokens": completion.usage.total_tokens,
    }


def call_qwen(provider: str, messages: list[dict], max_tokens: int) -> dict:
    """Return the Qwen server's answer and token counts for the given endpoint name."""
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
    }


def call_llm(messages: list[dict], max_tokens: int = 512) -> dict:
    """Return {model, content, prompt_tokens, completion_tokens, total_tokens, latency_ms} from LLM_PROVIDER.

    latency_ms is the round trip measured here, so it means the same thing for all three providers.
    """
    # Read on every call, so an evaluation script can switch providers between runs.
    provider = os.environ.get("LLM_PROVIDER", "openai")

    started = time.perf_counter()
    if provider == "openai":
        answer = call_openai(messages, max_tokens)
    elif provider in QWEN_ENDPOINTS:
        answer = call_qwen(provider, messages, max_tokens)
    else:
        raise ValueError(f"LLM_PROVIDER must be openai, qwen8B or lora-finetune, not {provider!r}")
    answer["latency_ms"] = int((time.perf_counter() - started) * 1000)
    return answer
