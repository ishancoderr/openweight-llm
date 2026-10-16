"""Load Qwen3-8B (GGUF) with and without the LoRA adapter and run chat generations."""

import os
import re
import threading
import time
from dataclasses import dataclass

from llama_cpp import Llama

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
BASE_GGUF = os.environ.get("BASE_GGUF", "Qwen3-8B-Q4_K_M.gguf")
ADAPTER_GGUF = os.environ.get("ADAPTER_GGUF", "adapter.gguf")
GPU_LAYERS = int(os.environ.get("GPU_LAYERS", "-1"))
THREADS = int(os.environ.get("THREADS", "4"))
CTX_SIZE = int(os.environ.get("CTX_SIZE", "4096"))
DISABLE_THINKING = os.environ.get("DISABLE_THINKING", "true").lower() in ("1", "true", "yes")

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class LoadedModel:
    name: str
    llm: Llama
    lock: threading.Lock  # a Llama object can only run one generation at a time


def load_model(name: str, lora_file: str | None) -> LoadedModel:
    """Return one Qwen3-8B instance fully on the GPU, with the LoRA adapter applied if given."""
    base_path = os.path.join(MODELS_DIR, BASE_GGUF)
    lora_path = os.path.join(MODELS_DIR, lora_file) if lora_file else None
    for path in (base_path, lora_path):
        if path and not os.path.isfile(path):
            raise FileNotFoundError(f"{path} not found - run ./prepare_model.sh first")

    print(f"Loading {name}: {base_path}" + (f" + {lora_path}" if lora_path else ""), flush=True)
    llm = Llama(
        model_path=base_path,
        lora_path=lora_path,
        n_gpu_layers=GPU_LAYERS,
        n_threads=THREADS,
        n_ctx=CTX_SIZE,
        chat_format="chatml",
    )
    return LoadedModel(name=name, llm=llm, lock=threading.Lock())


def load_models() -> dict[str, LoadedModel]:
    """Return both models keyed by endpoint name: plain Qwen3-8B and Qwen3-8B + LoRA."""
    # Two separate copies of the 4-bit weights (~5 GB each) plus their KV caches fit in the T4's 16 GB.
    return {
        "qwen8B": load_model("qwen8B", lora_file=None),
        "lora-finetune": load_model("lora-finetune", lora_file=ADAPTER_GGUF),
    }


def add_no_think(messages: list[dict]) -> list[dict]:
    """Return a copy of the messages with Qwen3's " /no_think" switch appended to the last one."""
    prompt_messages = [dict(message) for message in messages]
    prompt_messages[-1]["content"] += " /no_think"
    return prompt_messages


def strip_think(content: str) -> str:
    """Return the answer with any <think>...</think> block removed."""
    return THINK_BLOCK.sub("", content).strip()


def generate(model: LoadedModel, messages: list[dict], max_tokens: int, temperature: float) -> dict:
    """Return the answer text, prompt/completion/total token counts and generation time in ms."""
    prompt_messages = add_no_think(messages) if DISABLE_THINKING else messages

    # Timed inside the lock, so time spent queueing behind another request is not counted.
    with model.lock:
        started = time.perf_counter()
        completion = model.llm.create_chat_completion(
            messages=prompt_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

    content = completion["choices"][0]["message"]["content"] or ""
    if DISABLE_THINKING:
        content = strip_think(content)
    usage = completion["usage"]
    return {
        "content": content,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "latency_ms": latency_ms,
    }
