"""Routes, auth, validation and prompt format of the FastAPI service (fake llama_cpp, no GPU)."""

import os

import jinja2.ext
import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from app.llm import build_prompt
from app.main import MODELS
from tests.conftest import API_HEADERS, PARSER_MESSAGES

CHAT_PATHS = [
    ("/openWeightllm/qwen8B", "qwen8B", '{"model": "base"}'),
    ("/openWeightllm/qwen8B/", "qwen8B", '{"model": "base"}'),
    ("/openWeightllm/qwen8B/lora-finetune", "lora-finetune", '{"model": "lora"}'),
    ("/openWeightllm/qwen8B/lora-finetune/", "lora-finetune", '{"model": "lora"}'),
]


def render_qwen3_template(messages: list[dict]) -> str:
    """Return the prompt transformers' apply_chat_template(..., enable_thinking=False) builds.

    qwen3_chat_template.jinja is the chat_template.jinja that notebook 02 saved next to the adapter.
    """
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=[jinja2.ext.loopcontrols])
    template_path = os.path.join(os.path.dirname(__file__), "qwen3_chat_template.jinja")
    with open(template_path, encoding="utf-8") as template_file:
        template = environment.from_string(template_file.read())
    return template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)


def test_both_models_loaded_on_gpu(client):
    base, lora = MODELS["qwen8B"].llm, MODELS["lora-finetune"].llm
    assert base.lora_path is None
    assert lora.lora_path.endswith("adapter.gguf")
    assert base.model_path == lora.model_path
    assert base.settings["n_gpu_layers"] == -1 and lora.settings["n_gpu_layers"] == -1


@pytest.mark.parametrize("path, model_name, expected_content", CHAT_PATHS)
def test_chat_endpoints(client, path, model_name, expected_content):
    response = client.post(path, json={"messages": PARSER_MESSAGES}, headers=API_HEADERS,
                           follow_redirects=False)
    assert response.status_code == 200
    answer = response.json()
    assert answer["model"] == model_name
    assert answer["content"] == expected_content  # <think> block removed
    assert (answer["prompt_tokens"], answer["completion_tokens"], answer["total_tokens"]) == (11, 7, 18)
    assert isinstance(answer["latency_ms"], int)


@pytest.mark.parametrize("path", ["/openWeightllm/qwen8B", "/openWeightllm/qwen8B/lora-finetune/"])
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong-key"}])
def test_wrong_or_missing_key_is_401(client, path, headers):
    response = client.post(path, json={"messages": PARSER_MESSAGES}, headers=headers)
    assert response.status_code == 401


def test_key_is_checked_before_the_body(client):
    response = client.post("/openWeightllm/qwen8B", json={"messages": []})
    assert response.status_code == 401


@pytest.mark.parametrize("body", [
    {"messages": []},
    {"messages": [{"role": "tool", "content": "x"}]},
    {"messages": PARSER_MESSAGES, "temperature": -1},
])
def test_invalid_body_is_422(client, body):
    response = client.post("/openWeightllm/qwen8B/lora-finetune", json=body, headers=API_HEADERS)
    assert response.status_code == 422


def test_prompt_longer_than_context_is_400(client):
    response = client.post("/openWeightllm/qwen8B", json={"messages": PARSER_MESSAGES, "max_tokens": 5000},
                           headers=API_HEADERS)
    assert response.status_code == 400


@pytest.mark.parametrize("path", ["/health", "/health/"])
def test_health(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models_loaded": ["qwen8B", "lora-finetune"]}


@pytest.mark.parametrize("messages", [
    PARSER_MESSAGES,
    [{"role": "user", "content": "only a user turn"}],
    [{"role": "system", "content": "S"}, {"role": "user", "content": "q1"},
     {"role": "assistant", "content": '{"a": 1}'}, {"role": "user", "content": "q2\nsecond line"}],
])
def test_prompt_matches_training_template(messages):
    assert build_prompt(messages) == render_qwen3_template(messages)


def test_server_sends_training_prompt(client):
    client.post("/openWeightllm/qwen8B/lora-finetune", json={"messages": PARSER_MESSAGES}, headers=API_HEADERS)
    lora = MODELS["lora-finetune"].llm
    assert lora.last_prompt == render_qwen3_template(PARSER_MESSAGES)
    assert lora.last_prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert lora.last_stop == ["<|im_end|>"]
