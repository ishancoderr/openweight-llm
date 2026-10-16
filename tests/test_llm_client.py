"""agent_side/llm_client.py: the same call_llm() result for openai, qwen8B and lora-finetune."""

import sys
from types import SimpleNamespace

import httpx
import pytest

from agent_side import llm_client
from tests.conftest import PARSER_MESSAGES

RESULT_KEYS = {"model", "content", "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms"}


@pytest.mark.parametrize("provider", ["qwen8B", "lora-finetune"])
def test_qwen_providers_call_the_server(client, monkeypatch, provider):
    sent_requests = []

    def post_to_test_client(url, headers, json, timeout):
        sent_requests.append({"url": url, "headers": headers, "timeout": timeout})
        return client.post(url.removeprefix("http://ec2.test"), headers=headers, json=json)

    monkeypatch.setattr(httpx, "post", post_to_test_client)
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("QWEN_HOST", "http://ec2.test/")
    monkeypatch.setenv("QWEN_API_KEY", "test-key")

    answer = llm_client.call_llm(PARSER_MESSAGES)

    assert set(answer) == RESULT_KEYS
    assert answer["model"] == provider
    assert sent_requests[0]["url"] == "http://ec2.test" + llm_client.QWEN_ENDPOINTS[provider]
    assert sent_requests[0]["headers"] == {"X-API-Key": "test-key"}
    assert sent_requests[0]["timeout"] == 300


def test_openai_provider(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=20, completion_tokens=5, total_tokens=25)
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"a": 1}'))],
                                 usage=usage)
    create = lambda **request: completion
    fake_openai = SimpleNamespace(OpenAI=lambda: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create))))
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    answer = llm_client.call_llm(PARSER_MESSAGES)

    assert set(answer) == RESULT_KEYS
    assert answer["model"] == "gpt-4o-mini"
    assert answer["content"] == '{"a": 1}'
    assert answer["total_tokens"] == 25


def test_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gpt5")
    with pytest.raises(ValueError):
        llm_client.call_llm(PARSER_MESSAGES)
