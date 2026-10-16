"""agent_side/llm_client.py: call_llm() for qwen8B and lora-finetune."""

import httpx
import pytest

from agent_side import llm_client
from tests.conftest import PARSER_MESSAGES

RESULT_KEYS = {"model", "content", "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms"}


@pytest.fixture
def sent_requests(client, monkeypatch):
    """Return the list of requests call_llm() sends, with httpx.post routed to the TestClient."""
    requests_seen = []

    def post_to_test_client(url, headers, json, timeout):
        requests_seen.append({"url": url, "headers": headers, "timeout": timeout})
        return client.post(url.removeprefix("http://ec2.test"), headers=headers, json=json)

    monkeypatch.setattr(httpx, "post", post_to_test_client)
    monkeypatch.setenv("QWEN_HOST", "http://ec2.test/")
    monkeypatch.setenv("QWEN_API_KEY", "test-key")
    return requests_seen


@pytest.mark.parametrize("provider", ["qwen8B", "lora-finetune"])
def test_call_llm(sent_requests, monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)

    answer = llm_client.call_llm(PARSER_MESSAGES)

    assert set(answer) == RESULT_KEYS
    assert answer["model"] == provider
    assert sent_requests[0]["url"] == "http://ec2.test" + llm_client.QWEN_ENDPOINTS[provider]
    assert sent_requests[0]["headers"] == {"X-API-Key": "test-key"}
    assert sent_requests[0]["timeout"] == 300


def test_default_provider_is_lora_finetune(sent_requests, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm_client.call_llm(PARSER_MESSAGES)["model"] == "lora-finetune"


@pytest.mark.parametrize("provider", ["openai", "gpt5"])
def test_unknown_provider(monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    with pytest.raises(ValueError):
        llm_client.call_llm(PARSER_MESSAGES)
