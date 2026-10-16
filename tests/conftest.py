"""Test setup: empty model files and settings, set before app.llm reads them at import time.

pytest.ini puts tests/fake_llama_cpp on the path, so `import llama_cpp` gets the fake, not the GPU build.
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

TEST_MODELS_DIR = tempfile.mkdtemp()
for model_file in ("Qwen3-8B-Q4_K_M.gguf", "adapter.gguf"):
    open(os.path.join(TEST_MODELS_DIR, model_file), "w").close()

os.environ.update(
    MODELS_DIR=TEST_MODELS_DIR,
    BASE_GGUF="Qwen3-8B-Q4_K_M.gguf",
    ADAPTER_GGUF="adapter.gguf",
    CTX_SIZE="4096",
    DISABLE_THINKING="true",
    API_KEY="test-key",
)

API_HEADERS = {"X-API-Key": "test-key"}
PARSER_MESSAGES = [
    {"role": "system", "content": "Return QueryParams as JSON."},
    {"role": "user", "content": "Population of Bavaria in 2020?"},
]


@pytest.fixture(scope="module")
def client():
    """Return a TestClient whose startup has loaded both (fake) models."""
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
