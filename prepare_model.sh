#!/usr/bin/env bash
# Run once on EC2 from the project folder, after copying the PEFT adapter into ./adapter:
#     ./prepare_model.sh
# 1. downloads the 4-bit GGUF base model into ./models
# 2. converts ./adapter (adapter_config.json + adapter_model.safetensors) to ./models/adapter.gguf
# Both steps run inside python:3.11-slim containers, so nothing is installed on the host.
set -euo pipefail

BASE_REPO="${BASE_REPO:-Qwen/Qwen3-8B-GGUF}"
BASE_FILE="${BASE_FILE:-Qwen3-8B-Q4_K_M.gguf}"
BASE_HF_ID="${BASE_HF_ID:-Qwen/Qwen3-8B}"
# Same llama.cpp version that llama-cpp-python 0.3.36 is built from, so the adapter format matches.
LLAMA_CPP_REF="${LLAMA_CPP_REF:-b11312}"
HF_HUB_VERSION="2.1.1"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_DIR="$PROJECT_DIR/adapter"
MODELS_DIR="$PROJECT_DIR/models"
# Files written by the containers (which run as root) are handed back to this user.
HOST_OWNER="$(id -u):$(id -g)"

echo "== 1/3 Adapter"
for adapter_file in adapter_config.json adapter_model.safetensors; do
    if [ ! -f "$ADAPTER_DIR/$adapter_file" ]; then
        echo "ERROR: $ADAPTER_DIR/$adapter_file not found." >&2
        echo "Copy your PEFT adapter folder to $ADAPTER_DIR first, e.g. from your laptop:" >&2
        echo "  scp -i key.pem -r ./adapter ubuntu@<ec2-ip>:~/openweight-llm/adapter" >&2
        exit 1
    fi
done
ADAPTER_BASE="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("base_model_name_or_path", "(not set)"))' \
    "$ADAPTER_DIR/adapter_config.json")"
echo "Adapter was trained on : $ADAPTER_BASE"
echo "Converting against     : $BASE_HF_ID"
if [ "$ADAPTER_BASE" != "$BASE_HF_ID" ]; then
    # A 4-bit re-upload of the same model (e.g. unsloth/qwen3-8b-bnb-4bit) is fine; another model is not.
    # (Loading Qwen/Qwen3-8B in 4-bit with bitsandbytes, as notebook 02 does, still records Qwen/Qwen3-8B.)
    echo "WARNING: these differ. Continue only if the adapter's base is a variant of $BASE_HF_ID."
fi

mkdir -p "$MODELS_DIR"

echo "== 2/3 Base model $BASE_REPO/$BASE_FILE (~5 GB)"
if [ -f "$MODELS_DIR/$BASE_FILE" ]; then
    echo "Already downloaded: $MODELS_DIR/$BASE_FILE"
else
    docker run --rm \
        -v "$MODELS_DIR:/models" \
        -e BASE_REPO="$BASE_REPO" -e BASE_FILE="$BASE_FILE" -e HOST_OWNER="$HOST_OWNER" -e HF_TOKEN \
        python:3.11-slim bash -c '
            set -euo pipefail
            pip install --quiet --root-user-action=ignore "huggingface_hub=='"$HF_HUB_VERSION"'"
            hf download "$BASE_REPO" "$BASE_FILE" --local-dir /models
            rm -rf /models/.cache
            chown "$HOST_OWNER" "/models/$BASE_FILE"
        '
fi

echo "== 3/3 Adapter -> $MODELS_DIR/adapter.gguf (llama.cpp $LLAMA_CPP_REF)"
docker run --rm \
    -v "$ADAPTER_DIR:/adapter:ro" \
    -v "$MODELS_DIR:/models" \
    -e BASE_HF_ID="$BASE_HF_ID" -e LLAMA_CPP_REF="$LLAMA_CPP_REF" -e HOST_OWNER="$HOST_OWNER" -e HF_TOKEN \
    python:3.11-slim bash -c '
        set -euo pipefail
        cd /tmp
        python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], \"llama.cpp.tar.gz\")" \
            "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$LLAMA_CPP_REF.tar.gz"
        mkdir llama.cpp && tar -xzf llama.cpp.tar.gz -C llama.cpp --strip-components=1
        # The requirements file adds the CPU-only PyTorch index, so no CUDA wheels are pulled in.
        pip install --quiet --root-user-action=ignore \
            -r llama.cpp/requirements/requirements-convert_lora_to_gguf.txt "huggingface_hub=='"$HF_HUB_VERSION"'"

        # Only the config and tokenizer of the base model are needed, not its 16 GB of weights.
        hf download "$BASE_HF_ID" config.json generation_config.json tokenizer.json tokenizer_config.json \
            vocab.json merges.txt --local-dir /tmp/base-config

        python llama.cpp/convert_lora_to_gguf.py \
            --base /tmp/base-config --outtype f16 --outfile /models/adapter.gguf /adapter
        chown "$HOST_OWNER" /models/adapter.gguf
    '

echo
ls -lh "$MODELS_DIR"
echo "Models ready. Next: cp .env.example .env, edit API_KEY, then docker compose up -d --build"
