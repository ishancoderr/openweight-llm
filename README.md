# openweight-llm

A FastAPI service that serves **Qwen3-8B** with and without the thesis' **QLoRA adapter**, from one
Docker container on a single AWS EC2 **g4dn.xlarge** (1× NVIDIA T4 16 GB).

It is part of the MSc thesis *"Multi-Agent System for Retrieving Missing Geospatial Data"*.
Agent-001 and Agent-002 turn natural-language questions into `QueryParams` JSON in
`pipeline/query_parser.py::parse_query()`. With `agent_side/llm_client.py`, that step runs on one of
the two models served here:

| Configuration | `LLM_PROVIDER` | Where it runs |
|---|---|---|
| Qwen3-8B (base) | `qwen8B` | this service |
| Qwen3-8B + LoRA adapter | `lora-finetune` (default) | this service |

## Endpoints

| Method | Path | Model | Auth |
|---|---|---|---|
| `POST` | `/openWeightllm/qwen8B` | Qwen3-8B, no adapter | `X-API-Key` |
| `POST` | `/openWeightllm/qwen8B/lora-finetune` | Qwen3-8B + LoRA adapter | `X-API-Key` |
| `GET` | `/health` | – | none |

All paths also work with a trailing slash. A wrong or missing `X-API-Key` returns **401**, an empty
`messages` list returns **422**, and a prompt that doesn't fit into `CTX_SIZE` returns **400**.
Interactive docs: `http://<ec2-ip>/docs`.

Request:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "max_tokens": 512, "temperature": 0.0}
```

Response (`latency_ms` = generation time on the server, without time spent waiting for the lock):

```json
{"model": "lora-finetune", "content": "{...}", "prompt_tokens": 412,
 "completion_tokens": 58, "total_tokens": 470, "latency_ms": 2130}
```

`GET /health` → `{"status": "ok", "models_loaded": ["qwen8B", "lora-finetune"]}`

## Architecture

```
 Agent-001 / Agent-002                        EC2 g4dn.xlarge (eu-central-1)
 pipeline/query_parser.py                     ┌──────────────────────────────────────────────┐
   parse_query()                              │ docker container "openweight-llm"            │
     └─ call_llm(messages)                    │  uvicorn :8000  (host port 80)               │
                                              │  app/main.py   X-API-Key check, routes       │
          LLM_PROVIDER=qwen8B ───── HTTP ───► │  app/llm.py                                  │
          LLM_PROVIDER=lora-finetune ───────► │    "qwen8B"        Llama(Q4_K_M)       + Lock│
                                              │    "lora-finetune" Llama(Q4_K_M, LoRA) + Lock│
                                              │         both fully on the T4 (16 GB)         │
                                              │  /models (read-only) ◄── ./models on EBS     │
                                              └──────────────────────────────────────────────┘
```

- Serving uses **llama-cpp-python 0.3.36** (llama.cpp `b11312`) compiled with CUDA for the T4
  (`-DCMAKE_CUDA_ARCHITECTURES=75`).
- At startup, two `Llama` instances are loaded from the same GGUF file: one plain, one with
  `lora_path=adapter.gguf`. The adapter is applied when the model loads, so the two instances never
  interfere with each other.
- Each instance has its own `threading.Lock`, because a `Llama` object can't run two generations at
  once. Route handlers are plain `def`, so FastAPI runs them in its thread pool and the event loop
  stays free while the GPU works.
- The prompt is built exactly like in the training notebooks: Qwen3's chat template with
  `enable_thinking=False` (`DISABLE_THINKING=true`), which pre-fills an empty `<think>` block.
  Any `<think>...</think>` block is also removed from the answer, so the parser receives clean JSON.

GPU memory (approximate, `CTX_SIZE=4096`):

| Item | Per instance | Both |
|---|---|---|
| Q4_K_M weights | 4.7 GiB | 9.4 GiB |
| KV cache (36 layers × 8 KV heads × 128 × K+V × f16 × 4096 tokens) | 0.56 GiB | 1.1 GiB |
| Compute buffers + CUDA context | ~0.6 GiB | ~1.2 GiB |
| LoRA adapter (f16, depends on rank) | – | ~0.1–0.3 GiB |
| **Total** | | **~12–12.5 GiB of the T4's ~15 GiB usable** |

## Folder layout

```
openweight-llm/
├── app/                       the service (the only code that goes into the Docker image)
│   ├── __init__.py
│   ├── main.py                FastAPI app, request/response models, routes, API key check
│   └── llm.py                 load_models(), build_prompt(), generate()
├── agent_side/
│   └── llm_client.py          call_llm() to copy into Agent-001/Agent-002 pipeline/
├── tests/                     pytest, runs without a GPU
│   ├── fake_llama_cpp/        stand-in for the CUDA llama_cpp package
│   ├── qwen3_chat_template.jinja   Qwen3 chat template saved by notebook 02, to check build_prompt()
│   ├── conftest.py            test settings + TestClient fixture
│   ├── test_api.py            routes, auth, validation, prompt format
│   └── test_llm_client.py     call_llm() for both models
├── .github/workflows/ci.yml   on every push: tests, shell script syntax, compose file check
├── adapter/                   PEFT adapter from notebook 02 (MyDrive/kqml_finetune/adapters/qwen3-8b-v3/);
│                              adapter_model.safetensors (167 MB) is stored with Git LFS
├── models/                    created by prepare_model.sh on EC2 (git-ignored)
│   ├── Qwen3-8B-Q4_K_M.gguf
│   └── adapter.gguf
├── Dockerfile                 CUDA 12.4 devel image + CUDA build of llama-cpp-python, non-root user
├── docker-compose.yml         port 80 → 8000, ./models:/models:ro, GPU reservation, healthcheck
├── requirements.txt           server dependencies (pinned)
├── requirements-dev.txt       test dependencies (pinned)
├── pytest.ini
├── .env.example               every setting, with comments
├── setup_ec2.sh               once per instance: Docker + NVIDIA container toolkit
└── prepare_model.sh           once per adapter: download base GGUF, convert adapter to GGUF
```

## Running the tests

The tests replace `llama_cpp` with a small fake (`tests/fake_llama_cpp`), so they run on a laptop
without a GPU or a llama.cpp build. GitHub Actions runs them on every push (`.github/workflows/ci.yml`).

```bash
python -m venv venv
venv\Scripts\activate            # Windows; on Linux/Mac: source venv/bin/activate
pip install -r requirements-dev.txt
pytest -v
```

They check both endpoints with and without a trailing slash, 401 for a missing or wrong key, 422 for
an invalid body, 400 for a prompt that is too long, `/health`, `call_llm()` for both models,
and that `build_prompt()` gives exactly the same text as the Qwen3 chat template used in training.

## Cost

| Item | Price (eu-central-1, Frankfurt) |
|---|---|
| g4dn.xlarge on-demand, running | ≈ **$0.66/h** (≈ €0.60/h, under the €1/h budget) |
| Public IPv4 address, running | $0.005/h |
| 50 GB gp3 root volume | ≈ $4.80/month, **also while the instance is stopped** |

Left running 24/7 the instance costs ≈ $480/month, so:

- **Stop the instance whenever you are not running experiments** (`aws ec2 stop-instances --instance-ids <id>`
  or the console). A stopped instance only costs the EBS volume. Terminate it when the thesis is done.
- **Set a budget alert:** Billing and Cost Management → Budgets → Create budget → *Monthly cost budget*,
  e.g. $30, with an e-mail alert at 80 %.

## Before the first launch: GPU quota

New AWS accounts usually have a quota of **0 vCPUs** for GPU instances.

1. Switch the console to **Europe (Frankfurt) eu-central-1**.
2. Service Quotas → AWS services → Amazon Elastic Compute Cloud (Amazon EC2).
3. Search **"Running On-Demand G and VT instances"** → *Request increase at account level* → **4**
   (the quota counts vCPUs; a g4dn.xlarge has 4).
4. Approval takes from minutes to a couple of days.

## EC2 launch settings

| Setting | Value |
|---|---|
| Region | eu-central-1 (Frankfurt) |
| AMI | **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)** |
| Instance type | g4dn.xlarge |
| Key pair | create one, keep the `.pem` |
| Security group | inbound **22** (SSH) from *My IP*; inbound **80** (HTTP) from the agents' IP(s) |
| Storage | **50 GB gp3** root volume (if the AMI's snapshot needs more, use the minimum the console shows) |

The g4dn's local NVMe disk is wiped on every stop; keep everything on the EBS root volume.

## Deploy step by step

```bash
# 1. On your laptop: connect
ssh -i key.pem ubuntu@<ec2-ip>

# 2. On EC2: get the code (public repo, no login needed)
git clone https://github.com/ishancoderr/openweight-llm.git
cd openweight-llm
chmod +x setup_ec2.sh prepare_model.sh

# 3. git-lfs (downloads the 167 MB adapter weights), Docker + NVIDIA container toolkit
#    (skips whatever the AMI already has), then log in again
./setup_ec2.sh
exit
ssh -i key.pem ubuntu@<ec2-ip>
cd openweight-llm

# 4. Check the adapter arrived as real weights (167M), not a small Git LFS pointer file
ls -lh adapter/adapter_model.safetensors

# 5. Download the base GGUF (~5 GB) and convert the adapter to models/adapter.gguf
./prepare_model.sh

# 6. Settings: set a long random API key
cp .env.example .env
sed -i "s/^API_KEY=.*/API_KEY=$(openssl rand -hex 32)/" .env
grep API_KEY .env          # give this value to the agents as QWEN_API_KEY

# 7. Build and start (the first build compiles llama.cpp: ~10-20 min)
df -h /                    # "Avail" should be at least 25 GB (CUDA image ~12 GB + build); else grow the volume
docker compose up -d --build
docker compose logs -f     # wait for "Application startup complete", then Ctrl+C
docker compose ps          # STATUS should become "healthy"
```

`prepare_model.sh` can be pointed at another GGUF file through environment variables, e.g.
`BASE_FILE=Qwen3-8B-Q5_K_M.gguf ./prepare_model.sh` (then set `BASE_GGUF` in `.env` to match).
`BASE_REPO`, `BASE_FILE`, `BASE_HF_ID` and `LLAMA_CPP_REF` can all be overridden this way.

## Testing with curl

```bash
HOST=http://<ec2-ip>
KEY=<API_KEY from .env>

curl $HOST/health

curl -X POST $HOST/openWeightllm/qwen8B \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Return JSON {\"capital\": ...} for Bavaria."}], "max_tokens": 64}'

curl -X POST $HOST/openWeightllm/qwen8B/lora-finetune \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Population of Saxony in 2020?"}]}'

# expect 401
curl -i -X POST $HOST/openWeightllm/qwen8B -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hi"}]}'
```

## Connecting the agents

1. Copy `agent_side/llm_client.py` into `Agent-001/pipeline/` and `Agent-002/pipeline/`.
   It needs `httpx` (`pip install httpx`; already installed if the agent has the `openai` package).
2. Add to each agent's `.env`:

   ```
   LLM_PROVIDER=lora-finetune        # qwen8B | lora-finetune
   QWEN_HOST=http://<ec2-ip>
   QWEN_API_KEY=<API_KEY from the server's .env>
   ```

3. In `pipeline/query_parser.py::parse_query()`, replace the OpenAI call with `call_llm()`:

   ```python
   from pipeline.llm_client import call_llm

   llm_answer = call_llm(messages, max_tokens=512)
   raw_json = llm_answer["content"]
   ```

   The rest of `parse_query()` stays the same. `call_llm()` returns the same shape for both models:
   `model`, `content`, `prompt_tokens`, `completion_tokens`, `total_tokens` and `latency_ms` (the
   round trip measured in the agent). Pass these to `evaluation/metrics_logger.py` for the comparison.

`LLM_PROVIDER` is read on every call, so an evaluation script can switch between the two models by
setting `os.environ["LLM_PROVIDER"]` between runs.

The public IP changes every time the instance is stopped and started; update `QWEN_HOST` (or attach
an Elastic IP, which costs $0.005/h while the instance is stopped).

## Operations

```bash
docker compose ps                    # running? healthy?
docker compose logs -f --tail 100    # server log, including llama.cpp timings per request
docker compose restart               # reload models, e.g. after a new adapter.gguf or .env change
docker compose down                  # stop the container (the instance keeps running and billing!)
nvidia-smi                           # GPU memory: ~12 GiB used when both models are loaded

# update the code
git pull && docker compose up -d --build

# new adapter version: replace adapter/ on your laptop, commit and push (Git LFS stores the weights), then on EC2
git pull && ./prepare_model.sh && docker compose restart

# after an instance stop/start the container comes back by itself (restart: unless-stopped)
```

## Notes and limitations

- **One request at a time per model.** Each model has a lock, so concurrent requests to the same
  endpoint queue up; the two endpoints can generate at the same time but share the GPU. The agents
  send one parse request per query, so this is enough for the evaluation. Keep the client timeout
  long (300 s in `llm_client.py`).
- **Out of GPU memory** at startup (`Failed to load model` in the logs): lower `CTX_SIZE` in `.env`
  (e.g. 2048; the longest training example was 1,830 tokens) and `docker compose restart`.
- **4-bit quantization (thesis limitation).** Training (notebook 02) and the notebook evaluations
  (01, 03) loaded `Qwen/Qwen3-8B` in 4-bit **NF4** with bitsandbytes. The server runs the same model
  in 4-bit too, but as llama.cpp's **Q4_K_M** GGUF. The LoRA matrices themselves are not quantized
  (they are saved in 32-bit and converted to 16-bit), so the adapter works on any version of
  Qwen3-8B. But the rounding error of the base weights it was trained against (NF4) is not exactly
  the one it meets in serving (Q4_K_M). Scores from the server may therefore differ slightly from
  the notebook scores, and GPT-4o mini is compared against a 4-bit open-weight model.
- **Prompt format = training format.** `build_prompt()` in `app/llm.py` writes the same text as the
  notebooks' `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`, including
  the pre-filled empty `<think>\n\n</think>\n\n`, so base and fine-tuned model get exactly the prompt
  they were evaluated and trained with. Greedy decoding (`temperature: 0.0`) also matches the
  notebooks. One difference: the notebooks' `clean()` also removes a surrounding ```` ``` ```` code
  fence from the answer; the server does not, so `parse_query()` should strip one if present.
- **HTTP only.** The API key travels unencrypted. Limit port 80 in the security group to the agents'
  IP addresses.
