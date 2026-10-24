"""Check a running openweight-llm server: health, API key, and real answers from both models.

Sends a few validation questions (from the training repo) to qwen8B and lora-finetune and scores the
answers against the gold answers the same way notebooks 01/03 do. Needs only the Python standard library.

    python check_deployment.py --host http://<ec2-ip> --api-key <API_KEY>
    python check_deployment.py --host http://<ec2-ip> --api-key <API_KEY> --per-task 5
"""

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_VAL_FILE = Path(__file__).parent.parent / "kqml-geo-qwen3-qlora" / "data" / "questions_v3_val.jsonl"
ENDPOINTS = {"qwen8B": "/openWeightllm/qwen8B", "lora-finetune": "/openWeightllm/qwen8B/lora-finetune"}


def send(url: str, api_key: str | None, body: dict | None = None) -> tuple[int, dict]:
    """Return (HTTP status, JSON body) for a GET (no body) or POST request."""
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["X-API-Key"] = api_key
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, {}


def clean(text: str) -> str:
    """Return the answer without a <think> block or a surrounding ``` code fence (as in the notebooks)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fenced = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, flags=re.S)
    return fenced.group(1).strip() if fenced else text


def is_correct(task: str, gold: str, answer: str) -> bool:
    """Return True if the answer matches the gold answer (same rule as the notebooks' score())."""
    try:
        predicted, expected = json.loads(answer), json.loads(gold)
    except json.JSONDecodeError:
        return False
    if task == "sql":
        normalize = lambda sql: re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()
        return isinstance(predicted, dict) and isinstance(predicted.get("sql"), str) \
            and normalize(predicted["sql"]) == normalize(expected["sql"])
    return predicted == expected


def load_examples(val_file: Path, per_task: int) -> list[dict]:
    """Return the first `per_task` validation examples of each task (classify, extract, sql)."""
    examples, taken = [], {}
    for line in val_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task = record["meta"].get("task", "all")
        if taken.get(task, 0) < per_task:
            taken[task] = taken.get(task, 0) + 1
            examples.append({"qid": record["meta"].get("qid", ""), "task": task,
                             "messages": record["messages"][:2], "gold": record["messages"][2]["content"]})
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="e.g. http://3.120.45.67")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--val-file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--per-task", type=int, default=2, help="validation questions per task (default 2)")
    args = parser.parse_args()
    host = args.host.rstrip("/")

    print("1. Health")
    status, health = send(host + "/health", api_key=None)
    print(f"   {status} {health}")
    if status != 200 or set(health.get("models_loaded", [])) != set(ENDPOINTS):
        raise SystemExit("   FAIL: server not ready - check 'docker compose logs' on EC2")

    print("2. API key")
    status, _ = send(host + ENDPOINTS["qwen8B"], api_key="wrong-key", body={"messages": [{"role": "user", "content": "hi"}]})
    print(f"   wrong key -> {status} ({'OK' if status == 401 else 'FAIL: expected 401'})")
    status, _ = send(host + ENDPOINTS["qwen8B"], api_key=args.api_key, body={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    print(f"   your key  -> {status} ({'OK' if status == 200 else 'FAIL: check API_KEY in both .env files'})")
    if status != 200:
        raise SystemExit(1)

    examples = load_examples(args.val_file, args.per_task)
    print(f"3. {len(examples)} validation questions from {args.val_file.name} (each takes a few seconds)")
    print(f"   {'qid':<10}{'task':<10}{'qwen8B':<16}{'lora-finetune':<16}")
    correct = {name: 0 for name in ENDPOINTS}
    answers_differ = 0
    for example in examples:
        row, answers = f"   {example['qid']:<10}{example['task']:<10}", {}
        for name, path in ENDPOINTS.items():
            status, result = send(host + path, args.api_key, {"messages": example["messages"], "max_tokens": 512})
            answers[name] = clean(result.get("content", "")) if status == 200 else f"HTTP {status}"
            ok = status == 200 and is_correct(example["task"], example["gold"], answers[name])
            correct[name] += ok
            row += f"{('correct' if ok else 'wrong') + ' ' + str(result.get('latency_ms', '-')) + 'ms':<16}"
        answers_differ += answers["qwen8B"] != answers["lora-finetune"]
        print(row)

    print("4. Result")
    for name in ENDPOINTS:
        print(f"   {name:<14} {correct[name]}/{len(examples)} correct")
    print(f"   answers differed on {answers_differ}/{len(examples)} questions")
    if answers_differ == 0:
        print("   WARNING: both models always answer the same - the adapter may not be active")
    elif correct["lora-finetune"] > 0 and correct["lora-finetune"] >= correct["qwen8B"]:
        print("   OK: the adapter is active and the fine-tuned model scores at least as well as the base model")
    else:
        print("   CHECK: the fine-tuned model scores lower than expected - compare with notebook 03's results")


if __name__ == "__main__":
    main()
