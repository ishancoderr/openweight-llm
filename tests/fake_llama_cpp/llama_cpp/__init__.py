"""Fake llama_cpp for tests without a GPU: remembers the last call and answers with a <think> block + JSON."""


class Llama:
    def __init__(self, model_path, lora_path=None, **settings):
        self.model_path = model_path
        self.lora_path = lora_path
        self.settings = settings
        self.last_prompt = None
        self.last_stop = None

    def create_completion(self, prompt, max_tokens, temperature, stop):
        self.last_prompt = prompt
        self.last_stop = stop
        if max_tokens > self.settings["n_ctx"]:
            # The real Llama raises the same error type when prompt + answer don't fit the context.
            raise ValueError(f"Requested tokens exceed context window of {self.settings['n_ctx']}")
        tag = "lora" if self.lora_path else "base"
        return {
            "choices": [{"text": f'<think>\n\n</think>\n\n{{"model": "{tag}"}}'}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }
