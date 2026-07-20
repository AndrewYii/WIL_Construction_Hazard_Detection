"""
Client for the Ollama server on the NVIDIA DGX Spark.

Wraps ollama.Client(host=...) with:
  - health check / model listing (cached briefly for dashboard polling)
  - preference-chain resolution: first model in the chain that is actually
    pulled on the server wins, so a missing pull degrades instead of failing
  - chat with timeout and a single retry

Set OLLAMA_HOST (or SPARK_OLLAMA_HOST) to point at the Spark, e.g.
    OLLAMA_HOST=http://192.168.1.50:11434
"""

import time

import config


class SparkLLM:
    def __init__(self, host: str | None = None, timeout: float | None = None):
        self.host = host or config.OLLAMA_HOST
        self.timeout = timeout or config.OLLAMA_TIMEOUT_SEC
        self._client = None
        self._models_cache: tuple[float, list[str]] | None = None
        self._caps: dict[str, set[str]] = {}

    def _get_client(self):
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=self.host, timeout=self.timeout)
        return self._client

    def available_models(self, cache_sec: float = 10.0) -> list[str]:
        """Model names pulled on the server. Empty list if unreachable.
        Also refreshes self._caps ({model: set(capabilities)}) from /api/tags
        so vision-capable models can be told apart."""
        now = time.time()
        if self._models_cache and now - self._models_cache[0] < cache_sec:
            return self._models_cache[1]
        models = []
        try:
            import json as _json
            import urllib.request
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as r:
                data = _json.load(r)
            for m in data.get("models", []):
                name = m.get("model") or m.get("name", "")
                if name:
                    models.append(name)
                    if "capabilities" in m:
                        self._caps[name] = set(m["capabilities"])
        except Exception:
            models = []
        self._models_cache = (now, models)
        return models

    def is_up(self) -> bool:
        return bool(self.available_models())

    def resolve(self, chain: list[str], need: str | None = None) -> str | None:
        """First model from the preference chain present on the server.
        Matches with or without an explicit tag (qwen3:32b == qwen3:32b-*).
        `need` filters on a server-reported capability (e.g. "vision") —
        models whose capability list is known and lacks it are skipped."""
        available = self.available_models()
        for want in chain:
            for have in available:
                if have == want or have.startswith(want + "-") or have.split(":")[0] == want:
                    if need and have in self._caps and need not in self._caps[have]:
                        break  # this chain entry can't do the job; try next
                    return have
        return None

    def status(self) -> dict:
        """Snapshot for the dashboard / Streamlit sidebar."""
        models = self.available_models()
        return {
            "host": self.host,
            "up": bool(models),
            "models": models,
            "report_model": self.resolve(config.REPORT_MODELS),
            "vlm_model": self.resolve(config.VLM_MODELS, need="vision"),
        }

    def chat(self, chain: list[str], messages: list[dict], retries: int = 1,
             need: str | None = None, num_predict: int | None = None) -> str | None:
        """Chat against the first available model in the chain.
        keep_alive pins the model in the server's memory so repeat calls skip
        the multi-GB reload; num_predict caps output length for speed.
        Returns the reply text, or None if every attempt failed.

        think=False (2026-07-20): several models in the preference chains
        (gemma4, qwen3.x) are "thinking" models that spend a chunk of
        num_predict on an invisible chain-of-thought field before writing
        the actual answer — measured on gemma4:31b, a trivial one-sentence
        reply used 103 tokens of thinking for 2 words of real content. With
        a tight num_predict (see config.REPORT_NUM_PREDICT) that could burn
        the whole budget and return empty content, which silently fell back
        to the template every time while still paying the full ~27s latency
        for nothing. Disabling it dropped the same call to ~4s and freed the
        full token budget for real content. Ollama no-ops this flag for
        models without a thinking mode, so it's safe chain-wide."""
        model = self.resolve(chain, need=need)
        candidates = [model] if model else list(chain)
        options = {"num_predict": num_predict} if num_predict else None
        for candidate in candidates:
            for _ in range(retries + 1):
                try:
                    response = self._get_client().chat(
                        model=candidate, messages=messages, think=False,
                        options=options, keep_alive="30m")
                    text = response["message"]["content"].strip()
                    if text:
                        return text
                except Exception:
                    continue
        return None

    def warm(self):
        """Preload the report + VLM models on the server (empty generate) so
        neither pays a multi-GB cold-load cost on first real use — a slow
        first /report click, or a slow first height-hazard PPE check. Cheap
        to call repeatedly; runs in the caller's thread — use a background
        one.

        This only stays safe on the Spark's unified memory (no separate VRAM
        pool) because REPORT_MODELS defaults to a *small* first choice
        (gemma4:31b, ~19GB) — together with VLM_MODELS' resolved pick
        (qwen3.6:35b-a3b-bf16, ~71GB) that's ~90GB against 121GB total, with
        real headroom left. 2026-07-20: this used to default to gpt-oss:120b
        (~65GB) first, and warming *that* alongside the VLM model got the
        whole live.py process OOM-killed outright (not a catchable
        exception — see the Gotchas page). Left as a preference-chain
        fallback for anyone who explicitly opts into REPORT_MODELS=
        gpt-oss:120b,... for its extra quality — that's a real memory
        trade-off they're accepting, not a bug; the report path degrades
        gracefully either way (generate_report()/chat() return None on
        timeout/failure, callers fall back to a template — see
        docs/SPARK_SETUP.md's "never crash because of the LLM" rule)."""
        for chain, need in ((config.REPORT_MODELS, None), (config.VLM_MODELS, "vision")):
            model = self.resolve(chain, need=need)
            if model:
                try:
                    self._get_client().generate(model=model, prompt="", think=False,
                                                keep_alive="30m")
                except Exception:
                    pass

    def generate_report(self, prompt: str) -> str | None:
        return self.chat(config.REPORT_MODELS, [{"role": "user", "content": prompt}],
                         num_predict=config.REPORT_NUM_PREDICT)

    def describe_image(self, prompt: str, jpeg_bytes: bytes) -> str | None:
        return self.chat(
            config.VLM_MODELS,
            [{"role": "user", "content": prompt, "images": [jpeg_bytes]}],
            need="vision", num_predict=220,
        )


_default: SparkLLM | None = None


def get_client() -> SparkLLM:
    global _default
    if _default is None:
        _default = SparkLLM()
    return _default


if __name__ == "__main__":
    import json
    print(json.dumps(get_client().status(), indent=2))
