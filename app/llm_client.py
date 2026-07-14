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
        Returns the reply text, or None if every attempt failed."""
        model = self.resolve(chain, need=need)
        candidates = [model] if model else list(chain)
        options = {"num_predict": num_predict} if num_predict else None
        for candidate in candidates:
            for _ in range(retries + 1):
                try:
                    response = self._get_client().chat(
                        model=candidate, messages=messages,
                        options=options, keep_alive="30m")
                    text = response["message"]["content"].strip()
                    if text:
                        return text
                except Exception:
                    continue
        return None

    def warm(self):
        """Preload the report + VLM models on the server (empty generate) so
        the first real request doesn't pay the multi-GB model load. Cheap to
        call repeatedly; runs in the caller's thread — use a background one."""
        for chain, need in ((config.REPORT_MODELS, None), (config.VLM_MODELS, "vision")):
            model = self.resolve(chain, need=need)
            if model:
                try:
                    self._get_client().generate(model=model, prompt="", keep_alive="30m")
                except Exception:
                    pass

    def generate_report(self, prompt: str) -> str | None:
        return self.chat(config.REPORT_MODELS, [{"role": "user", "content": prompt}],
                         num_predict=900)

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
