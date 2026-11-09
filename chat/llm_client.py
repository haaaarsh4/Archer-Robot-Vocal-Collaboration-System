from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import requests
from loguru import logger

from chat.archer_prompt import ARCHER_SYSTEM_PROMPT


@dataclass
class ChatTurn:
    role: str    # "user" | "assistant"
    content: str


@dataclass
class LLMConfig:
    """One model, two ways to run it.

    ``model`` is the Ollama tag; ``gguf_path`` is the same model as a file for
    the llama_cpp backend. There is no second chat model and no fallback: if the
    Archer model is not installed, the request fails with the install command
    rather than being answered by a different model.
    """
    backend: str = "ollama"                      # "ollama" | "llama_cpp"
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "archer:14b"
    gguf_path: str | None = None
    n_ctx: int = 4096
    temperature: float = 0.0
    max_tokens: int = 260
    timeout_s: float = 120.0
    # OLLAMA KEEPS A LOADED MODEL FOR FIVE MINUTES BY DEFAULT, and then unloads
    # it. Reloading 4.7 GB of weights is the single slowest thing this app can
    # do: on a machine that is already using its swap it is minutes, not
    # seconds, and it lands on whichever message arrives after the pause. The
    # request asks for an hour instead, so the model stays warm for a working
    # session and the cost is paid once, at startup. It is a hint to Ollama, not
    # a promise: `ollama ps` is the truth about what is resident.
    keep_alive: str = "1h"
    # THREADS ARE NOT FREE ON A BUSY MACHINE. Ollama's default is every core it
    # can see, which on a laptop that is also running the neural stages and a
    # browser is oversubscription: the cores spend their time contending and the
    # model generates slower the more of them it is given. 0 leaves the choice to
    # Ollama; a positive number is passed through as `num_thread`.
    num_thread: int = 0
    # === how the next token is chosen ===
    # MIN-P, NOT TOP-P. Nguyen et al., "Min-p sampling" (ICLR 2025), keep every
    # token whose probability is at least min_p times the top token's, and drop
    # the rest. Because the cut moves with how confident the model is, it does
    # not have to be retuned when temperature changes, which top-p does; the
    # paper measures it beating top-p and top-k on quality and diversity from 1B
    # to 123B and in a human preference study. 0.05 is their default.
    #
    # IT ONLY WORKS IF TOP-P AND TOP-K ARE OFF. Ollama applies all three filters
    # in sequence, so leaving its defaults (top_p 0.9, top_k 40) in place means
    # the 0.9 cut decides the tail and min_p never gets to. 1.0 and 0 are the
    # "disabled" values, and that is what makes min_p the only truncation here.
    min_p: float = 0.05
    top_p: float = 1.0
    top_k: int = 0
    # DRY: DON'T REPEAT YOURSELF (arXiv 2608.22761, 2026). Penalise a token that
    # would continue a span already written, scaled by how long the match is. It
    # is aimed at exactly the failure this app showed: a clause or the visitor's
    # own sentence repeated back. allowed_length 2 means a single repeated token
    # ("ê ê") is left alone and only a three-token match starts costing, which is
    # what a stuck loop looks like and a particle is not. multiplier 0 turns it
    # off. base 1.75 and allowed_length 2 are llama.cpp's own defaults.
    dry_multiplier: float = 0.8
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    # Left to the Modelfile's own PARAMETER repeat_penalty 1.08 unless a value
    # is set here, so the exported tag and the app agree by default.
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None


class LocalLLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.cfg = config or LLMConfig()
        self._llama_cpp_model = None
        self._checked = False

    def installed(self) -> bool | None:
        """Is the one model there? None means the backend could not be asked."""
        if self.cfg.backend == "llama_cpp":
            return bool(self.cfg.gguf_path) and Path(self.cfg.gguf_path).exists()
        try:
            resp = requests.get(f"{self.cfg.ollama_url}/api/tags", timeout=2.0)
            resp.raise_for_status()
            return self.cfg.model in [m["name"] for m in resp.json().get("models", [])]
        except requests.exceptions.RequestException:
            return None

    def _ensure_backend(self) -> None:
        if self.cfg.backend == "llama_cpp":
            return
        if self.cfg.backend != "ollama":
            raise ValueError(f"Unknown LLM backend: {self.cfg.backend!r}")
        if self._checked:
            return
        try:
            resp = requests.get(f"{self.cfg.ollama_url}/api/tags", timeout=3.0)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Can't reach Ollama at {self.cfg.ollama_url}: {e}\n"
                "Start it with `ollama serve`, or switch chat.backend to 'llama_cpp' "
                "in config.yaml and set chat.gguf_path."
            ) from e
        if self.cfg.model not in available:
            # The file name follows the model actually trained, so the hint is
            # built from the tag rather than naming one build: a 14B export is
            # archer-14b.Q4_K_M.gguf and creating archer:7b from it is wrong.
            tag = self.cfg.model.split(":")[-1]
            raise RuntimeError(
                f"The Archer model '{self.cfg.model}' is not installed in Ollama. "
                "Install it once:\n"
                "  mkdir -p assets/archer_llm\n"
                f"  cp archer-{tag}.Q4_K_M.gguf Modelfile assets/archer_llm/\n"
                f"  cd assets/archer_llm && ollama create {self.cfg.model} -f Modelfile\n"
                "See docs/cree-llm.md. There is no second chat model to fall back to."
            )
        self._checked = True

    def chat(self, user_message: str, history: list[ChatTurn],
             system_prompt: str = ARCHER_SYSTEM_PROMPT) -> str:
        self._ensure_backend()
        temperature = self.cfg.temperature
        max_tokens = self.cfg.max_tokens

        if self.cfg.backend == "ollama":
            return self._chat_ollama(user_message, history, system_prompt,
                                     temperature, max_tokens)
        return self._chat_llama_cpp(user_message, history, system_prompt,
                                    temperature, max_tokens)

    def _messages(self, user_message: str, history: list[ChatTurn], system_prompt: str):
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-8:]:      # tone and reference, not evidence
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user_message})
        return messages

    def _options(self, temperature: float, max_tokens: int) -> dict:
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": self.cfg.n_ctx,
        }
        if self.cfg.num_thread > 0:
            options["num_thread"] = self.cfg.num_thread
        # See the LLMConfig comment: min-p is the truncation, so the other two
        # are sent at the values that disable them rather than left to Ollama's
        # defaults, which would cut first and hide the effect.
        if self.cfg.min_p > 0:
            options["min_p"] = self.cfg.min_p
            options["top_p"] = self.cfg.top_p
            options["top_k"] = self.cfg.top_k
        if self.cfg.dry_multiplier > 0:
            options["dry_multiplier"] = self.cfg.dry_multiplier
            options["dry_base"] = self.cfg.dry_base
            options["dry_allowed_length"] = self.cfg.dry_allowed_length
        if self.cfg.repeat_penalty is not None:
            options["repeat_penalty"] = self.cfg.repeat_penalty
        if self.cfg.repeat_last_n is not None:
            options["repeat_last_n"] = self.cfg.repeat_last_n
        return options

    def _ollama_body(self, user_message: str, history: list[ChatTurn],
                     system_prompt: str, temperature: float, max_tokens: int,
                     stream: bool) -> dict:
        return {
            "model": self.cfg.model,
            "messages": self._messages(user_message, history, system_prompt),
            "stream": stream,
            "keep_alive": self.cfg.keep_alive,
            "options": self._options(temperature, max_tokens),
        }

    def _chat_ollama(self, user_message: str, history: list[ChatTurn],
                     system_prompt: str, temperature: float, max_tokens: int) -> str:
        try:
            resp = requests.post(
                f"{self.cfg.ollama_url}/api/chat",
                json=self._ollama_body(user_message, history, system_prompt,
                                       temperature, max_tokens, stream=False),
                timeout=(10.0, self.cfg.timeout_s),
            )
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                "The model did not answer within "
                f"{self.cfg.timeout_s:.0f}s. If the machine is short of memory "
                "Ollama reloads the weights on the next message, which is slow: "
                "check `free -m` and `ollama ps`."
            ) from e
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()

    def stream_chat(self, user_message: str, history: list[ChatTurn],
                    system_prompt: str = ARCHER_SYSTEM_PROMPT) -> Iterator[str]:
        """Yield the reply as it is written, one piece at a time.

        The panel does not have to wait for the last token to show the first
        one, which is the difference between a chat that looks stuck for seven
        seconds and one that starts answering immediately. Only Ollama can be
        streamed this way; llama.cpp has no such path here and hands back one
        piece, so the caller still gets a correct answer, just not gradually.
        """
        self._ensure_backend()
        if self.cfg.backend != "ollama":
            yield self.chat(user_message, history, system_prompt)
            return
        with requests.post(
            f"{self.cfg.ollama_url}/api/chat",
            json=self._ollama_body(user_message, history, system_prompt,
                                   self.cfg.temperature, self.cfg.max_tokens, stream=True),
            stream=True,
            timeout=(10.0, self.cfg.timeout_s),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                piece = (chunk.get("message") or {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break

    def warm(self) -> bool:
        """Load the weights once, at startup, so the first message is not the load.

        The request asks for a single token, so it costs a prompt evaluation and
        nothing else. It uses a different prompt prefix from a real question, so
        it does not seed Ollama's cache, it only makes sure the model is in
        memory and held for ``keep_alive``.
        """
        if self.cfg.backend != "ollama":
            return False
        try:
            self._ensure_backend()
            resp = requests.post(
                f"{self.cfg.ollama_url}/api/chat",
                json={
                    "model": self.cfg.model,
                    "messages": [{"role": "user", "content": "<|song|>\nhello"}],
                    "stream": False,
                    "keep_alive": self.cfg.keep_alive,
                    "options": self._options(0.0, 1),
                },
                timeout=(5.0, 600.0),
            )
            resp.raise_for_status()
            return True
        except Exception as e:  # noqa: BLE001 - warming is best effort
            logger.warning(f"[chat] model warm-up failed: {e}")
            return False

    def _load_llama_cpp(self):
        if not self.cfg.gguf_path:
            raise RuntimeError(
                "chat.backend is 'llama_cpp' but chat.gguf_path is not set. "
                "Point it at the exported Q4_K_M GGUF, the same model the Ollama "
                "backend serves as chat.model."
            )
        if self._llama_cpp_model is not None:
            return self._llama_cpp_model
        from llama_cpp import Llama
        logger.info(f"Loading GGUF model from {self.cfg.gguf_path} (CPU, this can take a bit)...")
        self._llama_cpp_model = Llama(
            model_path=self.cfg.gguf_path,
            n_ctx=self.cfg.n_ctx,
            n_threads=None,   # let llama.cpp pick based on available cores
            verbose=False,
        )
        return self._llama_cpp_model

    def _chat_llama_cpp(self, user_message: str, history: list[ChatTurn],
                        system_prompt: str, temperature: float, max_tokens: int) -> str:
        model = self._load_llama_cpp()
        sampling = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # The same truncation policy as the Ollama path, in llama.cpp's own
        # parameter names. DRY is left out here: it only exists in recent
        # llama-cpp-python builds, and passing an unknown sampler to an older
        # one raises instead of being ignored.
        if self.cfg.min_p > 0:
            sampling["min_p"] = self.cfg.min_p
            sampling["top_p"] = self.cfg.top_p
            sampling["top_k"] = self.cfg.top_k
        if self.cfg.repeat_penalty is not None:
            sampling["repeat_penalty"] = self.cfg.repeat_penalty
        if self.cfg.repeat_last_n is not None:
            sampling["repeat_last_n"] = self.cfg.repeat_last_n
        result = model.create_chat_completion(
            messages=self._messages(user_message, history, system_prompt),
            **sampling,
        )
        return result["choices"][0]["message"]["content"].strip()
