from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    model: str = "archer:7b"
    gguf_path: str | None = None
    n_ctx: int = 4096
    temperature: float = 0.3
    max_tokens: int = 500
    timeout_s: float = 120.0


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
            raise RuntimeError(
                f"The Archer model '{self.cfg.model}' is not installed in Ollama. "
                "Install it once:\n"
                "  mkdir -p assets/archer_llm\n"
                "  cp archer-7b.Q4_K_M.gguf Modelfile assets/archer_llm/\n"
                "  cd assets/archer_llm && ollama create archer:7b -f Modelfile\n"
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

    def _chat_ollama(self, user_message: str, history: list[ChatTurn],
                     system_prompt: str, temperature: float, max_tokens: int) -> str:
        resp = requests.post(
            f"{self.cfg.ollama_url}/api/chat",
            json={
                "model": self.cfg.model,
                "messages": self._messages(user_message, history, system_prompt),
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": self.cfg.n_ctx,
                },
            },
            timeout=self.cfg.timeout_s,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()

    def _load_llama_cpp(self):
        if not self.cfg.gguf_path:
            raise RuntimeError(
                "chat.backend is 'llama_cpp' but chat.gguf_path is not set. "
                "Point it at archer-7b.Q4_K_M.gguf, the same model the Ollama "
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
        result = model.create_chat_completion(
            messages=self._messages(user_message, history, system_prompt),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return result["choices"][0]["message"]["content"].strip()
