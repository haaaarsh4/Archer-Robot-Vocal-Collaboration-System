from __future__ import annotations

import json
from dataclasses import dataclass, field

import requests
from loguru import logger


DEFAULT_SYSTEM_PROMPT = """You are Archer's music assistant: a helpful, warm guide to Archer's \
songs, running entirely on the user's own laptop. Answer questions about song meaning, \
backstory, and creation using ONLY the context passages you are given below each question. \
If the context doesn't contain the answer, say you don't have notes on that yet instead of \
guessing. Keep answers conversational and concise. Some songs reference Cree phrases or \
titles; if the user asks for an actual Cree translation (not just discussion), tell them you're \
routing that through the app's dedicated Cree translator instead of answering yourself."""


@dataclass
class ChatTurn:
    role: str    # "user" | "assistant"
    content: str


@dataclass
class LLMConfig:
    backend: str = "ollama"                      # "ollama" | "llama_cpp"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b-instruct-q4_K_M"
    gguf_path: str | None = None                  # used only for llama_cpp backend
    n_ctx: int = 4096
    temperature: float = 0.4
    max_tokens: int = 500
    timeout_s: float = 120.0


class LocalLLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.cfg = config or LLMConfig()
        self._llama_cpp_model = None
        self._backend_verified = False

    def _ensure_backend(self):
        if self._backend_verified:
            return
        if self.cfg.backend == "ollama":
            try:
                resp = requests.get(f"{self.cfg.ollama_url}/api/tags", timeout=3.0)
                resp.raise_for_status()
                models = [m["name"] for m in resp.json().get("models", [])]
                if self.cfg.ollama_model not in models:
                    logger.warning(
                        f"Ollama is running but model '{self.cfg.ollama_model}' isn't pulled yet. "
                        f"Run: ollama pull {self.cfg.ollama_model}"
                    )
            except requests.exceptions.RequestException as e:
                raise RuntimeError(
                    f"Can't reach Ollama at {self.cfg.ollama_url}: {e}\n"
                    "Is it running? Try `ollama serve` in a terminal, or switch chat.backend "
                    "to 'llama_cpp' in config.yaml if you'd rather run in-process."
                ) from e
        self._backend_verified = True

    def chat(self, user_message: str, history: list[ChatTurn], context_chunks: list[str],
              system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        self._ensure_backend()

        context_block = ""
        if context_chunks:
            joined = "\n\n---\n\n".join(context_chunks)
            context_block = f"\n\nContext from your song notes:\n{joined}"

        if self.cfg.backend == "ollama":
            return self._chat_ollama(user_message, history, system_prompt, context_block)
        elif self.cfg.backend == "llama_cpp":
            return self._chat_llama_cpp(user_message, history, system_prompt, context_block)
        raise ValueError(f"Unknown LLM backend: {self.cfg.backend!r}")

    def _chat_ollama(self, user_message: str, history: list[ChatTurn],
                      system_prompt: str, context_block: str) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-8:]:      # keep the last few turns; RAG context is per-question anyway
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user_message + context_block})

        resp = requests.post(
            f"{self.cfg.ollama_url}/api/chat",
            json={
                "model": self.cfg.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.cfg.temperature,
                    "num_predict": self.cfg.max_tokens,
                    "num_ctx": self.cfg.n_ctx,
                },
            },
            timeout=self.cfg.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()

    def _load_llama_cpp(self):
        if self._llama_cpp_model is not None:
            return self._llama_cpp_model
        if not self.cfg.gguf_path:
            raise RuntimeError(
                "chat.backend is 'llama_cpp' but chat.gguf_path isn't set in config.yaml. "
                "Point it at a local GGUF file, e.g. a Llama-3.1-8B-Instruct Q4_K_M download."
            )
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
                         system_prompt: str, context_block: str) -> str:
        model = self._load_llama_cpp()
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-8:]:
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user_message + context_block})

        result = model.create_chat_completion(
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return result["choices"][0]["message"]["content"].strip()
