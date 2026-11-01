from __future__ import annotations

from dataclasses import dataclass

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
    # The nêhiyawêwin model trained by Cree_LLM_Training.ipynb. It is a separate
    # model with its own prompt; an empty value keeps every request on the
    # general model, so a checkout without the GGUF still runs.
    cree_model: str = ""
    cree_gguf_path: str | None = None
    cree_temperature: float = 0.2
    cree_max_tokens: int = 400
    n_ctx: int = 4096
    temperature: float = 0.4
    max_tokens: int = 500
    timeout_s: float = 120.0


class LocalLLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.cfg = config or LLMConfig()
        self._llama_cpp_models: dict[str, object] = {}
        # Each model is checked against Ollama once. The Cree model is a second
        # tag, so a single "backend is up" flag would hide the case where the
        # general model is pulled and the Cree one is not.
        self._checked_models: set[str] = set()

    def _ensure_backend(self, model: str | None = None):
        if self.cfg.backend != "ollama":
            return
        wanted = model or self.cfg.ollama_model
        if wanted in self._checked_models:
            return
        try:
            resp = requests.get(f"{self.cfg.ollama_url}/api/tags", timeout=3.0)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            if wanted not in available:
                logger.warning(
                    f"Ollama is running but model '{wanted}' isn't pulled yet. "
                    f"Run: ollama pull {wanted}"
                )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Can't reach Ollama at {self.cfg.ollama_url}: {e}\n"
                "Is it running? Try `ollama serve` in a terminal, or switch chat.backend "
                "to 'llama_cpp' in config.yaml if you'd rather run in-process."
            ) from e
        self._checked_models.add(wanted)

    def chat(self, user_message: str, history: list[ChatTurn], context_chunks: list[str],
             system_prompt: str = DEFAULT_SYSTEM_PROMPT, model: str | None = None,
             temperature: float | None = None, max_tokens: int | None = None) -> str:
        self._ensure_backend(model)

        context_block = ""
        if context_chunks:
            joined = "\n\n===\n\n".join(context_chunks)
            context_block = f"\n\nContext from your song notes:\n{joined}"

        temperature = self.cfg.temperature if temperature is None else temperature
        max_tokens = self.cfg.max_tokens if max_tokens is None else max_tokens

        if self.cfg.backend == "ollama":
            return self._chat_ollama(user_message, history, system_prompt, context_block,
                                     model or self.cfg.ollama_model, temperature, max_tokens)
        if self.cfg.backend == "llama_cpp":
            path = self.cfg.gguf_path
            if model and model == self.cfg.cree_model:
                path = self.cfg.cree_gguf_path or path
            return self._chat_llama_cpp(user_message, history, system_prompt, context_block,
                                        path, temperature, max_tokens)
        raise ValueError(f"Unknown LLM backend: {self.cfg.backend!r}")

    def _chat_ollama(self, user_message: str, history: list[ChatTurn],
                     system_prompt: str, context_block: str, model: str,
                     temperature: float, max_tokens: int) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-8:]:      # keep the last few turns; RAG context is per-question anyway
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user_message + context_block})

        resp = requests.post(
            f"{self.cfg.ollama_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
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
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()

    def _load_llama_cpp(self, path: str | None):
        if not path:
            raise RuntimeError(
                "chat.backend is 'llama_cpp' but no GGUF path is set in config.yaml. "
                "Set chat.gguf_path for the general model and chat.cree_gguf_path for the Cree model."
            )
        if path in self._llama_cpp_models:
            return self._llama_cpp_models[path]
        from llama_cpp import Llama
        logger.info(f"Loading GGUF model from {path} (CPU, this can take a bit)...")
        model = Llama(
            model_path=path,
            n_ctx=self.cfg.n_ctx,
            n_threads=None,   # let llama.cpp pick based on available cores
            verbose=False,
        )
        self._llama_cpp_models[path] = model
        return model

    def _chat_llama_cpp(self, user_message: str, history: list[ChatTurn],
                        system_prompt: str, context_block: str, gguf_path: str | None,
                        temperature: float, max_tokens: int) -> str:
        model = self._load_llama_cpp(gguf_path)
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-8:]:
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user_message + context_block})

        result = model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return result["choices"][0]["message"]["content"].strip()
