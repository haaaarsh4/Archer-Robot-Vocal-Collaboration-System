## 7. Export

The app runs on a laptop, so the model has to leave Colab as a quantised GGUF.

There is one complication. llama.cpp can apply a LoRA to a base GGUF, but not a LoRA that added vocabulary, and this one added several thousand Cree tokens. The adapter is therefore merged into a full-precision copy of the base model, and that merged directory is converted. Merging needs room for a second copy of the model; the cell reports the device it chose and stops with a clear message if neither the GPU nor the CPU has the memory.

The quantisation is Q4_K_M, which is the usual quality-per-byte point for a 7B model: about 4.5 GB, small enough for a laptop and large enough that the Cree tokens are not crushed. The Modelfile carries the system prompt and the sampling settings, so `ollama create` reproduces the behaviour measured here rather than a different model with the same weights.

The last cell writes the report card and the exact commands for wiring the model into this project's config.
