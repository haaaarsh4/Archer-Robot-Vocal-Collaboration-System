## 7. Export

The app runs on a laptop, so the one model has to leave Colab as a quantised GGUF.

There is one complication. llama.cpp can apply a LoRA to a base GGUF, but not a LoRA that added vocabulary, and this one added several thousand Cree tokens. The adapter is therefore merged into a full-precision copy of the base model, and that merged directory is converted. Merging needs room for a second copy of the model; the cell reports the device it chose and stops with a clear message if neither the GPU nor the CPU has the memory.

The quantisation is Q4_K_M, which is the usual quality-per-byte point: about 4.5 GB for the 7B and about 9 GB for the 14B, large enough that the Cree tokens are not crushed. The Modelfile carries the sampling settings. The system prompt is sent by the app on every request rather than baked into the tag, because Ollama would otherwise place it in the context twice. There is one artifact and one tag, `archer:<size>`, where `<size>` comes from the base model: `archer:14b` for the 14B run and `archer:7b` for the 7B run. It serves every chat request, including translation and song-note questions.

The last cell writes the report card and the exact commands for wiring the model into this project's config.
