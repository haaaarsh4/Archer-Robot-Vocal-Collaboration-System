# Archer Chat — setup guide

Everything below runs on your machine only. Nothing in this feature calls
a third-party API or uploads Archer's music, notes, or vocals anywhere.

## What you're getting

- **A chat panel** (bottom-right button on the site) that answers questions
  about Archer's songs using notes you write yourself, via a small local
  LLM + local retrieval (RAG) — no cloud model, no fine-tuning.
- **Optional spoken replies** via a local TTS voice (Piper).
- **"Sing my idea"**: hum/sing a rough version into the mic, get it back
  sung in Archer's voice via your existing trained RVC model.

## 1. Install the new Python deps

In the **same** environment your main `server.py` already runs in (not
`neural_env`, which is RVC's separate venv):

```bash
pip install -r requirements-chat.txt
```

## 2. Set up the local LLM (Ollama, recommended)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b-instruct-q4_K_M
ollama serve   # or it may already be running as a background service
```

That's a ~4.9GB one-time download. After that, `ollama serve` needs no
internet. CPU-only is fine — expect a few seconds per reply, not instant,
but comfortably usable.

*(If you'd rather avoid a second daemon process, you can use
`llama-cpp-python` in-process instead — see the comment block at the top
of `chat/llm_client.py` and set `chat.backend: llama_cpp` below.)*

## 3. Write your song notes

Put your own notes about each song — meaning, backstory, creation
details — as `.md`/`.txt` files in `song_notes/`. See
`song_notes/README.md` for the format and an important note about not
pasting in text copied from elsewhere (apxo.net or anywhere else) —
write your own summary in your own words instead. That keeps the index
genuinely local-only content you have full rights to, and tends to
produce better, more focused answers than a raw copy-paste would anyway.

The index builds itself automatically the first time `/api/chat` is
called. Edit notes any time; call `POST /api/chat/reindex` to force an
immediate rebuild instead of waiting for the automatic change-detection.

## 4. (Optional) Spoken replies

```bash
pip install piper-tts
python -m piper.download_voices en_US-lessac-medium
```

Then set `tts.voice_path` in `config.yaml` to the downloaded `.onnx`
file. Without this, "Spoken" mode in the chat panel just shows an
explanatory message and still gives you the text reply.

## 5. Add these keys to `config.yaml`

```yaml
chat:
  backend: ollama                                  # "ollama" | "llama_cpp"
  ollama_url: "http://127.0.0.1:11434"
  ollama_model: "llama3.1:8b-instruct-q4_K_M"
  gguf_path: null                                   # only used if backend: llama_cpp
  temperature: 0.4
  max_tokens: 500

tts:
  voice_path: null                                  # e.g. "assets/piper/en_US-lessac-medium.onnx"

svs:
  diffsinger_url: "http://127.0.0.1:8802"           # only needed for the lyrics+melody path, see below
  diffsinger_timeout_s: 600
  default_voice_index: 0
```

## 6. Mount the router / wire the widget

Already done for you in the updated `server.py` (chat router mounted
right after `neural_timbre` is built) and `index.html` (floating button +
panel, bottom-right to open, panel opens on the left as requested). If
you're hand-merging into your own working copy instead of replacing the
files outright, the two touch points are:

- `server.py`: the `from chat.chat_api import router as chat_router` /
  `app.include_router(chat_router)` block, placed after `neural_timbre`
  is constructed (the chat router needs that instance).
- `index.html`: the `#archer-chat-fab` / `#archer-chat-panel` block plus
  its `<style>` and `<script>` sections.

## 7. Run it

```bash
python server.py
```

Open the site, click the chat button bottom-right. `GET /api/chat/health`
tells you whether the RAG index and TTS are ready.

---

## "Sing my idea" — the two paths, and which one you actually have today

You have a **generate-audio-from-scratch** requirement, and it's worth
being precise about what RVC can and can't do for that, since it changes
what's realistic to ship now vs. later.

**RVC is voice *conversion*, not voice *synthesis*.** It re-skins the
timbre of audio that already exists — it can't produce a melody out of
text or silence. So "generate audio from scratch" needs a second stage
in front of RVC that actually *makes* the singing performance:

### Path A — hum/sing it yourself (ships today)
```
your hummed/sung scratch vocal → RVC (your trained Archer model) → done
```
This is what the chat panel's "Sing my idea" tab does right now. It
reuses your existing `NeuralTimbreConverter`/RVC sidecar exactly as your
offline-track-render feature already does — no new ML pipeline, no
training, just a new UI + endpoint calling code you already have working.
Melody, rhythm, and words all come from your own performance; RVC changes
only the timbre.

### Path B — lyrics + melody, no scratch vocal (bigger lift, scaffolded not shipped)
```
lyrics + melody (MIDI) → DiffSinger (generates a sung scratch vocal) → RVC → done
```
This is genuinely a second full local ML pipeline, not a config change:

1. Install DiffSinger from https://github.com/openvpi/DiffSinger locally.
2. Get (or fine-tune) an acoustic model + vocoder checkpoint. The
   out-of-the-box pretrained checkpoints are English/Mandarin general
   singing voices — good enough as the *scratch* vocal, since RVC repaints
   the timbre afterward. You don't need to fine-tune DiffSinger on Archer's
   voice; you only need it to produce *a* clean sung performance of your
   lyrics+melody for RVC to then re-voice.
3. Wrap DiffSinger's inference in a small local FastAPI sidecar, exactly
   like `neural_env/rvc_server.py` already does for RVC — a `/synthesize`
   endpoint that takes `{lyrics, notes}` and returns WAV. That's the
   contract `chat/svs_pipeline.py` already expects (`svs.diffsinger_url`
   in config.yaml); once that sidecar is up, `POST /api/chat/sing-from-lyrics`
   in this feature will work with zero further code changes on this side.
4. For turning a melody idea into the `notes` format DiffSinger needs
   (per-note pitch/duration/lyric), the natural on-ramp is: let the user
   upload a MIDI file, parse it with `pretty_midi`, and align syllables to
   notes yourself (simple 1 syllable → 1 note heuristic is a fine start).

This is real work — plan on it being its own multi-day-to-multi-week
project separate from everything else here, mainly around getting
DiffSinger's environment and checkpoints working locally and getting
lyric-to-note alignment feeling right. Path A ships today and covers the
"take my rough idea, sing it back as Archer" use case without any of that;
Path B is the way to get there without needing a scratch performance from
you at all, when/if you want to invest in it.

## Privacy checklist (recap from the earlier conversation)

- [ ] `song_notes/`, any isolated vocal stems, and the trained `.pth`/
      `.index` RVC files are in `.gitignore` (repo is public).
- [ ] No cloud-synced folder (iCloud/Dropbox/OneDrive) contains the
      project directory.
- [ ] Disk encryption (FileVault/BitLocker) is on.
- [ ] Everything above talks only to `127.0.0.1` — Ollama, the RAG index,
      Piper, the RVC sidecar, and (if you build Path B) the DiffSinger
      sidecar. Worth firewalling outbound network access for the app
      process once everything's confirmed working, per your original
      requirement.
