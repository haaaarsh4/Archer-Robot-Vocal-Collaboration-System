"""Baseline 2: correct vocab wiring, dual adapters, real audio + TTS ground truth."""
import json, os, sys, time
import numpy as np, soundfile as sf, torch

sys.path.insert(0, os.getcwd())
from analysis.cree_orthography import to_sro, to_syllabics, anglicize

BASE = "data/models/cree/facebook--mms-1b-all"
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]
torch.set_num_threads(int(os.environ.get("MMS_THREADS", "8")))


def build_vocab_json(lang):
    """MMS ships a frequency-ordered `token count` TSV per language, whose rows are
    the adapter's lm_head rows offset by the four special tokens."""
    src = f"{BASE}/vocabs/{lang}.txt"
    out = f"{BASE}/vocabs/{lang}.json"
    toks = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # token is everything before the last space-separated integer
            tok = line.rsplit(" ", 1)[0]
            toks.append(tok)
    vocab = {t: i for i, t in enumerate(SPECIALS + toks)}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    return out, len(toks)


def load_audio(path, start_s=0.0, dur_s=8.0, sr=16000):
    import librosa
    audio, src = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if src != sr:
        audio = librosa.resample(audio, orig_sr=src, target_sr=sr)
    return np.ascontiguousarray(audio[int(start_s * sr):int((start_s + dur_s) * sr)], dtype=np.float32)


def load_model():
    from transformers import Wav2Vec2ForCTC
    t0 = time.time()
    model = Wav2Vec2ForCTC.from_pretrained(BASE, local_files_only=True,
                                            low_cpu_mem_usage=True, dtype=torch.float32)
    print(f"[load] base {time.time()-t0:.1f}s", flush=True)
    return model


def use_adapter(model, lang):
    from transformers import Wav2Vec2CTCTokenizer
    json_path, n = build_vocab_json(lang)
    tok = Wav2Vec2CTCTokenizer(vocab_file=json_path, unk_token="<unk>", pad_token="<pad>",
                               word_delimiter_token="|", do_lower_case=False)
    model.config._name_or_path = BASE
    model.load_adapter(lang, local_files_only=True)
    model.eval()
    assert tok.vocab_size == model.config.vocab_size, (tok.vocab_size, model.config.vocab_size)
    return tok, n


def greedy(model, tok, audio):
    inputs = torch.from_numpy(audio)[None]
    t0 = time.time()
    with torch.no_grad():
        logits = model(inputs).logits[0]
    el = time.time() - t0
    ids = logits.argmax(-1)
    probs = logits.softmax(-1)
    return tok.decode(ids), float(probs.max(-1).values.mean()), el, logits


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "real"
    model = load_model()

    if mode == "tts":
        synth = sys.argv[2]
        from transformers import VitsModel, AutoTokenizer
        tdir = "data/models/cree/facebook--mms-tts-crk-script_latin"
        tts_tok = AutoTokenizer.from_pretrained(tdir, local_files_only=True)
        tts = VitsModel.from_pretrained(tdir, local_files_only=True)
        tts.eval()
        ids = tts_tok(synth, return_tensors="pt").input_ids
        with torch.no_grad():
            wav = tts(ids).waveform[0, 0].numpy()
        if tts.config.sampling_rate != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=tts.config.sampling_rate, target_sr=16000)
        print(f"[tts] {synth!r} -> {len(wav)/16000:.2f}s", flush=True)
        sf.write("scratch/tts_ref.wav", wav, 16000)
        audio = wav.astype(np.float32)
        truth = synth
    else:
        path = sys.argv[2] if len(sys.argv) > 2 else "frontend/assets/audio/memory-v2.mp3"
        start = float(sys.argv[3]) if len(sys.argv) > 3 else 34.0
        dur = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0
        audio = load_audio(path, start, dur)
        truth = None
        print(f"[audio] {path} {start}s +{dur}s rms={float(np.sqrt((audio**2).mean())):.4f}", flush=True)

    for lang in ("crk-script_latin", "crk-script_syllabics"):
        print(f"\n===== {lang}", flush=True)
        tok, n = use_adapter(model, lang)
        raw, conf, el, logits = greedy(model, tok, audio)
        print(f"[greedy] {el:.1f}s ({el/ (len(audio)/16000):.2f}x RT) frames={logits.shape[1]} "
              f"vocab={tok.vocab_size} ({n} letters) mean_conf={conf:.3f}", flush=True)
        print(f"  RAW : {raw!r}", flush=True)
        sro = to_sro(raw) if "syllabics" in lang else raw
        print(f"  SRO : {sro!r}", flush=True)
        print(f"  CANS: {to_syllabics(sro)!r}", flush=True)
        print(f"  ANGL: {anglicize(sro)!r}", flush=True)
        if truth:
            from analysis.cree_orthography import normalize_sro
            t = normalize_sro(truth)
            s = normalize_sro(sro)
            import difflib
            cer = 1 - difflib.SequenceMatcher(None, t, s).ratio()
            print(f"  TRUTH: {t!r}  CER={cer:.3f}", flush=True)
        np.save(f"scratch/logits_{lang}.npy", logits.numpy())
        with open(f"scratch/vocabids_{lang}.json", "w") as f:
            json.dump({str(i): tok.convert_ids_to_tokens(i) for i in range(tok.vocab_size)}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
