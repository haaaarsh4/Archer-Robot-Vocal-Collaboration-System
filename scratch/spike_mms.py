"""Baseline: MMS-1b + crk adapters on real Archer singing. Greedy decode, timed."""
import os, sys, time
import numpy as np, soundfile as sf, torch

sys.path.insert(0, os.getcwd())
from analysis.cree_orthography import to_sro, to_syllabics, anglicize

BASE = "data/models/cree/facebook--mms-1b-all"
torch.set_num_threads(int(os.environ.get("MMS_THREADS", "8")))


def load_audio(path, start_s=0.0, dur_s=8.0, sr=16000):
    import librosa
    audio, src = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if src != sr:
        audio = librosa.resample(audio, orig_sr=src, target_sr=sr)
    return np.ascontiguousarray(audio[int(start_s * sr):int((start_s + dur_s) * sr)], dtype=np.float32)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "frontend/assets/audio/memory-v2.mp3"
    start = float(sys.argv[2]) if len(sys.argv) > 2 else 34.0
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    audio = load_audio(path, start, dur)
    print(f"audio {len(audio)/16000:.2f}s rms={float(np.sqrt((audio**2).mean())):.4f}", flush=True)

    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC
    t0 = time.time()
    model = Wav2Vec2ForCTC.from_pretrained(BASE, local_files_only=True,
                                            low_cpu_mem_usage=True, dtype=torch.float32)
    print(f"model load {time.time()-t0:.1f}s", flush=True)

    for lang in ("crk-script_latin", "crk-script_syllabics"):
        print(f"\n===== {lang}", flush=True)
        tok = Wav2Vec2CTCTokenizer(
            vocab_file=f"{BASE}/vocabs/{lang}.txt",
            unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|",
            target_lang=lang)
        model.config._name_or_path = BASE
        model.config.vocab_size = tok.vocab_size
        model.load_adapter(lang, local_files_only=True)
        model.eval()
        inputs = torch.from_numpy(audio)[None]
        t0 = time.time()
        with torch.no_grad():
            logits = model(inputs).logits
        el = time.time() - t0
        probs = logits[0].softmax(-1)
        ids = logits[0].argmax(-1)
        raw = tok.decode(ids)
        conf = probs.max(-1).values
        print(f"forward {el:.1f}s for {dur}s ({el/dur:.2f}x RT) frames={logits.shape[1]} "
              f"mean_conf={float(conf.mean()):.3f} min_conf={float(conf.min()):.3f}", flush=True)
        print(f"RAW      : {raw!r}", flush=True)
        sro = to_sro(raw) if "syllabics" in lang else raw
        print(f"SRO      : {sro!r}", flush=True)
        print(f"SYLLABICS: {to_syllabics(sro)!r}" if "syllabics" not in lang else f"SYLLABICS: {raw!r}", flush=True)
        print(f"ANGLICIZED: {anglicize(sro)!r}", flush=True)
        np.save(f"scratch/logits_{lang}.npy", logits[0].numpy())
        np.save(f"scratch/probs_{lang}.npy", probs.numpy())
        with open(f"scratch/vocab_{lang}.txt", "w") as f:
            for i in range(tok.vocab_size):
                f.write(tok.convert_ids_to_tokens(i) + "\n")
        print("saved logits", flush=True)


if __name__ == "__main__":
    main()
