"""Temporary: first real decode of a native-speaker recording through the stack.

Prints, for each adapter, the greedy CTC decode (what the acoustics actually
hear) next to the beam search's ideas, so the language model's contribution is
visible instead of inferred.
"""
import sys, time, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.cree_asr import CreeTranscriber, CreeASRConfig, SAMPLE_RATE, _tidy, isolate_vocals, _condition

path = sys.argv[1] if len(sys.argv) > 1 else \
    "data/PlainsCree/CreeLiteracyOrg/CongratsGrads/Grad_JOkimasis_line2.wav"
isolate = sys.argv[2] if len(sys.argv) > 2 else "harmonic"
ref = sys.argv[3] if len(sys.argv) > 3 else "(unknown)"

import librosa
y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
print(f"file: {path}")
print(f"ref : {ref}")
print(f"loaded {len(y)/sr:.2f}s")

cfg = CreeASRConfig(isolation=isolate, beam_width=24, symbols_per_frame=8, n_best=6)
tr = CreeTranscriber(cfg)
assert tr.available, tr._asr.load_error
vocals, method = isolate_vocals(y, SAMPLE_RATE, isolate)
cond, _ = _condition(vocals, cfg)
spans = tr.cfg and None

import analysis.cree_asr as M
lo, hi = 0, len(cond)
clip = cond[lo:hi]

for adapter in cfg.adapters:
    posts = tr._asr.posteriors(clip, SAMPLE_RATE)
    logp = posts[adapter]
    pieces = tr._asr.tokenizer_for(adapter)[1]
    vocab = [p[1] for p in pieces]
    # greedy CTC: argmax, collapse repeats, drop blank
    ids = np.argmax(logp, axis=-1)
    blank = next(i for i, p in enumerate(pieces) if p[2])
    prev, greedy = None, []
    for i in ids:
        if i != prev:
            prev = i
            if i != blank:
                greedy.append(vocab[i])
    print(f"\n===== {adapter}  frames={logp.shape[0]} vocab={len(pieces)}")
    print(f"GREEDY: {_tidy(''.join(greedy))!r}")
    t0 = time.time()
    hyp = tr.decoder().decode(logp, pieces, n_best=8)
    dt = time.time() - t0
    print(f"beam took {dt:.2f}s")
    for s, text in hyp:
        words = [w for w in text.split(" ") if w]
        known = sum(1 for w in words if tr.text_model.is_known(w))
        print(f"  {s:9.2f}  known={known}/{len(words)}  {text}")

print("\n--- vocable model ---")
v = tr._vocables
print("available:", v.load(), "error:", v.error)
if v.available:
    ph = v.phones(clip, SAMPLE_RATE)
    print("phones:", ph[:60])
    for i, syl in enumerate(v._syllabify(ph)[0]):
        print(f"   syl {i}: {syl}  -> snap {v._snap([syl])}")
