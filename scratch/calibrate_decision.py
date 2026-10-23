"""Temporary: gather the numbers the lexical-vs-vocable decision has to be based on."""
import sys, pathlib, re, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.cree_asr import (CreeTranscriber, CreeASRConfig, SAMPLE_RATE,
                               isolate_vocals, _condition, find_voiced_spans)
import librosa

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRADS = ROOT / "data/PlainsCree/CreeLiteracyOrg/CongratsGrads"
lines = [l.strip() for l in (GRADS / "congrats_grads_cr.txt").read_text().splitlines() if l.strip()]

pairs = []
for p in sorted(GRADS.iterdir()):
    if p.suffix.lower() not in (".m4a", ".wav", ".mp3"):
        continue
    m = re.search(r"line(\d+)(?:_(\d+))?", p.stem)
    if not m:
        continue
    idx = [int(m.group(1))] + ([int(m.group(2))] if m.group(2) else [])
    ref = " ".join(lines[i-1] for i in idx if i-1 < len(lines))
    pairs.append((p, ref))

cfg = CreeASRConfig(isolation="harmonic", beam_width=40, symbols_per_frame=10, n_best=6)
tr = CreeTranscriber(cfg)
assert tr.available
lm = tr.text_model
dec = tr.decoder()

print(f"{'file':26s} {'lexical best hypothesis':60s} {'tq':>6s} {'kn':>3s} {'lg':>3s} | {'vocable':34s} {'vtq':>6s} {'vlg':>3s}")
for path, ref in pairs:
    y, _sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    vocals, _m = isolate_vocals(y, SAMPLE_RATE, cfg.isolation)
    cond, _n = _condition(vocals, cfg, SAMPLE_RATE)
    spans = find_voiced_spans(cond, cfg, SAMPLE_RATE)
    # transcribe all spans, but report span 0 in detail
    for span_i, (a, b) in enumerate(spans[:1]):
        clip = cond[int(a*SAMPLE_RATE):int(b*SAMPLE_RATE)]
        posts = tr._asr.posteriors(clip, SAMPLE_RATE)
        best = None
        for adapter in cfg.adapters:
            pieces = tr._asr.tokenizer_for(adapter)[1]
            hyps = dec.decode(posts[adapter], pieces, n_best=6)
            for score, text in hyps:
                words = [w for w in text.split() if w]
                kn = sum(1 for w in words if lm.is_known(w))
                lg = sum(1 for w in words if lm.is_legal_word(w))
                tq = lm.text_quality(text)
                cand = (score, text, tq, kn, lg, adapter)
                if best is None or score > best[0]:
                    best = cand
        _s, text, tq, kn, lg, adapter = best
        vtext, vconf, vnotes = tr._vocables.transcribe(clip, SAMPLE_RATE)
        vwords = [w for w in vtext.split() if w]
        vlg = sum(1 for w in vwords if lm.is_legal_word(w))
        print(f"{path.name[:26]:26s} {text[:60]:60s} {tq:6.2f} {kn:3d} {lg:3d} | {vtext[:34]:34s} {lm.text_quality(vtext):6.2f} {vlg:3d}")
        print(f"    ref: {ref}   [{adapter}]")
print()
print("reference lines, for the record:")
for i, l in enumerate(lines, 1):
    print(f"  {i}: {l}   (tq {lm.text_quality(l):.2f})")
