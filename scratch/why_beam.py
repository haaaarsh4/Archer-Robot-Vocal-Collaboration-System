"""Temporary: why does the beam produce worse text than greedy, and what ranks where?"""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from analysis.cree_asr import (CreeTranscriber, CreeASRConfig, SAMPLE_RATE,
                               isolate_vocals, _condition, find_voiced_spans)
import librosa

PATH = pathlib.Path("data/PlainsCree/CreeLiteracyOrg/CongratsGrads/Grad_JOkimasis_line2.wav")
REF = "mistahi kikihcêyimitinân êyahkohtêyêk"

for lm_weight, ins, word_w, phono_w in [(0.45, 1.7, 2.0, 1.6), (0.0, 0.6, 2.0, 1.6), (0.15, 0.6, 2.0, 1.6)]:
    cfg = CreeASRConfig(isolation="off", beam_width=48, symbols_per_frame=10, n_best=8,
                        lm_weight=lm_weight, insertion_bonus=ins,
                        word_weight=word_w, phono_weight=phono_w)
    tr = CreeTranscriber(cfg); assert tr.available
    y, _ = librosa.load(str(PATH), sr=SAMPLE_RATE, mono=True)
    v, _m = isolate_vocals(y, SAMPLE_RATE, "harmonic")
    cond, _ = _condition(v, cfg, SAMPLE_RATE)
    spans = find_voiced_spans(cond, cfg, SAMPLE_RATE)
    a, b = spans[0]
    clip = cond[int(a*SAMPLE_RATE):int(b*SAMPLE_RATE)]
    posts = tr._asr.posteriors(clip, SAMPLE_RATE)
    dec = tr.decoder()
    print(f"\n########## lm={lm_weight} ins={ins} word={word_w} phono={phono_w}")
    for adapter in cfg.adapters:
        logp = posts[adapter]; pieces = tr._asr.tokenizer_for(adapter)[1]
        frames = logp.shape[0]
        g = dec.greedy(logp, pieces)
        ga, gl, gt = dec.score(logp, pieces, g)
        ra, rl, rt = dec.score(logp, pieces, REF)
        print(f"--- {adapter} frames={frames}")
        print(f"  greedy={g!r}")
        print(f"     ac={ga/frames:7.3f}/f  lang={gl:8.2f} total/f={gt/frames:7.3f}")
        print(f"  REFERENCE={REF!r}")
        print(f"     ac={ra/frames:7.3f}/f  lang={rl:8.2f} total/f={rt/frames:7.3f}")
        hyps = dec.decode(logp, pieces, n_best=8)
        for s, text in hyps:
            print(f"     {s/frames:8.3f}/f  {text}")
