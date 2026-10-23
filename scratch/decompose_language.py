"""Temporary: which term of the language objective is failing?"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from analysis.cree_asr import (CreeTranscriber, CreeASRConfig, SAMPLE_RATE,
                               isolate_vocals, _condition, find_voiced_spans,
                               _text_to_pieces, _CANONICAL_PIECES)
import librosa

PATH = pathlib.Path("data/PlainsCree/CreeLiteracyOrg/CongratsGrads/Grad_JOkimasis_line2.wav")
REF = "mistahi kikihcêyimitinân êyahkohtêyêk"
CFG = CreeASRConfig(isolation="off")
tr = CreeTranscriber(CFG); assert tr.available
lm = tr.text_model; dec = tr.decoder()

TEXTS = [REF, "miti êkikihciyêyimitinân êwiyâhkohtânêk", "ti kitinê iskok", "ta", "mê"]

def decompose(text):
    pieces = _text_to_pieces(text, _CANONICAL_PIECES)
    total_char, hist, word = 0.0, "", ""
    word_charge = 0.0
    phono_total = 0.0
    words = []
    for piece in pieces:
        if piece == " ":
            if word:
                c = dec._word_charge(word)
                word_charge += c
                phono_total += CFG.phono_weight * lm.phonotactic_score(word)
                words.append(word)
            hist, word = "", ""
            continue
        total_char += lm.char_logprob(hist, piece)
        hist += piece
        word += piece
    if word:
        word_charge += dec._word_charge(word)
        phono_total += CFG.phono_weight * lm.phonotactic_score(word)
        words.append(word)
    n = len(pieces)
    return {
        "chars": n,
        "char_lm": total_char,
        "char_per": total_char / max(1, n),
        "word_charge": word_charge,
        "phono": phono_total,
        "words": words,
        "known_words": [lm.is_known(w) for w in words],
        "text_quality": lm.text_quality(text),
    }

print(f"{'text':52s} {'n':>3s} {'charLM':>8s} {'/ch':>6s} {'word':>7s} {'phono':>8s} {'total':>8s} {'tq':>6s}")
for t in TEXTS:
    d = decompose(t)
    tot = d["char_lm"] + d["word_charge"] + d["phono"]
    print(f"{t[:52]:52s} {d['chars']:3d} {d['char_lm']:8.1f} {d['char_per']:6.2f} "
          f"{d['word_charge']:7.1f} {d['phono']:8.1f} {tot:8.1f} {d['text_quality']:6.2f}")
    print(f"     words={d['words']} known={d['known_words']}")

# now the acoustics of the same texts
y, _ = librosa.load(str(PATH), sr=SAMPLE_RATE, mono=True)
v, _m = isolate_vocals(y, SAMPLE_RATE, "harmonic")
cond, _ = _condition(v, CFG, SAMPLE_RATE)
a, b = find_voiced_spans(cond, CFG, SAMPLE_RATE)[0]
clip = cond[int(a*SAMPLE_RATE):int(b*SAMPLE_RATE)]
posts = tr._asr.posteriors(clip, SAMPLE_RATE)
print()
for adapter in CFG.adapters:
    logp = posts[adapter]; pieces = tr._asr.tokenizer_for(adapter)[1]
    frames = logp.shape[0]
    print(f"--- {adapter} frames={frames}")
    for t in TEXTS:
        ac = dec.acoustic_value(logp, pieces, t)
        print(f"    {t[:40]:42s} ac/f = {ac/frames:8.4f}   ac = {ac:9.1f}")
