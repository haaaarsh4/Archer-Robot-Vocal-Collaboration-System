import sys, pathlib, numpy as np, librosa
sys.path.insert(0,'.')
from analysis.cree_asr import SAMPLE_RATE, CreeASRConfig, _condition, find_voiced_spans, isolate_vocals
G=pathlib.Path('data/PlainsCree/CreeLiteracyOrg/CongratsGrads')
lines=(G/'congrats_grads_cr.txt').read_text().strip().split('\n')
files=sorted(p for p in G.iterdir() if p.suffix.lower() in ('.m4a','.wav','.mp3'))
for margin in (30.0, 24.0, 20.0, 16.0, 12.0):
    print(f"\n=== silence_margin_db={margin}")
    for p in files:
        y,_=librosa.load(str(p), sr=SAMPLE_RATE, mono=True)
        cfg=CreeASRConfig(isolation='off', silence_margin_db=margin)
        v,_=isolate_vocals(y,SAMPLE_RATE,'off'); c,_=_condition(v,cfg,SAMPLE_RATE)
        spans=find_voiced_spans(c,cfg,SAMPLE_RATE)
        voiced=sum(b-a for a,b in spans)
        print(f"  {p.name:26s} dur {y.size/SAMPLE_RATE:6.2f}s  voiced {voiced:6.2f}s "
              f"({voiced/(y.size/SAMPLE_RATE)*100:5.1f}%)  spans {[(round(a,2),round(b,2)) for a,b in spans]}")
