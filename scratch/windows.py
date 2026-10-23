import sys, pathlib, numpy as np, librosa
sys.path.insert(0,'.')
from analysis.cree_asr import SAMPLE_RATE, CreeASRConfig, _condition, isolate_vocals, get_transcriber
G=pathlib.Path('data/PlainsCree/CreeLiteracyOrg/CongratsGrads')
cfg=CreeASRConfig(isolation='off')
tr=get_transcriber(cfg)
d=tr.decoder()
for name in ['Grad_SBird_line6.m4a','Grad_WJackson_line5.m4a']:
    y,_=librosa.load(str(G/name), sr=SAMPLE_RATE, mono=True)
    v,_=isolate_vocals(y,SAMPLE_RATE,'off'); c,_=_condition(v,cfg,SAMPLE_RATE)
    dur=y.size/SAMPLE_RATE
    print(f"\n=== {name} ({dur:.2f}s)")
    # 2-second windows with 0.5 s hop
    for a in np.arange(0, dur-1.0, 0.5):
        b=min(dur, a+2.5)
        clip=c[int(a*SAMPLE_RATE):int(b*SAMPLE_RATE)]
        if clip.size < int(0.5*SAMPLE_RATE): continue
        post=tr._asr.posteriors(clip,SAMPLE_RATE)
        pieces=tr._asr.tokenizer_for(cfg.adapters[0])[1]
        g=d.greedy(post[cfg.adapters[0]], pieces)
        cands=tr.candidates(post,[])
        best=cands[0][1] if cands else ''
        print(f"  {a:5.1f}-{b:5.1f} greedy: {g[:70]}")
