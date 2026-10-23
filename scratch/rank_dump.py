import sys, pathlib, numpy as np, librosa
sys.path.insert(0,'.')
from analysis.cree_asr import SAMPLE_RATE, CreeASRConfig, _condition, find_voiced_spans, isolate_vocals, get_transcriber
G=pathlib.Path('data/PlainsCree/CreeLiteracyOrg/CongratsGrads')
cfg=CreeASRConfig(isolation='off')
tr=get_transcriber(cfg)
lines=(G/'congrats_grads_cr.txt').read_text().strip().split('\n')
for name,ln in [('Grad_SBird_line6.m4a',6),('Grad_WJackson_line5.m4a',5)]:
    ref=lines[ln-1]
    y,_=librosa.load(str(G/name), sr=SAMPLE_RATE, mono=True)
    v,_=isolate_vocals(y,SAMPLE_RATE,'off'); c,_=_condition(v,cfg,SAMPLE_RATE)
    spans=find_voiced_spans(c,cfg,SAMPLE_RATE)
    print(f"\n=== {name}  ref: {ref}")
    d=tr.decoder()
    for (a,b) in spans:
        clip=c[int(a*SAMPLE_RATE):int(b*SAMPLE_RATE)]
        post=tr._asr.posteriors(clip,SAMPLE_RATE)
        cands=tr.candidates(post,[])
        for i,(rank,text,agree) in enumerate(cands):
            parts=[]
            for ad in cfg.adapters:
                lp=post.get(ad)
                if lp is None: continue
                ac,lang,tot=d.score(lp,tr._asr.tokenizer_for(ad)[1],text)
                parts.append(f"{ad.split('_')[-1]}:ac={ac:8.1f},lang={lang:6.1f}")
            print(f" {i:2d} rank={rank:9.1f} | " + " | ".join(parts) + " | " + text[:100])
