import sys, pathlib, numpy as np, librosa
sys.path.insert(0,'.')
from analysis.cree_asr import SAMPLE_RATE, CreeASRConfig, _condition, find_voiced_spans, isolate_vocals, get_transcriber
G=pathlib.Path('data/PlainsCree/CreeLiteracyOrg/CongratsGrads')
cfg=CreeASRConfig(isolation='off')
tr=get_transcriber(cfg)
for name in ['Grad_SBird_line6.m4a','Grad_WJackson_line5.m4a']:
    p=G/name
    y,_=librosa.load(str(p), sr=SAMPLE_RATE, mono=True)
    v,_=isolate_vocals(y,SAMPLE_RATE,'off')
    c,_=_condition(v,cfg,SAMPLE_RATE)
    hop=int(0.01*SAMPLE_RATE); win=int(0.025*SAMPLE_RATE)
    n=1+(c.size-win)//hop
    fr=np.lib.stride_tricks.as_strided(c,shape=(n,win),strides=(hop*c.dtype.itemsize,c.dtype.itemsize)).copy()
    db=20*np.log10(np.sqrt(np.mean(fr**2,axis=1)+1e-12)+1e-12)
    step=max(1,int(0.5*SAMPLE_RATE/hop))
    print(f"\n{name}: {y.size/SAMPLE_RATE:.2f}s spans={[(round(a,2),round(b,2)) for a,b in find_voiced_spans(c,cfg,SAMPLE_RATE)]}")
    print("  dB/0.5s:", " ".join(f"{db[i]:.0f}" for i in range(0,len(db),step)))
    dur=y.size/SAMPLE_RATE
    for label,(a,b) in [('first half',(0,dur/2)),('second half',(dur/2,dur))]:
        clip=c[int(a*SAMPLE_RATE):int(b*SAMPLE_RATE)]
        post=tr._asr.posteriors(clip,SAMPLE_RATE)
        txt=' '.join(t for _s,t in tr._decoder.decode(post, post.get('_pieces')) if False) if False else None
        seg=tr._transcribe_span(clip,a,b,[],posteriors=post)
        print(f"  {label}: {seg.sro if seg else ''}")
