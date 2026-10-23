import sys, numpy as np, librosa
sys.path.insert(0,'.'); sys.path.insert(0,'tools')
from check_cree_transcription import synthesise, normalise, cer, coverage_error, extra_rate
from analysis.cree_asr import SAMPLE_RATE, CreeASRConfig, _condition, find_voiced_spans, isolate_vocals, get_transcriber
cfg=CreeASRConfig(isolation='off')
tr=get_transcriber(cfg)
refs=["ê kî sâkihêt ,", "kî miskawâw ,", "maskosis otâpâsow kiskinohamâtôtâpânâsk",
      "nipostisken nipakâsimowayiwinisa ekwa nimaskisinihkanisa"]
for ref in refs:
    audio,sr=synthesise(ref)
    if sr!=SAMPLE_RATE:
        audio=librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    dur=len(audio)/SAMPLE_RATE
    v,_=isolate_vocals(audio,SAMPLE_RATE,'off'); c,_=_condition(v,cfg,SAMPLE_RATE)
    spans=find_voiced_spans(c,cfg,SAMPLE_RATE)
    print(f"\n=== ref: {ref}   tts_dur {dur:.2f}s  spans {[(round(a,2),round(b,2)) for a,b in spans]}")
    for ad in cfg.adapters:
        post=tr._asr.posteriors(c,SAMPLE_RATE)
        pieces=tr._asr.tokenizer_for(ad)[1]
        g=tr.decoder().greedy(post[ad],pieces)
        print(f"   greedy[{ad.split('_')[-1]}]: {g}")
    # how many chars per second is the ref?
    n=len(normalise(ref).replace(' ',''))
    print(f"   ref chars {n} -> {n/dur:.1f} chars/s ; hyp chars/s check")
    res=tr.transcribe(audio,SAMPLE_RATE,isolate='off')
    print(f"   chosen: {res.sro}  CER {cer(ref,res.sro)*100:.1f}% cov {coverage_error(ref,res.sro)*100:.1f}% extra {extra_rate(ref,res.sro)*100:.1f}%")
    print(f"   segments: {[(round(s.start,2),round(s.end,2),s.kind,s.sro) for s in res.segments]}")
