"""One live hum chunk, straight from the endpoint the page posts to.

8 seconds of hummingbird.mp3 where the singing is, exactly the body
_convertAudioChunkViaNeural builds, and the same measurements as
diag_live_hum_chunks on whatever comes back, next to the murmur the DSP engine
renders for the same seconds of the track.
"""
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, "scratch")
from diag_live_hum_chunks import load, measure, post_multipart, encode_wav, SR, BASE  # noqa: E402

track = "frontend/assets/audio/hummingbird.mp3"
audio = load(track)

for start in (16.0, 24.0):
    slice_ = audio[int(start * SR):int((start + 8) * SR)]
    print(f"\n=== live hum chunk @ {start:.0f}s (8 s of the mix) ===")
    measure("the chunk sent", slice_)
    fields = {"voice_index": "0", "transpose_semitones": "0", "pad_seconds": "0.3",
              "instruments_enabled": "false", "vibrato_amount": "45", "mode": "solo_humming"}
    data, headers = post_multipart(f"{BASE}/api/neural/convert-track", fields,
                                   {"file": ("chunk.wav", encode_wav(slice_))})
    out = load(io_bytes := __import__("io").BytesIO(data)) if False else None
    out, sr = sf.read(__import__("io").BytesIO(data), dtype="float32", always_2d=False)
    out = np.asarray(out, dtype=np.float32)
    print(f"    headers: X-Neural-Applied={headers.get('X-Neural-Applied')} "
          f"X-Backing-Bed={headers.get('X-Backing-Bed')} len {len(out) / sr:.2f}s")
    measure("  what comes back", out)
