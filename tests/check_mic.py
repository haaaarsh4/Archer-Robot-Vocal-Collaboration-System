import sounddevice as sd
import numpy as np
import time

DEVICE = 6
SAMPLE_RATE = 44100
FRAME_SIZE = 1024
DURATION = 10

print(f"Listening for {DURATION}s - play song on laptop speakers NOW")
print(f"{'Time':>6}  {'RMS':>10}  Bar")
print("-" * 50)

start = time.perf_counter()
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                    blocksize=FRAME_SIZE, device=DEVICE) as stream:
    while time.perf_counter() - start < DURATION:
        frame, _ = stream.read(FRAME_SIZE)
        rms = float(np.sqrt(np.mean(frame.flatten() ** 2)))
        elapsed = time.perf_counter() - start
        bar = "#" * int(rms * 50)
        print(f"{elapsed:6.1f}s  {rms:10.6f}  {bar}")
        time.sleep(0.1)