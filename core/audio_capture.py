import queue
import numpy as np
import pyaudio
from loguru import logger
from config.config_loader import get_config

class AudioCapture:

    def __init__(self):
        cfg = get_config()
        self.sample_rate  = cfg["audio"]["sample_rate"]
        self.frame_size   = cfg["audio"]["frame_size"]
        self.channels     = cfg["audio"]["channels"]
        self.input_device = cfg["audio"]["input_device"]

        self.queue = queue.Queue()
        self._p      = pyaudio.PyAudio()
        self._stream = None
        self._running = False

    def list_devices(self):
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            print(f"Device number ({i}): {p.get_device_info_by_index(i).get('name')}")
        p.terminate()

    def _callback(self, in_data, frame_count, time_info, status):
        samples = np.frombuffer(in_data, dtype=np.float32).copy()
        self.queue.put_nowait(samples)
        return (None, pyaudio.paContinue)

    def start(self):
        logger.info(f"Opening audio stream: device={self.input_device}, rate={self.sample_rate}Hz, frame={self.frame_size}")
        self._stream = self._p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device,
            frames_per_buffer=self.frame_size,
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        self._running = True
        logger.info("Audio capture started")

    def stop(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._p.terminate()
        self._running = False
        logger.info("Audio capture stopped")