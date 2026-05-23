import queue
import threading
import numpy as np
import sounddevice as sd
from loguru import logger
from config.config_loader import get_config

class AudioCapture:

    # Load up the audio settings from the config file.
    # Each incoming frame is placed on self.queue (dumped from the callback function).
    # Also creates a placeholder for the sounddevice microphone connection
    def __init__(self):
        cfg = get_config()
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.frame_size = cfg["audio"]["frame_size"]
        self.channels = cfg["audio"]["channels"]
        self.dtype = cfg["audio"]["dtype"]
        self.input_device = cfg["audio"]["input_device"]

        self.queue: queue.Queue = queue.Queue()

        self._stream = None
        self._running = False

    # soundrive will call this function everytime the mic has a new chunk ready.
    # takes the first channel (as per our config) of the audio chunk and drops it into the queue
    def _callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio input status: {status}")

        frame = indata[:, 0].copy().astype(np.float32)
        self.queue.put_nowait(frame)

    # Create the sounddevice sd microphone connection
    def start(self):
        if self._running:
            logger.warning("AudioCapture already running")
            return

        logger.info(
            f"Opening audio stream: "
            f"device={self.input_device or 'default'}, "
            f"rate={self.sample_rate}Hz, "
            f"frame={self.frame_size} samples"
        )

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            channels=self.channels,
            dtype=self.dtype,
            device=self.input_device,
            callback=self._callback,
        )
        self._stream.start()
        self._running = True
        logger.info("Audio capture started")

    # Stop the microphone
    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
        logger.info("Audio capture stopped")

    @property
    def is_running(self):
        return self._running

    def list_devices(self):
        print(sd.query_devices())