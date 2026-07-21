import queue
import numpy as np
import pyaudio
from loguru import logger
from config.config_loader import get_config

# Maps the dtype string in config.yaml to the matching pyaudio format
# constant, instead of hardcoding paFloat32 and silently ignoring whatever
# audio.dtype is actually set to.
_DTYPE_TO_PYAUDIO_FORMAT = {
    "float32": pyaudio.paFloat32,
    "int16":   pyaudio.paInt16,
    "int32":   pyaudio.paInt32,
}
_DTYPE_TO_NUMPY = {
    "float32": np.float32,
    "int16":   np.int16,
    "int32":   np.int32,
}


class AudioCapture:

    def __init__(self):
        cfg = get_config()
        self.sample_rate  = cfg["audio"]["sample_rate"]
        self.frame_size   = cfg["audio"]["frame_size"]
        self.channels     = cfg["audio"]["channels"]
        self.input_device = cfg["audio"]["input_device"]
        self.dtype_name   = cfg["audio"].get("dtype", "float32")
        self._pa_format   = _DTYPE_TO_PYAUDIO_FORMAT.get(self.dtype_name, pyaudio.paFloat32)
        self._np_dtype    = _DTYPE_TO_NUMPY.get(self.dtype_name, np.float32)

        # Bounded, not unbounded. If whatever's reading this queue falls
        # behind (a slow frame, a GC pause, anything), an unbounded queue
        # just keeps growing and the audio you eventually process is
        # further and further behind real time. Dropping the oldest frame
        # once the queue is full keeps latency bounded instead, which
        # matters a lot more than never losing a frame for a live pipeline.
        self._queue_maxsize = int(cfg["audio"].get("queue_maxsize", 64))
        self.queue: queue.Queue = queue.Queue(maxsize=self._queue_maxsize)

        self._p: pyaudio.PyAudio | None = None
        self._stream = None
        self._running = False

    def list_devices(self):
        # Reuses self._p instead of spinning up a second, separate
        # PyAudio/PortAudio session just to list devices.
        owns_instance = self._p is None
        p = self._p or pyaudio.PyAudio()
        try:
            for i in range(p.get_device_count()):
                print(f"Device number ({i}): {p.get_device_info_by_index(i).get('name')}")
        finally:
            if owns_instance:
                p.terminate()

    def _callback(self, in_data, frame_count, time_info, status):
        samples = np.frombuffer(in_data, dtype=self._np_dtype).copy()
        try:
            self.queue.put_nowait(samples)
        except queue.Full:
            # Drop the oldest queued frame to make room, rather than
            # blocking the audio callback thread (which must never block)
            # or growing the queue without limit.
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(samples)
            except queue.Full:
                pass
        return (None, pyaudio.paContinue)

    def start(self):
        if self._running:
            logger.warning("AudioCapture.start() called while already running, ignoring")
            return

        # PyAudio.terminate() shuts down the underlying PortAudio session
        # for good, a PyAudio instance can't be reused to open a new stream
        # after that. Re-creating it here means start() -> stop() -> start()
        # actually works, instead of failing the second time.
        self._p = pyaudio.PyAudio()

        logger.info(f"Opening audio stream: device={self.input_device}, rate={self.sample_rate}Hz, "
                    f"frame={self.frame_size}, format={self.dtype_name}")
        try:
            self._stream = self._p.open(
                format=self._pa_format,
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
        except Exception as e:
            logger.error(f"Failed to open audio stream (device={self.input_device}): {e}")
            self._p.terminate()
            self._p = None
            raise

    def stop(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._p:
            self._p.terminate()
            self._p = None
        self._running = False
        logger.info("Audio capture stopped")
