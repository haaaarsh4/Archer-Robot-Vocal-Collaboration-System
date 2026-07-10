from __future__ import annotations

import threading
import time
from typing import Optional
from gpiozero import Button  # only present on Raspberry Pi-style hardware

from loguru import logger


class ProtocolGuard:
    def __init__(self, cfg: dict):
        pcfg = cfg.get("protocol", {})
        self._lock = threading.Lock()
        self._enabled: bool = bool(pcfg.get("start_enabled", True))

        self.sensitive_phoneme_classes = set(pcfg.get("sensitive_phoneme_classes", []))

        self.sound_cue_enabled   = bool(pcfg.get("sound_cue_enabled", True))
        self._cue_hz_range       = tuple(pcfg.get("sound_cue_hz_range", [80, 140]))
        self._cue_reps_required  = int(pcfg.get("sound_cue_repetitions", 3))
        self._cue_max_gap_s      = float(pcfg.get("sound_cue_max_gap_s", 0.6))
        self._cue_cooldown_s     = float(pcfg.get("sound_cue_cooldown_s", 2.0))
        self._cue_hits: list[float] = []
        self._last_toggle_time: float = 0.0

        self._gpio_pin = pcfg.get("physical_cue_gpio_pin", None)
        self._gpio = None
        if self._gpio_pin is not None:
            self._init_physical_cue()

        logger.info(f"ProtocolGuard initialized — protocol_enabled={self._enabled}")

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def enable(self, source: str = "manual") -> None:
        self._set(True, source)

    def disable(self, source: str = "manual") -> None:
        self._set(False, source)

    def toggle(self, source: str = "manual") -> bool:
        with self._lock:
            new_state = not self._enabled
        self._set(new_state, source)
        return new_state

    def status(self) -> dict:
        with self._lock:
            return {
                "enabled": self._enabled,
                "last_toggle_time": self._last_toggle_time,
                "physical_cue_armed": self._gpio is not None,
                "sound_cue_enabled": self.sound_cue_enabled,
            }

    def _set(self, state: bool, source: str) -> None:
        with self._lock:
            if state == self._enabled:
                return
            self._enabled = state
            self._last_toggle_time = time.time()
        logger.info(f"Protocol {'ENABLED' if state else 'DISABLED'} via {source} cue")

    def _init_physical_cue(self) -> None:
        try:
            self._gpio = Button(self._gpio_pin, bounce_time=0.05)
            self._gpio.when_pressed = lambda: self.toggle(source="physical")
            logger.info(f"Physical sovereignty cue armed on GPIO pin {self._gpio_pin}")
        except Exception as e:
            logger.warning(
                f"Physical cue unavailable ({e}). This is expected on a cloud "
                "server (e.g. Render) with no attached hardware — use the sound "
                "cue or the manual /protocol/toggle endpoint instead."
            )
            self._gpio = None

    def check_sound_cue(self, archer_hz: Optional[float], is_voiced: bool, now_s: float) -> None:
        if not self.sound_cue_enabled:
            return

        if now_s - self._last_toggle_time < self._cue_cooldown_s:
            return  # debounce: ignore further hits right after a toggle

        in_cue_range = (
            is_voiced and archer_hz is not None
            and self._cue_hz_range[0] <= archer_hz <= self._cue_hz_range[1]
        )

        if in_cue_range:
            if self._cue_hits and (now_s - self._cue_hits[-1]) > self._cue_max_gap_s:
                self._cue_hits.clear()
            self._cue_hits.append(now_s)
            if len(self._cue_hits) >= self._cue_reps_required:
                self._cue_hits.clear()
                self.toggle(source="sound")
        elif self._cue_hits and (now_s - self._cue_hits[-1]) > self._cue_max_gap_s:
            self._cue_hits.clear()
