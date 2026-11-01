"""End-to-end: a sung take in, a finished accompaniment and mix out.

This is the layer the app and the command line both use.  It keeps the three
stages separate so each can be tested and reused:

``analyze``  -> :class:`mysong.melody.AudioAnalysis`
``harmonize`` -> :class:`mysong.harmonize.Harmonization`
``arrange``  -> :class:`mysong.accompaniment.Arrangement`

The mix places the accompaniment on the recording's beat grid.  A take that
starts with a pickup before the first beat is handled by padding the
accompaniment by the estimated beat offset; the vocal is never time-stretched.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mysong.accompaniment import (
    BAND_OFFSET_DB,
    DEFAULT_BAND,
    Arrangement,
    ArrangementStyle,
    arrange,
    band_placement_gain_db,
    level_band_stem,
    mix_with_vocal,
    render_arrangement,
)
from mysong.harmonize import Harmonizer, Harmonization, Style
from mysong.melody import AudioAnalysis, analyze_audio
from mysong.model import MySongModel
from mysong.theory import Key, parse_key


@dataclass
class SongProject:
    analysis: AudioAnalysis
    harmony: Harmonization
    arrangement: Arrangement
    mix: np.ndarray
    sample_rate: int
    vocal: np.ndarray | None = None
    #: The generated band as it sits in the mix (padded to the take's beat
    #: grid, before the voice is added).  Kept because the app plays the band
    #: under the untouched take as two sources instead of rendering it twice.
    backing: np.ndarray | None = None
    #: The same band written at the app's one reference level, which is what the
    #: page's ROBOT VOLUME player actually uses (``backing`` keeps the
    #: renderer's own level for the CLI and for the WAV stems on disk).
    band_stem: np.ndarray | None = None
    #: How far the stem has to come down to sit ``BAND_OFFSET_DB`` under this
    #: take, in dB.  Always negative for a real recording; the page applies it
    #: together with the ROBOT VOLUME slider, so the two sources are the mix.
    band_placement_db: float = 0.0
    paths: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    renderer: str = "auto"

    def to_dict(self) -> dict:
        return {
            "analysis": {
                "duration_seconds": round(float(self.analysis.duration), 3),
                "tempo_bpm": round(float(self.analysis.tempo.bpm), 3),
                "beat_offset_seconds": round(float(self.analysis.tempo.beat_offset), 3),
                "tempo_confidence": round(float(self.analysis.tempo.confidence), 3),
                "tuning_cents": round(float(self.analysis.tuning_cents), 2),
                "voiced_ratio": round(float(self.analysis.voiced_ratio), 3),
                "key_guess": self.analysis.key_guess.name(),
                "warnings": list(self.analysis.warnings),
            },
            "harmony": self.harmony.to_dict(),
            "arrangement": {
                "style": self.arrangement.style.name,
                "notes": len(self.arrangement.notes),
                "duration_seconds": round(self.arrangement.duration_seconds, 3),
                "tempo_bpm": round(float(self.arrangement.tempo_bpm), 3),
                "renderer": self.renderer,
            },
            "warnings": self.warnings,
        }

    def save(self, out_dir: str | Path, *, sample_rate: int | None = None) -> dict[str, str]:
        import soundfile as sf

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        sr = sample_rate or self.sample_rate
        paths: dict[str, str] = {}
        mix_path = out / "mix.wav"
        sf.write(mix_path, self.mix, sr, subtype="PCM_16")
        paths["mix"] = str(mix_path)
        accompaniment = render_arrangement(
            self.arrangement, sample_rate=sr, renderer=self.renderer
        )
        accompaniment_path = out / "accompaniment.wav"
        sf.write(accompaniment_path, accompaniment, sr, subtype="PCM_16")
        paths["accompaniment"] = str(accompaniment_path)
        if self.vocal is not None:
            vocal_path = out / "vocal.wav"
            sf.write(vocal_path, self.vocal, sr, subtype="PCM_16")
            paths["vocal"] = str(vocal_path)
        midi_path = out / "song.mid"
        self.arrangement.save_midi(str(midi_path), melody_notes=self.analysis.notes)
        paths["midi"] = str(midi_path)
        json_path = out / "chords.json"
        json_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        paths["json"] = str(json_path)
        self.paths = paths
        return paths


def analyze_song(
    audio: np.ndarray | str | Path,
    *,
    model: MySongModel | None = None,
    key: Key | str | None = None,
    style: Style | None = None,
    bpm: float | None = None,
    beat_offset: float | None = None,
    beats_per_bar: float = 4.0,
    sample_rate: int | None = None,
    backend: str = "pyin",
) -> tuple[AudioAnalysis, Harmonization]:
    """Analyse and harmonize a take without rendering audio."""
    model = model or MySongModel.load()
    analysis = analyze_audio(
        audio,
        sr=sample_rate,
        bpm=bpm,
        beat_offset=beat_offset,
        beats_per_bar=beats_per_bar,
        backend=backend,
    )
    harmony = Harmonizer(model).harmonize(
        analysis.melody,
        key=key,
        style=style,
        beats_per_bar=beats_per_bar,
        tempo_bpm=analysis.tempo.bpm,
    )
    return analysis, harmony


def create_song(
    audio: np.ndarray | str | Path,
    *,
    model: MySongModel | None = None,
    key: Key | str | None = None,
    style: Style | ArrangementStyle | str | None = None,
    arrangement_style: str | None = None,
    bpm: float | None = None,
    beat_offset: float | None = None,
    beats_per_bar: float = 4.0,
    sample_rate: int = 44100,
    mix_vocal: bool = True,
    vocal_gain_db: float = 0.0,
    band_gain_db: float = 0.0,
    duck_db: float = 6.0,
    count_in: bool = False,
    render: bool = True,
    backend: str = "pyin",
    renderer: str = "auto",
) -> SongProject:
    """Run the full pipeline.

    ``style`` is the *harmony* style (:class:`mysong.harmonize.Style`);
    ``arrangement_style`` names the backing band preset (``"pop"``, ``"jazz"``,
    ...).  Keeping the two separate matters: the harmony model's observation
    weight is not the same knob as "play brushes instead of a backbeat".
    """
    model = model or MySongModel.load()
    if isinstance(style, ArrangementStyle):
        arrangement_style = style.name
        style = None
    elif isinstance(style, str):
        arrangement_style = style
        style = None

    import librosa

    if isinstance(audio, (str, Path)):
        vocal, native_sr = librosa.load(str(audio), sr=sample_rate, mono=True)
    else:
        vocal = np.asarray(audio, dtype=np.float32)
        if vocal.ndim > 1:
            vocal = vocal.mean(axis=1)
        native_sr = int(sample_rate)
        if len(vocal) == 0:
            raise ValueError("empty audio")
    if native_sr != sample_rate:
        vocal = librosa.resample(vocal, orig_sr=native_sr, target_sr=sample_rate)

    analysis, harmony = analyze_song(
        vocal,
        model=model,
        key=key,
        style=style,
        bpm=bpm,
        beat_offset=beat_offset,
        beats_per_bar=beats_per_bar,
        sample_rate=sample_rate,
        backend=backend,
    )

    arrangement = arrange(
        harmony,
        style=arrangement_style or DEFAULT_BAND,
        melody=analysis.notes,
    )
    arrangement.sample_rate = sample_rate
    backing = (
        render_arrangement(arrangement, sample_rate=sample_rate, renderer=renderer)
        if render else np.zeros((1, 2), dtype=np.float32)
    )

    # The vocal file starts at recording time zero; beat zero is
    # ``beat_offset`` seconds later.  Pad the accompaniment so its first beat
    # lands on the recording's first beat, then place the vocal at zero.  The
    # voice is never stretched to fit the grid.
    lead_in = max(0.0, float(analysis.tempo.beat_offset))
    # The band's own trim (a dense band sits further back) is part of where the
    # band belongs, so it applies to the rendered mix and to the level the page
    # plays the band at -- otherwise the download and the panel would disagree.
    band_offset = BAND_OFFSET_DB + float(band_gain_db) + float(arrangement.style.level_db)
    if render:
        if lead_in > 0:
            pad = np.zeros((int(round(lead_in * sample_rate)), 2), dtype=np.float32)
            backing = np.vstack([pad, backing])
        mixed = (
            mix_with_vocal(
                backing,
                vocal,
                sample_rate=sample_rate,
                offset_seconds=0.0,
                vocal_gain_db=vocal_gain_db,
                band_offset_db=band_offset,
                duck_db=duck_db,
            )
            if mix_vocal
            else backing
        )
    else:
        mixed = backing

    band_stem = level_band_stem(backing) if render else None
    # No ``band_gain_db`` here: that is the panel's ROBOT VOLUME, applied to the
    # player, and baking it in as well would count the slider twice.
    placement_db = (
        band_placement_gain_db(
            vocal,
            sample_rate=sample_rate,
            band_offset_db=BAND_OFFSET_DB + float(arrangement.style.level_db),
        )
        if render
        else 0.0
    )

    warnings = list(analysis.warnings) + list(harmony.warnings)
    if count_in:
        warnings.append("count-in is not implemented; the take starts on its first beat")
    return SongProject(
        analysis=analysis,
        harmony=harmony,
        arrangement=arrangement,
        mix=mixed,
        sample_rate=sample_rate,
        vocal=vocal,
        backing=backing if render else None,
        band_stem=band_stem,
        band_placement_db=placement_db,
        warnings=warnings,
        renderer=renderer,
    )


def render_song(
    audio: np.ndarray | str | Path,
    out_dir: str | Path,
    **kwargs: Any,
) -> SongProject:
    """One-call convenience used by the CLI and the HTTP API."""
    project = create_song(audio, **kwargs)
    project.save(out_dir)
    return project
