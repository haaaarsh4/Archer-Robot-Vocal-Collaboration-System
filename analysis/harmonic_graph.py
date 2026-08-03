from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import yaml

@dataclass
class NoteEvent:
    start_s: float
    end_s: float
    hz: float
    duration_s: float
    followed_by_silence: bool = False


def hz_to_cents(hz: float, ref_hz: float) -> float:
    return 1200.0 * np.log2(hz / ref_hz)


def track_pitch(y: np.ndarray, sr: int, fmin: float, fmax: float,
                 frame_length: int = 2048, hop_length: int = 512):
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    return f0, voiced_flag, times


def segment_notes(f0: np.ndarray, voiced: np.ndarray, times: np.ndarray,
                   cents_tolerance: float, min_duration_s: float,
                   silence_gap_s: float) -> list[NoteEvent]:
    events: list[NoteEvent] = []
    current_hzs: list[float] = []
    current_start: Optional[float] = None

    def flush(end_time: float):
        nonlocal current_hzs, current_start
        if current_start is not None and current_hzs:
            dur = end_time - current_start
            if dur >= min_duration_s:
                med = float(np.median(current_hzs))
                events.append(NoteEvent(current_start, end_time, med, dur))
        current_hzs, current_start = [], None

    for i, t in enumerate(times):
        hz = f0[i]
        is_voiced = (voiced[i] if voiced is not None else True) and hz and not np.isnan(hz)
        if is_voiced:
            if current_start is None:
                current_start, current_hzs = t, [hz]
            else:
                ref = np.median(current_hzs)
                if abs(hz_to_cents(hz, ref)) <= cents_tolerance:
                    current_hzs.append(hz)
                else:
                    flush(t)
                    current_start, current_hzs = t, [hz]
        else:
            flush(t)
    flush(float(times[-1]) if len(times) else 0.0)

    for i, ev in enumerate(events):
        gap_after = (events[i + 1].start_s if i + 1 < len(events) else 999) - ev.end_s
        ev.followed_by_silence = gap_after >= silence_gap_s

    return events


@dataclass
class CandidateTone:
    tone_id: str
    reference_hz: float
    total_duration_s: float
    occurrence_count: int
    max_single_hold_s: float
    is_low_register: bool


def cluster_tones(events: list[NoteEvent], cents_tolerance: float, low_register_hz: float) -> list[CandidateTone]:
    if not events:
        return []
    sorted_events = sorted(events, key=lambda e: e.hz)
    clusters: list[list[NoteEvent]] = []
    for ev in sorted_events:
        placed = False
        for cluster in clusters:
            ref = float(np.median([e.hz for e in cluster]))
            if abs(hz_to_cents(ev.hz, ref)) <= cents_tolerance:
                cluster.append(ev)
                placed = True
                break
        if not placed:
            clusters.append([ev])

    clusters.sort(key=lambda c: np.median([e.hz for e in c]))
    tones = []
    for i, cluster in enumerate(clusters):
        hzs = [e.hz for e in cluster]
        durations = [e.duration_s for e in cluster]
        ref_hz = float(np.median(hzs))
        tones.append(CandidateTone(
            tone_id=f"tone_{i + 1:02d}",
            reference_hz=ref_hz,
            total_duration_s=float(sum(durations)),
            occurrence_count=len(cluster),
            max_single_hold_s=float(max(durations)),
            is_low_register=ref_hz < low_register_hz,
        ))
    return tones


def assign_tone_id(hz: float, tones: list[CandidateTone]) -> str:
    best_id, best_diff = tones[0].tone_id, float("inf")
    for tone in tones:
        diff = abs(hz_to_cents(hz, tone.reference_hz))
        if diff < best_diff:
            best_diff, best_id = diff, tone.tone_id
    return best_id


def main():
    parser = argparse.ArgumentParser(description="Build Archer's harmonic graph from a folder of recordings, end to end.")
    parser.add_argument("input_dir", help="Folder of mp3/wav/m4a/flac recordings")
    parser.add_argument("--output", default="data/cree_harmony/archer_harmonic_graph.yaml")
    parser.add_argument("--fmin", type=float, default=65.0, help="Lowest pitch to track, Hz")
    parser.add_argument("--fmax", type=float, default=800.0, help="Highest pitch to track, Hz")
    parser.add_argument("--note-tolerance-cents", type=float, default=35.0,
                         help="How much pitch can drift within one held note")
    parser.add_argument("--tone-cluster-cents", type=float, default=50.0,
                         help="How close two notes must be to count as the same tone")
    parser.add_argument("--min-note-duration-s", type=float, default=0.12)
    parser.add_argument("--silence-gap-s", type=float, default=0.35,
                         help="Gap length that counts as a phrase ending")
    parser.add_argument("--low-register-hz", type=float, default=165.0,
                         help="Tones below this are eligible as the drone root")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".flac")
    )
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return

    print(f"Found {len(audio_files)} recording(s). Tracking pitch through each one...")

    per_file_events: dict[str, list[NoteEvent]] = {}
    all_events: list[NoteEvent] = []
    for path in audio_files:
        print(f"  {path.name}")
        y, sr = librosa.load(str(path), sr=None, mono=True)
        f0, voiced, times = track_pitch(y, sr, args.fmin, args.fmax)
        events = segment_notes(f0, voiced, times, args.note_tolerance_cents,
                                args.min_note_duration_s, args.silence_gap_s)
        per_file_events[path.name] = events
        all_events.extend(events)

    if not all_events:
        print("No stable pitched notes detected in any recording. Nothing to build.")
        return

    print(f"Clustering {len(all_events)} sung notes across all recordings into shared tones...")
    tones = cluster_tones(all_events, args.tone_cluster_cents, args.low_register_hz)

    tones_by_duration = sorted(tones, key=lambda t: -t.total_duration_s)
    drone_tone = next((t for t in tones_by_duration if t.is_low_register), None)
    if drone_tone:
        drone_tone.tone_id = "drone_root"

    print(f"Found {len(tones)} distinct tones across the corpus.")
    if drone_tone:
        print(f"  Drone root: {drone_tone.reference_hz:.1f} Hz, held {drone_tone.total_duration_s:.1f}s total across all songs")

    print("Counting transitions and phrase endings across every recording...")
    transition_counts: dict[tuple[str, str], int] = {}
    phrase_final_counts: dict[tuple[str, str], int] = {}

    for filename, events in per_file_events.items():
        tone_ids = [assign_tone_id(e.hz, tones) for e in events]
        for i in range(len(events) - 1):
            a, b = tone_ids[i], tone_ids[i + 1]
            if a != b:
                transition_counts[(a, b)] = transition_counts.get((a, b), 0) + 1
        for i, ev in enumerate(events):
            if ev.followed_by_silence and i > 0:
                key = (tone_ids[i - 1], tone_ids[i])
                phrase_final_counts[key] = phrase_final_counts.get(key, 0) + 1

    relations = []
    if transition_counts:
        max_t = max(transition_counts.values())
        for (a, b), count in sorted(transition_counts.items(), key=lambda kv: -kv[1]):
            relations.append({
                "from_tone": a, "to_tone": b, "relation_type": "transitions_to",
                "weight": round(count / max_t, 2), "protocol_sensitive": False,
                "source_note": f"seen {count}x across {len(audio_files)} recording(s)",
            })
    if phrase_final_counts:
        max_p = max(phrase_final_counts.values())
        for (a, b), count in sorted(phrase_final_counts.items(), key=lambda kv: -kv[1]):
            relations.append({
                "from_tone": a, "to_tone": b, "relation_type": "phrase_final_candidate",
                "weight": round(count / max_p, 2), "protocol_sensitive": False,
                "source_note": f"phrase ends here {count}x across {len(audio_files)} recording(s)",
            })

    graph = {
        "_generated_from": [p.name for p in audio_files],
        "_generated_by": "analysis/build_archer_harmonic_graph.py",
        "_note": (
            "Built automatically from acoustic pitch patterns. Tones and "
            "transitions reflect what was actually sung and how often. "
            "protocol_sensitive fields default to false and can be edited "
            "directly in this file at any time."
        ),
        "tones": [
            {
                "id": t.tone_id,
                "reference_hz": round(t.reference_hz, 2),
                "role": "drone_root" if t is drone_tone else "",
                "protocol_sensitive": False,
                "occurrence_count": t.occurrence_count,
                "total_duration_s": round(t.total_duration_s, 2),
                "max_single_hold_s": round(t.max_single_hold_s, 2),
            }
            for t in tones
        ],
        "relations": relations,
    }

    with open(output_path, "w") as f:
        yaml.dump(graph, f, sort_keys=False, allow_unicode=True)

    print(f"\nDone. Wrote {output_path}")
    print(f"  {len(tones)} tones, {len(relations)} relations, built from {len(audio_files)} recording(s).")
    print("HarmonyEngine will pick this file up automatically on next run.")


if __name__ == "__main__":
    main()