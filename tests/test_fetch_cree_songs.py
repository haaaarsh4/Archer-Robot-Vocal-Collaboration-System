from __future__ import annotations

import json
from collections import defaultdict

from tools.fetch_cree_songs import LICENCE, LICENCE_URL, TRACKS, _write_split_views


def test_track_ids_and_titles_are_unique():
    ids = [track.youtube_id for track in TRACKS]
    titles = [track.title for track in TRACKS]
    assert len(ids) == len(set(ids))
    assert len(titles) == len(set(titles))
    assert len(TRACKS) == 27


def test_every_track_has_a_real_cc_by_licence_and_source():
    assert LICENCE == "CC BY"
    assert LICENCE_URL.startswith("https://creativecommons.org/licenses/by/")
    for track in TRACKS:
        assert track.youtube_id
        assert track.title
        assert track.singer
        assert track.style
        assert track.dialect == "Plains Cree"
        assert track.url == f"https://www.youtube.com/watch?v={track.youtube_id}"


def test_singer_splits_are_disjoint():
    splits = defaultdict(set)
    for track in TRACKS:
        splits[track.singer].add(track.split)
    leaked = {singer: values for singer, values in splits.items() if len(values) > 1}
    assert not leaked, f"singers appear in multiple splits: {leaked}"


def test_all_tracks_have_a_training_split():
    assert {track.split for track in TRACKS} <= {"train", "dev", "test"}


def test_split_views_do_not_copy_audio(tmp_path):
    paired = tmp_path / "paired"
    for song_id in ("train_song", "test_song"):
        folder = paired / "audio" / song_id
        folder.mkdir(parents=True)
        (folder / "vocals.wav").write_bytes(b"vocal")
        (folder / "instrumental.wav").write_bytes(b"instrumental")
    rows = [
        {"song_id": "train_song", "split": "train"},
        {"song_id": "test_song", "split": "test"},
    ]
    (paired / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    _write_split_views(paired)
    assert (paired / "train_audio" / "train_song").is_symlink()
    assert (paired / "test_audio" / "test_song").is_symlink()
    assert not (paired / "train_audio" / "test_song").exists()
