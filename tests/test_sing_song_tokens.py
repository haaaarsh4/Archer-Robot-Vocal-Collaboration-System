"""The SingSong vocabulary is an interface, so it gets exact assertions.

A 10 s clip must produce 250 semantic frames at 25 Hz and 2000 coarse codes at
50 Hz (four interleaved books), and the stage 1+2 target must therefore be
2250 tokens.  The fine stage must produce 4000 tokens.  These are the numbers
the paper reports; an off-by-one here silently shifts every code by one
quantizer and the model can never learn the task.
"""
from __future__ import annotations

import pytest

from sing_song.tokens import (
    CODEBOOK_SIZE,
    FINE_BOS,
    FINE_PAD,
    FINE_VOCAB_SIZE,
    SEMANTIC_HZ,
    ACOUSTIC_HZ,
    STAGE12_BOS,
    STAGE12_PAD,
    STAGE12_VOCAB_SIZE,
    FineLayout,
    Stage12Layout,
)


def test_paper_rates_and_vocabulary_sizes():
    assert SEMANTIC_HZ == 25
    assert ACOUSTIC_HZ == 50
    assert CODEBOOK_SIZE == 1024
    assert STAGE12_VOCAB_SIZE == 5122
    assert FINE_VOCAB_SIZE == 8194
    assert STAGE12_BOS == 5120 and STAGE12_PAD == 5121
    assert FINE_BOS == 8192 and FINE_PAD == 8193


def test_ten_second_layout_has_the_paper_sequence_lengths():
    layout = Stage12Layout.for_seconds(10.0)
    fine = FineLayout.for_seconds(10.0)
    assert layout.semantic_frames == 250
    assert layout.coarse_frames == 500
    assert layout.coarse_length == 2000
    assert layout.target_length == 2250
    assert fine.coarse_prefix_length == 2000
    assert fine.fine_length == 4000


def test_target_round_trip_preserves_every_code():
    layout = Stage12Layout(semantic_frames=3, coarse_frames=2)
    semantic = [1, 2, 1023]
    coarse = [
        [10, 11],
        [20, 21],
        [30, 31],
        [40, 41],
    ]
    packed = layout.build_target(semantic, coarse)
    assert len(packed) == 3 + 4 * 2
    sem_out, coarse_out = layout.split_target(packed)
    assert sem_out == semantic
    assert coarse_out == coarse


def test_interleaved_coarse_order_is_frame_major():
    layout = Stage12Layout(semantic_frames=1, coarse_frames=2)
    packed = layout.build_target([7], [[1, 2], [3, 4], [5, 6], [7, 8]])
    # semantic, then (q0,q1,q2,q3) for frame 0, then frame 1
    assert packed == [
        7,
        1024 + 1, 2048 + 3, 3072 + 5, 4096 + 7,
        1024 + 2, 2048 + 4, 3072 + 6, 4096 + 8,
    ]


def test_every_position_only_allows_its_own_quantizer():
    layout = Stage12Layout(semantic_frames=2, coarse_frames=3)
    assert layout.position_range(0) == (0, 1024)
    assert layout.position_range(1) == (0, 1024)
    # first coarse frame, four quantizers
    assert layout.position_range(2) == (1024, 2048)
    assert layout.position_range(3) == (2048, 3072)
    assert layout.position_range(4) == (3072, 4096)
    assert layout.position_range(5) == (4096, 5120)
    # second coarse frame
    assert layout.position_range(6) == (1024, 2048)
    with pytest.raises(IndexError):
        layout.position_range(layout.target_length)


def test_invalid_tokens_are_rejected_not_silently_reinterpreted():
    layout = Stage12Layout(semantic_frames=2, coarse_frames=1)
    with pytest.raises(ValueError):
        layout.build_target([0, 1024], [[1], [2], [3], [4]])
    with pytest.raises(ValueError):
        layout.split_target([0, 1, 2, 3, 4])
    # a coarse id in a semantic slot is not accepted either
    with pytest.raises(ValueError):
        layout.split_target([0, 1024, 1024, 2048, 3072, 4096])


def test_fine_layout_offsets_and_round_trip():
    layout = FineLayout(coarse_frames=2, fine_frames=2)
    coarse_prefix = layout.build_coarse_prefix([[1, 2], [3, 4], [5, 6], [7, 8]])
    assert coarse_prefix[:4] == [1, 1024 + 3, 2048 + 5, 3072 + 7]
    assert layout.split_coarse_prefix(coarse_prefix) == [[1, 2], [3, 4], [5, 6], [7, 8]]

    fine = [[i for i in range(2)] for _ in range(8)]
    packed = layout.build_fine(fine)
    assert len(packed) == 16
    assert packed[0] == 0
    assert packed[1] == 1024
    assert packed[7] == 7 * 1024
    assert layout.split_fine(packed) == fine
    assert layout.fine_position_range(0) == (0, 1024)
    assert layout.fine_position_range(7) == (7 * 1024, 8 * 1024)


def test_fine_requires_exactly_eight_quantizers():
    with pytest.raises(ValueError):
        FineLayout(coarse_frames=1, fine_frames=1, num_fine=7)
