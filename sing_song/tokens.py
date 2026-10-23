"""The discrete vocabulary SingSong actually models, in one place.

The paper (Donahue et al., 2023) factorises instrumental audio into

    semantic  : 25 codes/second, 1024-entry codebook (w2v-BERT + k-means)
    coarse    : 4 x 50 codes/second, 1024-entry codebooks (SoundStream RVQ 0-3)
    fine      : 8 x 50 codes/second, 1024-entry codebooks (SoundStream RVQ 4-11)

and models ``P(Sem(y), Coarse(y) | Feats(x))`` followed by
``P(Fine(y) | Coarse(y))``.  The two modelled sequences therefore have
different lengths and different valid vocabularies at every position:

    stage 1+2 target, 10 s : [250 semantic][500 q0 q1 q2 q3 ...]  = 2250
    stage 3 target,   10 s : [2000 coarse prefix][4000 fine]      = 6000

The acoustic codes in the paper are flattened in *frame-major* order: all
four coarse books for time step 0, then all four for time step 1, and so on.
That ordering is what keeps a 50 Hz stream aligned with the 25 Hz semantic
stream and is what the constrained sampler below assumes.  Nothing in this
module imports torch, so the layout can be tested without a model or weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# The paper uses 1024 codes per book and two sentinels (start-of-sequence and
# padding; padding is unused because every sequence has a fixed length).
CODEBOOK_SIZE = 1024
NUM_COARSE_QUANTIZERS = 4
NUM_FINE_QUANTIZERS = 8
SEMANTIC_HZ = 25
ACOUSTIC_HZ = 50

STAGE12_VOCAB_SIZE = (
    CODEBOOK_SIZE + NUM_COARSE_QUANTIZERS * CODEBOOK_SIZE + 2
)  # 5122
FINE_VOCAB_SIZE = NUM_FINE_QUANTIZERS * CODEBOOK_SIZE + 2  # 8194

STAGE12_BOS = STAGE12_VOCAB_SIZE - 2  # 5120
STAGE12_PAD = STAGE12_VOCAB_SIZE - 1  # 5121
FINE_BOS = FINE_VOCAB_SIZE - 2        # 8192
FINE_PAD = FINE_VOCAB_SIZE - 1        # 8193


def _round_frames(seconds: float, hz: int) -> int:
    return int(round(float(seconds) * hz))


@dataclass(frozen=True)
class Stage12Layout:
    """Sequence layout for the joint semantic + coarse model.

    ``semantic_frames`` and ``coarse_frames`` are both derived from the same
    clip length, but kept explicit so a caller can build a short prompt (the
    5 s continuation window) without silently changing the frame rate.
    """

    semantic_frames: int
    coarse_frames: int
    codebook_size: int = CODEBOOK_SIZE
    num_coarse: int = NUM_COARSE_QUANTIZERS
    interleaved: bool = True

    def __post_init__(self) -> None:
        if self.semantic_frames <= 0 or self.coarse_frames <= 0:
            raise ValueError("semantic_frames and coarse_frames must be positive")
        if self.codebook_size & (self.codebook_size - 1):
            raise ValueError("codebook_size must be a power of two")
        if not 1 <= self.num_coarse <= NUM_COARSE_QUANTIZERS:
            raise ValueError(f"num_coarse must be 1..{NUM_COARSE_QUANTIZERS}")

    @classmethod
    def for_seconds(cls, seconds: float, **kwargs) -> "Stage12Layout":
        return cls(
            semantic_frames=_round_frames(seconds, SEMANTIC_HZ),
            coarse_frames=_round_frames(seconds, ACOUSTIC_HZ),
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self.codebook_size + self.num_coarse * self.codebook_size + 2

    @property
    def bos_id(self) -> int:
        return self.vocab_size - 2

    @property
    def pad_id(self) -> int:
        return self.vocab_size - 1

    @property
    def target_length(self) -> int:
        return self.semantic_frames + self.num_coarse * self.coarse_frames

    @property
    def coarse_length(self) -> int:
        return self.num_coarse * self.coarse_frames

    def coarse_offset(self, quantizer: int) -> int:
        if not 0 <= quantizer < self.num_coarse:
            raise IndexError(quantizer)
        return self.codebook_size + quantizer * self.codebook_size

    def position_range(self, position: int) -> tuple[int, int]:
        """Half-open vocabulary range allowed at one target position."""
        if not 0 <= position < self.target_length:
            raise IndexError(position)
        if position < self.semantic_frames:
            return 0, self.codebook_size
        rel = position - self.semantic_frames
        if self.interleaved:
            quantizer = rel % self.num_coarse
        else:
            quantizer = rel // self.coarse_frames
        start = self.coarse_offset(quantizer)
        return start, start + self.codebook_size

    def allowed_mask(self) -> list[list[bool]]:
        """The full target-length mask, useful for tests and debugging.

        Building 2250 x 5122 booleans is harmless for inspection but should not
        be used inside the sampling loop; ``position_range`` is the hot path.
        """
        mask: list[list[bool]] = []
        for i in range(self.target_length):
            lo, hi = self.position_range(i)
            row = [False] * self.vocab_size
            for j in range(lo, hi):
                row[j] = True
            mask.append(row)
        return mask

    def build_target(self, semantic: Iterable[int], coarse: Iterable[Iterable[int]]) -> list[int]:
        """Pack semantic ids and per-quantizer coarse ids into one sequence."""
        semantic = [int(x) for x in semantic]
        coarse = [[int(x) for x in q] for q in coarse]
        if len(semantic) != self.semantic_frames:
            raise ValueError(
                f"expected {self.semantic_frames} semantic frames, got {len(semantic)}"
            )
        if len(coarse) != self.num_coarse:
            raise ValueError(
                f"expected {self.num_coarse} coarse quantizers, got {len(coarse)}"
            )
        for q, row in enumerate(coarse):
            if len(row) != self.coarse_frames:
                raise ValueError(
                    f"coarse quantizer {q}: expected {self.coarse_frames} frames, got {len(row)}"
                )
        for i, token in enumerate(semantic):
            if not 0 <= token < self.codebook_size:
                raise ValueError(f"semantic token {i} out of range: {token}")
        for q, row in enumerate(coarse):
            for i, token in enumerate(row):
                if not 0 <= token < self.codebook_size:
                    raise ValueError(f"coarse q{q} token {i} out of range: {token}")

        out = list(semantic)
        if self.interleaved:
            for frame in range(self.coarse_frames):
                for q in range(self.num_coarse):
                    out.append(self.coarse_offset(q) + coarse[q][frame])
        else:
            for q in range(self.num_coarse):
                off = self.coarse_offset(q)
                out.extend(off + token for token in coarse[q])
        return out

    def split_target(self, tokens: Iterable[int]) -> tuple[list[int], list[list[int]]]:
        """Inverse of ``build_target``, validating every block."""
        tokens = [int(x) for x in tokens]
        if len(tokens) != self.target_length:
            raise ValueError(
                f"expected {self.target_length} target tokens, got {len(tokens)}"
            )
        semantic: list[int] = []
        coarse: list[list[int]] = [[] for _ in range(self.num_coarse)]
        for i in range(self.semantic_frames):
            lo, hi = self.position_range(i)
            token = tokens[i]
            if not lo <= token < hi:
                raise ValueError(f"token {i}={token} is not semantic ({lo}..{hi - 1})")
            semantic.append(token)
        if self.interleaved:
            for frame in range(self.coarse_frames):
                base = self.semantic_frames + frame * self.num_coarse
                for q in range(self.num_coarse):
                    lo, hi = self.position_range(base + q)
                    token = tokens[base + q]
                    if not lo <= token < hi:
                        raise ValueError(f"token {base + q}={token} is not coarse q{q}")
                    coarse[q].append(token - lo)
        else:
            for q in range(self.num_coarse):
                base = self.semantic_frames + q * self.coarse_frames
                for frame in range(self.coarse_frames):
                    lo, hi = self.position_range(base + frame)
                    token = tokens[base + frame]
                    if not lo <= token < hi:
                        raise ValueError(f"token {base + frame}={token} is not coarse q{q}")
                    coarse[q].append(token - lo)
        return semantic, coarse


@dataclass(frozen=True)
class FineLayout:
    """Sequence layout for the fine acoustic model.

    The coarse prefix keeps the *coarse* vocabulary ids (0..4095, interleaved)
    and the generated fine tokens use a separate 8194-entry vocabulary with
    eight disjoint 1024-token blocks.  The model has a separate coarse
    embedding table, so the two vocabularies never collide.
    """

    coarse_frames: int
    fine_frames: int
    codebook_size: int = CODEBOOK_SIZE
    num_coarse: int = NUM_COARSE_QUANTIZERS
    num_fine: int = NUM_FINE_QUANTIZERS
    interleaved: bool = True

    def __post_init__(self) -> None:
        if self.coarse_frames <= 0 or self.fine_frames <= 0:
            raise ValueError("coarse_frames and fine_frames must be positive")
        if not 1 <= self.num_coarse <= NUM_COARSE_QUANTIZERS:
            raise ValueError("num_coarse out of range")
        if self.num_fine != NUM_FINE_QUANTIZERS:
            raise ValueError(
                f"exactly {NUM_FINE_QUANTIZERS} fine quantizers are required by SoundStream"
            )

    @classmethod
    def for_seconds(cls, seconds: float, **kwargs) -> "FineLayout":
        frames = _round_frames(seconds, ACOUSTIC_HZ)
        return cls(coarse_frames=frames, fine_frames=frames, **kwargs)

    @property
    def coarse_vocab_size(self) -> int:
        return self.num_coarse * self.codebook_size

    @property
    def fine_vocab_size(self) -> int:
        return self.num_fine * self.codebook_size + 2

    @property
    def bos_id(self) -> int:
        return self.fine_vocab_size - 2

    @property
    def pad_id(self) -> int:
        return self.fine_vocab_size - 1

    @property
    def coarse_prefix_length(self) -> int:
        return self.num_coarse * self.coarse_frames

    @property
    def fine_length(self) -> int:
        return self.num_fine * self.fine_frames

    def fine_offset(self, quantizer: int) -> int:
        if not 0 <= quantizer < self.num_fine:
            raise IndexError(quantizer)
        return quantizer * self.codebook_size

    def fine_position_range(self, position: int) -> tuple[int, int]:
        if not 0 <= position < self.fine_length:
            raise IndexError(position)
        rel = position
        if self.interleaved:
            quantizer = rel % self.num_fine
        else:
            quantizer = rel // self.fine_frames
        start = self.fine_offset(quantizer)
        return start, start + self.codebook_size

    def build_coarse_prefix(self, coarse: Iterable[Iterable[int]]) -> list[int]:
        coarse = [[int(x) for x in q] for q in coarse]
        if len(coarse) != self.num_coarse:
            raise ValueError(f"expected {self.num_coarse} coarse quantizers")
        for q, row in enumerate(coarse):
            if len(row) != self.coarse_frames:
                raise ValueError(f"coarse q{q}: expected {self.coarse_frames} frames")
            for i, token in enumerate(row):
                if not 0 <= token < self.codebook_size:
                    raise ValueError(f"coarse q{q} frame {i} out of range: {token}")
        if self.interleaved:
            return [
                coarse[q][frame] + q * self.codebook_size
                for frame in range(self.coarse_frames)
                for q in range(self.num_coarse)
            ]
        return [token + q * self.codebook_size for q in range(self.num_coarse) for token in coarse[q]]

    def split_coarse_prefix(self, tokens: Iterable[int]) -> list[list[int]]:
        """Inverse of ``build_coarse_prefix``."""
        tokens = [int(x) for x in tokens]
        if len(tokens) != self.coarse_prefix_length:
            raise ValueError(
                f"expected {self.coarse_prefix_length} coarse prefix tokens, got {len(tokens)}"
            )
        coarse: list[list[int]] = [[] for _ in range(self.num_coarse)]
        if self.interleaved:
            for frame in range(self.coarse_frames):
                base = frame * self.num_coarse
                for q in range(self.num_coarse):
                    token = tokens[base + q]
                    lo = q * self.codebook_size
                    if not lo <= token < lo + self.codebook_size:
                        raise ValueError(f"token {base + q}={token} is not coarse q{q}")
                    coarse[q].append(token - lo)
        else:
            for q in range(self.num_coarse):
                base = q * self.coarse_frames
                lo = q * self.codebook_size
                for frame in range(self.coarse_frames):
                    token = tokens[base + frame]
                    if not lo <= token < lo + self.codebook_size:
                        raise ValueError(f"token {base + frame}={token} is not coarse q{q}")
                    coarse[q].append(token - lo)
        return coarse

    def build_fine(self, fine: Iterable[Iterable[int]]) -> list[int]:
        fine = [[int(x) for x in q] for q in fine]
        if len(fine) != self.num_fine:
            raise ValueError(f"expected {self.num_fine} fine quantizers")
        for q, row in enumerate(fine):
            if len(row) != self.fine_frames:
                raise ValueError(f"fine q{q}: expected {self.fine_frames} frames")
            for i, token in enumerate(row):
                if not 0 <= token < self.codebook_size:
                    raise ValueError(f"fine q{q} frame {i} out of range: {token}")
        if self.interleaved:
            return [
                fine[q][frame] + self.fine_offset(q)
                for frame in range(self.fine_frames)
                for q in range(self.num_fine)
            ]
        return [token + self.fine_offset(q) for q in range(self.num_fine) for token in fine[q]]

    def split_fine(self, tokens: Iterable[int]) -> list[list[int]]:
        tokens = [int(x) for x in tokens]
        if len(tokens) != self.fine_length:
            raise ValueError(f"expected {self.fine_length} fine tokens, got {len(tokens)}")
        fine: list[list[int]] = [[] for _ in range(self.num_fine)]
        if self.interleaved:
            for frame in range(self.fine_frames):
                base = frame * self.num_fine
                for q in range(self.num_fine):
                    lo, hi = self.fine_position_range(base + q)
                    token = tokens[base + q]
                    if not lo <= token < hi:
                        raise ValueError(f"token {base + q}={token} is not fine q{q}")
                    fine[q].append(token - lo)
        else:
            for q in range(self.num_fine):
                base = q * self.fine_frames
                for frame in range(self.fine_frames):
                    lo, hi = self.fine_position_range(base + frame)
                    token = tokens[base + frame]
                    if not lo <= token < hi:
                        raise ValueError(f"token {base + frame}={token} is not fine q{q}")
                    fine[q].append(token - lo)
        return fine


def stage12_layout(seconds: float = 10.0) -> Stage12Layout:
    return Stage12Layout.for_seconds(seconds)


def fine_layout(seconds: float = 10.0) -> FineLayout:
    return FineLayout.for_seconds(seconds)
