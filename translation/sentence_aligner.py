import math
from pathlib import Path
from typing import List, Tuple

from loguru import logger

ALIGNMENT_TYPES = [
    (1, 1, 0.89),
    (1, 0, 0.01),
    (0, 1, 0.01),
    (2, 1, 0.045),
    (1, 2, 0.045),
]
DELETION_PENALTY = 25.0  # fixed cost for dropping an unmatched line
S2 = 6.8  # variance parameter, standard value from the original paper


def _norm_tail_prob(z: float) -> float:
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def align_sequences(source: List[str], target: List[str],
                     band: int = None) -> List[Tuple[str, str]]:
    n, m = len(source), len(target)
    src_lens = [len(s) for s in source]
    tgt_lens = [len(t) for t in target]
    total_src = sum(src_lens) or 1
    total_tgt = sum(tgt_lens) or 1
    c = total_tgt / total_src  # expected chars-per-char ratio, this file

    if band is None:
        band = max(n, m) + 1
    ratio = m / n if n else 1.0

    NEG = float("inf")
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(n + 1):
        j_center = int(i * ratio)
        j_lo, j_hi = max(0, j_center - band), min(m, j_center + band)
        for j in range(j_lo, j_hi + 1):
            if i == 0 and j == 0:
                continue
            best_cost, best_prev = NEG, None
            for di, dj, prior in ALIGNMENT_TYPES:
                if di > i or dj > j:
                    continue
                prev = dp[i - di][j - dj]
                if prev == NEG:
                    continue
                if di == 0 or dj == 0:
                    step_cost = DELETION_PENALTY
                else:
                    l1 = sum(src_lens[i - di:i])
                    l2 = sum(tgt_lens[j - dj:j])
                    z = (l2 - l1 * c) / math.sqrt(max(l1, 1) * S2)
                    prob = max(_norm_tail_prob(z), 1e-30)
                    step_cost = -math.log(prob) - math.log(prior)
                total = prev + step_cost
                if total < best_cost:
                    best_cost, best_prev = total, (di, dj)
            dp[i][j] = best_cost
            back[i][j] = best_prev

    if dp[n][m] == NEG:
        logger.warning("Alignment failed within band, retrying with unrestricted search")
        return align_sequences(source, target, band=max(n, m) + 1) if band <= max(n, m) else []

    i, j = n, m
    pairs = []
    while i > 0 or j > 0:
        step = back[i][j]
        if step is None:
            break
        di, dj = step
        if di and dj:
            src_chunk = " ".join(source[i - di:i])
            tgt_chunk = " ".join(target[j - dj:j])
            pairs.append((src_chunk, tgt_chunk))
        i -= di
        j -= dj
    pairs.reverse()
    return pairs


def align_file_pair(cr_path: Path, en_path: Path, band: int = 200) -> List[Tuple[str, str]]:
    cr_lines = [l.strip() for l in open(cr_path, encoding="utf-8") if l.strip()]
    en_lines = [l.strip() for l in open(en_path, encoding="utf-8") if l.strip()]
    if not cr_lines or not en_lines:
        return []
    pairs = align_sequences(cr_lines, en_lines, band=band)
    logger.info(f"{cr_path.name}: aligned {len(cr_lines)} cr / {len(en_lines)} en lines "
                f"-> {len(pairs)} sentence pairs (was 1 whole-file pair under the old fallback)")
    return pairs
