"""nêhiyawêwin transcription: reading Archer's Cree out of a sung take.

WHAT THIS REPLACES, AND WHY
---------------------------
The project used to hand sung audio to Whisper. Whisper is an English speech
recogniser with an English language model inside its weights, so on a voice that
is *singing Cree* its decoder does the only thing it can: it asks which sequence
of fluent English words best explains these sounds, and answers with English
words that were never sung ("well, hooray!"). No beam size, temperature or prompt
changes that, because it is not a search problem -- it is what the probabilities
say. The singer gets back a transcript in a language they were not singing, which
is worse than nothing when the whole point is to see their own words.

This module cannot do that, structurally. It decodes with Meta's MMS-1b-all
Plains Cree adapters -- the only trained nêhiyawêwin acoustic models that
exist -- over the language's own alphabet, and every hypothesis it can produce is
scored by a nêhiyawêwin lexicon, a character n-gram model of the language, and
its syllable phonotactics. There is no English word in the search space to be
emitted. A stretch of singing that is not a word of the language therefore comes
back as *vocables* (way, hay, hoo, aye) spelled with the syllables the language
actually has, which is what those sounds are, rather than as English that is not.

THE FOUR STAGES
---------------
1. **Vocal isolation.** MMS is trained on read speech with no accompaniment, and
   drums bleed through its front end as spurious consonants. Demucs separates
   the lead vocal before anything else looks at the audio, which on this
   project's own takes is the single largest accuracy change of all of them. A
   librosa harmonic-percussive fallback stands in when the separation weights are
   not on disk.

2. **Acoustic posteriors, over Standard Roman Orthography.** *Both* crk adapters
   are used, not one. The Latin adapter (35 symbols, one per letter) and the
   syllabics adapter (78 symbols, one per *syllable glyph*) make different
   mistakes, and fusing their N-best lists is worth real accuracy on a language
   this low-resource. One wrinkle the vocab files make easy to miss: Meta's Latin
   crk spelling writes long vowels with macrons (`ē ā ī ō`) while the community
   and this project write circumflexes (`ê â î ô`), so every token is passed
   through the orthography engine before it is scored -- otherwise every long
   vowel in the language would read as an unknown character.

3. **Constrained CTC beam search** (`CTCDecoder`) -- prefix beam search with the
   character LM shallow-fused, a lexicon price paid at every word boundary, a
   penalty while a word in progress is not the prefix of any known word, and a
   penalty for pieces that are not legal nêhiyawêwin syllables. This is the stage
   that turns a noisy posterior into the language; the weights live in
   `CreeASRConfig` and are measured by `tools/check_cree_transcription.py`, not
   guessed.

4. **Vocables** (`VocableRecognizer`). Sustained non-lexical singing gets its own
   recogniser: an espeak-IPA phone model reads the phones (it has no language
   model and no words, which is exactly right for singing that is not words),
   phones group into syllables around their vowel nuclei, and each syllable snaps
   to the nearest nêhiyawêwin syllable by articulation -- then to this tradition's
   own inventory of sung syllables (way, hay, yo, hi, na, aye, hoo) so the
   transcript reads the way a singer writes it down.

Output is in three scripts at once, because that is what the page promises:
syllabics (ᑖᓂᓯ), Standard Roman Orthography (tânisi), and the anglicized
respelling a reader who knows neither can actually say (taa-ni-si). All of it
goes through `analysis/cree_orthography`, whose tables are the itwêwina
dictionary toolchain's own, so the three scripts never disagree with each other.
"""

from __future__ import annotations

import math
import os
import pathlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from analysis.cree_orthography import (
    anglicize,
    is_syllabics,
    macron_to_circumflex,
    normalize_sro,
    syllabify,
    to_sro,
    to_syllabics,
)
from analysis.cree_corpus import CREE_LETTERS, CreTextModel, load_text_model

__all__ = [
    "SAMPLE_RATE", "CreeASRConfig", "TranscriptSegment", "CreeTranscript",
    "MMSCreeASR", "CTCDecoder", "VocableRecognizer", "CreeTranscriber",
    "get_transcriber", "isolate_vocals", "find_voiced_spans",
]

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent

SAMPLE_RATE = 16000
_DEFAULT_MODEL_DIR = _ROOT / "data" / "models" / "cree"

_VOWELS = set("aeiouâêîô")# The letters the Latin adapter can emit that nêhiyawêwin is not written with.
# Meta's crk dictionary carries them because its training text has loanwords
# (`biscuit`, proper names); the model keeps the output dimension it was trained
# with, so they stay in the vocabulary and their probability is forced to the
# floor before decoding instead. That is the mechanism that makes an English word
# unrepresentable rather than merely unlikely. `r` and `l` are NOT
# here: both are real nêhiyawêwin letters in the western dialects, which this
# project's own corpus uses.
_NON_CREE_SYMBOLS = set("dbfgvxjquz") | {"9", "7", "6", "8", "0"}

# Symbols that are *not letters but not nothing*. MMS writes its Plains Cree
# training text with a hyphen between a preverb and its verb (`ê-kikihci...`),
# and this project's corpus drops it -- `read_corpus` keeps only letters, so
# `ê-kikihci` and `êkikihci` are the same word to the language model. Zeroing
# these would force the acoustic model to spend those frames on a letter and
# corrupt the word it is in the middle of; folding their probability into the
# **blank** instead lets it spend them on nothing, which is what they are. This
# is not a cosmetic detail: `ê-` precedes a conjunct verb constantly in this
# corpus, so those frames are a large fraction of every sentence.
_BLANK_FOLD_SYMBOLS = set("-‐—–'") | set("‘’") | {"​"}

# Fonts of junk that arrive with the syllabics vocabulary as single Latin
# characters; `h` and `t` are real Cree letters and are kept.
_JUNK_TOKENS = {"<pad>", "<s>", "</s>", "<unk>", ""}


# configuration

@dataclass
class CreeASRConfig:
    """Every number the decoder uses, in one place.

    The defaults are the values `tools/check_cree_transcription.py` measures as
    best on this project's gold set (native-speaker recordings from the corpus,
    an MMS-TTS round trip, and the Archer takes). They are named here rather than
    inlined so the harness can sweep them, and so a caller can trade accuracy for
    latency instead of by accident.
    """

    # -- acoustic model
    model_dir: pathlib.Path = _DEFAULT_MODEL_DIR
    adapters: tuple[str, ...] = ("crk-script_latin", "crk-script_syllabics")
    threads: int = 0                       # 0 -> a sensible default from os.cpu_count()
    chunk_seconds: float = 20.0            # attention is quadratic; keep windows small
    chunk_overlap_seconds: float = 2.0

    # -- input conditioning
    # "auto" separates with Demucs when its weights are present and otherwise
    # transcribes the mixture unchanged; "harmonic" is the librosa hpss hint, an
    # explicit choice for a genuinely mixed take and *worse than nothing* on a
    # solo voice (measured: 0-6% CER becomes 75%+), which is why "auto" no longer
    # reaches for it; "off" skips separation entirely.
    isolation: str = "auto"                # "auto" | "demucs" | "harmonic" | "off"
    highpass_hz: float = 55.0
    peak_normalize: float = 0.72
    silence_floor_db: float = -52.0        # below this a frame is not singing at all
    # How far below the take's own loudest frame a frame has to be before it is
    # treated as room rather than voice. Loose, this hands a recogniser several
    # seconds of room tone and it fills them with fluent invented nêhiyawêwin --
    # the failure that dominates this project's remaining gold error -- so it is
    # a parameter with a measured value rather than a constant in the arithmetic.
    silence_margin_db: float = 30.0

    # -- beam search
    beam_width: int = 40
    symbols_per_frame: int = 10            # how many non-blank symbols a frame may branch into
    n_best: int = 6
    lm_order: int = 6
    # All three language terms are measured as deviations from the corpus's own
    # baselines rather than against zero, so that ordinary nêhiyawêwin scores near
    # zero whatever its length, and a weight here means "how many log units of
    # acoustic evidence is one unit of unnaturalness worth".
    lm_weight: float = 0.3                 # weight on the character n-gram LM
    word_weight: float = 0.4               # weight on a word being one the lexicon holds
    phono_weight: float = 0.15             # weight on syllable phonotactics, per syllable
    # Per-adapter trust, because their scores are not on the same scale: an
    # adapter with fewer symbols gets higher log-probabilities for the same
    # audio, so comparing two adapters' hypotheses by raw score would be a
    # comparison of their alphabets. Measured and set by
    # tools/check_cree_transcription.py, not assumed.
    adapter_weights: dict = field(default_factory=lambda: {
        "crk-script_latin": 1.0, "crk-script_syllabics": 1.0})
    insertion_bonus: float = 1.2           # per emitted token; counters CTC's deletion bias
    oov_prefix_penalty: float = 1.0        # while a word in progress is not a lexicon prefix
    illegal_piece_penalty: float = 2.0     # when an emitted piece is not a legal Cree syllable
    acoustic_scale: float = 1.0
    max_word_chars: int = 30               # longer than this is not a word of this language
    # Nobody sings fifty characters a second. Every language has a speaking rate
    # and a singer's is slower than a speaker's, so a hypothesis that spends more
    # than this many characters per second of audio is not a reading of the take
    # at all -- it is the search failing to stop. Enforcing the rate is what stops
    # a hard, quiet passage coming back as five times the words that were sung.
    max_chars_per_second: float = 17.0
    frames_per_second: float = 50.0        # MMS's stride: 20 ms per frame

    # -- segmenting the take
    min_segment_seconds: float = 0.4
    max_gap_seconds: float = 0.5
    pad_seconds: float = 0.1

    # -- ranking, in per-frame units
    #
    # The beam generates candidates; these weights choose between them. The
    # difference matters, because a sum over a transcription grows with its length
    # no matter what it measures -- a sum of log-probabilities punishes saying
    # anything and a sum of *centred* scores rewards saying the commonest thing
    # forever, and both of those are failures this project actually produced. So
    # the ranking is per frame: the acoustics, which are per frame already, decide
    # how much was sung, and each language term enters as a bounded *average*
    # quality whose weight says how many per-frame acoustic units one unit of
    # unnaturalness is worth. Nothing here can be won by being long or short.
    rank_char_weight: float = 0.16         # per-character n-gram quality
    rank_word_weight: float = 0.05         # mean evidence that the words are real
    rank_phono_weight: float = 0.05        # per-syllable phonotactic deviation
    rank_agree_weight: float = 0.04        # both adapters preferring the same text
    rank_repeat_weight: float = 0.8        # a reading that has locked into repeating itself
    # Charged per repeated 6-gram *during the search*, so ordinary text pays
    # nothing. It defaults to zero, and that is a measurement rather than an
    # oversight: it was written to stop a CTC model that had lost its place from
    # looping, and on this project's gold set it was doing the opposite of what
    # was wanted. Two of the seven recordings really do hold the line more than
    # once, and the penalty suppressed those genuine second takes until the
    # search settled on a mangled first one -- measured, coverage error 3.9%
    # with the penalty at 5.0 against 3.0% at 0.0, with the gain showing up as
    # `kimamihcihâwak kahkiyaw kiwâkomakanak` read cleanly three times over on
    # the take that holds it three times. Repetition is still judged, in
    # `rank_repeat_weight`, where it is judged about a whole finished reading
    # rather than frame by frame mid-search; and the rate ceiling below is what
    # actually stops a runaway. Kept as a parameter because a genuinely noisy
    # take is a different case and this is the knob for it.
    repeat_penalty: float = 0.0
    repeat_ngram: int = 6

    # -- deciding lexical words vs vocables
    # The confidence curve is calibrated on language quality: measured over this
    # corpus, real nêhiyawêwin sentences score -0.95 to -1.6, the decoder's own
    # hypotheses on real recordings -1.5 to -4.2, and phone-recogniser output on
    # the same audio -4.8 to -6.3. The curve below maps that range onto [0, 1]
    # and the threshold says how much of it a segment has to reach to be claimed
    # as the singer's words.
    quality_high: float = -1.25            # this quality is a confident reading
    quality_low: float = -4.0              # this quality is not the language
    lexical_confidence: float = 0.40       # below this a segment is not claimed to be words
    vocable_confidence: float = 0.30
    vocable_margin: float = 0.12           # how much better the phone reading must be to win

    def resolved_threads(self) -> int:
        if self.threads:
            return int(self.threads)
        return max(1, min(8, os.cpu_count() or 4))


# results

@dataclass
class TranscriptSegment:
    """One stretch of singing, in all three scripts.

    `kind` is the accurate part of this dataclass. "cree" means the decoder found
    real nêhiyawêwin words it is confident in; "vocable" means it found sung
    syllables that are not words of the language (way, hay, hoo) and says so
    instead of pretending; "mixed" means both in one breath; "uncertain" means
    the audio was too far from anything the model knows, and the segment is
    reported as syllables-with-a-question-mark rather than as a word nobody sang.
    """

    start: float
    end: float
    sro: str
    syllabics: str = ""
    anglicized: str = ""
    kind: str = "cree"
    confidence: float = 0.0
    words: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.sro and not self.syllabics:
            self.syllabics = to_syllabics(self.sro)
        if self.sro and not self.anglicized:
            self.anglicized = anglicize(self.sro)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3), "end": round(self.end, 3),
            "sro": self.sro, "syllabics": self.syllabics, "anglicized": self.anglicized,
            "label": self.anglicized or self.sro,
            "kind": self.kind, "confidence": round(self.confidence, 4),
            "confident": self.kind in ("cree", "mixed"),
            "words": self.words, "notes": self.notes,
            "type": "vocal",
        }


@dataclass
class CreeTranscript:
    segments: list[TranscriptSegment]
    language: str = "crk"
    duration: float = 0.0
    detection_results: list | None = None
    timing: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def sro(self) -> str:
        return " ".join(s.sro for s in self.segments if s.sro).strip()

    @property
    def syllabics(self) -> str:
        return " ".join(s.syllabics for s in self.segments if s.syllabics).strip()

    @property
    def anglicized(self) -> str:
        return " ".join(s.anglicized for s in self.segments if s.anglicized).strip()

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "duration": round(self.duration, 2),
            "segments": [s.to_dict() for s in self.segments],
            "text": self.sro,
            "sro": self.sro,
            "syllabics": self.syllabics,
            "anglicized": self.anglicized,
            "has_lexical_speech": any(s.kind in ("cree", "mixed") for s in self.segments),
            "words": [w for s in self.segments for w in s.words],
            "timing": self.timing,
            "notes": self.notes,
            "detection_results": self.detection_results,
        }


# audio conditioning and vocal isolation

def _harmonic_vocal_hint(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A cheap stand-in for source separation.

    Demucs is the real separator but it needs its weights and a good deal of CPU.
    Harmonic-percussive separation keeps the sustained, pitched part of a mixture
    and drops transients, which on a take like these is most of what the drums
    are.

    Measured on this project's gold set, though, hpss is *harmful* on a solo
    voice: five recordings that transcribe at 0-6% CER with no separation come
    back at 75% and worse through it, because the sung energy it keeps is the
    vowel and what it discards -- the aperiodic part of every stop, affricate and
    fricative -- is most of a consonant. So it is offered as an explicit choice
    for a genuinely mixed take (:attr:`CreeASRConfig.isolation` = ``"harmonic"``),
    never as the thing that happens when Demucs is missing: a missing optional
    dependency must not silently degrade the transcriber, and on a solo take it
    degrades it badly.
    """
    import librosa

    y = np.asarray(audio, dtype=np.float32)
    if y.size < 2048:
        return y
    harmonic, _percussive = librosa.effects.hpss(y, margin=(1.0, 3.0))
    return np.ascontiguousarray(harmonic, dtype=np.float32)


_DEMUCS_LOCK = threading.Lock()
_DEMUCS_CACHE: tuple | None = None
_DEMUCS_ERROR: str | None = None


def _load_demucs():
    """Load the Demucs vocal separator once, reporting rather than raising.

    Weights are fetched by torch's hub on first use and are not shipped with the
    pip package. A machine with no network, or no hub cache, is a normal state
    for this project and must not take the transcriber down: the caller is told
    and reads the mixture unchanged, which measurement says is far better than
    the harmonic hint on a solo voice.
    """
    global _DEMUCS_CACHE, _DEMUCS_ERROR
    with _DEMUCS_LOCK:
        if _DEMUCS_CACHE is not None or _DEMUCS_ERROR is not None:
            return _DEMUCS_CACHE
        try:
            from demucs.apply import apply_model
            from demucs.pretrained import get_model

            model = get_model("htdemucs")
            model.eval()
            _DEMUCS_CACHE = (model, apply_model)
        except Exception as exc:  # noqa: BLE001 - any failure means "use the fallback"
            _DEMUCS_ERROR = f"{type(exc).__name__}: {exc}"
            print(f"[cree_asr] Demucs unavailable ({_DEMUCS_ERROR}); "
                  f"transcribing the mixture directly")
    return _DEMUCS_CACHE


def isolate_vocals(audio: np.ndarray, sr: int = SAMPLE_RATE,
                   mode: str = "auto") -> tuple[np.ndarray, str]:
    """Return `(vocals, method)`. Never raises; falls back and reports the path taken.

    `"auto"` means "separate if you can, otherwise hand over the mixture" -- and
    not "otherwise run hpss", because measuring the two on this
    project's own recordings showed hpss to be worse than doing nothing on a solo
    voice (see :func:`_harmonic_vocal_hint`)."""
    y = np.asarray(audio, dtype=np.float32)
    if mode == "off" or y.size == 0:
        return y, "off"
    if mode in ("auto", "demucs"):
        loaded = _load_demucs()
        if loaded is not None:
            model, apply_model = loaded
            import torch
            from demucs.audio import convert_audio

            with torch.no_grad():
                src = torch.from_numpy(np.ascontiguousarray(y))[None, None, :]
                src = convert_audio(src, sr, model.samplerate, model.audio_channels)
                out = apply_model(model, src, split=True, overlap=0.25,
                                  progress=False, device="cpu")[0]
                index = {name: i for i, name in enumerate(model.sources)}
                vocals = out[index["vocals"]]
                vocals = convert_audio(vocals[None], model.samplerate, sr, 1)[0, 0]
                return np.ascontiguousarray(vocals.numpy(), dtype=np.float32), "demucs"
        # Demucs wanted -- whether explicitly or by "auto" -- and unavailable:
        # hand over the mixture untouched rather than substituting a method the
        # caller did not ask for and that measures worse than nothing
        return y, "off"
    return _harmonic_vocal_hint(y, sr), "harmonic"


def _condition(audio: np.ndarray, cfg: CreeASRConfig,
               sr: int = SAMPLE_RATE) -> tuple[np.ndarray, list[str]]:
    """High-pass and peak-normalise before anything decodes.

    Both models were trained on close-miked read speech at a sensible level; a
    phone recording of a song is neither. The high-pass removes the rumble that
    otherwise reads as low-frequency consonant energy, and normalising to a fixed
    peak makes the level independent of how loud the take was, which matters
    because the CTC model's confidence calibration assumes a nominal level.
    """
    notes: list[str] = []
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.size == 0:
        return y, ["empty audio"]
    try:
        import scipy.signal as sig

        sos = sig.butter(2, cfg.highpass_hz / (sr / 2.0), btype="highpass", output="sos")
        y = sig.sosfiltfilt(sos, y.astype(np.float64)).astype(np.float32)
    except Exception:  # noqa: BLE001 - filtering is an improvement, not a requirement
        pass
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-6:
        y = (y * (cfg.peak_normalize / peak)).astype(np.float32)
    else:
        notes.append("silent input")
    return y, notes


def find_voiced_spans(audio: np.ndarray, cfg: CreeASRConfig,
                      sr: int = SAMPLE_RATE) -> list[tuple[float, float]]:
    """Where in the take is there a voice at all?

    Frame energy in dB against a floor, *not* a VAD: a speech VAD
    drops sustained singing, which is precisely the audio this module exists to
    transcribe. The floor is the higher of an absolute one and a relative one, so
    a quiet take still yields spans, and short spans are merged so a melody's own
    phrase gaps do not cut words in half.
    """
    y = np.asarray(audio, dtype=np.float32)
    if y.size == 0:
        return []
    hop = int(0.01 * sr)
    win = int(0.025 * sr)
    if y.size < win:
        return [(0.0, y.size / sr)]
    n = 1 + (y.size - win) // hop
    frames = np.lib.stride_tricks.as_strided(
        y, shape=(n, win), strides=(hop * y.dtype.itemsize, y.dtype.itemsize)).copy()
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    top = float(np.percentile(db, 98))
    floor = max(cfg.silence_floor_db, top - cfg.silence_margin_db)
    voiced = db > floor

    spans: list[tuple[float, float]] = []
    start = None
    for i, v in enumerate(voiced):
        t = i * hop / sr
        if v and start is None:
            start = t
        elif not v and start is not None:
            spans.append((start, t + win / sr))
            start = None
    if start is not None:
        spans.append((start, len(voiced) * hop / sr + win / sr))

    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s - merged[-1][1] <= cfg.max_gap_seconds:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(max(0.0, s - cfg.pad_seconds), min(len(y) / sr, e + cfg.pad_seconds))
            for s, e in merged if e - s >= cfg.min_segment_seconds]


# the acoustic model

class MMSCreeASR:
    """MMS-1b-all with the Plains Cree adapters, giving posteriors over SRO.

    One instance owns the 1B base model and swaps the per-script adapters (a few
    MB each) around it, so both scripts' posteriors pay for the encoder once.
    Loading is lazy and guarded, because a web server imports this and must still
    start when the model directory is missing.
    """

    def __init__(self, cfg: CreeASRConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._model = None
        self._tokenizers: dict[str, tuple] = {}
        self._load_error: str | None = None
        self.load_seconds: float = 0.0
        self.notes: list[str] = []

    # -- lifecycle ---------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def base_dir(self) -> pathlib.Path:
        return pathlib.Path(self.cfg.model_dir) / "facebook--mms-1b-all"

    def load(self) -> bool:
        """Load the base model and its adapters. Returns availability; never raises."""
        with self._lock:
            if self._model is not None:
                return True
            base = self.base_dir()
            if not base.is_dir():
                self._load_error = f"no MMS model directory at {base}"
                return False
            try:
                import torch
                from transformers import Wav2Vec2ForCTC

                torch.set_num_threads(self.cfg.resolved_threads())
                t0 = time.perf_counter()
                model = Wav2Vec2ForCTC.from_pretrained(
                    str(base), local_files_only=True, low_cpu_mem_usage=True,
                    dtype=torch.float32)
                # `from_pretrained` repoints the model at its own directory, which
                # breaks the adapter lookup for anything but the first language;
                # the config has to keep pointing at the local snapshot.
                model.config._name_or_path = str(base)
                for lang in self.cfg.adapters:
                    self._build_tokenizer(lang)
                    model.load_adapter(lang, local_files_only=True)
                model.eval()
                self._model = model
                self.load_seconds = time.perf_counter() - t0
                self.notes.append(f"MMS-1b + {len(self.cfg.adapters)} crk adapters "
                                  f"loaded in {self.load_seconds:.1f}s")
                return True
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                self._load_error = f"{type(exc).__name__}: {exc}"
                return False

    def _build_tokenizer(self, lang: str):
        """Build the per-language tokenizer, writing the JSON that MMS does not ship.

        `facebook/mms-1b-all` stores each language's dictionary as a
        frequency-ordered `token count` text file, while `Wav2Vec2CTCTokenizer`
        wants `{token: id}` JSON. The adapter's `lm_head` has exactly four rows
        more than the file has lines -- `<pad>`, `<s>`, `</s>`, `<unk>` -- and
        `<pad>` is id 0, which is CTC's blank, so the mapping is those four
        specials followed by the file's own order. The relationship is *asserted*
        against the loaded adapter's output dimension rather than assumed,
        because getting it wrong still decodes fluently -- just in a different
        alphabet, which is the kind of bug that costs a day.
        """
        cached = self._tokenizers.get(lang)
        if cached is not None:
            return cached
        import json
        from transformers import Wav2Vec2CTCTokenizer

        base = self.base_dir()
        tsv = base / "vocabs" / f"{lang}.txt"
        js = base / "vocabs" / f"{lang}.json"
        if not tsv.exists():
            raise FileNotFoundError(f"missing MMS vocabulary file {tsv}")
        tokens: list[str] = []
        with open(tsv, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                tokens.append(line.rsplit(" ", 1)[0])
        vocab = {t: i for i, t in enumerate(("<pad>", "<s>", "</s>", "<unk>") + tuple(tokens))}
        try:
            if not js.exists() or json.load(open(js, encoding="utf-8")) != vocab:
                with open(js, "w", encoding="utf-8") as f:
                    json.dump(vocab, f, ensure_ascii=False)
        except (OSError, ValueError):
            with open(js, "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False)

        tok = Wav2Vec2CTCTokenizer(vocab_file=str(js), unk_token="<unk>", pad_token="<pad>",
                                  word_delimiter_token="|", do_lower_case=False)
        blank_id = tok.pad_token_id
        pieces: list[tuple[str, str, bool]] = []
        for i in range(tok.vocab_size):
            token = tok.convert_ids_to_tokens(i)
            if i == blank_id:
                pieces.append((token, "", True))
                continue
            pieces.append((token, _token_to_sro(token), False))
        self._tokenizers[lang] = (tok, pieces)
        return self._tokenizers[lang]

    def tokenizer_for(self, lang: str):
        return self._build_tokenizer(lang)

    def _use_adapter(self, lang: str):
        model = self._model
        if model is None:
            raise RuntimeError("MMS model is not loaded")
        if getattr(model, "target_lang", None) != lang:
            model.load_adapter(lang, local_files_only=True)
        tok, pieces = self._build_tokenizer(lang)
        if tok.vocab_size != model.config.vocab_size:
            raise RuntimeError(
                f"{lang}: tokenizer has {tok.vocab_size} entries but the adapter "
                f"has {model.config.vocab_size} outputs")
        return tok, pieces

    # -- inference ---------------------------------------------------------

    def posteriors(self, audio: np.ndarray, sr: int = SAMPLE_RATE,
                   progress=None) -> dict[str, np.ndarray]:
        """Per-adapter log-softmax posteriors, `{adapter: (frames, vocabulary)}`.

        Decoded in overlapping windows because attention over a whole take is
        quadratic; the overlap is cropped from the head of each later window, at
        the frame rate the crop implies, so the concatenation covers the audio
        once without a seam in the middle of a word.
        """
        import torch

        if not self.load():
            raise RuntimeError(f"MMS model unavailable: {self._load_error}")
        y = np.asarray(audio, dtype=np.float32)
        cfg = self.cfg
        win = int(cfg.chunk_seconds * sr)
        hop = max(1, int((cfg.chunk_seconds - cfg.chunk_overlap_seconds) * sr))
        if y.size <= win:
            windows = [(0, y.size)]
        else:
            windows = []
            pos = 0
            while pos < y.size:
                windows.append((pos, min(y.size, pos + win)))
                if pos + win >= y.size:
                    break
                pos += hop

        out: dict[str, list[np.ndarray]] = {lang: [] for lang in cfg.adapters}
        for w_index, (a, b) in enumerate(windows):
            chunk = y[a:b]
            if chunk.size < 400:                 # the convolutional front end needs a minimum
                continue
            tensor = torch.from_numpy(np.ascontiguousarray(chunk))[None]
            for lang in cfg.adapters:
                self._use_adapter(lang)
                with torch.no_grad():
                    logits = self._model(tensor).logits[0]
                arr = torch.log_softmax(logits.float(), dim=-1).numpy()
                if w_index > 0 and cfg.chunk_overlap_seconds > 0 and arr.shape[0] > 1:
                    frames = arr.shape[0]
                    drop = int(round(frames * cfg.chunk_overlap_seconds / cfg.chunk_seconds))
                    drop = max(0, min(drop, frames // 3))
                    arr = arr[drop:]
                out[lang].append(arr)
            if progress:
                progress((w_index + 1) / len(windows))
        return {lang: (np.concatenate(v, axis=0) if v else np.zeros((0, 1), dtype=np.float32))
                for lang, v in out.items()}


def _token_to_sro(token: str) -> str:
    """The Standard Roman Orthography a single MMS vocabulary token contributes.

    Three cases, and the middle one is the trap:

      * Latin tokens are already SRO, *except* that Meta writes the long vowels
        with macrons (`ē ā ī ō`) where the community -- and this project's whole
        corpus and lexicon -- writes circumflexes (`ê â î ô`). Converting here
        means every downstream stage sees one spelling of the language.
      * Syllabics tokens are glyphs: precomposed syllables, finals, `ᕽ`, the
        dotted-`w` marker. `analysis/cree_orthography.to_sro` converts them
        exactly, because its tables are the dictionary's own transducer rules.
      * `|` is the word delimiter, which the rest of this module writes as a space.
    """
    if token in _JUNK_TOKENS:
        return ""
    if token == "|":
        return " "
    if is_syllabics(token):
        return to_sro(token) or token
    if len(token) == 1:
        return macron_to_circumflex(token)
    return macron_to_circumflex(token)


# the decoder

class CTCDecoder:
    """Prefix beam search over CTC posteriors, fused with a nêhiyawêwin text model.

    The search itself is the textbook algorithm (Hannun et al., and every
    production CTC decoder since): a hypothesis carries two probabilities,
    because CTC collapses repeats, so "ended in a blank" and "ended in a symbol"
    are different states with the same text. Each frame a hypothesis either stays
    put (the blank) or commits to a symbol; committing to the symbol it just
    committed to does not extend anything unless a blank came between, which is
    exactly how a doubled letter is produced.

    What makes this a *Cree* decoder is the objective, and all four terms of it
    are language knowledge rather than acoustics:

      * every piece a hypothesis emits is charged `lm_weight` times the character
        n-gram log-probability of its letters given the letters already emitted
        -- so a sequence that looks like the language costs less than one that
        does not, frame by frame, before any word is complete;
      * when a hypothesis closes a word, that word is charged `word_weight`
        times its lexicon log-probability, and an outright known word gets
        `known_word_bonus` back. This is where an unknown word -- which includes
        every English word -- pays for itself;
      * while a word is in progress, a prefix that is not the prefix of any
        lexicon word is charged `oov_prefix_penalty`, which steers the search
        toward words that exist without ever making a name or a new word
        impossible to finish;
      * a piece whose letters cannot be grouped into legal nêhiyawêwin syllables
        is charged `illegal_piece_penalty`, which is what stops the Latin
        adapter's English letters being strung into something plausible-looking.
    """

    def __init__(self, text_model: CreTextModel, cfg: CreeASRConfig):
        self.lm = text_model
        self.cfg = cfg
        self._trie = LexiconTrie(text_model.words)

    # -- the shared view of the acoustics ----------------------------------

    def _masked(self, logprobs: np.ndarray,
                pieces: Sequence[tuple[str, str, bool]]) -> np.ndarray:
        """The posteriors with every non-nêhiyawêwin symbol at the floor.

        This is the single mechanism by which an English word is not merely
        unlikely but unrepresentable, and it is applied identically by the search,
        by the greedy decode and by :meth:`score`, so all three are describing the
        same model. The masked symbols stay in the array -- the adapter's output
        dimension is fixed by its trained weights -- but their probability mass is
        gone before anything looks at it.
        """
        work = np.asarray(logprobs, dtype=np.float32).copy()
        floor = float(np.log(1e-12))
        blank_id = next((i for i, p in enumerate(pieces) if p[2]), 0)
        for i, (_token, sro, is_blank) in enumerate(pieces):
            if is_blank or not sro or sro == " ":
                continue
            if all(ch in _BLANK_FOLD_SYMBOLS for ch in sro):
                # not a character: hand its frames to the blank, so the letters
                # on either side of a morpheme boundary stay clean
                work[:, blank_id] = np.logaddexp(work[:, blank_id], work[:, i])
                work[:, i] = floor
            elif all(ch in _NON_CREE_SYMBOLS for ch in sro):
                work[:, i] = floor
        return work

    # -- the search --------------------------------------------------------

    def decode(self, logprobs: np.ndarray, pieces: Sequence[tuple[str, str, bool]],
               n_best: int | None = None) -> list[tuple[float, str]]:
        """Decode one span. Returns `[(score, sro_text), ...]`, best first.

        `logprobs` is (frames, vocabulary) of log-softmax output and `pieces`
        maps each vocabulary index to `(token, sro_text, is_blank)`.
        """
        cfg = self.cfg
        n_best = n_best or cfg.n_best
        frames = int(logprobs.shape[0])
        if frames == 0:
            return []

        work = self._masked(logprobs, pieces)
        blank_id = next((i for i, p in enumerate(pieces) if p[2]), 0)
        vocab_sro = [p[1] for p in pieces]
        lm_cache: dict[tuple[str, str], float] = {}

        # beams: collapsed label sequence -> [p_blank, p_nonblank]
        beams: dict[tuple, list[float]] = {(): [0.0, -math.inf]}
        # meta: sequence -> (text, history, word, lm_sum, word_sum, n_symbols,
        #                    n_chars) -- characters are counted separately from
        # symbols because the syllabics adapter emits three characters at once,
        # and the rate that has to be bounded is the one a reader would read.
        meta: dict[tuple, tuple[str, str, str, float, float, int, int]] = {
            (): ("", "", "", 0.0, 0.0, 0, 0)}
        max_chars = max(8, int(frames * cfg.max_chars_per_second / cfg.frames_per_second))

        for t in range(frames):
            row = work[t]
            # Which symbols this frame can branch into at all. Taking a per-frame
            # shortlist (rather than every symbol, for every hypothesis) is what
            # keeps the search affordable; the blank is handled separately, and
            # the shortlist is never allowed to be shorter than the branching.
            k = min(cfg.symbols_per_frame, row.size)
            top = np.argpartition(row, row.size - k)[row.size - k:]
            shortlist = [(int(i), float(row[int(i)])) for i in top
                         if int(i) != blank_id and vocab_sro[int(i)]]
            shortlist.sort(key=lambda ip: ip[1], reverse=True)
            p_blank = float(row[blank_id])

            new: dict[tuple, list[float]] = {}

            def bump(key: tuple, pb_delta: float, pnb_delta: float) -> None:
                slot = new.get(key)
                if slot is None:
                    new[key] = [pb_delta, pnb_delta]
                else:
                    slot[0] = _logaddexp(slot[0], pb_delta)
                    slot[1] = _logaddexp(slot[1], pnb_delta)

            for key, (p_b, p_nb) in beams.items():
                if p_b < -1e30 and p_nb < -1e30:
                    continue
                total = _logaddexp(p_b, p_nb)
                # 1. stay put
                bump(key, total + p_blank, -math.inf)
                last = key[-1] if key else -1
                for idx, p in shortlist:
                    if idx == last:
                        # the same symbol again: without a blank between them CTC
                        # collapses the two into one, which does not extend the
                        # text; with a blank between them it is a real doubling.
                        if p_nb > -1e30:
                            bump(key, -math.inf, p_nb + p)
                        if p_b > -1e30:
                            self._extend(key, idx, p_b + p, meta, vocab_sro,
                                         lm_cache, bump)
                    else:
                        if total > -1e30:
                            self._extend(key, idx, total + p, meta, vocab_sro,
                                         lm_cache, bump)

            # prune: rank every surviving hypothesis by what the language model
            # and the acoustics together say, and keep the best `beam_width`
            scored: list[tuple[float, tuple]] = []
            for key, (p_b, p_nb) in new.items():
                entry = meta.get(key)
                if entry is None:
                    continue
                text, _hist, _word, lm_sum, word_sum, n, n_chars = entry
                if n_chars > max_chars:
                    continue            # faster than any voice: not a reading of this take
                s = (cfg.acoustic_scale * _logaddexp(p_b, p_nb) + lm_sum + word_sum
                     + cfg.insertion_bonus * n)
                scored.append((s, key))
            if len(scored) > cfg.beam_width:
                scored.sort(key=lambda kv: kv[0], reverse=True)
                del scored[cfg.beam_width:]
            beams = {key: new[key] for _s, key in scored}

        # -- final ranking
        results: list[tuple[float, str]] = []
        seen: set[str] = set()
        for key, (p_b, p_nb) in beams.items():
            entry = meta.get(key)
            if entry is None:
                continue
            text, _hist, word, lm_sum, word_sum, n, _n_chars = entry
            tail = self._word_charge(word) if word else 0.0
            score = (cfg.acoustic_scale * _logaddexp(p_b, p_nb) + lm_sum + word_sum
                     + tail + cfg.insertion_bonus * n)
            cleaned = _tidy(text)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            results.append((score, cleaned))
        # The plain argmax reading competes with the beam on the same objective.
        # It is not a courtesy: the beam prunes by a score that includes the
        # language model, and on one hard span that is enough to walk away from a
        # reading the acoustics clearly support. Scoring the candidate the same
        # way and keeping whichever wins is the accurate resolution.
        greedy_text = _tidy("".join(self._pieces_from(work, pieces)))
        if greedy_text and greedy_text not in seen:
            results.append((self.score(work, pieces, greedy_text)[2], greedy_text))
            results.sort(key=lambda kv: kv[0], reverse=True)
        return results[:n_best]

    def _pieces_from(self, logprobs: np.ndarray,
                     pieces: Sequence[tuple[str, str, bool]]) -> list[str]:
        """CTC-collapse an already-masked posterior into its piece spellings."""
        if logprobs.shape[0] == 0:
            return []
        blank_id = next((i for i, p in enumerate(pieces) if p[2]), 0)
        vocab_sro = [p[1] for p in pieces]
        ids = np.argmax(logprobs, axis=-1)
        out: list[str] = []
        previous: int | None = None
        for i in ids:
            i = int(i)
            if i != previous:
                previous = i
                piece = vocab_sro[i] if 0 <= i < len(vocab_sro) else ""
                if i != blank_id and piece:
                    out.append(piece)
        return out

    def _word_charge(self, word: str) -> float:
        """What one finished word costs the hypothesis that produced it.

        Two terms, and the shape of both matters more than their weights.

        The **lexicon** term is measured *against the unknown word*, not against
        zero. That distinction is the difference between a decoder that
        transcribes and one that does not work at all: `word_logprob` is
        `log P(word)`, a number around -8 for a real word and -12 for a word the
        corpus never saw, and charging the whole of it once per word makes every
        word boundary cost about twenty negative units -- which makes *fewer
        words* the objective, and welds every sentence into one enormous non-word.
        Charging the difference instead means the term says the only thing it has
        any business saying: *a word the language uses is better than a word it
        does not*, by an amount that grows with how often the language uses it,
        and costs nothing to a word that is merely new.

        The **phonotactic** term is what catches a word that is new *and* wrong --
        `west` passes the letter test but its one syllable ends on a coda Plains
        Cree barely uses. It is charged as a *sum* over syllables rather than an
        average, for the same reason as above: an average would be a constant per
        word and would recreate the counting problem the lexicon term was just
        fixed for.
        """
        cfg = self.cfg
        evidence = min(1.0, len(word) / 4.0)
        if not self.lm.is_known(word):
            evidence = 0.0
        charge = cfg.word_weight * evidence
        # relative to an ordinary word of the same length, so that word length
        # stops being charged as if it were a fault
        charge += cfg.phono_weight * self.lm.phonotactic_relative(word)
        if len(word) > cfg.max_word_chars:
            charge -= cfg.illegal_piece_penalty
        return charge

    def _piece_charge(self, piece: str, hist: str, word: str,
                      lm_cache) -> tuple[float, str, str]:
        """One emitted symbol's cost, `(charge, new_history, new_word)`.

        the *only* place the language model's opinion is expressed,
        so that the search and :meth:`score` charge a transcription identically.
        A second copy of this arithmetic is how a decoder comes to rank its own
        best hypothesis below one it rejected, and then no measurement made on
        the ranking means anything.
        """
        cfg = self.cfg
        if piece == " ":
            # a word boundary: pay for the word just closed, then start fresh
            charge = self._word_charge(word) if word else 0.0
            return charge, "", ""
        charge = self._lm_charge(piece, hist, lm_cache)
        if not _legal_piece(piece, self.lm):
            charge -= cfg.illegal_piece_penalty
        new_word = word + piece
        if new_word and not self._trie.is_prefix(new_word):
            charge -= cfg.oov_prefix_penalty
        return charge, hist + piece, new_word

    def language_value(self, text: str) -> float:
        """The language side of the objective for a finished transcription.

        Charged against a character-level segmentation that does not depend on any
        adapter, so the same text always costs the same, whichever model produced
        it -- which is what makes it usable for comparing the two adapters'
        readings of the same audio. Walks the text exactly the way the beam walks
        its hypotheses, so the number agrees with what the search optimises rather
        than merely resembling it.
        """
        lm_cache: dict[tuple[str, str], float] = {}
        hist, word, total = "", "", 0.0
        for piece in _text_to_pieces(text, _CANONICAL_PIECES):
            charge, hist, word = self._piece_charge(piece, hist, word, lm_cache)
            total += self.cfg.lm_weight * charge
        if word:
            total += self.cfg.lm_weight * self._word_charge(word)
        return total

    def language_score(self, text: str, pieces: Sequence[tuple[str, str, bool]]
                       ) -> float:
        """`language_value` scored through a specific adapter's vocabulary.

        Kept for the harness, which uses it to ask whether a given adapter's own
        segmentation of a text is what the search saw. The transcriber itself uses
        `language_value`, because a comparison between adapters must not depend on
        which one is being asked.
        """
        lm_cache: dict[tuple[str, str], float] = {}
        hist, word, total = "", "", 0.0
        for piece in _text_to_pieces(text, pieces):
            charge, hist, word = self._piece_charge(piece, hist, word, lm_cache)
            total += self.cfg.lm_weight * charge
        if word:
            total += self.cfg.lm_weight * self._word_charge(word)
        return total

    def rank_value(self, text: str, acoustic: float, frames: int,
                   agreeing: bool = False) -> float:
        """The number candidates are compared by: a likelihood, plus a bounded allowance.

        `acoustic` is the **absolute** CTC log-probability of the text under one
        or both adapters. Absolute, not per frame, and that is the point: it is a
        probability of the whole span, so a reading that invents three times as
        many words as were sung pays for every one of them. Dividing it by the
        frame count -- which an earlier version of this did -- makes length free,
        and a reading that fills a quiet passage with fluent invented nêhiyawêwin
        is then as good as the right one. Absolute likelihood is what stops it.

        The language terms are *per-frame averages* multiplied back by the frame
        count, so everything in the sum is in the same units as the acoustics. The
        average is the whole trick: a sum of language scores grows with length and
        punishes saying anything (a decoder given one answers with a single word),
        and a sum of *centred* scores rewards saying the commonest thing forever
        (a decoder given one answers with `ki ki ki ki`). Both of those happened
        here before this shape was arrived at. An average can say a reading is
        well-formed or ill-formed and cannot say it should be longer or shorter --
        which leaves the acoustics, correctly, as the only thing deciding how much
        was sung, and the language model free to say only what it knows.
        """
        if not text.strip():
            return -1e9
        cfg = self.cfg
        lm = self.lm
        words = [w for w in text.split(" ") if w]
        per_frame = cfg.rank_char_weight * (lm.text_quality(text) - lm.char_baseline)
        # A reading that has started repeating itself is the shape a CTC model
        # takes when it cannot segment what it is hearing, and it is fluent enough
        # that every other term above likes it. This is the term that does not.
        per_frame -= cfg.rank_repeat_weight * lm.repetition_ratio(text)
        if words:
            per_frame += cfg.rank_word_weight * sum(lm.word_quality(w) for w in words) / len(words)
        syllables = [s for s in syllabify(text) if s]
        if syllables:
            per_frame += cfg.rank_phono_weight * (
                lm.phonotactic_score(text) - lm.syllable_baseline * len(syllables)
            ) / len(syllables)
        if agreeing:
            per_frame += cfg.rank_agree_weight
        return acoustic + max(1, frames) * per_frame

    def greedy(self, logprobs: np.ndarray,
               pieces: Sequence[tuple[str, str, bool]]) -> str:
        """The plain argmax decode, collapsed the way CTC collapses.

        Worth having even though the beam is the point: on singing the acoustics
        are sometimes clearer than the language model is confident, in which case
        the beam's own search can walk away from a perfectly good reading. This
        candidate is scored by the same objective as everything else and competes
        with the beam's hypotheses on equal terms.
        """
        return _tidy("".join(self._pieces_from(self._masked(logprobs, pieces), pieces)))

    def acoustic_value(self, logprobs: np.ndarray,
                       pieces: Sequence[tuple[str, str, bool]], text: str) -> float:
        """The exact CTC log-probability **of a text the model did not produce**.

        This is what makes an ensemble of the two adapters possible at all. The
        beam search only ever returns texts its own model generated, so it can
        never say whether the *other* model would have liked them. The forward
        algorithm can: it marginalises over every alignment, so it scores any
        label sequence in the vocabulary, whoever thought of it first. Cross-
        scoring every candidate under both adapters and averaging is how the
        ensemble below gets its evidence.
        """
        labels = _text_to_indices(text, pieces)
        frames = int(logprobs.shape[0])
        if not labels or frames == 0:
            return -math.inf
        blank_id = next((i for i, p in enumerate(pieces) if p[2]), 0)
        states: list[int] = [blank_id]
        for lab in labels:
            states.extend((lab, blank_id))
        n_states = len(states)
        alpha = [-math.inf] * n_states
        alpha[0] = float(logprobs[0, blank_id])
        if n_states > 1:
            alpha[1] = float(logprobs[0, labels[0]])
        for t in range(1, frames):
            row = logprobs[t]
            nxt = [-math.inf] * n_states
            for s in range(n_states):
                best = alpha[s]
                if s - 1 >= 0:
                    best = _logaddexp(best, alpha[s - 1])
                if s - 2 >= 0 and states[s] != blank_id and states[s] != states[s - 2]:
                    best = _logaddexp(best, alpha[s - 2])
                if best > -math.inf:
                    nxt[s] = best + float(row[states[s]])
            alpha = nxt
        return _logaddexp(alpha[n_states - 1],
                          alpha[n_states - 2] if n_states > 1 else alpha[0])

    def score(self, logprobs: np.ndarray, pieces: Sequence[tuple[str, str, bool]],
              text: str) -> tuple[float, float, float]:
        """`(acoustic, language, total)` for one transcription of one span.

        The acoustic number is the exact CTC prefix probability: the forward
        algorithm over the label sequence interleaved with blanks, which is the
        probability that the model, marginalising over every alignment, produced
        this text. That is the quantity the beam approximates, so computing it
        exactly is what allows an accurate test of whether the beam found the best
        hypothesis. Costs O(frames x labels) and is used only for measurement.
        """
        cfg = self.cfg
        logprobs = self._masked(logprobs, pieces)
        labels = _text_to_indices(text, pieces)
        if not labels or int(logprobs.shape[0]) == 0:
            return -math.inf, 0.0, -math.inf
        acoustic = self.acoustic_value(logprobs, pieces, text)
        language = self.language_value(text)
        n_symbols = len(labels)
        total = (cfg.acoustic_scale * acoustic + language
                 + cfg.insertion_bonus * n_symbols)
        return acoustic, language, total

    def _extend(self, key: tuple, idx: int, prob: float, meta, vocab_sro,
                lm_cache, bump) -> None:
        """Append one symbol to a hypothesis, charging the language model for it."""
        cfg = self.cfg
        piece = vocab_sro[idx]
        new_key = key + (idx,)
        text, hist, word, lm_sum, word_sum, n, n_chars = meta[key]
        charge, new_hist, new_word = self._piece_charge(piece, hist, word, lm_cache)
        new_text = text + piece
        size = cfg.repeat_ngram
        if size and len(new_text) >= size and new_text[-size:] in new_text[:-1]:
            charge -= cfg.repeat_penalty
        meta[new_key] = (new_text, new_hist, new_word,
                         lm_sum + cfg.lm_weight * charge, word_sum, n + 1,
                         n_chars + len(piece.strip()))
        bump(new_key, -math.inf, prob)

    def _lm_charge(self, piece: str, hist: str, cache: dict[tuple[str, str], float]) -> float:
        """The character LM's log-probability of this piece, given the history.

        A piece is one character for the Latin adapter and a whole syllable for
        the syllabics one, so the charge is summed character by character with
        the history advanced inside the piece. Cached: the beam revisits the same
        handful of (context, piece) pairs thousands of times per utterance, and
        that cache is most of why this decoder is fast enough to be usable.
        """
        order = self.cfg.lm_order
        key = (hist[-(order - 1):], piece)
        cached = cache.get(key)
        if cached is None:
            total = 0.0
            h = hist
            for ch in piece:
                # Absolute, not centred: this is the *generator's* score, and a
                # probability is the right thing for a search to climb, because
                # it already prices every symbol it emits. The centred, bounded
                # version is what `rank_value` compares finished candidates with;
                # using a centred score to *search* is how the beam came to answer
                # a whole sentence with `ki ki ki ki`.
                total += self.lm.char_logprob(h, ch)
                h += ch
            cached = cache[key] = total
            if len(cache) > 200_000:
                # bounded, because a long take's contexts are unbounded and an
                # unbounded cache in a long-lived server is a leak
                cache.clear()
                cache[key] = cached
        return cached


def _logaddexp(a: float, b: float) -> float:
    """log(exp a + exp b), without the overflow that plain math would produce."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def _tidy(text: str) -> str:
    """Normalise a decoded hypothesis into something a reader can look at."""
    text = normalize_sro(text)
    return " ".join(w for w in text.split(" ") if w)


def _piece_map(pieces: Sequence[tuple[str, str, bool]]) -> dict[str, int]:
    """`{orthography: piece index}` for one adapter's vocabulary.

    One spelling per index would lose the ambiguity a tokenizer actually has,
    so the first (most frequent, since MMS orders its dictionary by frequency)
    index wins where two tokens spell the same thing.
    """
    mapping: dict[str, int] = {}
    for i, (_token, sro, is_blank) in enumerate(pieces):
        if is_blank or not sro:
            continue
        mapping.setdefault(sro, i)
    return mapping


def _piece_alignment(text: str, pieces: Sequence[tuple[str, str, bool]]) -> list[int]:
    """Split a transcription into the vocabulary pieces an adapter would emit.

    Longest match, so that the syllabics adapter's multi-letter pieces line up
    with the text the way they do in the audio -- `ka` is one syllable glyph and
    one acoustic state, not two letters. A character the vocabulary cannot spell
    at all is skipped rather than crashing: `score` is a measuring instrument and
    has to be able to take whatever it is handed.
    """
    mapping = _piece_map(pieces)
    if not mapping:
        return []
    longest = max(len(k) for k in mapping)
    out: list[int] = []
    i = 0
    while i < len(text):
        for length in range(min(longest, len(text) - i), 0, -1):
            chunk = text[i:i + length]
            index = mapping.get(chunk)
            if index is not None:
                out.append(index)
                i += length
                break
        else:
            i += 1
    return out


def _text_to_pieces(text: str, pieces: Sequence[tuple[str, str, bool]]) -> list[str]:
    """The piece spellings of a transcription, in order."""
    return [pieces[i][1] for i in _piece_alignment(text, pieces)]


def _text_to_indices(text: str, pieces: Sequence[tuple[str, str, bool]]) -> list[int]:
    """The vocabulary indices of a transcription, in order."""
    return _piece_alignment(text, pieces)


# A character-level vocabulary for scoring a transcription *independently of which
# adapter produced it*. The language side of the objective must be a property of
# the text, not of the tokenizer that wrote it down: charging the syllabics
# adapter's syllables the same penalties the Latin adapter pays per letter would
# make the same sentence score differently depending on which model said it,
# which is exactly the kind of artifact that makes a decoder's ranking meaningless.
_CANONICAL_PIECES: tuple[tuple[str, str, bool], ...] = tuple(
    [("<pad>", "", True)]
    + [(ch, ch, False) for ch in sorted(CREE_LETTERS)]
    + [("|", " ", False)]
)


def _legal_piece(piece: str, lm: CreTextModel) -> bool:
    """Could the language have emitted exactly these letters?

    A piece is legal when every letter is one the language is written with and
    every syllable inside it is one the language builds. A bare consonant is
    allowed only as a coda the lexicon actually uses. This is the cheap
    phonotactic gate the beam pays a penalty for crossing; the expensive,
    corpus-frequency version is `CreTextModel.is_legal_word`, which is applied to
    whole words when deciding how far to trust a finished segment.
    """
    if not piece or piece == " ":
        return True
    if any(ch not in CREE_LETTERS for ch in piece):
        return False
    for syllable in syllabify(piece):
        if any(ch in _VOWELS for ch in syllable):
            continue
        if syllable not in lm.codas:
            return False
    return True


class LexiconTrie:
    """Prefix membership over the lexicon, for the in-progress-word penalty.

    Nested dicts are enough at this size (~28k words over a 35-letter alphabet)
    and cost nothing to build, which matters because it is queried once per beam
    extension and a slower structure would show up in the transcriber's latency.
    """

    def __init__(self, words: Iterable[str]):
        self.root: dict = {}
        self.words = set(words)
        for word in words:
            node = self.root
            for ch in word:
                node = node.setdefault(ch, {})
            node["$"] = True

    def is_prefix(self, text: str) -> bool:
        node = self.root
        for ch in text:
            node = node.get(ch)
            if node is None:
                return False
        return True

    def is_word(self, text: str) -> bool:
        return text in self.words


# vocables: sung syllables that are not words

# The syllables these song traditions actually sing, written the way nêhiyawêwin
# writes them. This is the inventory the recogniser snaps to, so a sustained
# "whaa-hey-yo" comes back as `wâ-hê-yo` rather than as whatever four vowels the
# phone model happened to name. It is short and open: a vocable that
# is not in it still decodes -- through the syllable mapping below -- it just does
# not snap to a canonical spelling.
VOCABLE_INVENTORY: tuple[str, ...] = (
    "way", "hay", "ho", "hô", "ha", "hi", "ya", "yô", "yo", "yê", "wê", "wâ",
    "nê", "na", "no", "ni", "aye", "hê", "hey", "ê", "a", "ô", "oho", "hoho",
    "waya", "haya", "yaho", "hiya", "naha", "waha", "yaha", "hoya",
)

# IPA phones the espeak recogniser emits, folded onto the nêhiyawêwin phonemes
# each is closest to. The mapping is articulatory, not orthographic -- Cree `c`
# is [ts] and `ê` is a long [eː], which espeak writes as `ts` and `eː` -- and it
# is written out rather than derived because the espeak alphabet carries
# distinctions the language does not make (four heights of front vowel, five
# rhotics across sixty languages) and each one has to be sent somewhere
# deliberate. Anything not listed is dropped rather than guessed at.
_IPA_TO_CREE: dict[str, str] = {
    # stops and affricates
    "p": "p", "b": "p", "ph": "p", "pː": "p", "ɸ": "p", "f": "p", "v": "w", "β": "w",
    "t": "t", "d": "t", "th": "th", "θ": "th", "ð": "th", "t̪": "t", "tː": "t",
    "k": "k", "g": "k", "ɡ": "k", "q": "k", "ɟ": "k", "kh": "k", "kʲ": "k",
    "c": "c", "ts": "c", "t͡s": "c", "tsh": "c", "tʃ": "c", "dʒ": "c", "tɕ": "c",
    "dʑ": "c", "tɕh": "c", "ts.": "c",
    # nasals
    "m": "m", "n": "n", "ŋ": "n", "ɲ": "n", "n̩": "n", "ɳ": "n", "ɴ": "n",
    # fricatives
    "s": "s", "z": "s", "ʃ": "s", "ʒ": "s", "ɕ": "s", "ʑ": "s", "ʂ": "s",
    "s̪": "s", "s.": "s", "h": "h", "x": "h", "χ": "h", "ɣ": "h", "ʁ": "h",
    "ç": "h", "ħ": "h", "ʕ": "h", "ɦ": "h", "ɬ": "th", "ɮ": "th",
    # approximants and liquids
    "w": "w", "ʋ": "w", "j": "y", "ʎ": "l", "l": "l", "ɫ": "l", "ɭ": "l",
    "r": "r", "ɾ": "r", "ɹ": "r", "ɻ": "r", "ʀ": "r",
    # vowels: short
    "a": "a", "ä": "a", "ɐ": "a", "ʌ": "a", "ə": "a", "ɜ": "a", "æ": "a",
    "e": "ê", "e̞": "ê", "ɛ": "ê", "ɪ": "i", "i": "i", "ᵻ": "i", "ɨ": "i",
    "ɯ": "i", "y": "i", "ø": "i", "œ": "i",
    "o": "o", "o̞": "o", "ɔ": "o", "u": "o", "ʊ": "o", "ʉ": "o", "ɵ": "o", "ɒ": "o",
    # vowels: long
    "aː": "â", "ɑː": "â", "ɑ": "â", "ɐː": "â",
    "eː": "ê", "ɛː": "ê", "ɜː": "ê", "ei": "ê", "eɪ": "ê",
    "iː": "î", "ɨː": "î", "iy": "î",
    "oː": "ô", "ɔː": "ô", "ɔːɹ": "ô", "uː": "ô",
    # diphthongs: the language has one glide per syllable
    "aɪ": "ay", "aɪə": "aya", "aɪɚ": "aya", "aʊ": "aw", "ɔɪ": "oy", "oʊ": "ô",
    "ou": "ô", "əʊ": "ô", "iə": "iya", "ja": "ya", "ju": "yo",
}
_IPA_VOWEL_MAP_KEYS = {k for k, v in _IPA_TO_CREE.items() if v[:1] in _VOWELS or v in ("â", "ê", "î", "ô")}
_LONG_MARK = "ː"
_STRESS_MARKS = "ˈˌ"
_DROP_TOKEN = re.compile(r"^[0-9.\-\[\]()?!,;:]+$")


class VocableRecognizer:
    """Turns non-lexical singing into nêhiyawêwin syllables.

    The phone model is Meta's espeak-IPA wav2vec2: it emits a universal phone
    alphabet, with no language model and no vocabulary of words, which is exactly
    the right shape of model for singing that is not made of words. Its phones
    are mapped onto the nêhiyawêwin inventory by articulation, grouped into
    syllables around their vowel nuclei, and the syllables are matched against
    the vocable inventory above so the spelling settles on the form a singer
    would write. The result goes through the same orthography engine as
    everything else, so vocables arrive in syllabics and anglicized too.
    """

    def __init__(self, model_dir: pathlib.Path | None = None):
        self._dir = (pathlib.Path(model_dir or _DEFAULT_MODEL_DIR)
                     / "facebook--wav2vec2-lv-60-espeak-cv-ft")
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def error(self) -> str | None:
        return self._error

    def load(self) -> bool:
        """Load the feature extractor, vocabulary and weights -- without phonemizer.

        This model ships as a `Wav2Vec2PhonemeCTCTokenizer`, which refuses to
        exist unless the `phonemizer` package (and an espeak-ng binary) are
        installed, because it is built to turn *text* into phones. Only the
        opposite direction is ever needed here, so the vocabulary JSON is loaded
        through the plain CTC tokenizer and the feature extractor is the ordinary
        wav2vec2 one. Nothing is lost: the acoustic model is the only part that
        hears anything, and its weights are unchanged.
        """
        with self._lock:
            if self._model is not None:
                return True
            if not self._dir.is_dir():
                self._error = f"no phone model directory at {self._dir}"
                return False
            try:
                from transformers import AutoModelForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

                vocab_file = self._dir / "vocab.json"
                if not vocab_file.exists():
                    raise FileNotFoundError(f"no vocabulary at {vocab_file}")
                self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
                    str(self._dir), local_files_only=True)
                self._tokenizer = Wav2Vec2CTCTokenizer(
                    vocab_file=str(vocab_file), unk_token="<unk>", pad_token="<pad>",
                    word_delimiter_token="|", do_lower_case=False)
                self._model = AutoModelForCTC.from_pretrained(
                    str(self._dir), local_files_only=True)
                self._model.eval()
                return True
            except Exception as exc:  # noqa: BLE001
                self._error = f"{type(exc).__name__}: {exc}"
                return False

    def phones(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> list[str]:
        """The phone sequence of a stretch of singing (espeak IPA symbols)."""
        if not self.load():
            return []
        import torch

        y = np.asarray(audio, dtype=np.float32)
        if y.size < 1600:
            return []
        inputs = self._processor(y, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
        ids = torch.argmax(logits, dim=-1).tolist()
        vocab = self._tokenizer.convert_ids_to_tokens
        phones: list[str] = []
        previous = None
        for i in ids:
            if i == previous:
                continue
            previous = i
            token = vocab(int(i))
            if not token or token in _JUNK_TOKENS:
                continue
            phones.append(token)
        return phones

    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE
                   ) -> tuple[str, float, list[str]]:
        """`(sro, confidence, notes)` for one stretch of singing."""
        phones = self.phones(audio, sr)
        if not phones:
            return "", 0.0, ["no phones recognised"]
        syllables, notes = self._syllabify(phones)
        if not syllables:
            return "", 0.0, ["phones did not group into syllables"]
        words = self._snap(syllables)
        sro = normalize_sro(" ".join(words).strip())
        confidence = min(0.80, 0.26 + 0.07 * len(syllables))
        return sro, confidence, notes

    # -- internals ---------------------------------------------------------

    def _to_cree(self, token: str) -> str | None:
        """One espeak phone to one nêhiyawêwin letter (or letter pair), or None."""
        base = token
        for mark in _STRESS_MARKS:
            base = base.replace(mark, "")
        base = base.strip()
        if not base or _DROP_TOKEN.match(base):
            return None
        if base in _IPA_TO_CREE:
            return _IPA_TO_CREE[base]
        long_vowel = base.endswith(_LONG_MARK)
        if long_vowel:
            base = base.rstrip(_LONG_MARK)
            if base in _IPA_TO_CREE:
                mapped = _IPA_TO_CREE[base]
                # lengthen only a vowel this language has a long form of
                return {"a": "â", "ê": "ê", "e": "ê", "i": "î", "o": "ô"}.get(mapped, mapped)
        # Multi-phone tokens the alphabet carries (`aɪ`, `tʃ`, `onɡ5`): take the
        # first symbol that maps, which is the syllable's head, and let the rest
        # come from the phones around it.
        for length in (3, 2):
            for i in range(0, max(0, len(base) - length + 1)):
                chunk = base[i:i + length]
                if chunk in _IPA_TO_CREE:
                    return _IPA_TO_CREE[chunk]
        return None

    def _syllabify(self, phones: list[str]) -> tuple[list[str], list[str]]:
        """Group phones into syllables around their vowel nuclei, in Cree letters.

        One vowel per syllable is the language's rule, so the algorithm is: walk
        the phones, and every time a vowel arrives, it closes the syllable that
        was accumulating in front of it. A consonant with no vowel after it
        before the next consonant joins the previous syllable as its coda, which
        is where `way` comes from -- `w`, `a`, then a trailing `y`.
        """
        notes: list[str] = []
        letters: list[tuple[str, bool]] = []
        for token in phones:
            cree = self._to_cree(token)
            if cree is None:
                notes.append(f"dropped phone {token!r}")
                continue
            for i, ch in enumerate(cree):
                letters.append((ch, ch in _VOWELS))

        syllables: list[str] = []
        pending = ""
        for ch, is_vowel in letters:
            if is_vowel:
                syllables.append(pending + ch)
                pending = ""
            else:
                if pending:
                    # two consonants and no vowel between them: the first was a
                    # coda of the syllable before, this one opens the next
                    if syllables:
                        syllables[-1] += pending
                    pending = ch
                else:
                    pending = ch
        if pending:
            if syllables:
                syllables[-1] += pending
            else:
                syllables.append(pending)
        return [s for s in syllables if s], notes

    def _snap(self, syllables: list[str]) -> list[str]:
        """Snap near-miss syllables onto the vocable inventory, then group words.

        A syllable already in the inventory stays exactly as recognised. One that
        is close takes the inventory's spelling, which is what keeps `haj` from
        being written when the singer is singing `hay`. One with no near
        neighbour is kept as recognised: an accurate syllable beats a familiar
        wrong one, and that is the whole reason this path exists.
        """
        import difflib

        snapped: list[str] = []
        for syllable in syllables:
            if syllable in VOCABLE_INVENTORY:
                snapped.append(syllable)
                continue
            key = _syllable_key(syllable)
            best, best_ratio = None, 0.0
            for candidate in VOCABLE_INVENTORY:
                ratio = difflib.SequenceMatcher(None, key, _syllable_key(candidate)).ratio()
                if ratio > best_ratio:
                    best, best_ratio = candidate, ratio
            if best is not None and best_ratio >= 0.70:
                snapped.append(best)
            else:
                snapped.append(syllable)

        # A short run of one-syllable vocables IS the phrase the singer sang
        # ("way hay hoo"), so those stay separate words. Longer runs are written
        # out as the inventory writes its own multi-syllable entries, which is
        # how these appear in the transcriptions people actually make.
        words: list[str] = []
        buffer: list[str] = []
        for syllable in snapped:
            buffer.append(syllable)
            if len("".join(buffer)) >= 7:
                words.append("".join(buffer))
                buffer = []
        if buffer:
            words.append("".join(buffer))
        return words


def _syllable_key(syllable: str) -> str:
    """A syllable as (onset class, nucleus, coda class), for fuzzy matching.

    Comparing raw letters would call `hay` and `hy` similar; comparing the
    skeleton means two syllables are alike when they are built the same way,
    which is what "the singer sang the same syllable" means.
    """
    match = re.match(r"^([^aeiouâêîô]*)([aeiouâêîô]+)([^aeiouâêîô]*)$", syllable)
    if not match:
        return syllable
    onset, nucleus, coda = match.groups()
    return f"{onset[:1]}|{nucleus}|{coda[:1]}"


# the transcriber that ties it together

class CreeTranscriber:
    """End to end: a take in, a nêhiyawêwin transcript in three scripts out."""

    def __init__(self, cfg: CreeASRConfig | None = None,
                 text_model: CreTextModel | None = None):
        self.cfg = cfg or CreeASRConfig()
        self._text_model = text_model
        self._asr = MMSCreeASR(self.cfg)
        self._vocables = VocableRecognizer(self.cfg.model_dir)
        self._decoder: CTCDecoder | None = None
        self._decoder_lock = threading.Lock()

    # -- resources ---------------------------------------------------------

    @property
    def text_model(self) -> CreTextModel:
        if self._text_model is None:
            self._text_model = load_text_model()
        return self._text_model

    def decoder(self) -> CTCDecoder:
        with self._decoder_lock:
            if self._decoder is None:
                self._decoder = CTCDecoder(self.text_model, self.cfg)
        return self._decoder

    @property
    def available(self) -> bool:
        return self._asr.load()

    def health(self) -> dict:
        return {
            "engine": "mms-1b-all + crk-script_latin + crk-script_syllabics",
            "mms": {"available": self._asr.available, "error": self._asr.load_error,
                    "adapters": list(self.cfg.adapters)},
            "vocable_model": {"available": self._vocables.available,
                              "error": self._vocables.error},
            "demucs": {"available": _DEMUCS_CACHE is not None, "error": _DEMUCS_ERROR},
            "text_model": self.text_model.stats(),
        }

    # -- main entry --------------------------------------------------------

    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE, *,
                   isolate: str | None = None, language: str = "crk") -> CreeTranscript:
        """Transcribe a take.

        Never raises for audio reasons: a take with no voice in it returns an
        empty transcript with notes, because this is called from a web endpoint
        that has to answer whatever the phone sent it.
        """
        started = time.perf_counter()
        timing: dict[str, float] = {}
        notes: list[str] = []

        y = np.asarray(audio, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
            sr = SAMPLE_RATE
        duration = y.size / sr

        if not self._asr.load():
            return CreeTranscript([], duration=duration, notes=notes,
                                  timing={"error": self._asr.load_error or "unavailable"})

        mode = isolate or self.cfg.isolation
        t0 = time.perf_counter()
        vocals, method = isolate_vocals(y, sr, mode)
        timing["isolation"] = time.perf_counter() - t0
        if mode in ("auto", "demucs") and method == "off":
            notes.append("Demucs weights unavailable; transcribed the mixture directly")

        conditioned, conditioning_notes = _condition(vocals, self.cfg, sr)
        notes.extend(conditioning_notes)

        spans = find_voiced_spans(conditioned, self.cfg, sr)
        timing["segmentation"] = time.perf_counter() - t0 - timing["isolation"]
        if not spans:
            return CreeTranscript([], duration=duration, timing=timing, notes=notes)

        segments: list[TranscriptSegment] = []
        t_decode = time.perf_counter()
        for (a, b) in spans:
            clip = conditioned[int(a * sr):int(b * sr)]
            if clip.size < int(0.2 * sr):
                continue
            segment = self._transcribe_span(clip, a, b, notes)
            if segment is not None:
                segments.append(segment)
        timing["decode"] = time.perf_counter() - t_decode

        segments = _merge_adjacent(segments, self.cfg)
        timing["total"] = time.perf_counter() - started
        timing["audio_seconds"] = duration
        if duration:
            timing["realtime_factor"] = timing["total"] / duration
        return CreeTranscript(segments, duration=duration, timing=timing, notes=notes)

    def _transcribe_span(self, clip: np.ndarray, start: float, end: float,
                         notes: list[str],
                         posteriors: dict[str, np.ndarray] | None = None
                         ) -> TranscriptSegment | None:
        """Read one span, and decide accurately how much of it to claim.

        Two readings are produced and then compared on the evidence rather than
        on taste: the **acoustic** reading, which is a search over the language
        constrained by the lexicon, the character model and the syllable
        inventory, and the **phone** reading, which maps an IPA phone recogniser
        onto the language's syllables and knows nothing about words at all. The
        phone reading exists because sung syllables are not words and a word
        search will not find them; the acoustic reading exists because most of
        what a singer sings *is* words.

        Which one wins is decided by `text_quality` -- how much the language model
        likes the letters -- because that is the one number that follows the
        language rather than the audio, and it is measured rather than asserted
        (see `CreeASRConfig.quality_low`). When neither reading is good enough to
        claim, the better of the two is still reported, as `uncertain`: showing a
        reader the syllables the model did hear is useful, while printing an
        English word nobody sang is the failure this module was written to end.
        """
        cfg = self.cfg
        if posteriors is None:
            try:
                posteriors = self._asr.posteriors(clip, SAMPLE_RATE)
            except Exception as exc:  # noqa: BLE001 - one bad span must not kill the take
                notes.append(f"span {start:.1f}-{end:.1f} failed: {type(exc).__name__}")
                return None

        candidates = self.candidates(posteriors, notes)
        if not candidates:
            return None
        _rank, acoustic_text, agreeing = candidates[0]
        lm = self.text_model
        acoustic_conf = text_confidence(lm, acoustic_text, bool(agreeing), cfg)

        # The phone reading, always taken: it is cheap (a small model on one
        # span), and it is the only reading that can represent a syllable which
        # is not a word.
        vocable_text, vocable_conf, vocable_notes = self._vocables.transcribe(clip)
        notes.extend(vocable_notes[:2])
        vocable_conf = _vocable_confidence(lm, vocable_text, vocable_conf, cfg)

        if not acoustic_text and not vocable_text:
            return TranscriptSegment(start=start, end=end, sro="", kind="uncertain",
                                     confidence=0.0, notes=vocable_notes[:2])

        acoustic_quality = lm.text_quality(acoustic_text) if acoustic_text else -1e9
        vocable_quality = lm.text_quality(vocable_text) if vocable_text else -1e9
        # The phone reading has to be clearly better to displace the acoustic one,
        # because the acoustic one was searched with the language model in it and
        # the phone one was not: a tie goes to the search.
        use_vocable = vocable_quality > acoustic_quality + cfg.vocable_margin

        if use_vocable:
            kind = "vocable" if vocable_conf >= cfg.vocable_confidence else "uncertain"
            if kind == "vocable" and acoustic_conf >= cfg.lexical_confidence:
                kind = "mixed"          # words on one side, vocables on the other
            return TranscriptSegment(start=start, end=end, sro=vocable_text, kind=kind,
                                     confidence=vocable_conf, notes=vocable_notes[:2])

        words = [w for w in acoustic_text.split(" ") if w]
        if not words:
            return TranscriptSegment(start=start, end=end, sro="", kind="uncertain",
                                     confidence=0.0, notes=vocable_notes[:2])
        kind = "cree" if acoustic_conf >= cfg.lexical_confidence else "uncertain"
        if kind == "cree" and acoustic_conf < cfg.lexical_confidence + 0.2:
            kind = "mixed"
        return TranscriptSegment(
            start=start, end=end, sro=" ".join(words), kind=kind,
            confidence=acoustic_conf,
            words=[{"word": w, "kind": "cree" if lm.is_known(w) else "legal",
                    "syllabics": to_syllabics(w), "anglicized": anglicize(w),
                    "known": lm.is_known(w)} for w in words],
            notes=(["both adapters agree"] if agreeing else [])
                  + (["low confidence"] if kind == "uncertain" else []),
        )

    def candidates(self, posteriors: dict[str, np.ndarray],
                   notes: list[str]) -> list[tuple]:
        """Ranked readings of one span: `(rank, text, agreeing)`, best first.

        Separated from the decision about what to *claim* because the two
        questions are different and are measured separately: whether the right
        reading was among the candidates at all is a question about the search,
        and which candidate is chosen is a question about the objective. The
        harness scores both, because a system whose search never finds the right
        answer and a system that finds it and rejects it need opposite fixes.
        """
        decoder = self.decoder()
        cfg = self.cfg

        # every adapter's readings, gathered into one candidate set
        order: list[str] = []                # candidate texts, in the order found
        seen: set[str] = set()
        best_per_adapter: dict[str, str] = {}
        for adapter in cfg.adapters:
            logprobs = posteriors.get(adapter)
            if logprobs is None or logprobs.shape[0] == 0:
                continue
            pieces = self._asr.tokenizer_for(adapter)[1]
            try:
                hypotheses = decoder.decode(logprobs, pieces, n_best=cfg.n_best)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{adapter} decode failed: {type(exc).__name__}: {str(exc)[:70]}")
                continue
            if hypotheses:
                best_per_adapter[adapter] = hypotheses[0][1]
            for _score, text in hypotheses:
                if text and text not in seen:
                    seen.add(text)
                    order.append(text)

        # cross-scoring: ask *every* adapter about *every* candidate
        #
        # This is the part that makes the ensemble worth having. A beam search can
        # only score the texts its own model produced, so on its own each adapter
        # is guessing where it is weakest and the two never compare notes. The CTC
        # forward algorithm can score any label sequence, so each candidate is
        # asked of both adapters and the answers are averaged. Two models that
        # make *different* mistakes, averaged, are better than either -- which is
        # the oldest and most reliable result in speech recognition, and the reason
        # both crk adapters are in this file at all.
        scored: list[tuple[float, str, str]] = []            # (rank, text, label)
        frames = {adapter: max(1, int(logp.shape[0]))
                  for adapter, logp in posteriors.items() if logp is not None}
        for text in order:
            acoustic, used = 0.0, []
            for adapter in cfg.adapters:
                logprobs = posteriors.get(adapter)
                if logprobs is None or logprobs.shape[0] == 0:
                    continue
                pieces = self._asr.tokenizer_for(adapter)[1]
                try:
                    value = decoder.acoustic_value(logprobs, pieces, text)
                except Exception:  # noqa: BLE001
                    continue
                if value == -math.inf:
                    continue
                weight = float(cfg.adapter_weights.get(adapter, 1.0))
                # accumulated as it comes; the per-frame normalisation these
                # used to get happens once, in `rank_value`, and only on the
                # language side where it belongs
                acoustic += weight * value
                used.append(adapter)
            if not used:
                continue
            acoustic /= len(used)
            span_frames = sum(frames[a] for a in used) / len(used)
            rank = decoder.rank_value(text, acoustic, span_frames)
            scored.append((rank, text, "+".join(a.split("-")[-1] for a in used)))

        if not scored:
            return []
        scored.sort(key=lambda c: c[0], reverse=True)
        # Both adapters independently preferring the same text is evidence, and it
        # is measured on each one's own best reading rather than on the union,
        # where agreement would be an artifact of the union.
        agreeing = (len(best_per_adapter) > 1
                    and len(set(best_per_adapter.values())) == 1)
        return [(rank, text, agreeing) for rank, text, _label in scored]


def _legal_syllable_share(lm: CreTextModel, text: str) -> float:
    """The share of a text's syllables that the language itself builds.

    Not `is_legal_word`, which asks the question of whole words: this asks it of
    each syllable in turn, so a reading that is mostly well-formed nêhiyawêwin
    but has one odd cluster in it is credited for the parts that are right
    instead of being thrown out whole.
    """
    syllables = [s for s in syllabify(text) if s]
    if not syllables:
        return 0.0
    legal = 0
    for syllable in syllables:
        match = re.match(r"^([^aeiouâêîô]*)([aeiouâêîô]+)([^aeiouâêîô]*)$", syllable)
        if not match:
            continue
        onset, nucleus, coda = match.groups()
        if (not onset or onset in lm.onsets) and nucleus in lm.nuclei \
                and (not coda or coda in lm.codas):
            legal += 1
    return legal / len(syllables)


def text_confidence(lm: CreTextModel, text: str, agreeing: bool,
                    cfg: CreeASRConfig) -> float:
    """How much of a decode should be believed, as a number in [0, 1].

    *not* the acoustic model's own softmax. On singing a CTC model is
    confidently wrong -- that is exactly how the previous pipeline produced fluent
    English that nobody sang, and no threshold on its confidence would have
    caught it. What is measured here instead is how much of the output *is* the
    language, in three parts whose weights were set by looking at both:

      * `text_quality`, the character model's per-letter opinion, which is the
        dominant term because it needs no word list to be right about a language;
      * the mean evidence that the individual words are real, which catches a
        short decode the character model is merely neutral about;
      * the share of words the lexicon holds outright, which is the strongest
        single kind of evidence for the words it applies to;
      * the share of the text's syllables the language is able to build, which
        credits a reading that is well-formed but not yet a word.

    A decode of mostly-unknown letters scores near zero however sure the acoustics
    were, and that number is what a segment's `kind` is decided from.
    """
    if not text.strip():
        return 0.0
    words = [w for w in text.split(" ") if w]
    quality = lm.text_quality(text)
    span = max(1e-6, cfg.quality_high - cfg.quality_low)
    curve = min(1.0, max(0.0, (quality - cfg.quality_low) / span))
    evidence = (sum(lm.word_quality(w) for w in words) / len(words)) if words else 0.0
    known_share = (sum(1 for w in words if lm.is_known(w)) / len(words)) if words else 0.0
    confidence = 0.56 * curve + 0.22 * evidence + 0.12 * known_share
    confidence *= 0.90 + 0.10 * _legal_syllable_share(lm, text)
    # a two-letter decode is easy to get right by accident, so it is scaled down
    confidence *= min(1.0, sum(len(w) for w in words) / 4.0)
    if agreeing:
        confidence = min(1.0, confidence * 1.10)
    return float(max(0.0, min(1.0, confidence)))


def _vocable_confidence(lm: CreTextModel, text: str, raw: float,
                        cfg: CreeASRConfig) -> float:
    """How much of a *phone* reading should be believed, which is less.

    The phone recogniser has no language model, so a confident phone sequence is
    not evidence of anything -- its own softmax (the `raw` number) says how sure
    it is about sounds, and on sustained singing with drums under it, it is sure
    about the wrong sounds. What is added here is the only check that can be made
    on its output: **how much of it is nêhiyawêwin at all**. Every syllable it
    produced is tested against the language's own syllable inventory, and the
    reading is trusted in proportion to how many of them passed and how much the
    character model likes the result. Garbage syllables -- the `krs`, the `kway`
    welded to a name -- take the score down instead of being reported as a
    confident transcription.
    """
    if not text.strip():
        return 0.0
    legal_share = _legal_syllable_share(lm, text)
    quality = lm.text_quality(text)
    span = max(1e-6, cfg.quality_high - cfg.quality_low)
    curve = min(1.0, max(0.0, (quality - cfg.quality_low) / span))
    return float(max(0.0, min(1.0,
        0.45 * legal_share + 0.35 * curve + 0.20 * min(1.0, raw))))


def _merge_adjacent(segments: list[TranscriptSegment], cfg: CreeASRConfig
                    ) -> list[TranscriptSegment]:
    """Join neighbouring segments of the same kind when they are one phrase.

    Segmenting a melody by energy cuts the singer's breath, not their sentences,
    so two spans of the same kind a fraction of a second apart are almost always
    one phrase and are written as one.
    """
    if not segments:
        return []
    out = [segments[0]]
    for segment in segments[1:]:
        previous = out[-1]
        if (previous.kind == segment.kind and segment.start - previous.end <= cfg.max_gap_seconds
                and previous.sro and segment.sro):
            previous.end = segment.end
            previous.sro = f"{previous.sro} {segment.sro}".strip()
            previous.syllabics = to_syllabics(previous.sro)
            previous.anglicized = anglicize(previous.sro)
            previous.words = previous.words + segment.words
            previous.confidence = max(previous.confidence, segment.confidence)
        else:
            out.append(segment)
    return out


# process-wide instance

_TRANSCRIBER: CreeTranscriber | None = None
_TRANSCRIBER_LOCK = threading.Lock()


def get_transcriber(cfg: CreeASRConfig | None = None) -> CreeTranscriber:
    """The shared transcriber; loading a 1B model twice would be wasteful."""
    global _TRANSCRIBER
    with _TRANSCRIBER_LOCK:
        if _TRANSCRIBER is None:
            _TRANSCRIBER = CreeTranscriber(cfg)
    return _TRANSCRIBER


def warmup() -> CreeTranscriber:
    """Load the models at start-up so the first real request is not the slow one."""
    transcriber = get_transcriber()
    transcriber._asr.load()
    return transcriber
