"""Semantic tokens: a self-supervised speech/music encoder plus k-means.

The paper uses intermediate w2v-BERT representations at 25 Hz, quantized with a
1024-centre k-means trained on the corpus.  The encoder is deliberately a thin
adapter around Hugging Face's w2v-BERT/HuBERT implementations: the SingSong
contribution is the conditional cascade, not a new self-supervised objective,
and reimplementing w2v-BERT would only make the checkpoint incompatible with
the published one.

The adapter supports either a local Hugging Face directory or a model id.  This
project never downloads at inference time, so a local directory is the
production path; a model id is useful for the one-off k-means training step.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import SemanticConfig


@dataclass
class KMeansTokenizer:
    """A thin wrapper so both FAISS and scikit-learn k-means work."""

    model: object
    codebook_size: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.ascontiguousarray(features, dtype=np.float32)
        if hasattr(self.model, "search"):
            _, indices = self.model.search(features, 1)
            return indices[:, 0].astype(np.int64)
        return np.asarray(self.model.predict(features), dtype=np.int64)

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "KMeansTokenizer":
        import joblib

        obj = joblib.load(path)
        if isinstance(obj, cls):
            return obj
        # Older checkpoints may be a bare sklearn/faiss object.
        return cls(model=obj, codebook_size=int(getattr(obj, "n_clusters", 1024)))


def _resample(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    import torchaudio.functional as AF

    return AF.resample(wav, sr, target_sr)


def _zero_mean_unit_var(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + 1e-7)


class SemanticEncoder:
    def __init__(self, cfg: SemanticConfig, *, device: torch.device | None = None):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_extractor = None
        self.uses_features = False
        self.model_type = ""
        self.kmeans: KMeansTokenizer | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, *, kmeans_path: str | Path | None = None) -> "SemanticEncoder":
        from transformers import AutoConfig, AutoModel

        path = str(self.cfg.model_path)
        if not Path(path).exists() and not path.startswith(("facebook/", "m-a-p/", "microsoft/")):
            raise FileNotFoundError(
                f"semantic encoder not found at {path!r}; point singsong.semantic.model_path "
                "at a local w2v-BERT/HuBERT directory"
            )
        config = AutoConfig.from_pretrained(path, trust_remote_code=False)
        self.model_type = str(getattr(config, "model_type", ""))
        # w2v-BERT 2.0 consumes Kaldi-style mel features (two 80-bin frames
        # stacked into 160 bins), not a raw waveform.  HuBERT and wav2vec 2.0
        # consume ``input_values`` instead.  Passing the wrong one silently
        # feeds the wrong tensor or fails inside the model, so the adapter
        # chooses from the checkpoint's own config rather than guessing.
        self.uses_features = self.model_type == "wav2vec2-bert"
        if self.uses_features:
            from transformers import SeamlessM4TFeatureExtractor

            self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
                path, trust_remote_code=False
            )
        self.model = AutoModel.from_pretrained(path, trust_remote_code=False)
        self.model.to(self.device).eval()
        if kmeans_path is not None:
            self.kmeans = KMeansTokenizer.load(kmeans_path)
        return self

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def embeddings(
        self,
        wav: torch.Tensor,
        sample_rate: int = 16000,
        *,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        """Return [B, frames, hidden] at the encoder's native frame rate."""
        if self.model is None:
            raise RuntimeError("call .load() before extracting features")
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim != 2:
            raise ValueError("wav must be [B, T]")
        # ``native_hz`` is the feature frame rate, not an audio sample rate.
        # The encoder always consumes 16 kHz waveform (or 16 kHz mel frames);
        # treating 50 as a sample rate would throw away 99.7% of the signal.
        wav = _resample(wav.to(self.device), sample_rate, self.cfg.sample_rate)
        if self.uses_features:
            emb = self._features_from_mel(wav)
        else:
            outputs = self.model(
                input_values=wav,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states
            if hidden is None:
                raise RuntimeError("semantic encoder did not return hidden states")
            idx = min(max(int(self.cfg.hidden_layer), 0), len(hidden) - 1)
            emb = hidden[idx]
        if normalize is None:
            normalize = self.cfg.normalize_embeddings
        if normalize:
            emb = _zero_mean_unit_var(emb)
        return emb

    @torch.no_grad()
    def _features_from_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """Run the checkpoint's own mel frontend, one waveform at a time.

        ``SeamlessM4TFeatureExtractor`` expects NumPy samples in [-1, 1] and
        stacks two 10 ms Kaldi frames, so its 16 kHz output is 50 Hz.  The
        extractor pads a batch to its longest item; running each item
        separately keeps frame counts exact, and the caller's batch is only
        ever one 10 s vocal window in practice.
        """
        if self.feature_extractor is None:
            raise RuntimeError("feature extractor is not loaded")
        hidden_states: list[torch.Tensor] = []
        for one in wav:
            batch = self.feature_extractor(
                one.detach().float().cpu().numpy(),
                sampling_rate=int(self.cfg.sample_rate),
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = batch["input_features"].to(self.device)
            mask = batch.get("attention_mask")
            outputs = self.model(
                input_features=input_features,
                attention_mask=None if mask is None else mask.to(self.device),
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states
            if hidden is None:
                raise RuntimeError("semantic encoder did not return hidden states")
            idx = min(max(int(self.cfg.hidden_layer), 0), len(hidden) - 1)
            hidden_states.append(hidden[idx])
        if len(hidden_states) == 1:
            return hidden_states[0]
        longest = max(h.shape[1] for h in hidden_states)
        padded = [
            F.pad(h, (0, 0, 0, longest - h.shape[1])) for h in hidden_states
        ]
        return torch.cat(padded, dim=0)

    @torch.no_grad()
    def encode(
        self,
        wav: torch.Tensor,
        sample_rate: int = 16000,
        *,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return [B, T/40 ms] semantic token ids (25 Hz by default)."""
        emb = self.embeddings(wav, sample_rate)
        if return_embeddings:
            return emb
        if self.kmeans is None:
            raise RuntimeError("semantic k-means is not loaded")
        # Native frame rate is 50 Hz for the public w2v-BERT/HuBERT checkpoints;
        # the paper's semantic stream is 25 Hz.  Average pairs of frames, which
        # is the same low-pass the paper's 25 Hz w2v-BERT layer provides.
        if self.cfg.output_hz < self.cfg.native_hz:
            factor = max(1, int(round(self.cfg.native_hz / self.cfg.output_hz)))
            usable = (emb.shape[1] // factor) * factor
            if usable == 0:
                raise ValueError("audio is too short for the requested semantic rate")
            emb = emb[:, :usable].reshape(emb.shape[0], usable // factor, factor, emb.shape[-1]).mean(2)
        flat = emb.reshape(-1, emb.shape[-1]).float().cpu().numpy()
        ids = self.kmeans.predict(flat)
        if ids.size % emb.shape[0]:
            raise RuntimeError("k-means returned an unexpected number of tokens")
        return torch.from_numpy(ids.reshape(emb.shape[0], -1)).long().to(emb.device)

    @property
    def output_hz(self) -> int:
        return self.cfg.output_hz

    @property
    def codebook_size(self) -> int:
        return self.cfg.codebook_size


def train_kmeans(
    features: np.ndarray,
    *,
    codebook_size: int = 1024,
    seed: int = 0,
    use_faiss: bool = True,
    sample_limit: int | None = 2_000_000,
) -> KMeansTokenizer:
    """Fit the 1024-centre codebook on a (N, D) feature matrix.

    FAISS' k-means is dramatically faster on the tens of millions of frames a
    full corpus produces.  It is optional: scikit-learn's MiniBatchKMeans is a
    correct fallback and is what the unit tests use.
    """
    features = np.ascontiguousarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must be [N, D]")
    if sample_limit and len(features) > sample_limit:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(features), size=sample_limit, replace=False)
        features = features[idx]

    if use_faiss:
        try:
            import faiss

            kmeans = faiss.Kmeans(
                features.shape[1], codebook_size, niter=100, verbose=False,
                seed=seed, nredo=1, min_points_per_centroid=1,
            )
            kmeans.train(features)
            return KMeansTokenizer(model=kmeans, codebook_size=codebook_size)
        except ImportError:
            pass

    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(
        n_clusters=codebook_size,
        init="k-means++",
        batch_size=10000,
        max_iter=100,
        n_init=10,
        random_state=seed,
        reassignment_ratio=0.0,
    )
    model.fit(features)
    return KMeansTokenizer(model=model, codebook_size=codebook_size)


def load_semantic(cfg: SemanticConfig, kmeans_path: str | Path | None = None) -> SemanticEncoder:
    encoder = SemanticEncoder(cfg)
    return encoder.load(kmeans_path=kmeans_path)
