from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"
_CHUNK_SIZE_CHARS = 900
_CHUNK_OVERLAP_CHARS = 150


@dataclass
class Chunk:
    text: str
    source: str          # filename the chunk came from
    chunk_index: int      # position within that file


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def _chunk_text(text: str, source: str) -> list[Chunk]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 2 <= _CHUNK_SIZE_CHARS:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf)
            if len(para) <= _CHUNK_SIZE_CHARS:
                buf = para
            else:
                # a single paragraph longer than one chunk: slide a window over it
                start = 0
                while start < len(para):
                    end = start + _CHUNK_SIZE_CHARS
                    chunks.append(para[start:end])
                    start = end - _CHUNK_OVERLAP_CHARS
                buf = ""
    if buf:
        chunks.append(buf)

    return [Chunk(text=c, source=source, chunk_index=i) for i, c in enumerate(chunks)]


class SongNotesIndex:
    def __init__(self, notes_dir: str | Path = "song_notes", cache_dir: str | Path | None = None):
        self.notes_dir = Path(notes_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.notes_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(_MODEL_NAME, device="cpu")
        return self._model

    def build_or_load(self, force_rebuild: bool = False) -> int:
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(
            [*self.notes_dir.glob("*.md"), *self.notes_dir.glob("*.txt")]
        )

        manifest_path = self.cache_dir / "manifest.json"
        vectors_path = self.cache_dir / "embeddings.npy"
        manifest = {}
        if manifest_path.exists() and not force_rebuild:
            manifest = json.loads(manifest_path.read_text())

        all_chunks: list[Chunk] = []
        all_vecs: list[np.ndarray] = []
        cached_vecs = None
        if vectors_path.exists() and not force_rebuild:
            cached_vecs = np.load(vectors_path)

        new_chunks_needing_embed: list[Chunk] = []
        cache_offset_by_key: dict[str, tuple[int, int]] = manifest.get("_offsets", {})

        model = None  # loaded only if we actually need to embed something

        for f in files:
            content = f.read_text(encoding="utf-8", errors="ignore")
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            key = f"{f.name}:{digest}"

            if key in cache_offset_by_key and cached_vecs is not None:
                start, end = cache_offset_by_key[key]
                file_chunks = _chunk_text(content, f.name)
                if len(file_chunks) == (end - start):
                    all_chunks.extend(file_chunks)
                    all_vecs.append(cached_vecs[start:end])
                    continue

            file_chunks = _chunk_text(content, f.name)
            if not file_chunks:
                continue
            if model is None:
                model = self._get_model()
            vecs = model.encode([c.text for c in file_chunks], normalize_embeddings=True,
                                 show_progress_bar=False)
            all_chunks.extend(file_chunks)
            all_vecs.append(np.asarray(vecs, dtype=np.float32))

        if not all_chunks:
            self._chunks = []
            self._embeddings = np.zeros((0, 384), dtype=np.float32)
            return 0

        self._chunks = all_chunks
        self._embeddings = np.concatenate(all_vecs, axis=0)

        new_manifest = {"_offsets": {}}
        cursor = 0
        for f in files:
            content = f.read_text(encoding="utf-8", errors="ignore")
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            key = f"{f.name}:{digest}"
            n = len(_chunk_text(content, f.name))
            if n == 0:
                continue
            new_manifest["_offsets"][key] = [cursor, cursor + n]
            cursor += n

        np.save(vectors_path, self._embeddings)
        manifest_path.write_text(json.dumps(new_manifest))

        return len(self._chunks)

    def query(self, question: str, top_k: int = 4, min_score: float = 0.15) -> list[RetrievedChunk]:
        if self._embeddings is None or len(self._chunks) == 0:
            return []
        model = self._get_model()
        q_vec = model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self._embeddings @ q_vec  # cosine similarity (both sides normalized)
        top_idx = np.argsort(-scores)[:top_k]
        results = [
            RetrievedChunk(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_idx if scores[i] >= min_score
        ]
        return results

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)
