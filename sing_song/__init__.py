"""In-repo SingSong: musical accompaniment generation from singing.

The package follows Donahue et al., "SingSong: Generating musical
accompaniments from singing" (2023):

* ``tokens``    exact vocabulary layout and per-position constraints
* ``model``     T5-style semantic+coarse model and the fine decoder
* ``sampling``  constrained temperature sampling with KV caching
* ``codec``     SoundStream-shaped 16 kHz RVQ codec
* ``semantic``  w2v-BERT/HuBERT features + k-means tokenizer
* ``data``      separation, filtering, noising, token caching
* ``training``  stage 1+2 and stage 3 training loops
* ``inference`` sliding-window generation and vocal/instrumental mixing
"""
from .config import (
    CodecConfig,
    DataConfig,
    FineConfig,
    InferenceConfig,
    PathsConfig,
    SemanticConfig,
    SingSongConfig,
    Stage12Config,
    load_sing_song_config,
)
from .inference import SingSongEngine, SingSongNotReady

__all__ = [
    "CodecConfig",
    "DataConfig",
    "FineConfig",
    "InferenceConfig",
    "PathsConfig",
    "SemanticConfig",
    "SingSongConfig",
    "Stage12Config",
    "SingSongEngine",
    "SingSongNotReady",
    "load_sing_song_config",
]
__version__ = "1.0.0"
