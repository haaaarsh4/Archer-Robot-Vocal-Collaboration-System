"""MySong-style automatic accompaniment for a vocal melody.

The public entry points are:

``mysong.pipeline.create_song``
    audio in, chords + arrangement + mix out.
``mysong.pipeline.analyze_song``
    audio in, key + chords out (no rendering).
``mysong.harmonize.Harmonizer``
    a melody in, a chord sequence out.
``mysong.accompaniment.arrange``
    a chord sequence in, piano/bass/drums and a MIDI file out.

The model is trained from a lead-sheet corpus with
``python -m mysong.cli train``; the checked-in model in
``data/mysong/model/mysong.npz`` is what the app loads by default.
"""

from mysong.harmonize import Harmonization, Harmonizer, Style, harmonize
from mysong.model import MySongModel
from mysong.pipeline import SongProject, analyze_song, create_song, render_song

__version__ = "1.0.0"

__all__ = [
    "Harmonization",
    "Harmonizer",
    "Style",
    "harmonize",
    "MySongModel",
    "SongProject",
    "analyze_song",
    "create_song",
    "render_song",
]
