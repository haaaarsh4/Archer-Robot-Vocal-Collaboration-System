"""What plays UNDER the robot, held to the one rule the panel promises.

The Neural engine does not add a part beside the singer: it re-sings the
recording's own line, off the same notes, a few milliseconds behind the
original. Playing the uploaded file underneath that put the recording's lead and
the robot's copy of it on top of each other, which is a comb filter across the
whole vocal range -- the words inside it are harder to read than either voice
alone, and the track's own instruments sit on top of both. That is the
"muddier than it used to be, and there is music I cannot place under it" report.

The page already fetches the separated accompaniment, and the server already
records at length why it exists, so these assertions are what notice if the
playback path starts ignoring it again. They read the page's source because the
decision is a line of browser JavaScript; there is no server-side equivalent to
test, and a regression here is silent -- the panel simply plays the wrong file.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _index_html() -> str:
    with open(os.path.join(ROOT, "frontend", "index.html"), encoding="utf-8") as fh:
        return fh.read()


def _function_body(source: str, name: str) -> str:
    """The text of one top-level function, up to the next top-level `}`."""
    start = source.index(f"function {name}(")
    end = source.index("\n}\n", start)
    return source[start:end]


def test_resinging_modes_are_the_ones_that_duplicate_the_line():
    body = _function_body(_index_html(), "_robotSingsTheLine")
    assert "demoVoiceEngine === 'neural'" in body
    # The robot is not singing the singer's line in these two, so the singer is
    # exactly what should be heard under it.
    assert "'call_and_response'" in body
    assert "'solo_humming'" in body


def test_the_upload_is_not_the_track_when_the_robot_sings_the_line():
    body = _function_body(_index_html(), "_playbackTrackBuffer")
    assert "if (!_robotSingsTheLine()) return demoAudioBuffer;" in body
    assert "if (!demoInstrumentsEnabled) return null;" in body
    assert "demoBandBuffer" in body, (
        "the separated accompaniment is what replaces the upload; without it the "
        "page has nothing to play that does not contain the original lead"
    )


def test_playback_analyses_the_upload_but_plays_the_track_it_chose():
    source = _index_html()
    # The render path no longer starts the upload as the audible track: it is the
    # analyser's source, and its gain is what says whether it is heard.
    assert "demoSource.buffer = _playbackTrackBuffer();" not in source
    assert "demoSource.buffer = demoAudioBuffer;" in source
    assert "(underBuffer === demoAudioBuffer) ? demoSourceVolume : 0" in source


def test_pausing_stops_the_separated_band_too():
    source = _index_html()
    # A band left running under a paused panel is a track with no robot on it.
    assert source.count("demoBandSource.stop()") >= 2


def test_the_download_is_the_mix_that_was_heard():
    body = _function_body(_index_html(), "_mixStemsForDownload")
    assert "singsTheLine" in body
    assert "bedGain.gain.value = demoSourceVolume;" in body
    assert "const trackBuffer = singsTheLine ? null : _playbackTrackBuffer();" in body
