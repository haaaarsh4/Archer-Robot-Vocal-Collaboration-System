"""
neural_env/i18n/i18n.py

Pass-through stand-in for the original RVC-Project's i18n.i18n. It's only
used for translating a handful of printt() log-line strings in infer/rtrvc.py
(RVC.__init__/RVC.infer). This sidecar doesn't use rtrvc.RVC directly (see
rvc_pipeline.py -- it calls infer/rtrvc.py's get_synthesizer() helper and
otherwise talks to infer/hubert.py and infer/rmvpe.py directly), but keeping
this here means rtrvc.py itself needs zero edits to import cleanly.
"""


class I18nAuto:
    def __call__(self, key):
        return key
