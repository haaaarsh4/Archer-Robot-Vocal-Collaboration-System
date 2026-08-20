import asyncio
import io
import os
import sys
import threading
import time

import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Leave 1-2 cores free for the main app process, the mic capture thread,
# and this process's own FastAPI event loop -- letting torch grab every
# core for matmuls causes contention that shows up as slower, jittery
# inference, not faster. Tune this number to your actual core count.
import os as _os
torch.set_num_threads(max(1, (_os.cpu_count() or 4) - 2))

import numpy as np
import soundfile as sf
import uvicorn
import yaml
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import Response, JSONResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `infer.*`/`rvc_pipeline` import cleanly

from rvc_pipeline import RVCOfflineVoice, RMVPEPitchExtractor

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG_PATH = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "..", "config", "config.yaml")
)
CONFIG_PATH = os.path.abspath(os.environ.get("ARCHER_ROBOT_CONFIG", _DEFAULT_CONFIG_PATH))

app = FastAPI(title="Archer-Robot Neural Timbre Sidecar")

_voices: list = []           # loaded RVCOfflineVoice instances, index-aligned with _model_paths (or None on load failure)
_model_paths: list = []
_load_errors: list = []
_rmvpe: RMVPEPitchExtractor | None = None
_voice_params: list = []     # per-voice {index_rate, protect, transpose_semitones}, index-aligned with _voices
_warmup_ms: dict = {}        # voice_index -> warm-up inference time in ms, so /health can show real device speed
_inference_lock = threading.Lock()


def _load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"Could not find config.yaml at '{CONFIG_PATH}'. Set the "
            f"ARCHER_ROBOT_CONFIG environment variable to its exact path if "
            f"your layout differs from neural_env/../config/config.yaml."
        )
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _resolve_asset_path(p: str, project_root: str) -> str:
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(project_root, p))


def _load_voices():
    global _model_paths, _rmvpe

    cfg = _load_config()
    ncfg = cfg.get("synthesis", {}).get("neural", {})
    project_root = os.path.dirname(os.path.dirname(CONFIG_PATH))
    _model_paths = [_resolve_asset_path(p, project_root) for p in ncfg.get("model_paths", [])]
    index_paths = [_resolve_asset_path(p, project_root) for p in ncfg.get("index_paths", [])]
    device = ncfg.get("device", "cpu")
    is_half = bool(ncfg.get("is_half", False))
    index_rate = float(ncfg.get("index_rate", 0.66))
    protect = float(ncfg.get("protect", 0.33))
    transpose = float(ncfg.get("transpose_semitones", 0))
    rms_mix_rate = float(ncfg.get("rms_mix_rate", 0.25))
    pad_seconds = float(ncfg.get("pad_seconds", 1.0))

    # config.yaml can ask for cuda/rocm, but if this Python environment's
    # torch build doesn't actually have it (e.g. the CPU-only wheel from
    # https://download.pytorch.org/whl/cpu instead of a ROCm/CUDA build),
    # torch.device("cuda") constructs fine but the FIRST .to(device) call
    # deep inside model loading throws -- which previously meant the whole
    # sidecar process died on startup with a confusing stack trace instead
    # of a clear one-line explanation. Check up front and fall back loudly.
    if device not in ("cpu", "") and not torch.cuda.is_available():
        print(f"[WARN] config.yaml requests synthesis.neural.device={device!r}, but "
              f"torch.cuda.is_available() is False in this environment (torch {torch.__version__}). "
              "Falling back to CPU for this run -- conversions will be slow. This almost always "
              "means torch was installed from the CPU wheel index instead of a ROCm/CUDA build; "
              "see GPU_SETUP.md to fix it, then restart this sidecar.")
        device = "cpu"
        is_half = False  # fp16 on CPU is a pure loss, not just a no-op -- don't carry the request over

    if not _model_paths:
        print("[WARN] No synthesis.neural.model_paths configured in config.yaml — "
              "sidecar will start with zero voices loaded. /convert will 404 until "
              "you add at least one trained model path and restart.")
        return

    rmvpe_path = _resolve_asset_path(
        cfg.get("pitch", {}).get("rmvpe", {}).get("model_path", "assets/rmvpe/rmvpe.pt"),
        project_root,
    )
    if not os.path.exists(rmvpe_path):
        print(f"[FATAL] RMVPE checkpoint not found at '{rmvpe_path}' — the sidecar can't "
              "extract pitch without it. Check pitch.rmvpe.model_path in config.yaml.")
        sys.exit(1)
    print(f"Loading RMVPE from {rmvpe_path} ({device}, half={is_half})...")
    _rmvpe = RMVPEPitchExtractor(rmvpe_path, device=device, is_half=is_half)

    for i, model_path in enumerate(_model_paths):
        if not os.path.exists(model_path):
            print(f"[ERROR] Voice {i}: model file not found at '{model_path}' — skipping.")
            _voices.append(None)
            _load_errors.append(f"model not found: {model_path}")
            _voice_params.append(None)
            continue
        index_path = index_paths[i] if i < len(index_paths) else None
        try:
            voice = RVCOfflineVoice(model_path, index_path=index_path, device=device, is_half=is_half)
            _voices.append(voice)
            _load_errors.append(None)
            _voice_params.append({
                "index_rate": index_rate if voice.index is not None else 0.0,
                "protect": protect,
                "transpose_semitones": transpose,
                "rms_mix_rate": rms_mix_rate,
                "pad_seconds": pad_seconds,
            })
            has_index = "with index" if voice.index is not None else "no index"
            print(f"[OK] Voice {i} loaded: {model_path} ({has_index}, version={voice.version}, "
                  f"f0={bool(voice.if_f0)}, tgt_sr={voice.tgt_sr}) on {device} "
                  f"(half={voice.is_half}{' -- requested but not applied on CPU' if is_half and not voice.is_half else ''})")

            _warm_up_voice(i, voice, index_rate, protect, rms_mix_rate, pad_seconds)
        except Exception as e:
            print(f"[ERROR] Voice {i} failed to load ({model_path}): {e}")
            _voices.append(None)
            _load_errors.append(str(e))
            _voice_params.append(None)


def _warm_up_voice(index: int, voice: RVCOfflineVoice, index_rate: float, protect: float,
                    rms_mix_rate: float, pad_seconds: float):
    """
    'Loaded' (weights sitting in memory) and 'warm' (already run through
    once) are not the same thing -- the first real inference call pays
    for things that only happen once per process: PyTorch's lazy kernel/
    op dispatch setup, any first-call CUDA context/kernel compilation on
    GPU, thread-pool spin-up for BLAS/OpenMP on CPU, page faults as
    tensors actually get touched for the first time, etc. None of that
    shows up in "model loaded" -- it shows up as one abnormally slow
    request, and previously that request was whichever real chunk a real
    person's Play press happened to trigger. This pays that cost now,
    at startup, on a throwaway half-second of near-silent audio, so it's
    gone before anyone's around to notice it.
    """
    try:
        dummy = (np.random.randn(8000).astype(np.float32)) * 0.001  # 0.5s @16kHz, near-silent but non-zero (real energy through the pipeline, not a degenerate all-zero edge case)
        start = time.perf_counter()
        voice.convert(
            dummy, 16000, _rmvpe,
            f0_up_key=0.0, index_rate=index_rate, protect=protect,
            rms_mix_rate=rms_mix_rate, pad_seconds=min(pad_seconds, 0.3),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        _warmup_ms[index] = round(elapsed_ms, 1)
        print(f"[OK] Voice {index} warmed up in {elapsed_ms:.0f}ms — first real request won't pay this cost.")
    except Exception as e:
        # Warm-up is a pure optimization -- if it fails for any reason, the
        # voice is still loaded and usable, it just eats the cold-start
        # cost on the first real request instead. Not fatal.
        print(f"[WARN] Voice {index} warm-up failed (non-fatal, voice is still usable): {e}")


@app.on_event("startup")
def startup():
    print(f"Loading neural voices from config: {os.path.abspath(CONFIG_PATH)}")
    _load_voices()
    loaded = sum(1 for v in _voices if v is not None)
    print(f"Neural sidecar ready: {loaded}/{len(_voices)} voice(s) loaded.")


@app.get("/health")
def health():
    return {
        "voices_configured": len(_model_paths),
        "voices_loaded": sum(1 for v in _voices if v is not None),
        "errors": [{"voice_index": i, "error": err}
                   for i, err in enumerate(_load_errors) if err],
        # so the frontend (and you, in a browser tab) can see at a glance
        # whether this is actually running on GPU or silently fell back to
        # CPU -- e.g. requested device="cuda" but torch.cuda.is_available()
        # was False because of a CPU-only torch build.
        "voices": [
            {
                "index": i,
                "device": str(v.device) if v is not None else None,
                "half_precision": v.is_half if v is not None else None,
                "hubert_quantized_int8": v.quantized if v is not None else None,
                "warmup_ms": _warmup_ms.get(i),
            }
            for i, v in enumerate(_voices)
        ],
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }


class _InferenceBusy(Exception):
    pass


class _InferenceTimedOut(Exception):
    pass


def _run_conversion(voice, audio, in_sr, params, pad_seconds_override, wait):
    """
    Runs entirely inside a worker thread (see asyncio.to_thread call in
    convert() below) -- lock acquisition AND the actual inference both
    block, and previously ran directly on the async event loop, which
    meant the whole process (including /health, and accepting new
    connections for the NEXT chunk request) was frozen for the full
    duration of every single conversion. Moving this to a thread means
    the event loop stays responsive the entire time; the lock still
    correctly serializes actual model compute (only one inference can
    physically run at a time on this hardware), it just no longer also
    serializes everything else about the process along with it.
    """
    if wait:
        if not _inference_lock.acquire(timeout=1800):
            raise _InferenceTimedOut()
    elif not _inference_lock.acquire(blocking=False):
        raise _InferenceBusy()

    try:
        pad_seconds = params["pad_seconds"] if pad_seconds_override is None else pad_seconds_override
        # Clamp rather than trust blindly: too little padding risks
        # scipy's filtfilt padlen requirements on very short/silent audio,
        # and reflect-padding needs at least a small amount of real signal
        # to reflect. 0.15s floor is comfortably safe for that while still
        # being a big win over the 1.0s default on short Live chunks.
        pad_seconds = max(0.15, float(pad_seconds))

        start = time.perf_counter()
        converted, out_sr = voice.convert(
            audio, in_sr, _rmvpe,
            f0_up_key=params["transpose_semitones"],
            index_rate=params["index_rate"],
            protect=params["protect"],
            rms_mix_rate=params["rms_mix_rate"],
            pad_seconds=pad_seconds,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return converted, out_sr, elapsed_ms
    finally:
        _inference_lock.release()


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    voice_index: int = Form(0),
    sample_rate: int = Form(44100),
    wait: bool = Form(False),
    pad_seconds: float | None = Form(None),
):
    if not _voices:
        return JSONResponse({"error": "no voices loaded"}, status_code=503)

    voice = _voices[voice_index % len(_voices)]
    if voice is None:
        return JSONResponse(
            {"error": f"voice {voice_index} failed to load at startup, see /health"},
            status_code=503,
        )
    params = _voice_params[voice_index % len(_voices)]

    try:
        raw = await file.read()
        audio, in_sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        converted, out_sr, elapsed_ms = await asyncio.to_thread(
            _run_conversion, voice, audio, in_sr, params, pad_seconds, wait
        )
        print(f"[convert] voice={voice_index} {len(audio)/in_sr:.2f}s in -> "
              f"{elapsed_ms:.0f}ms (pad={pad_seconds if pad_seconds is not None else params['pad_seconds']}s)")

        out_buf = io.BytesIO()
        sf.write(out_buf, converted, out_sr, format="WAV")
        out_buf.seek(0)

        return Response(
            content=out_buf.read(),
            media_type="audio/wav",
            headers={"X-Inference-Ms": f"{elapsed_ms:.1f}", "X-Sample-Rate": str(out_sr)},
        )
    except _InferenceTimedOut:
        return JSONResponse({"error": "inference lock timed out after 1800s"}, status_code=503)
    except _InferenceBusy:
        return JSONResponse({"error": "inference busy with another request"}, status_code=503)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"conversion failed: {e}"}, status_code=500)


if __name__ == "__main__":
    port = int(os.environ.get("RVC_SIDECAR_PORT", 8801))
    print(f"\n  Neural sidecar starting on http://127.0.0.1:{port}")
    print(f"  Config: {os.path.abspath(CONFIG_PATH)}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
