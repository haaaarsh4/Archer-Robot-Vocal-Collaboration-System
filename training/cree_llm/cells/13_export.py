# Export: adapter -> merged weights -> GGUF -> Ollama.
#
# llama.cpp cannot apply a LoRA that adds vocabulary, and this model adds Cree
# tokens. The adapter is therefore merged into a full-precision copy of the base
# model first, and the merged directory is what gets converted. Merging needs
# roughly twice the model's fp16 size in RAM or VRAM; the cell says which device
# it chose and stops with a clear message if neither has room.
from peft import PeftModel

# The merged fp16 model is an intermediate, not an artifact: it is 15.2 GB for
# the 7B and can always be rebuilt from the adapter. On Colab it goes to the
# local disk, because a free Drive is 15 GB in total and filling it during the
# longest run in the notebook would fail the export after hours of training.
# Only the GGUF, the Modelfile and the archive are written to Drive.
EXPORT = Path("/content/cree_llm_export") if IN_COLAB else OUT / "export"
EXPORT.mkdir(parents=True, exist_ok=True)

# The file this cell actually shipped. Stays None on a scoring run or a smoke
# run, where nothing is quantised and the closing instructions must still print
# something accurate instead of raising.
final_gguf: Path | None = None


def _cmake() -> str | None:
    """A cmake to build the quantiser with, installed without root if needed.

    The export is the only step in this pipeline that needs a toolchain, and a
    shared cluster login node often has a compiler but no cmake. `pip install
    cmake` puts one in the run's own virtualenv, which is the one place this
    script is allowed to write. Nothing here touches the system Python.
    """
    found = shutil.which("cmake")
    if found:
        return found

    def _from_the_package() -> str | None:
        try:
            import cmake as _cmake_pkg
            candidate = Path(_cmake_pkg.CMAKE_BIN_DIR) / "cmake"
            return str(candidate) if candidate.exists() else None
        except Exception:
            return None

    packaged = _from_the_package()
    if packaged:
        return packaged
    log("cmake: not on PATH, installing it into this virtualenv")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cmake"],
                       check=True, timeout=900)
    except Exception as exc:
        log(f"cmake: installation failed ({exc})")
        return None
    return shutil.which("cmake") or _from_the_package()


def _quantiser(llama: Path) -> Path | None:
    """The llama-quantize binary, built once, or None if this host cannot build it.

    Returning None is not a failure. The Python converter can write a Q8_0 file
    on its own, and that is a model Ollama runs exactly the same way. A finished
    model that is 4 GB larger beats no model at all.
    """
    for candidate in (llama / "build" / "bin" / "llama-quantize",
                      llama / "build" / "llama-quantize"):
        if candidate.exists():
            return candidate
    on_path = shutil.which("llama-quantize")
    if on_path:
        return Path(on_path)

    cmake = _cmake()
    if cmake is None:
        return None
    log("llama.cpp: building llama-quantize (once per work folder)")
    try:
        subprocess.run([cmake, "-B", "build", "-DGGML_NATIVE=OFF", "-DLLAMA_CURL=OFF"],
                       cwd=llama, check=True)
        subprocess.run([cmake, "--build", "build", "--config", "Release",
                        "--target", "llama-quantize",
                        "--parallel", str(max(1, os.cpu_count() or 4))],
                       cwd=llama, check=True)
    except subprocess.CalledProcessError as exc:
        log(f"llama.cpp: build failed ({exc}); the export will write Q8_0 instead")
        return None
    for candidate in (llama / "build" / "bin" / "llama-quantize",
                      llama / "build" / "llama-quantize"):
        if candidate.exists():
            return candidate
    return None

if os.environ.get("CREE_ADAPTER_STAGE", "").strip():
    # A scoring run must not overwrite the shipped weights with an adapter it
    # was only asked to measure.
    log("export skipped: scoring run (CREE_ADAPTER_STAGE is set)")
elif SMOKE:
    log("SMOKE: skipping merge and GGUF; packaging the adapter only")
    zip_path = OUT / "archer-smoke.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for path in (CKPT / "dpo").rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(ROOT))
        zf.write(OUT / "creebench_report.json", "creebench_report.json")
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
else:
    # Free the training model before asking for a second full copy.
    try:
        del model
    except NameError:
        pass
    gc.collect()

    # This cell is often the first one run after a reconnect, so nothing it
    # needs is assumed to be in the kernel except the files on Drive.
    from transformers import AutoTokenizer as _AutoTokenizer

    tok = _AutoTokenizer.from_pretrained(PROC / "tokenizer")

    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 if torch.cuda.is_available() else 0
    import psutil

    ram_gb = psutil.virtual_memory().total / 1e9
    # Merging holds the fp16 base and the fp16 merged copy at the same time,
    # plus the adapter and transient buffers: roughly 4.4 GB per billion
    # parameters. A 24 GB threshold that ignores model size is how a 14B or 32B
    # export ends in an OOM after hours of training.
    params_b = {"0.5b": 0.5, "1.5b": 1.5, "3b": 3, "7b": 7, "14b": 14, "32b": 32}.get(MODEL_TAG, 7)
    need_gb = 4.4 * params_b
    force = os.environ.get("CREE_MERGE_DEVICE", "")
    merge_on_gpu = False if force == "cpu" else (gpu_gb >= need_gb)
    if force == "cuda" and gpu_gb < need_gb:
        raise RuntimeError(
            f"Merging {MODEL_STEM} needs about {need_gb:.0f} GB of GPU memory; this GPU has {gpu_gb:.0f} GB. "
            "Unset CREE_MERGE_DEVICE to let the cell choose CPU, or use a larger GPU."
        )
    target = "cuda" if merge_on_gpu else "cpu"
    if not merge_on_gpu and ram_gb < need_gb:
        raise RuntimeError(
            f"Merging {MODEL_STEM} needs about {need_gb:.0f} GB of CPU RAM (fp16 base plus fp16 merged copy); "
            f"this runtime has {ram_gb:.0f} GB and {gpu_gb:.0f} GB of GPU memory. "
            "Use Runtime -> Change runtime type -> High-RAM, or a larger GPU for the merge. "
            "The trained adapter and report are already saved; nothing is lost."
        )

    # Every artifact is checked against the adapter's own timestamp, not just
    # against existence. Retraining one stage and re-running this cell used to
    # keep the previous GGUF, because a file was there and large: the export
    # would report success while shipping the old weights.
    def _newest(path: Path) -> float:
        return max((p.stat().st_mtime for p in path.rglob("*") if p.is_file()), default=0.0)

    adapter_mtime = _newest(CKPT / "dpo")
    merged_dir = EXPORT / "merged"
    merged_shards = sorted(merged_dir.glob("*.safetensors"))
    merged_ok = bool(merged_shards) and (merged_dir / "config.json").exists() and not any(
        safetensors_problem(p) for p in merged_shards
    ) and _newest(merged_dir) >= adapter_mtime
    if merged_shards and not merged_ok:
        log("merged model is older than the adapter it should contain; merging again")
    if merged_ok:
        log(f"merged model already complete at {merged_dir}, skipping the merge")
    else:
        log(f"merging on {target} (GPU {gpu_gb:.0f} GB, RAM {ram_gb:.0f} GB)")

        base = AutoModelForCausalLM.from_pretrained(
            ensure_base_weights(),
            **_dtype_kwargs(torch.float16),
            low_cpu_mem_usage=True,
            device_map={"": 0} if merge_on_gpu else {"": "cpu"},
        )
        # The graft is restored here rather than resized blank, otherwise the
        # merge would write random rows for every Cree token the adapter was
        # trained on.
        graft_rows(base, tok)
        merged = PeftModel.from_pretrained(base, str(CKPT / "dpo"))
        merged = merged.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="4GB")
        tok.save_pretrained(merged_dir)
        del merged, base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"merged model saved to {merged_dir}")

    q4 = EXPORT / f"{MODEL_STEM}.Q4_K_M.gguf"
    q8 = EXPORT / f"{MODEL_STEM}.Q8_0.gguf"

    def _fresh(path: Path) -> bool:
        return (path.exists() and path.stat().st_size > 1_000_000_000
                and path.stat().st_mtime >= adapter_mtime)

    for candidate in (q4, q8):
        if candidate.exists() and not _fresh(candidate):
            log(f"{candidate.name} predates the adapter; removing it")
            candidate.unlink()

    final_gguf = q4 if _fresh(q4) else (q8 if _fresh(q8) else None)
    if final_gguf is not None:
        log(f"{final_gguf.name} already built ({final_gguf.stat().st_size / 1e9:.2f} GB), skipping conversion")
    else:
        # llama.cpp: convert, then quantise
        LLAMA = RAW / "llama.cpp"
        if not (LLAMA / "convert_hf_to_gguf.py").exists():
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/ggml-org/llama.cpp.git", str(LLAMA)], check=True)
        # Only the one package the converter needs and the runtime does not
        # already have. Installing llama.cpp's full requirements here can upgrade
        # numpy or transformers under the not-yet-saved report, so it is
        # scoped.
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf"], check=True)

        # The f16 is an intermediate, and a run that died after converting
        # leaves one behind. Quantising a file that predates the adapter would
        # write the previous weights under the new name, so it follows the same
        # rule as the merged copy: it must be newer than what it came from.
        f16 = EXPORT / f"{MODEL_STEM}-f16.gguf"
        if f16.exists() and f16.stat().st_mtime < adapter_mtime:
            log(f"{f16.name} predates the adapter; converting it again")
            f16.unlink()
        if not f16.exists():
            subprocess.run([
                sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(merged_dir),
                "--outfile", str(f16), "--outtype", "f16",
            ], check=True)

        quant = _quantiser(LLAMA)
        if quant is None:
            log("llama.cpp cannot be built here; writing Q8_0 with the Python converter instead of Q4_K_M")
            subprocess.run([
                sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(merged_dir),
                "--outfile", str(q8), "--outtype", "q8_0",
            ], check=True)
            final_gguf = q8
        else:
            subprocess.run([str(quant), str(f16), str(q4), "Q4_K_M"], check=True)
            final_gguf = q4
        f16.unlink(missing_ok=True)
        if not final_gguf.exists() or final_gguf.stat().st_size < 1_000_000_000:
            raise RuntimeError(f"quantisation did not produce a usable GGUF at {final_gguf}")

    modelfile = EXPORT / "Modelfile"
    # The system prompt lives in the app (chat/archer_prompt.py) and is sent on
    # every request. Baking it into the Modelfile as well would put it in the
    # context twice, once from .System and once as the system message.
    modelfile.write_text(
        f"FROM ./{final_gguf.name}\n"
        "PARAMETER temperature 0.3\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER repeat_penalty 1.08\n"
        "PARAMETER num_ctx 4096\n",
        encoding="utf-8",
    )

    zip_path = OUT / f"{MODEL_STEM}.zip"
    room_gb = shutil.disk_usage(str(OUT)).free / 1e9
    if zip_path.exists() and zip_path.stat().st_mtime >= final_gguf.stat().st_mtime:
        log(f"{zip_path.name} already packaged, skipping the archive")
    elif room_gb < final_gguf.stat().st_size / 1e9 * 1.1:
        # The archive is a copy of the GGUF plus the report. Copying it with no
        # room left leaves a truncated zip and no way to finish the export.
        log(f"skipping the archive: {room_gb:.1f} GB free on Drive, {final_gguf.stat().st_size / 1e9:.1f} GB needed")
        log(f"the GGUF itself is complete at {final_gguf}; move it off Drive to make room for the zip")
    else:
        digest = hashlib.sha256()
        with final_gguf.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)

        manifest = {
            "gguf": final_gguf.name,
            "quantisation": final_gguf.name.split(".")[-2],
            "size_gb": round(final_gguf.stat().st_size / 1e9, 2),
            "sha256": digest.hexdigest(),
            "base_model": CFG.base_model,
            "profile": ACTIVE_PROFILE,
            "created": time.strftime("%Y-%m-%d %H:%M"),
        }
        jdump(EXPORT / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2))

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            zf.write(final_gguf, final_gguf.name)
            zf.write(modelfile, "Modelfile")
            zf.write(EXPORT / "manifest.json", "manifest.json")
            zf.write(OUT / "creebench_report.json", "creebench_report.json")
            zf.write(OUT / "report_card.md", "report_card.md")
            for png in PLOTS.glob("*.png"):
                zf.write(png, f"plots/{png.name}")
        print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")

        if IN_COLAB:
            try:
                from google.colab import files

                files.download(str(final_gguf))
                files.download(str(zip_path))
            except Exception as exc:
                print(f"[note] automatic download failed ({exc}); use the Files pane")

print("\nInstall the model on the machine that runs the app")
_shipped = final_gguf.name if final_gguf is not None else f"{MODEL_STEM}.Q4_K_M.gguf"
print(textwrap.dedent(f"""
    mkdir -p assets/archer_llm
    cp {_shipped} assets/archer_llm/
    cp Modelfile assets/archer_llm/
    cd assets/archer_llm && ollama create archer:{MODEL_TAG} -f Modelfile

    # config.yaml
    chat:
      model: archer:{MODEL_TAG}
"""))
