# ─────────────────────────────────────────────────────────────────────────────
# Export: adapter -> merged weights -> GGUF -> Ollama.
#
# llama.cpp cannot apply a LoRA that adds vocabulary, and this model adds Cree
# tokens. The adapter is therefore merged into a full-precision copy of the base
# model first, and the merged directory is what gets converted. Merging needs
# roughly twice the model's fp16 size in RAM or VRAM; the cell says which device
# it chose and stops with a clear message if neither has room.
# ─────────────────────────────────────────────────────────────────────────────
from peft import PeftModel

# The merged fp16 model is an intermediate, not an artifact: it is 15.2 GB for
# the 7B and can always be rebuilt from the adapter. On Colab it goes to the
# local disk, because a free Drive is 15 GB in total and filling it during the
# longest run in the notebook would fail the export after hours of training.
# Only the GGUF, the Modelfile and the archive are written to Drive.
EXPORT = Path("/content/cree_llm_export") if IN_COLAB else OUT / "export"
EXPORT.mkdir(parents=True, exist_ok=True)

if SMOKE:
    log("SMOKE: skipping merge and GGUF; packaging the adapter only")
    zip_path = OUT / "archer-cree-smoke.zip"
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

    merged_dir = EXPORT / "merged"
    merged_shards = sorted(merged_dir.glob("*.safetensors"))
    merged_ok = bool(merged_shards) and (merged_dir / "config.json").exists() and not any(
        safetensors_problem(p) for p in merged_shards
    )
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
    if q4.exists() and q4.stat().st_size > 1_000_000_000:
        log(f"{q4.name} already built ({q4.stat().st_size / 1e9:.2f} GB), skipping conversion")
    else:
        # ── llama.cpp: convert, then quantise ────────────────────────────────
        LLAMA = RAW / "llama.cpp"
        if not (LLAMA / "convert_hf_to_gguf.py").exists():
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/ggml-org/llama.cpp.git", str(LLAMA)], check=True)
        # Only the one package the converter needs and the runtime does not
        # already have. Installing llama.cpp's full requirements here can upgrade
        # numpy or transformers under the not-yet-saved report, so it is
        # deliberately scoped.
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf"], check=True)

        f16 = EXPORT / f"{MODEL_STEM}-f16.gguf"
        if not f16.exists():
            subprocess.run([
                sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(merged_dir),
                "--outfile", str(f16), "--outtype", "f16",
            ], check=True)

        quant = LLAMA / "build" / "bin" / "llama-quantize"
        if not quant.exists():
            if shutil.which("cmake") is None:
                raise RuntimeError("cmake is not installed; install it, or use the merged model with "
                                   "transformers instead of GGUF")
            subprocess.run(["cmake", "-B", "build", "-DGGML_NATIVE=OFF"], cwd=LLAMA, check=True)
            subprocess.run(["cmake", "--build", "build", "--config", "Release", "-j", str(os.cpu_count() or 4)],
                           cwd=LLAMA, check=True)
        subprocess.run([str(quant), str(f16), str(q4), "Q4_K_M"], check=True)
        f16.unlink(missing_ok=True)
        if not q4.exists() or q4.stat().st_size < 1_000_000_000:
            raise RuntimeError(f"quantisation did not produce a usable GGUF at {q4}")

    modelfile = EXPORT / "Modelfile"
    modelfile.write_text(
        f"FROM ./{q4.name}\n"
        "PARAMETER temperature 0.2\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER repeat_penalty 1.08\n"
        "PARAMETER num_ctx 4096\n"
        f'SYSTEM """{SYSTEM}"""\n',
        encoding="utf-8",
    )

    zip_path = OUT / f"{MODEL_STEM}.zip"
    room_gb = shutil.disk_usage(str(OUT)).free / 1e9
    if zip_path.exists() and zip_path.stat().st_mtime >= q4.stat().st_mtime:
        log(f"{zip_path.name} already packaged, skipping the archive")
    elif room_gb < q4.stat().st_size / 1e9 * 1.1:
        # The archive is a copy of the GGUF plus the report. Copying it with no
        # room left leaves a truncated zip and no way to finish the export.
        log(f"skipping the archive: {room_gb:.1f} GB free on Drive, {q4.stat().st_size / 1e9:.1f} GB needed")
        log(f"the GGUF itself is complete at {q4}; move it off Drive to make room for the zip")
    else:
        digest = hashlib.sha256()
        with q4.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)

        manifest = {
            "gguf": q4.name,
            "size_gb": round(q4.stat().st_size / 1e9, 2),
            "sha256": digest.hexdigest(),
            "base_model": CFG.base_model,
            "profile": ACTIVE_PROFILE,
            "created": time.strftime("%Y-%m-%d %H:%M"),
        }
        jdump(EXPORT / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2))

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            zf.write(q4, q4.name)
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

                files.download(str(q4))
                files.download(str(zip_path))
            except Exception as exc:
                print(f"[note] automatic download failed ({exc}); use the Files pane")

print("\n── install on the machine that runs the app ──")
print(textwrap.dedent(f"""
    mkdir -p assets/cree_llm
    cp {MODEL_STEM}.Q4_K_M.gguf assets/cree_llm/
    cp Modelfile assets/cree_llm/
    cd assets/cree_llm && ollama create archer-cree:{MODEL_TAG} -f Modelfile

    # config.yaml
    chat:
      cree_model: archer-cree:{MODEL_TAG}
"""))
