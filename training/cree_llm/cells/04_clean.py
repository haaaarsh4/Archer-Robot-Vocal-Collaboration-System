# ─────────────────────────────────────────────────────────────────────────────
# Cleaning, deduplication and splitting.
#
# Three failures destroy a low-resource model, and each is handled explicitly:
#   1. English hiding inside "Cree" text  -> language check per sentence.
#   2. The same sentence in train and test -> exact + near-duplicate removal,
#      then a document-disjoint split, then a second leakage check.
#   3. Bag-of-words files pretending to be prose -> rejected by shape.
# ─────────────────────────────────────────────────────────────────────────────

DOCS = read_jsonl(RAW / "docs.jsonl")
PAIRS = read_jsonl(RAW / "pairs.jsonl")
LEXICON = jload(RAW / "lexicon.json")

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<[^>]+>")
_WIKI = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]|\[[0-9]+\]|\{\{[^}]*\}\}")
_SPACE = re.compile(r"[ \t\u00a0]+")
_BOW_LINE = re.compile(r"^\s*[^\W\d_][^\s]*\s+\d+\s*$")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = _HTML.sub(" ", text)
    text = _WIKI.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if looks_like_syllabics(text):
        text = to_sro(text)
    text = text.replace("᙮", ".")
    # Paragraph breaks are sentence breaks even without punctuation.
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    for piece in pieces:
        piece = re.sub(r"^[\s\"'\-–—•*]+|[\s\"'\-–—•*]+$", "", piece)
        if not piece:
            continue
        words = piece.split()
        if 4 <= len(words) <= 80:
            out.append(piece)
    return out


def bow_share(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return 0.0
    return sum(bool(_BOW_LINE.match(ln)) for ln in lines) / len(lines)


class NearDup:
    """MinHash + banded LSH, 32 permutations, 8 bands of 4 rows.

    A sentence is a duplicate when its estimated Jaccard similarity to an
    indexed sentence is at least ``threshold``. This catches the repeated
    refrains, quoted verses and copied dictionary examples that exact hashing
    misses, without pulling in a dependency.
    """

    _M = (1 << 61) - 1

    def __init__(self, threshold: float = 0.82, ngram: int = 5, perms: int = 32, bands: int = 8):
        self.threshold = threshold
        self.ngram = ngram
        self.perms = perms
        self.bands = bands
        self.rows = perms // bands
        rng = random.Random(4242)
        self.a = [rng.randrange(1, self._M - 1) for _ in range(perms)]
        self.b = [rng.randrange(0, self._M - 1) for _ in range(perms)]
        self.buckets: list[dict[tuple, list[int]]] = [{} for _ in range(bands)]
        self.sigs: list[tuple[int, ...]] = []
        self.texts: list[str] = []

    def _signature(self, text: str) -> tuple[int, ...]:
        words = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
        if len(words) < self.ngram:
            grams = [" ".join(words)] if words else [text.lower()]
        else:
            grams = [" ".join(words[i:i + self.ngram]) for i in range(len(words) - self.ngram + 1)]
        hashes = {int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big") for g in grams}
        return tuple(min((h * a + b) % self._M for h in hashes) for a, b in zip(self.a, self.b))

    def find(self, text: str) -> int | None:
        sig = self._signature(text)
        for band in range(self.bands):
            key = tuple(sig[band * self.rows:(band + 1) * self.rows])
            for idx in self.buckets[band].get(key, ()):  # candidates only
                other = self.sigs[idx]
                same = sum(x == y for x, y in zip(sig, other))
                if same / self.perms >= self.threshold:
                    return idx
        return None

    def add(self, text: str) -> bool:
        if self.find(text) is not None:
            return False
        sig = self._signature(text)
        idx = len(self.sigs)
        self.sigs.append(sig)
        self.texts.append(text)
        for band in range(self.bands):
            key = tuple(sig[band * self.rows:(band + 1) * self.rows])
            self.buckets[band].setdefault(key, []).append(idx)
        return True


def norm_key(text: str) -> str:
    return re.sub(r"[^\wâêîô]+", "", normalize_sro(text))


# ── documents -> sentences ───────────────────────────────────────────────────
# The splits are the thing every later cell reads, so they are built once and
# then read back. Rebuilding them is not just slow: the near-duplicate filter is
# randomised by insertion order, so a second build could produce slightly
# different splits under the same names and quietly invalidate a benchmark that
# was already computed.
_CLEAN_KEY = fingerprint(
    docs=(RAW / "docs.jsonl").stat().st_size,
    pairs=(RAW / "pairs.jsonl").stat().st_size,
    version="clean-v1",
)
_SPLIT_FILES = [PROC / f"lm_{s}.jsonl" for s in ("train", "dev", "test")]
_SPLIT_FILES += [PROC / "parallel_train.jsonl", PROC / "parallel_test.jsonl"]

if all(f.exists() for f in _SPLIT_FILES) and stage_done("clean", _CLEAN_KEY, requires=PROC / "stats.json"):
    stats = jload(PROC / "stats.json")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
else:
    # ── documents -> sentences ───────────────────────────────────────────────────
    NOTES = [d for d in DOCS if d["kind"] == "song-notes"]
    DOCS = [d for d in DOCS if d["kind"] != "song-notes"]

    sentences: list[dict] = []
    seen_exact: set[str] = set()
    near = NearDup()
    rejected = {"english": 0, "short": 0, "bow": 0, "duplicate": 0, "near": 0}

    for doc in DOCS:
        if bow_share(doc["text"]) > 0.35:
            rejected["bow"] += 1
            continue
        historical = doc.get("dialect") in {"historical", "watkins1865"}
        for sent in split_sentences(doc["text"]):
            guess, ratio = lang_guess(sent), cree_word_ratio(sent)
            # Nineteenth-century texts were printed without the long-vowel marks
            # the modern language detector expects, so a good historical sentence
            # can come back as "mixed". It still has to clear a higher Cree-letter
            # ratio, which keeps genuinely bilingual pages out.
            if guess != "crk" and not (historical and guess == "mixed" and ratio >= 0.9):
                rejected["english"] += 1
                continue
            if ratio < (0.75 if historical else 0.8):
                rejected["english"] += 1
                continue
            key = norm_key(sent)
            if len(key) < 12:
                rejected["short"] += 1
                continue
            if key in seen_exact:
                rejected["duplicate"] += 1
                continue
            seen_exact.add(key)
            if not near.add(sent):
                rejected["near"] += 1
                continue
            sentences.append({
                "text": sent,
                "source": doc["source"],
                "doc_id": doc["doc_id"],
                "licence": doc["licence"],
                "dialect": doc["dialect"],
                "kind": doc["kind"],
            })

    log(f"sentences kept {len(sentences)}  rejected {rejected}")

    # ── pairs: deduplicate, then split by document ───────────────────────────────
    _PRIORITY = {"edtekla": 0, "figurative": 1, "project": 2, "watkins1865": 3}

    pairs_by_cree: dict[str, dict] = {}
    for pair in PAIRS:
        cree, english = clean_text(pair["cree"]), clean_text(pair["english"])
        if len(cree.split()) < 2 or len(english.split()) < 2:
            continue
        if cree_word_ratio(cree) < 0.55 or lang_guess(english) != "eng":
            continue
        key = norm_key(cree)
        if not key:
            continue
        rank = _PRIORITY.get(pair["source"].split("/")[0], 9)
        if key not in pairs_by_cree or rank < _PRIORITY.get(pairs_by_cree[key]["source"].split("/")[0], 9):
            pairs_by_cree[key] = {**pair, "cree": cree, "english": english}

    pairs = list(pairs_by_cree.values())
    log(f"aligned pairs kept {len(pairs)} of {len(PAIRS)}")

    # A near-duplicate check inside the pair set would delete legitimate repeated
    # short sentences ("tânisi" appears in many lessons), so pairs are deduped on
    # the exact Cree string only and leakage is handled at the group level below.


    def group_of(row: dict) -> str:
        src = row["source"].split("/")[0]
        doc = str(row.get("doc_id", ""))
        if src == "figurative":
            return f"{src}|{doc.split('/')[0]}"
        if src == "project":
            # All project pairs share one bucket; split by stable sentence hash.
            return f"{src}|{int(sha1(norm_key(row.get('cree', row.get('text', ''))))[:8], 16) % 20}"
        return f"{src}|{doc.split('#')[0]}"


    def assign(group: str) -> str:
        h = int(sha1(group)[:8], 16) % 100
        return "test" if h < 5 else ("dev" if h < 12 else "train")


    for row in sentences:
        row["split"] = assign(group_of(row))
    for row in pairs:
        row["split"] = assign(group_of(row))

    # ── leakage guard: nothing in dev/test may be a near-duplicate of train ──────
    train_guard = NearDup(threshold=0.8)
    for row in sentences:
        if row["split"] == "train":
            train_guard.add(row["text"])

    leaked = 0
    for row in sentences:
        if row["split"] != "train" and train_guard.find(row["text"]) is not None:
            row["split"] = "train"
            leaked += 1
    for row in pairs:
        if row["split"] == "train":
            continue
        if train_guard.find(row["text"] if "text" in row else row["cree"]) is not None:
            row["split"] = "train"
            leaked += 1
    log(f"moved {leaked} near-duplicates out of dev/test")

    # ── a tiny dev/test split must still exist ───────────────────────────────────
    def ensure_split(rows: list[dict], minimum: int) -> None:
        """Top up the test split by moving whole groups, never single rows.

        Moving one sentence out of a document breaks the document-disjoint
        property the split is supposed to have, and a benchmark scored on a
        sibling of a training sentence is not held out in any meaningful sense.
        """
        if sum(r["split"] == "test" for r in rows) >= minimum:
            return
        groups: dict[str, list[dict]] = {}
        for row in rows:
            if row["split"] == "train":
                groups.setdefault(group_of(row), []).append(row)
        for group in sorted(groups, key=sha1):
            for row in groups[group]:
                row["split"] = "test"
            if sum(r["split"] == "test" for r in rows) >= minimum:
                break


    ensure_split(sentences, 40 if SMOKE else 300)
    ensure_split(pairs, 30 if SMOKE else 200)

    lm = {s: [r for r in sentences if r["split"] == s] for s in ("train", "dev", "test")}
    tr = {s: [r for r in pairs if r["split"] == s] for s in ("train", "dev", "test")}

    # The evaluation set for translation is modern Cree only; Watkins 1865 is a
    # separate historical capability and would distort the modern score.
    tr_test_modern = [r for r in tr["test"] if r["source"] != "watkins1865"]
    if len(tr_test_modern) < 50 and not SMOKE:
        tr_test_modern = tr["test"]

    write_jsonl(PROC / "lm_train.jsonl", lm["train"])
    write_jsonl(PROC / "lm_dev.jsonl", lm["dev"])
    write_jsonl(PROC / "lm_test.jsonl", lm["test"])
    write_jsonl(PROC / "parallel_train.jsonl", tr["train"])
    write_jsonl(PROC / "parallel_dev.jsonl", tr["dev"])
    write_jsonl(PROC / "parallel_test.jsonl", tr_test_modern)
    write_jsonl(PROC / "parallel_test_historical.jsonl", [r for r in tr["test"] if r["source"] == "watkins1865"])
    write_jsonl(PROC / "notes.jsonl", NOTES)
    jdump(PROC / "lexicon.json", LEXICON)

    stats = {
        "documents": len(DOCS),
        "sentences": {k: len(v) for k, v in lm.items()},
        "sentence_words": {k: sum(len(r["text"].split()) for r in v) for k, v in lm.items()},
        "pairs": {k: len(v) for k, v in tr.items()},
        "pair_test_modern": len(tr_test_modern),
        "lexicon_forms": len(LEXICON),
        "rejected": rejected,
        "leakage_moved": leaked,
        "licences": sorted({r["licence"] for r in sentences})[:20],
    }
    jdump(PROC / "stats.json", stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # ── integrity assertions: the rest of the notebook relies on these ───────────
    assert lm["train"] and lm["test"], "empty LM split"
    assert tr["train"] and tr["test"], "empty pair split"
    _train_keys = {norm_key(r["text"]) for r in lm["train"]}
    assert not (_train_keys & {norm_key(r["text"]) for r in lm["test"]}), "exact LM leakage"
    assert all(len(r["cree"].split()) >= 2 for r in tr["train"]), "bad pair"
    assert all(lang_guess(r["text"]) in {"crk", "mixed"} and cree_word_ratio(r["text"]) >= 0.7
               for r in lm["test"]), "English in LM test"
    print("\nintegrity checks passed: splits are disjoint and language-checked")

    # ── composition figure ───────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    counts = {k: len(v) for k, v in lm.items()}
    ax[0].bar(counts.keys(), counts.values(), color=["#2f5d50", "#c98b3a", "#8c3b2e"])
    ax[0].set_title("Cree sentences by split")
    ax[0].set_ylabel("sentences")
    for i, v in enumerate(counts.values()):
        ax[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    by_src: dict[str, int] = {}
    for row in lm["train"]:
        by_src[row["source"].split("/")[0]] = by_src.get(row["source"].split("/")[0], 0) + 1
    ax[1].barh(list(by_src)[::-1], list(by_src.values())[::-1], color="#2f5d50")
    ax[1].set_title("Training sentences by source")
    ax[1].set_xlabel("sentences")
    fig.tight_layout()
    fig.savefig(PLOTS / "corpus_composition.png", dpi=160)
    plt.show()
    print(f"saved {PLOTS / 'corpus_composition.png'}")

    mark_stage("clean", _CLEAN_KEY, sentences=sum(len(v) for v in lm.values()),
               pairs=sum(len(v) for v in tr.values()))
