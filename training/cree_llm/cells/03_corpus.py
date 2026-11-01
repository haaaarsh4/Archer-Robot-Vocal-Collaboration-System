# ─────────────────────────────────────────────────────────────────────────────
# Corpus acquisition.
#
# Every loader returns (documents, pairs) and writes one line to the provenance
# ledger. Documents are running Cree text; pairs are aligned Cree/English
# sentences. A loader that fails is recorded as failed and the run continues:
# a missing source must never be papered over with invented text.
#
# Licence rule, enforced here rather than mentioned in a footnote:
#   * NoDerivatives material is never loaded (it forbids adapted models).
#   * NonCommercial ShareAlike material is loaded only when
#     CFG.include_noncommercial is True, and the model is then released under
#     the same licence in the report card.
# ─────────────────────────────────────────────────────────────────────────────

import bz2
import unicodedata
import urllib.error
import urllib.parse

DOCS: list[dict] = []
PAIRS: list[dict] = []
LEXICON: dict[str, float] = {}
LEDGER: list[dict] = []
STATUS: dict[str, str] = {}

PROJECT_REPO = "haaaarsh4/Archer-Robot-Vocal-Collaboration-System"
PROJECT_RAW = f"https://raw.githubusercontent.com/{PROJECT_REPO}/main"
_ND_MARKERS = ("noderiv", "noderat", "no-deriv", "no deriv", "nc-nd", "nd-4.0")


def ledger(name: str, url: str, licence: str, dialect: str, script: str,
           note: str = "", allowed: bool = True) -> None:
    LEDGER.append({
        "source": name, "url": url, "licence": licence, "dialect": dialect,
        "script": script, "note": note, "allowed": allowed,
    })


def _readme_licence(path: Path) -> str:
    """The nearest README's licence block, or the repository default.

    Only the closest README is consulted. Walking further up would apply the
    root README's bag-of-words licence (CC BY-NC 4.0) to texts whose own
    README states CC BY 4.0 or says nothing at all, which would either
    mislicense permissive text or let restricted text through.
    """
    for parent in [path.parent, *path.parents]:
        readme = parent / "README.md"
        if readme.exists():
            text = readme.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"#\s*Licen[cs]e(.{0,700})", text, re.I | re.S)
            if m:
                return " ".join(m.group(1).split())[:400]
            # The repository README is explicit: unless a collection states
            # otherwise, it is CC BY-NC-SA 4.0. Assuming CC0 here would quietly
            # relicense community material, so the fallback is the restrictive
            # one and the run's licence ledger inherits it.
            return "CC BY-NC-SA 4.0 (repository default; no subcollection licence stated)"
    return "CC BY-NC-SA 4.0 (repository default; no subcollection licence stated)"


def _blocked_by_licence(licence: str) -> bool:
    low = licence.lower()
    if any(marker in low for marker in _ND_MARKERS):
        return True
    if "noncommercial" in low or "non-commercial" in low or "nc-" in low:
        return not CFG.include_noncommercial
    return False


def _add_doc(text: str, **meta) -> None:
    text = unicodedata.normalize("NFC", str(text)).strip()
    if len(text) < 12:
        return
    DOCS.append({"text": text, **meta})


def _add_pair(cree: str, english: str, **meta) -> None:
    cree = unicodedata.normalize("NFC", str(cree)).strip()
    english = unicodedata.normalize("NFC", str(english)).strip()
    if len(cree) < 2 or len(english) < 2:
        return
    PAIRS.append({"cree": cree, "english": english, **meta})


# ── 1. EdTeKLA Cree Corpus ───────────────────────────────────────────────────
def load_edtekla() -> None:
    url = "https://codeload.github.com/EdTeKLA/IndigenousLanguages_Corpora/tar.gz/refs/heads/master"
    archive = download(url, RAW / "edtekla.tar.gz")
    root = RAW / "edtekla"
    if not (root / "IndigenousLanguages_Corpora-master").exists():
        with tarfile.open(archive) as tf:
            tf.extractall(root)
    base = root / "IndigenousLanguages_Corpora-master" / "PlainsCree"

    blocked_dirs = {"Facebook", "CreeLiteracyOrg", "Twitter", "Bible"}
    used = skipped_lic = skipped_bow = 0
    for path in sorted(base.rglob("*.txt")):
        rel = path.relative_to(base)
        if path.name.lower().startswith("readme"):
            continue
        if rel.parts[0] in blocked_dirs:
            skipped_lic += 1
            continue
        licence = _readme_licence(path)
        if _blocked_by_licence(licence):
            skipped_lic += 1
            continue
        if "_bow" in path.stem:
            # Bag-of-words files carry counts, not sentences. They are still
            # useful as a vocabulary check, never as training text.
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.rsplit(None, 1)
                if len(parts) == 2 and parts[0].strip() and parts[1].isdigit():
                    w = normalize_sro(parts[0])
                    if 2 <= len(w) <= 40 and re.fullmatch(r"[a-zâêîô'-]+", w):
                        LEXICON[w] = max(LEXICON.get(w, 0.0), float(parts[1]))
            skipped_bow += 1
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        script = "Cans" if path.stem.endswith("_syllabics") else "Latn"
        if script == "Cans":
            text = to_sro(text)
        # Keep a mixed document: the sentence filter in the next cell removes
        # its English lines, while its Cree paragraphs are still worth having.
        if cree_word_ratio(text) < 0.25 and lang_guess(text) != "crk":
            continue
        doc_id = str(rel.with_suffix(""))
        _add_doc(text, lang="crk", script="Latn", source=f"edtekla/{doc_id}",
                 doc_id=doc_id, licence=licence, dialect="plains", kind="prose")
        used += 1

        # A Cree file and its English sibling, line-aligned.
        en_path = path.with_name(path.name.replace("_cr.txt", "_en.txt").replace("_syllabics.txt", "_en.txt"))
        if en_path.exists():
            cr_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            en_lines = en_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if len(cr_lines) == len(en_lines):
                for i, (cr, en) in enumerate(zip(cr_lines, en_lines)):
                    cr = to_sro(cr) if script == "Cans" else cr
                    if len(cr.split()) >= 2 and len(en.split()) >= 2 and lang_guess(en) == "eng" and cree_word_ratio(cr) > 0.6:
                        _add_pair(cr, en, source="edtekla", doc_id=f"{doc_id}#{i}",
                                  licence=licence, dialect="plains", origin="human")

    ledger("EdTeKLA Cree Corpus", "https://github.com/EdTeKLA/IndigenousLanguages_Corpora",
           "CC BY-NC-SA 4.0 repo default; per-text READMEs; NC-ND subcollections skipped",
           "plains (y-dialect)", "Latn + Cans",
           f"{used} texts used; {skipped_lic} skipped on licence; {skipped_bow} BoW files used for vocabulary only")
    STATUS["edtekla"] = f"ok: {used} texts"


# ── 2. Plains Cree figurative corpus (Bloomfield 1934 + audits) ──────────────
def load_figurative() -> None:
    from datasets import load_dataset

    n = 0
    for config in ("gold", "silver"):
        ds = load_dataset("KonradBRG/plains-cree-figurative", config, split=config)
        if SMOKE:
            ds = ds.select(range(min(120, len(ds))))
        for row in ds:
            cree, english = row.get("text_cree", ""), row.get("text_en", "")
            if not cree or not english:
                continue
            src = row.get("source_file") or f"figurative-{config}"
            _add_pair(cree, english, source="figurative", doc_id=f"{src}/{row.get('sentence_id', n)}",
                      licence="CC BY 4.0", dialect="plains", origin="human" if config == "gold" else "silver")
            _add_doc(cree, lang="crk", script="Latn", source=f"figurative/{src}",
                     doc_id=f"{src}/{row.get('sentence_id', n)}", licence="CC BY 4.0",
                     dialect="plains", kind="prose")
            n += 1
    ledger("Plains Cree Figurative Language corpus", "https://huggingface.co/datasets/KonradBRG/plains-cree-figurative",
           "CC BY 4.0", "plains", "Latn", f"{n} sentence pairs and sentences")
    STATUS["figurative"] = f"ok: {n} pairs"


# ── 3. Wikipedia in Cree ─────────────────────────────────────────────────────
def load_wikipedia() -> None:
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.cr", split="train")
    n = 0
    for row in ds:
        text = row.get("text", "")
        if len(text) < 80 or cree_word_ratio(text) < 0.2:
            continue
        _add_doc(text, lang="crk", script="Latn", source="wikipedia",
                 doc_id=str(row.get("id", n)), licence="CC BY-SA 4.0",
                 dialect="plains", kind="encyclopedic")
        n += 1
        if SMOKE and n >= 20:
            break
    ledger("Cree Wikipedia", "https://huggingface.co/datasets/wikimedia/wikipedia",
           "CC BY-SA 4.0", "mixed Cree", "Latn", f"{n} articles")
    STATUS["wikipedia"] = f"ok: {n} articles"


# ── 4. FineWeb-2 Cree slices ─────────────────────────────────────────────────
def load_fineweb() -> None:
    from datasets import load_dataset

    n = 0
    for config, script in (("crk_Latn", "Latn"), ("crk_Cans", "Cans")):
        try:
            ds = load_dataset("HuggingFaceFW/fineweb-2", config, split="train", streaming=True)
        except Exception as exc:
            STATUS[f"fineweb:{config}"] = f"unavailable: {exc}"
            continue
        for row in ds:
            text = row.get("text", "")
            if len(text) < 120:
                continue
            if script == "Cans":
                text = to_sro(text)
            if cree_word_ratio(text) < 0.35:
                continue
            _add_doc(text, lang="crk", script=script, source=f"fineweb2/{config}",
                     doc_id=str(row.get("id", n)), licence="ODC-BY 1.0",
                     dialect="plains", kind="web")
            n += 1
            if SMOKE and n >= 30:
                break
    ledger("FineWeb-2 Cree slices", "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
           "ODC-BY 1.0", "plains", "Latn + Cans", f"{n} documents after language filtering")
    STATUS["fineweb2"] = f"ok: {n} docs"


# ── 4a. FinePDFs Cree slices ─────────────────────────────────────────────────
#
# FinePDFs extracts text from Common Crawl PDFs. Its Cree slices are small but
# they are the densest modern prose available: 430 Latin-script documents and
# 330 syllabic documents, roughly 1.2 million words, mostly published Cree
# rather than forum fragments. The text layer is already extracted, so this is
# clean copy, not OCR.
def load_finepdfs() -> None:
    from datasets import load_dataset

    n = 0
    for config, script in (("crk_Latn", "Latn"), ("crk_Cans", "Cans")):
        try:
            ds = load_dataset("HuggingFaceFW/finepdfs", config, split="train", streaming=True)
        except Exception as exc:
            STATUS[f"finepdfs:{config}"] = f"unavailable: {exc}"
            continue
        for row in ds:
            text = row.get("text", "")
            if len(text) < 150:
                continue
            if script == "Cans":
                text = to_sro(text)
            if cree_word_ratio(text) < 0.30:
                continue
            _add_doc(text, lang="crk", script=script, source=f"finepdfs/{config}",
                     doc_id=str(row.get("id", n)), licence="ODC-BY 1.0 (PDF text layer)",
                     dialect="plains", kind="pdf")
            n += 1
            if SMOKE and n >= 20:
                break
    ledger("FinePDFs Cree slices", "https://huggingface.co/datasets/HuggingFaceFW/finepdfs",
           "ODC-BY 1.0", "plains", "Latn + Cans", f"{n} documents from PDF text layers")
    STATUS["finepdfs"] = f"ok: {n} docs"


# ── 4b. FineTranslations Cree documents ──────────────────────────────────────
#
# FineTranslations pairs each FineWeb-2 document with an English translation.
# The Cree side is human web text; the English side is machine translation, so
# every pair is tagged `origin=mined` and `translation=machine` and the
# benchmark reports human-origin and mined results separately. Chunk alignment
# is used, never a whole article, because a 40-line page is not one sentence.
def load_finetranslations() -> None:
    from datasets import load_dataset

    n_docs = n_pairs = 0
    for config, script in (("crk_Latn", "Latn"), ("crk_Cans", "Cans")):
        try:
            ds = load_dataset("HuggingFaceFW/finetranslations", config, split="train", streaming=True)
        except Exception as exc:
            STATUS[f"finetranslations:{config}"] = f"unavailable: {exc}"
            continue
        for row in ds:
            cree = row.get("og_full_text", "") or ""
            english = row.get("translated_text", "") or ""
            if len(cree) < 80 or not english:
                continue
            if script == "Cans":
                cree = to_sro(cree)
            if cree_word_ratio(cree) < 0.30:
                continue
            doc_id = str(row.get("id", f"{config}-{n_docs}"))
            # The Cree document itself is already loaded by `load_fineweb`;
            # adding it again here would double-count the same text in the
            # ledger and make the corpus look larger than it is. Only the
            # aligned chunks are new.
            n_docs += 1
            chunks_cr = row.get("og_chunks") or []
            chunks_en = row.get("translated_chunks") or []
            for i, (cr, en) in enumerate(zip(chunks_cr, chunks_en)):
                cr = to_sro(cr) if script == "Cans" else cr
                if not (3 <= len(cr.split()) <= 80 and 3 <= len(en.split()) <= 120):
                    continue
                if cree_word_ratio(cr) < 0.40 or lang_guess(en) != "eng":
                    continue
                _add_pair(cr, en, source="finetranslations", doc_id=f"{doc_id}#{i}",
                          licence="ODC-BY 1.0 (machine English)", dialect="plains",
                          origin="mined", translation="machine")
                n_pairs += 1
            if SMOKE and n_docs >= 20:
                break
    ledger("FineTranslations Cree", "https://huggingface.co/datasets/HuggingFaceFW/finetranslations",
           "ODC-BY 1.0; English side is machine translation", "plains", "Latn + Cans",
           f"{n_docs} documents, {n_pairs} aligned chunks")
    STATUS["finetranslations"] = f"ok: {n_pairs} aligned chunks from {n_docs} documents"


# ── 4c. GlotCC Cree slices ───────────────────────────────────────────────────
def load_glotcc() -> None:
    from datasets import load_dataset

    n = 0
    for config, script in (("crk-Latn", "Latn"), ("crk-Cans", "Cans")):
        try:
            ds = load_dataset("cis-lmu/GlotCC-V1", config, split="train", streaming=True)
        except Exception as exc:
            STATUS[f"glotcc:{config}"] = f"unavailable: {exc}"
            continue
        for row in ds:
            text = row.get("content", "")
            if len(text) < 80:
                continue
            if script == "Cans":
                text = to_sro(text)
            if cree_word_ratio(text) < 0.30:
                continue
            _add_doc(text, lang="crk", script=script, source=f"glotcc/{config}",
                     doc_id=str(row.get("id", n)), licence="ODC-BY 1.0",
                     dialect="plains", kind="web")
            n += 1
    ledger("GlotCC Cree slices", "https://huggingface.co/datasets/cis-lmu/GlotCC-V1",
           "ODC-BY 1.0", "plains", "Latn + Cans", f"{n} documents")
    STATUS["glotcc"] = f"ok: {n} docs"


# ── 4d. Tatoeba Cree sentences ───────────────────────────────────────────────
def load_tatoeba() -> None:
    url = "https://downloads.tatoeba.org/exports/per_language/crk/crk_sentences.tsv.bz2"
    path = download(url, RAW / "tatoeba_crk.tsv.bz2")
    n = 0
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sent_id, _lang, text = parts[0], parts[1], "\t".join(parts[2:])
            if len(text) < 12:
                continue
            _add_doc(text, lang="crk", script="Cans" if looks_like_syllabics(text) else "Latn",
                     source="tatoeba", doc_id=sent_id, licence="CC BY 2.0 FR",
                     dialect="plains", kind="sentence")
            n += 1
    ledger("Tatoeba Cree sentences", "https://tatoeba.org/", "CC BY 2.0 FR",
           "plains", "Latn + Cans", f"{n} sentences")
    STATUS["tatoeba"] = f"ok: {n} sentences"


# ── 4e. GlotStoryBook Cree rows ──────────────────────────────────────────────
def load_storybook() -> None:
    from datasets import load_dataset

    n = 0
    try:
        ds = load_dataset("cis-lmu/GlotStoryBook", "default", split="train", streaming=True)
    except Exception as exc:
        STATUS["storybook"] = f"unavailable: {exc}"
        return
    for row in ds:
        code = str(row.get("ISO639-3", "")).lower()
        if code not in {"crk", "cre"}:
            continue
        text = row.get("Text", "")
        if len(text) < 40:
            continue
        _add_doc(text, lang="crk", script="Cans" if looks_like_syllabics(text) else "Latn",
                 source="storybook", doc_id=f"{row.get('Text Number')}-{n}",
                 licence=str(row.get("License", "CC BY-NC-SA")), dialect="plains", kind="story")
        n += 1
    ledger("GlotStoryBook Cree", "https://huggingface.co/datasets/cis-lmu/GlotStoryBook",
           "CC BY-NC-SA (per row)", "plains", "Latn + Cans", f"{n} story paragraphs")
    STATUS["storybook"] = f"ok: {n} paragraphs"


# ── 4f. Public-domain historical Cree books ───────────────────────────────────
#
# The modern web has almost no Plains Cree prose: FineWeb-2 carries 192 Cree
# documents in total, and Cree Wikipedia has 187 articles. The nineteenth
# century printed far more. These books are public domain, their Cree is real,
# and they are used for language exposure only, never as modern translation
# targets; every one carries the `historical` dialect tag.
_HIST_TRANS = str.maketrans({
    "û": "ô", "Û": "Ô", "ù": "ô", "Ù": "Ô",
    "ā": "â", "ē": "ê", "ī": "î", "ō": "ô",
    "à": "â", "è": "ê", "ì": "î", "ò": "ô", "é": "ê", "ó": "ô",
})
_WATKINS_EN_STOP = {
    "or", "and", "the", "a", "an", "to", "of", "in", "with", "by", "for",
    "is", "are", "was", "were", "this", "that", "when", "past", "present",
    "future", "more", "less", "little", "small", "large", "see", "also",
    "same", "as", "it", "he", "she", "they", "his", "her", "their", "not",
}


def _clean_watkins_form(form: str) -> str:
    """One modern-spelling Cree headword, or an empty string.

    Watkins' dictionary is OCR'd and annotated: entries arrive as
    ``Jspimik. (more) awusim4`` or with 19th-century vowel marks. This keeps
    only a short, unmixed, normalised form, because a dirty gloss teaches the
    model a spelling nobody uses.
    """
    form = unicodedata.normalize("NFC", str(form))
    form = re.sub(r"\([^)]*\)", " ", form)
    form = form.translate(_HIST_TRANS)
    form = form.replace("‘", "'").replace("’", "'")
    form = re.sub(r"[^A-Za-zâêîô'\- ]+", " ", form)
    form = re.sub(r"\s+", " ", form).strip().lower()
    if not form:
        return ""
    words = form.split()
    if not 1 <= len(words) <= 4:
        return ""
    if any(w.strip(".,") in _WATKINS_EN_STOP for w in words):
        return ""
    if cree_word_ratio(form) < 0.8 or lang_guess(form) != "crk":
        return ""
    return form


def _clean_watkins_english(gloss: str) -> str:
    gloss = re.sub(r"\([^)]*\)", " ", str(gloss))
    gloss = re.sub(r"[^A-Za-z'\- ]+", " ", gloss)
    gloss = re.sub(r"\s+", " ", gloss).strip().lower()
    if not gloss or len(gloss.split()) > 6:
        return ""
    return gloss


# Books known to be worth the download even if the archive search ranks them
# low. The rest of the list is discovered, not hand-picked.
_ARCHIVE_SEED = {
    "P005757": "Plains Cree texts (Bloomfield, 1934)",
    "P005409": "Sacred stories of the Sweet Grass Cree (Bloomfield, 1930)",
    "cihm_07195": "Gospel of Matthew, 1877",
    "cihm_01021": "Gospel of Mark, 1876",
    "P001735": "Cree hymn book, 1888",
    "P002107": "Cree primer, 1893",
    "cihm_30552": "Primer and language lessons, 1890",
    "cihm_30323": "Indian child's book, 1890",
    "P004640": "Cree hymn book, 1920",
    "creehymnbook00huntgoog": "Cree hymn book, 1888",
    "grammarofcreelan00hord_0": "Grammar of the Cree language, 1881",
    "P001885": "Primer and language lessons in English and Cree, 1890",
    "P002037": "Indian child's book",
    "P002871": "Psalms in Plains Cree, 1905",
    "cihm_05447": "Psalter in Cree, 1875",
    "P000574": "Cree hymn book, 1870",
}
# Moosonee, James Bay and East Cree material is a different dialect. It is
# excluded rather than blended, because a y-dialect model that quietly learns
# n-dialect morphology is worse than one that says which data it used.
_ARCHIVE_BLOCK = (
    "moosonee", "moose cree", "rupert's land", "rupert ", "james bay",
    "east main", "eastmain", "albany", "chisasibi", "mistissini", "waskaganish",
    "wemindji", "nemaska", "ouje", "whapmagoostui", "fort george", "east cree",
    # French-language devotional books and English grammars are about Cree
    # rather than in it; their OCR is mostly the surrounding language.
    "français", "francais", "langue crise", "prières", "prieres", "cantiques",
    "catéchisme", "catechisme", "dictionnaire", "grammaire", "grammar of the cree",
)
_ARCHIVE_EN = {
    "the", "and", "of", "to", "in", "is", "was", "were", "that", "this", "for",
    "with", "he", "she", "they", "his", "her", "their", "you", "not", "but",
    "from", "have", "has", "had", "will", "would", "there", "what", "when",
    "who", "how", "all", "one", "two", "on", "at", "by", "as", "it", "be",
}


def _english_share(text: str) -> float:
    words = re.findall(r"[a-z']+", text.lower())
    return sum(w in _ARCHIVE_EN for w in words) / max(1, len(words))


def _lex_hit(text: str) -> float:
    """Fraction of words the GiellaLT lexicon recognises.

    This is the check that catches syllabic OCR: a scan of a syllabic book
    produces page after page of Latin letters that no Cree word list contains.
    English pages fail it too, for the same reason.
    """
    words = [w for w in re.findall(r"[a-zâêîô'-]+", normalize_sro(text)) if len(w) > 1]
    if not words:
        return 0.0
    hits = 0
    for word in words:
        parts = [p for p in word.replace("-", " ").split() if p]
        if any(p in LEXICON for p in parts):
            hits += 1
    return hits / len(words)


def _archive_candidates() -> list[tuple[str, str, int]]:
    params = {
        "q": "language:(cree) AND mediatype:texts",
        "fl[]": ["identifier", "title", "year", "downloads", "description"],
        "rows": "500",
        "output": "json",
    }
    url = "https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers={"User-Agent": "archer-llm/1.0"})
    payload = json.loads(urllib.request.urlopen(req, timeout=90).read())
    found: dict[str, tuple[str, str, int]] = {}
    for row in payload["response"]["docs"]:
        ident = str(row.get("identifier", ""))
        if not ident:
            continue
        title = str(row.get("title", "") or "")
        description = str(row.get("description", "") or "")
        low = f"{title} {description}".lower()
        if any(marker in low for marker in _ARCHIVE_BLOCK):
            continue
        match = re.search(r"(1[5-9]\d\d|20[0-2]\d)", str(row.get("year", "")))
        year = int(match.group(1)) if match else 0
        if year and year > 1934:
            continue
        found[ident] = (title, description, int(row.get("downloads", 0) or 0))
    ordered = sorted(found.items(), key=lambda kv: (kv[0] not in _ARCHIVE_SEED, -kv[1][2]))
    return [(ident, title, downloads) for ident, (title, _d, downloads) in ordered]


def load_archive() -> None:
    try:
        candidates = _archive_candidates()
    except Exception as exc:
        STATUS["archive_search"] = f"unavailable: {exc}"
        candidates = [(ident, title, 0) for ident, title in _ARCHIVE_SEED.items()]
    log(f"archive: {len(candidates)} candidate books after dialect filter")
    n = rejected = attempts = 0
    budget = 6 if SMOKE else 65
    max_attempts = 12 if SMOKE else 110
    for ident, label, _downloads in candidates:
        if n >= budget or attempts >= max_attempts:
            break
        attempts += 1
        try:
            meta = json.loads(urllib.request.urlopen(
                f"https://archive.org/metadata/{ident}", timeout=60
            ).read())
            names = [f["name"] for f in meta.get("files", []) if f["name"].endswith("_djvu.txt")]
            if not names:
                rejected += 1
                continue
            url = f"https://archive.org/download/{ident}/{urllib.parse.quote(names[0])}"
            path = download(url, RAW / f"archive_{ident}.txt", timeout=180)
            text = path.read_text(encoding="utf-8", errors="ignore").translate(_HIST_TRANS)
            if len(text.split()) < 1500:
                rejected += 1
                continue
            # Bilingual books (Bloomfield faces Cree with English) and books
            # with a long English introduction cannot pass a whole-book test.
            # Judge each 700-word window instead, then let the sentence filter
            # in the next cell remove the English half of a mixed window.
            words = text.split()
            keep: list[tuple[int, str]] = []
            for i in range(0, len(words), 700):
                chunk = " ".join(words[i:i + 700])
                if len(chunk.split()) < 150:
                    continue
                if _english_share(chunk) > 0.5 or cree_word_ratio(chunk) < 0.5:
                    continue
                if _lex_hit(chunk) < 0.2:
                    continue
                keep.append((i // 700, chunk))
            # A book that yields one plausible window is usually a bad scan
            # with a stray Cree quotation. Require at least three.
            if len(keep) >= 3:
                for index, chunk in keep:
                    _add_doc(chunk, lang="crk", script="Latn", source=f"archive/{ident}",
                             doc_id=f"{ident}-{index:03d}",
                             licence="public domain (published before 1935; traditional texts)",
                             dialect="historical", kind="historical")
                n += 1
            else:
                rejected += 1
                STATUS[f"archive:{ident}"] = "rejected: OCR not usable (syllabic scan or English)"
        except Exception as exc:
            STATUS[f"archive:{ident}"] = f"unavailable: {exc}"
            rejected += 1
    ledger("Internet Archive historical Cree", "https://archive.org/",
           "public domain (published before 1935; traditional texts)", "historical", "Latn",
           f"{n} books used, {rejected} rejected (syllabic OCR, English, or too short)")
    STATUS["archive"] = f"ok: {n} books, {rejected} rejected"


# ── 4g. Plains Cree phrase recordings ────────────────────────────────────────
def load_plainscree() -> None:
    from datasets import load_dataset

    n = 0
    try:
        ds = load_dataset("verayang/plainscree", split="train", streaming=True)
        # The audio column needs torchcodec; the transcriptions do not.
        ds = ds.select_columns(["audio_id", "cree_transcription", "english_transcription"])
    except Exception as exc:
        STATUS["plainscree"] = f"unavailable: {exc}"
        return
    for row in ds:
        cree = row.get("cree_transcription", "") or ""
        english = row.get("english_transcription", "") or ""
        if len(cree) < 4 or not english:
            continue
        _add_pair(cree, english, source="plainscree", doc_id=f"audio-{row.get('audio_id', n)}",
                  licence="research dataset (audio-derived transcriptions)", dialect="plains",
                  origin="human", translation="human")
        n += 1
    ledger("Plains Cree phrase recordings", "https://huggingface.co/datasets/verayang/plainscree",
           "research dataset; transcriptions only, audio not redistributed", "plains", "Latn",
           f"{n} short phrases")
    STATUS["plainscree"] = f"ok: {n} phrases"


# ── 5. This project's own data and notes ─────────────────────────────────────
def load_project() -> None:
    # 5a. The aligned sentences already curated in the repository. The file also
    #     contains dictionary frequency rows; those are filtered below.
    url = f"{PROJECT_RAW}/data/processed/parallel_cr_en.jsonl"
    path = download(url, RAW / "project_parallel.jsonl")
    n = 0
    skipped_rows = 0
    for row in read_jsonl(path):
        cree, english = row.get("cree", ""), row.get("english", "")
        # Most of this file is a dictionary alignment ("a 218" <-> "abiathar
        # 1"), not sentence pairs. Those rows teach the model to emit stray
        # numbers and short glosses, and they poison the benchmark because the
        # test split is drawn from the same pool. A translation pair has no
        # digits and at least two words on each side.
        if re.search(r"\d", f"{cree} {english}"):
            skipped_rows += 1
            continue
        if len(cree.split()) < 2 or len(english.split()) < 2:
            skipped_rows += 1
            continue
        if cree_word_ratio(cree) > 0.55 and lang_guess(english) == "eng":
            _add_pair(cree, english, source="project", doc_id=f"parallel-{n}",
                      licence="project-owned", dialect="plains", origin="human")
            n += 1
    STATUS["project_parallel"] = f"ok: {n} pairs ({skipped_rows} numeric/short rows skipped)"

    # 5b. Archer's song notes: the grounding text for the assistant.
    notes = []
    try:
        listing = json.loads(urllib.request.urlopen(
            f"https://api.github.com/repos/{PROJECT_REPO}/contents/song_notes", timeout=30
        ).read())
        for item in listing:
            if item["name"].endswith((".md", ".txt")) and item["name"].lower() != "readme.md":
                body = urllib.request.urlopen(item["download_url"], timeout=30).read().decode("utf-8", "ignore")
                notes.append({"name": item["name"], "text": body})
    except Exception as exc:
        STATUS["song_notes"] = f"unavailable: {exc}"
    for note in notes:
        _add_doc(note["text"], lang="eng", script="Latn", source=f"song_notes/{note['name']}",
                 doc_id=note["name"], licence="project-owned", dialect="plains", kind="song-notes")
    STATUS["song_notes"] = f"ok: {len(notes)} notes"

    # 5c. Watkins 1865 dictionary entries, kept under their own dialect tag.
    try:
        wpath = download(f"{PROJECT_RAW}/data/processed/watkins_entries.jsonl", RAW / "watkins.jsonl")
        n = 0
        skipped = 0
        for row in read_jsonl(wpath):
            english = _clean_watkins_english(row.get("english", ""))
            if not english:
                continue
            for form in row.get("cree_forms", []):
                cree = _clean_watkins_form(form)
                if not cree:
                    skipped += 1
                    continue
                _add_pair(cree, english, source="watkins1865", doc_id=f"watkins-{n}",
                          licence="public domain (1865)", dialect="watkins1865",
                          origin="historical")
                n += 1
        STATUS["watkins"] = f"ok: {n} historical pairs ({skipped} OCR/annotation forms rejected)"
    except Exception as exc:
        STATUS["watkins"] = f"unavailable: {exc}"

    ledger("Archer project data + song notes", f"https://github.com/{PROJECT_REPO}",
           "project-owned", "plains", "Latn", f"{len(notes)} song notes, parallel pairs, Watkins 1865")


# ── 6. GiellaLT lexicon (validity only, not running text) ────────────────────
def load_giellalt() -> None:
    url = "https://codeload.github.com/giellalt/lang-crk/tar.gz/refs/heads/main"
    archive = download(url, RAW / "lang-crk.tar.gz")
    root = RAW / "lang-crk"
    if not (root / "lang-crk-main").exists():
        with tarfile.open(archive) as tf:
            tf.extractall(root)
    base = root / "lang-crk-main"

    n = 0
    weighted = base / "tools" / "spellcheckers" / "prefix_weighting.lexc"
    if weighted.exists():
        for line in weighted.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r'([^\s#]+)\s*[#$]?\s*#?\s*"weight:\s*([0-9.]+)"', line.strip())
            if not m:
                continue
            word, weight = normalize_sro(m.group(1).split("+")[0].rstrip("$")), float(m.group(2))
            if 2 <= len(word) <= 40 and re.fullmatch(r"[a-zâêîô'-]+", word):
                LEXICON[word] = max(LEXICON.get(word, 0.0), weight)
                n += 1
    for lexc in (base / "src" / "fst" / "morphology" / "stems").glob("*.lexc"):
        for line in lexc.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("!") or line.startswith("LEXICON"):
                continue
            surface = line.split()[0].split(":")[0].split("+")[0].rstrip("$")
            word = normalize_sro(surface)
            if 2 <= len(word) <= 40 and re.fullmatch(r"[a-zâêîô'-]+", word):
                LEXICON.setdefault(word, 0.1)
    public_texts = base / "devtools" / "testdata" / "crk_public_texts.txt"
    if public_texts.exists():
        _add_doc(public_texts.read_text(encoding="utf-8", errors="ignore"), lang="crk",
                 script="Latn", source="giellalt/public-texts", doc_id="crk_public_texts",
                 licence="AGPL-3.0 repository; public texts", dialect="plains", kind="prose")
        n += 1

    ledger("GiellaLT lang-crk lexicon", "https://github.com/giellalt/lang-crk",
           "AGPL-3.0 source; word list used for validation and tokenizer weighting",
           "plains", "Latn", f"{len(LEXICON)} word forms; {n} public text files")
    STATUS["giellalt"] = f"ok: {len(LEXICON)} forms"


# ── 7. Anything the user drops in ────────────────────────────────────────────
def load_uploads() -> None:
    dirs = [Path(os.environ.get("CREE_UPLOAD_DIR", ROOT / "uploads"))]
    if IN_COLAB:
        dirs.append(Path("/content/drive/MyDrive/cree_corpus"))
    n = 0
    for d in dirs:
        if not d.exists():
            continue
        for path in sorted(d.rglob("*")):
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for para in re.split(r"\n\s*\n", text):
                if len(para.strip()) > 20:
                    _add_doc(para, lang=lang_guess(para), script="Cans" if looks_like_syllabics(para) else "Latn",
                             source="community", doc_id=f"{path.name}#{n}", licence="user-provided",
                             dialect="unspecified", kind="community")
                    n += 1
    if n:
        ledger("Community uploads", str(dirs[0]), "user-provided (confirm before publishing)",
               "unspecified", "Latn + Cans", f"{n} paragraphs")
        STATUS["uploads"] = f"ok: {n} paragraphs"


# ── run every loader independently ───────────────────────────────────────────
# On Drive this whole section is skipped once it has produced files. It is the
# slowest part of the notebook (network, archive index, PDF extraction) and none
# of it changes between sessions, so repeating it after a disconnect only burns
# the time the checkpointing is meant to save.
_CORPUS_KEY = fingerprint(corpus="v1", noncommercial=CFG.include_noncommercial)
_READY = (RAW / "docs.jsonl", RAW / "pairs.jsonl", RAW / "lexicon.json")

if all(f.exists() for f in _READY) and stage_done("corpus", _CORPUS_KEY, requires=RAW / "ledger.json"):
    DOCS = read_jsonl(RAW / "docs.jsonl")
    PAIRS = read_jsonl(RAW / "pairs.jsonl")
    LEXICON = jload(RAW / "lexicon.json")
    LEDGER = jload(RAW / "ledger.json")
    STATUS = jload(RAW / "source_status.json")
    log(f"corpus loaded from Drive: {len(DOCS)} documents, {len(PAIRS)} pairs, {len(LEXICON)} lexicon forms")
else:
    for name, fn in [
        ("edtekla", load_edtekla),
        ("figurative", load_figurative),
        ("project", load_project),
        ("wikipedia", load_wikipedia),
        ("fineweb2", load_fineweb),
        ("finepdfs", load_finepdfs),
        ("finetranslations", load_finetranslations),
        ("glotcc", load_glotcc),
        ("tatoeba", load_tatoeba),
        ("storybook", load_storybook),
        ("plainscree", load_plainscree),
        # GiellaLT runs before the archive because its lexicon is what separates a
        # real 19th-century Cree page from a syllabic scan OCR turned into noise.
        ("giellalt", load_giellalt),
        ("archive", load_archive),
        ("uploads", load_uploads),
    ]:
        t0 = time.time()
        try:
            fn()
            log(f"{name}: {STATUS.get(name, 'ok')}  ({time.time() - t0:.1f}s)")
        except Exception as exc:
            STATUS[name] = f"failed: {type(exc).__name__}: {exc}"
            log(f"{name}: FAILED — {exc}")

    write_jsonl(RAW / "docs.jsonl", DOCS)
    write_jsonl(RAW / "pairs.jsonl", PAIRS)
    jdump(RAW / "lexicon.json", LEXICON)
    jdump(RAW / "ledger.json", LEDGER)
    jdump(RAW / "source_status.json", STATUS)
    mark_stage("corpus", _CORPUS_KEY, documents=len(DOCS), pairs=len(PAIRS))


import pandas as pd

by_source: dict[str, list[dict]] = {}
for doc in DOCS:
    by_source.setdefault(doc["source"].split("/")[0], []).append(doc)
rows = []
for name, subset in sorted(by_source.items()):
    rows.append({
        "source": name,
        "documents": len(subset),
        "cree words": sum(len(d["text"].split()) for d in subset),
        "script": ", ".join(sorted({d["script"] for d in subset})),
    })
ledger_df = pd.DataFrame(rows)
print(f"\n{len(DOCS)} documents, {len(PAIRS)} aligned pairs, {len(LEXICON)} lexicon forms")
print(ledger_df.to_string(index=False))
print("\nsource status:", json.dumps(STATUS, indent=2, ensure_ascii=False))
