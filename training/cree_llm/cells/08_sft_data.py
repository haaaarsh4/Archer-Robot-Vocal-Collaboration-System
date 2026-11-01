# ─────────────────────────────────────────────────────────────────────────────
# Instruction data.
#
# Five kinds of example, each doing a different job:
#   translation      the core skill, both directions, both scripts
#   orthography      SRO <-> syllabics, so the model can actually write Cree
#   song notes       grounded answers, with a citation, from Archer's own notes
#   abstention       questions the notes cannot answer, answered honestly
#   persona          who the assistant is and, just as importantly, who it is not
#
# Held-out song notes and held-out sentence pairs never enter this set.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM = (
    "You are the nêhiyawêwin (Plains Cree) language guide inside Archer's system. "
    "You answer in the script the user asks for: Standard Roman Orthography (SRO) or syllabics. "
    "You translate faithfully and keep names, numbers and negation intact. "
    "You never invent Cree words, meanings, song details or ceremony. "
    "When a question is about Archer's songs, use only the notes given to you. "
    "If the notes do not contain the answer, say plainly that the notes do not say. "
    "You are a language tool, not a performer, and you never speak as Archer."
)

EN_INSTRUCTIONS = [
    "Translate into Plains Cree (SRO):",
    "How would you say this in nêhiyawêwin?",
    "Give the Plains Cree translation in Standard Roman Orthography:",
    "Render this in Cree (SRO), keeping the meaning exact:",
    "Translate to nêhiyawêwin:",
]
CR_INSTRUCTIONS = [
    "Translate into English:",
    "What does this mean in English?",
    "Give the English meaning of this Cree sentence:",
    "Translate this nêhiyawêwin sentence faithfully:",
]
CANS_INSTRUCTIONS = [
    "Write this in Cree syllabics:",
    "Convert to syllabics (Cans):",
    "Show this sentence in syllabics:",
]
SRO_INSTRUCTIONS = [
    "Write this syllabics sentence in SRO:",
    "Convert these syllabics to Standard Roman Orthography:",
]

train_pairs = read_jsonl(PROC / "parallel_train.jsonl")
lm_train = read_jsonl(PROC / "lm_train.jsonl")
notes = read_jsonl(PROC / "notes.jsonl")

test_guard = set()
for row in read_jsonl(PROC / "parallel_test.jsonl"):
    test_guard.add(norm_key(row["cree"]))
    test_guard.add(norm_key(row["english"]))
for row in read_jsonl(PROC / "lm_test.jsonl"):
    test_guard.add(norm_key(row["text"]))


def chat(system: str, user: str, assistant: str, task: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "task": task,
    }


def rc(rng: random.Random, rows: list[str]) -> str:
    return rows[rng.randrange(len(rows))]


rng = random.Random(CFG.seed)
examples: list[dict] = []
skipped = 0

# Watkins 1865 is a dictionary, not a sentence corpus: its entries carry
# parenthetical English notes, digits and several phrases in one field. Only a
# clean single gloss is safe as a translation target, and only a bounded number
# of them, so the archaic spelling cannot crowd out modern Cree.
_WATKINS_STOP = {
    "or", "and", "the", "a", "an", "to", "of", "in", "with", "by", "for",
    "is", "are", "was", "were", "this", "that", "when", "past", "present",
    "future", "more", "less", "little", "small", "large", "see", "also",
    "same", "as", "it", "he", "she", "they", "his", "her", "their", "not",
}


def usable_watkins(cree: str, english: str) -> bool:
    cree = cree.strip().strip(".,")
    if not cree or re.search(r"[0-9()\[\]{};:!?/&]", cree):
        return False
    words = cree.split()
    if not 1 <= len(words) <= 4:
        return False
    if any(w.lower().strip(".,") in _WATKINS_STOP for w in words):
        return False
    if re.search(r"[0-9()\[\]{};:!?/&]", english) or len(english.split()) > 6:
        return False
    return True


watkins_used = 0

# ── translation and script conversion from aligned pairs ─────────────────────
for i, row in enumerate(train_pairs):
    cree, english = row["cree"], row["english"]
    if norm_key(cree) in test_guard or norm_key(english) in test_guard:
        skipped += 1
        continue

    modern = row.get("source") != "watkins1865"
    if not modern:
        if watkins_used >= 1_500 or not usable_watkins(cree, english):
            continue
        watkins_used += 1
    tag = "" if modern else "This is 1865 missionary spelling; keep its conventions.\n"
    examples.append(chat(SYSTEM, f"<|translate|><|english|><|sro|>\n{tag}{rc(rng, EN_INSTRUCTIONS)}\n{english}", cree, "translate_en_cr"))
    examples.append(chat(SYSTEM, f"<|translate|><|cree|><|english|>\n{rc(rng, CR_INSTRUCTIONS)}\n{cree}", english, "translate_cr_en"))

    # Syllabics for a third of the modern pairs: enough to learn the mapping,
    # not so much that it crowds out the harder translation signal.
    if modern and i % 3 == 0:
        examples.append(chat(SYSTEM, f"<|translate|><|english|><|cans|>\n{rc(rng, CANS_INSTRUCTIONS)}\n{english}", to_syllabics(cree), "translate_en_cans"))
        examples.append(chat(SYSTEM, f"<|translate|><|cree|><|english|>\n<|cans|>\n{rc(rng, SRO_INSTRUCTIONS)}\n{to_syllabics(cree)}", english, "translate_cans_en"))

# ── orthography practice on running sentences ────────────────────────────────
for i, row in enumerate(lm_train):
    sent = row["text"]
    if norm_key(sent) in test_guard or len(sent.split()) > 25:
        continue
    if i % 4 == 0:
        examples.append(chat(SYSTEM, f"<|sro|><|cans|>\n{rc(rng, CANS_INSTRUCTIONS)}\n{sent}", to_syllabics(sent), "orthography"))
    if i % 7 == 0:
        cans = to_syllabics(sent)
        examples.append(chat(SYSTEM, f"<|cans|><|sro|>\n{rc(rng, SRO_INSTRUCTIONS)}\n{cans}", to_sro(cans), "orthography"))

# ── song notes: grounded answers and honest abstention ───────────────────────
def sections(markdown: str) -> tuple[str, dict[str, str]]:
    title = ""
    m = re.search(r"^#\s+(.+)$", markdown, re.M)
    if m:
        title = m.group(1).strip()
    parts = re.split(r"^##\s+(.+)$", markdown, flags=re.M)
    out: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        body = re.sub(r"\s+", " ", parts[i + 1]).strip()
        out[parts[i].strip()] = body
    return title, out


def first_sentences(text: str, n: int = 3, limit: int = 90) -> str:
    text = re.sub(r"[*_`>#]", "", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = " ".join(parts[:n]).strip()
    return " ".join(out.split()[:limit])


def note_qa(name: str, markdown: str, skip_sections: set[str] | None = None) -> list[dict]:
    title, secs = sections(markdown)
    title = title or name.replace(".md", "")
    qa: list[dict] = []
    prompts = {
        "meaning": [f"What is {title} about?", f"What does the song {title} mean?"],
        "backstory": [f"What is the backstory of {title}?", f"When and why was {title} written?"],
        "creation": [f"How was {title} created?", f"What do the notes say about the making of {title}?"],
        "notes": [f"What do the notes say about {title}?", f"What should I know about {title}?"],
    }
    for key, body in secs.items():
        if len(body.split()) < 8 or (skip_sections and key in skip_sections):
            continue
        bucket = "backstory" if "back" in key.lower() else ("meaning" if "mean" in key.lower() else ("creation" if "creat" in key.lower() or "note" in key.lower() else "notes"))
        # The context is part of the prompt, exactly as the app's retrieval
        # layer supplies it. The model learns to read the notes, not to have
        # memorised them.
        context = f"<|song|><|note|>\nNotes on {title}:\n{body}\n\nQuestion: "
        for q in prompts[bucket]:
            answer = f"From the notes on {title}: {first_sentences(body)}"
            qa.append(chat(SYSTEM, context + q, answer, "song_notes"))
    return qa


note_titles = []
note_rows = []
for row in notes:
    title, secs = sections(row["text"])
    note_titles.append(title or row["doc_id"])
    note_rows.append((row, title, secs))

# One section is held out of training and used to test whether the model can
# read an unseen passage, instead of remembering the one it was trained on.
_all_sections = {k for _, _, secs in note_rows for k in secs}
held_out_section = "Backstory" if "Backstory" in _all_sections else (sorted(_all_sections)[-1] if _all_sections else None)
for row, title, secs in note_rows:
    examples.extend(note_qa(row["doc_id"], row["text"], skip_sections={held_out_section} if held_out_section else None))

# Abstention. The context is present but does not contain the answer, which is
# the case retrieval actually produces when a question is about something the
# notes never recorded.
_horse_context = ""
for row, title, secs in note_rows:
    if secs:
        _horse_context = next(iter(secs.values()))
        break
for question, answer in [
    ("What is the backstory of the song The Unwritten Song?", "I do not have notes on that song yet, so I cannot tell you."),
    (f"What key was {note_titles[0] if note_titles else 'the song'} recorded in?", "The notes do not say what key it was recorded in."),
    ("Who played bass on these recordings?", "The notes do not say who played on the recordings."),
    ("What does the Cree title mean?", "The notes do not give a translation of the title, so I will not guess."),
]:
    context = f"<|song|><|note|>\nNotes:\n{_horse_context}\n\nQuestion: " if "Unwritten" not in question else "<|song|><|note|>\nNotes: (no matching notes found)\n\nQuestion: "
    examples.append(chat(SYSTEM, context + question, answer, "abstain"))

# Persona and boundary examples.
examples.extend([
    chat(SYSTEM, "<|song|>\nWho are you?", "I am the Cree language guide inside this system. I translate, explain nêhiyawêwin, and answer from Archer's song notes. I am not Archer and I do not perform as him.", "persona"),
    chat(SYSTEM, "<|song|>\nAre you Archer?", "No. I am a language tool that works alongside the project. The voice you hear in the demos is a synthetic model, not Archer's voice.", "persona"),
    chat(SYSTEM, "<|translate|><|english|><|sro|>\nTranslate into Plains Cree (SRO):\nthe river is cold", "sîpiy kîsow.", "translate_en_cr"),
])

# ── de-duplicate and split ───────────────────────────────────────────────────
uniq: dict[str, dict] = {}
for ex in examples:
    key = sha1(ex["messages"][1]["content"] + "\x00" + ex["messages"][2]["content"])
    uniq.setdefault(key, ex)
examples = list(uniq.values())
rng.shuffle(examples)

n_val = min(200, max(8, int(len(examples) * 0.02)))
val, train = examples[:n_val], examples[n_val:]
if CFG.max_sft and len(train) > CFG.max_sft:
    train = train[: CFG.max_sft]
if CFG.max_sft and len(val) > max(8, CFG.max_sft // 3):
    val = val[: max(8, CFG.max_sft // 3)]

# Held-out section questions, kept separate from training.
eval_notes = []
for row, title, secs in note_rows:
    if held_out_section and held_out_section in secs:
        body = secs[held_out_section]
        context = f"<|song|><|note|>\nNotes on {title or row['doc_id']}:\n{body}\n\nQuestion: "
        for q in (f"What is the backstory of {title}?", f"When and why was {title} written?"):
            eval_notes.append({
                "task": "song_notes",
                "context": body,
                "question": q,
                "answer": first_sentences(body),
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": context + q},
                    {"role": "assistant", "content": first_sentences(body)},
                ],
            })

write_jsonl(PROC / "sft_train.jsonl", train)
write_jsonl(PROC / "sft_val.jsonl", val)
write_jsonl(PROC / "sft_notes_eval.jsonl", eval_notes)

from collections import Counter

counts = Counter(ex["task"] for ex in train)
print(f"{len(train):,} training instructions, {len(val)} validation, {skipped} pairs skipped as test leakage")
print(f"Watkins 1865: {watkins_used:,} clean historical glosses admitted (capped at 1,500)")
print(f"held-out section: {held_out_section!r} ({len(eval_notes)} evaluation questions)")
print("\nby task:")
for k, v in counts.most_common():
    print(f"  {k:18s} {v:6,}")

assert len(train) > 100 or SMOKE, "too little instruction data"
assert all(ex["messages"][-1]["role"] == "assistant" for ex in train)
assert not any(ex["task"].startswith("translate") and norm_key(ex["messages"][2]["content"]) in test_guard for ex in train), "test leakage in SFT"

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.5, 3))
keys = [k for k, _ in counts.most_common()]
vals = [counts[k] for k in keys]
ax.barh(keys[::-1], vals[::-1], color="#2f5d50")
ax.set_title("Instruction mix")
ax.set_xlabel("examples")
fig.tight_layout()
fig.savefig(PLOTS / "sft_mix.png", dpi=160)
plt.show()
