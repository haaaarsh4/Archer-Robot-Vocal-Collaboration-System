# Instruction data for the one Archer model.
#
# There is a single model. It is not a Cree model sitting beside a general one,
# and it is not specialised into a router. It learns every job the assistant
# does, from one instruction set, in one adapter:
#
#   translate      English <-> Plains Cree, both directions, both scripts
#   orthography    SRO <-> syllabics, so it can write Cree rather than name it
#   explain        a Cree word and its English meaning, from the aligned pairs
#   song notes     grounded answers, with the song named, from Archer's notes
#   abstention     questions the notes cannot answer, answered accurately
#   project        conversation about Archer, this system and its limits
#   persona        who the assistant is, and just as importantly who it is not
#
# Held-out song notes and held-out sentence pairs never enter this set.

# The same string the app sends, in chat/archer_prompt.py. A test compares the
# two, because a system prompt that drifts from training moves the model off the
# distribution it learned.
SYSTEM = (
    "You are Archer's assistant, running entirely on the user's own machine. "
    "You are warm and plain spoken. If someone greets you or asks how you are, "
    "answer in one or two short sentences in the first person, and ask what they "
    "need. Do not describe the project or list your jobs unless they ask. "
    "You do four jobs: you talk about Archer's work and this system, you answer "
    "questions about Archer's songs from the notes you are given, you translate "
    "between English and nêhiyawêwin in Standard Roman Orthography and syllabics, "
    "and you explain and convert Cree words. "
    "When notes are given, they are the only evidence for anything about a song: "
    "answer from them, name the song, and if they do not contain the answer say "
    "plainly that the notes do not say. "
    "When no notes are given, answer about this project and about Plains Cree from "
    "what you know, and say when you are unsure instead of inventing a word, a "
    "meaning, a name, a number or a ceremony. "
    "Translate faithfully: keep names, numbers, negation and tense intact, and "
    "never leave an English word inside a Cree sentence. "
    "You never speak as Archer and you never claim his voice is yours."
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

# translation and script conversion from aligned pairs
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

# word explanation, from short aligned pairs
# The lexicon the pipeline builds records frequencies, not meanings. The short
# aligned pairs are the accurate source of a gloss: a single word mapped to a
# short English phrase by the corpus itself. Long pairs are skipped, so this
# cannot turn a sentence into a dictionary entry.
_gloss_seen: set[str] = set()
gloss_used = 0
for row in train_pairs:
    if gloss_used >= 900:
        break
    cree, english = row["cree"].strip(), row["english"].strip()
    if row.get("source") == "watkins1865":
        continue
    if not (1 <= len(cree.split()) <= 2 and 1 <= len(english.split()) <= 3):
        continue
    if norm_key(cree) in _gloss_seen or norm_key(cree) in test_guard:
        continue
    if not re.fullmatch(r"[a-zâêîô'-]+(?: [a-zâêîô'-]+)?", cree, re.I):
        continue
    _gloss_seen.add(norm_key(cree))
    gloss_used += 1
    for question in (
        f"What does {cree} mean?",
        f"Explain the Cree word {cree}.",
        f"Give the English meaning of {cree}.",
    ):
        examples.append(chat(SYSTEM, f"<|song|>\n{question}", f"{cree} means {english}.", "explain"))

# orthography practice on running sentences
for i, row in enumerate(lm_train):
    sent = row["text"]
    if norm_key(sent) in test_guard or len(sent.split()) > 25:
        continue
    if i % 4 == 0:
        examples.append(chat(SYSTEM, f"<|sro|><|cans|>\n{rc(rng, CANS_INSTRUCTIONS)}\n{sent}", to_syllabics(sent), "orthography"))
    if i % 7 == 0:
        cans = to_syllabics(sent)
        examples.append(chat(SYSTEM, f"<|cans|><|sro|>\n{rc(rng, SRO_INSTRUCTIONS)}\n{cans}", to_sro(cans), "orthography"))

# song notes: grounded answers and accurate abstention
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


def note_qa(name: str, markdown: str, skip_sections: set[str] | None = None,
            distractors: list[tuple[str, str]] | None = None) -> list[dict]:
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

        # Retrieval returns up to four passages, and one of them can belong to
        # another song. The model has to pick the passage that answers the
        # question, not read every paragraph as if it were about the same song.
        if distractors:
            other_title, other_body = distractors[rng.randrange(len(distractors))]
            mixed = (
                f"<|song|><|note|>\nNotes on {other_title}:\n{other_body}\n\n"
                f"Notes on {title}:\n{body}\n\nQuestion: "
            )
            for q in prompts[bucket]:
                answer = f"From the notes on {title}: {first_sentences(body)}"
                qa.append(chat(SYSTEM, mixed + q, answer, "song_notes"))
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
# The held-out section never appears here, not even as a distractor: the note
# reading score is only meaningful if that passage is genuinely unseen.
_section_pool = [
    (title or row["doc_id"], body)
    for row, title, secs in note_rows
    for name, body in secs.items()
    if len(body.split()) >= 8 and name != held_out_section
]
for row, title, secs in note_rows:
    own = title or row["doc_id"]
    others = [(t, body) for t, body in _section_pool if t != own]
    examples.extend(note_qa(
        row["doc_id"], row["text"],
        skip_sections={held_out_section} if held_out_section else None,
        distractors=others,
    ))

# Abstention. The context is present but does not contain the answer, which is
# the case retrieval actually produces when a question is about something the
# notes never recorded.
# The passage used for abstention and multi-turn examples must not be the
# held-out one, or the note reading score would be measuring a memorised text.
_horse_context = ""
_horse_title = note_titles[0] if note_titles else "the song"
for row, title, secs in note_rows:
    for name, body in secs.items():
        if name != held_out_section and body.strip():
            _horse_context = body
            _horse_title = title or row["doc_id"]
            break
    if _horse_context:
        break
# The retrieval layer can also return only passages about other songs. That is
# a different failure from an empty result, and the accurate answer names the
# mismatch instead of answering with the wrong song's notes.
for row, title, secs in note_rows[:3]:
    own = title or row["doc_id"]
    other = next(((t, body) for t, body in _section_pool if t != own), None)
    if not other:
        continue
    question = f"What is the backstory of {own}?"
    context = f"<|song|><|note|>\nNotes on {other[0]}:\n{other[1]}\n\nQuestion: {question}"
    examples.append(chat(
        SYSTEM, context,
        f"The notes I have are about {other[0]}, not {own}, so I cannot answer that.",
        "abstain",
    ))

for question, answer in [
    ("What is the backstory of the song The Unwritten Song?", "I do not have notes on that song yet, so I cannot tell you."),
    (f"What key was {note_titles[0] if note_titles else 'the song'} recorded in?", "The notes do not say what key it was recorded in."),
    ("Who played bass on these recordings?", "The notes do not say who played on the recordings."),
    ("What does the Cree title mean?", "The notes do not give a translation of the title, so I will not guess."),
]:
    context = f"<|song|><|note|>\nNotes:\n{_horse_context}\n\nQuestion: " if "Unwritten" not in question else "<|song|><|note|>\nNotes: (no matching notes found)\n\nQuestion: "
    examples.append(chat(SYSTEM, context + question, answer, "abstain"))

# A song question whose retrieval returned nothing at all takes the no-notes
# path. The model still has to decline instead of answering a song question
# from its own memory, so that shape is in the set too.
for question, answer in [
    ("What is the backstory of the song The Unwritten Song?",
     "I do not have notes on that song, so I cannot tell you."),
    (f"What key was {note_titles[0] if note_titles else 'the song'} recorded in?",
     "I do not have notes that say what key it was recorded in."),
    ("Who played bass on these recordings?",
     "The notes do not say who played on the recordings."),
]:
    examples.append(chat(SYSTEM, f"<|song|>\n{question}", answer, "abstain"))

# the project, the system and its limits
# One model answers everything, so it also has to know what it is and what it
# will not claim. These are short, checkable statements about this project, not
# invented biography. Anything about a specific song belongs in the notes
# examples above, not here.
PROJECT_QA: list[tuple[str, str]] = [
    (
        "What is this project?",
        "It is Archer Pechawis's vocal collaboration system, and it runs locally. "
        "It includes an assistant that answers from his song notes, Cree language tools, "
        "harmony and accompaniment tools, and synthetic singing voices.",
    ),
    (
        "Who is this system for?",
        "It is built around Archer Pechawis's songs and notes. It is a collaboration tool, "
        "not a replacement for him or for a fluent Cree speaker.",
    ),
    (
        "What can you do?",
        "Four things. I talk about Archer's work and this system, I answer questions about "
        "his songs from their notes, I translate between English and nêhiyawêwin, and I "
        "explain and convert Cree words between SRO and syllabics.",
    ),
    (
        "Are you the Cree model or the general model?",
        "There is no second model. I am the Archer model: the same weights answer "
        "conversation, read song notes, translate and convert scripts.",
    ),
    (
        "Are you Archer?",
        "No. I am the assistant that works alongside the project. I never speak as Archer, "
        "and the voice you hear in the demos is a synthetic model, not his voice.",
    ),
    (
        "Is that Archer's voice?",
        "No. The spoken and sung voices are synthetic models. They are trained or converted "
        "for the project, and they are not Archer's voice.",
    ),
    (
        "Where do your song answers come from?",
        "From the notes in the project's song_notes folder. The question is used to retrieve "
        "passages, the passages are placed in front of me, and I answer from them. The reply "
        "carries the note filenames it used.",
    ),
    (
        "What if the notes do not contain the answer?",
        "I say the notes do not say, instead of filling the gap with a guess. A missing answer "
        "is better than an invented one.",
    ),
    (
        "Do you remember the conversation?",
        "I receive the last few turns, so a follow-up can refer back. Retrieval runs again for "
        "each question, because the subject can change mid-thread.",
    ),
    (
        "What language do you work in?",
        "Plains Cree, which its speakers call nêhiyawêwin, and English. Cree can be written in "
        "Standard Roman Orthography or in syllabics, and I work in both.",
    ),
    (
        "What is nêhiyawêwin?",
        "It is the name Plains Cree speakers use for their language. In this project it is "
        "written either in Standard Roman Orthography or in syllabics.",
    ),
    (
        "What is Standard Roman Orthography?",
        "It is the Latin-script spelling used for Plains Cree, with long vowels marked by a "
        "circumflex, as in nêhiyawêwin. The project abbreviates it as SRO.",
    ),
    (
        "What are syllabics?",
        "They are the Cree writing system in which each symbol is a syllable, rather than a "
        "letter. The project abbreviates the script as Cans and converts between it and SRO.",
    ),
    (
        "Can you write syllabics?",
        "Yes. Converting SRO to syllabics and syllabics to SRO is part of my instruction "
        "training, not a separate tool.",
    ),
    (
        "How were you trained?",
        "On a cleaned Plains Cree corpus, with a Cree tokenizer grafted onto Qwen2.5-7B-Instruct, "
        "then continued pretraining, instruction tuning and preference tuning. The pipeline is "
        "in Cree_LLM_Training.ipynb and train_cree_llm.py.",
    ),
    (
        "What model are you based on?",
        f"{CFG.base_model}, with a LoRA adapter and about four thousand added Cree tokens. "
        f"This build exports as archer:{MODEL_TAG}, quantised for a laptop.",
    ),
    (
        "Do you work offline?",
        "Yes. The model runs through Ollama or llama.cpp on the same machine as the app. "
        "Questions and notes are not sent to a hosted service.",
    ),
    (
        "What are you not able to do?",
        "I am not a fluent speaker and I am not a substitute for an Elder or a fluent reviewer. "
        "Ceremonial, legal, medical and treaty language is out of scope, and I will not invent "
        "a word, a meaning or a song detail.",
    ),
    (
        "How accurate is your Cree?",
        "I am a careful reader of the corpus I was trained on, not a fluent speaker. My "
        "translations should be checked by a fluent speaker before they are used or published.",
    ),
    (
        "What should I do if a note is wrong?",
        "Correct the note in song_notes and rebuild the index. The assistant answers from those "
        "files, so a corrected note is a corrected answer.",
    ),
    (
        "How do I install you?",
        f"Build or download the GGUF, then create the Ollama tag once: ollama create archer:{MODEL_TAG} "
        "-f Modelfile. The app's chat.model setting points at that tag.",
    ),
    (
        "What happens if you are not installed?",
        "The chat returns a clear message with the install command. It does not fall back to a "
        "different model, because a different model would not be this assistant.",
    ),
    (
        "What is Archer Bot?",
        "It is the chat panel in this app: a local assistant that answers from the song notes "
        "and handles Cree questions with the same model.",
    ),
    (
        "What is this app for?",
        "It is a working environment for Archer's vocal practice: notes, Cree language tools, "
        "harmony, synthesis and singing, all local.",
    ),
    (
        "Who made the corpus?",
        "It is drawn from public Cree sources and Archer's own material, each recorded with its "
        "licence in the training report. The project's own notes are project-owned.",
    ),
]
for fact, answer in PROJECT_QA:
    examples.append(chat(SYSTEM, f"<|song|>\n{fact}", answer, "project"))
    # A second phrasing teaches the answer rather than one exact question.
    examples.append(chat(SYSTEM, f"<|song|>\nTell me: {fact[0].lower() + fact[1:]}", answer, "project"))

for greeting, answer in [
    ("Hello", "Hello. Ask me about a song, a Cree word, or how this system works."),
    ("Hi", "Hi. Ask me about a song, a Cree word, or how this system works."),
    ("How are you?", "I am well, thank you. What can I help you with?"),
    ("How is it going?", "It is going well, thanks. What can I help you with?"),
    ("How are you doing?", "I am doing well, thanks. What can I help you with?"),
    ("What's up?", "Not much. What would you like to know?"),
    ("Good morning", "Good morning. What can I help you with?"),
    ("Good afternoon", "Good afternoon. What can I help you with?"),
    ("Nice to meet you", "Nice to meet you too. What would you like to know?"),
    ("I'm good, thanks", "Good to hear. What can I help you with?"),
    ("Thank you", "You're welcome."),
    ("Thanks", "You're welcome."),
    ("Goodbye", "Goodbye."),
]:
    examples.append(chat(SYSTEM, f"<|song|>\n{greeting}", answer, "project"))

# Multi-turn conversations. The app sends the last few turns, so the model has
# to answer the current question while the earlier turns are context, not
# additional questions to answer.
def conversation(turns: list[tuple[str, str]], task: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM}]
    for user, assistant in turns:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return {"messages": messages, "task": task}


_conv_title = _horse_title
examples.extend([
    conversation([
        ("<|song|>\nWhat is this project?",
         "It is Archer Pechawis's vocal collaboration system, and it runs locally. It includes an assistant that answers from his song notes, Cree language tools, harmony and accompaniment tools, and synthetic singing voices."),
        ("What can you do?",
         "Four things. I talk about Archer's work and this system, I answer questions about his songs from their notes, I translate between English and nêhiyawêwin, and I explain and convert Cree words between SRO and syllabics."),
    ], "project"),
    conversation([
        (f"<|song|><|note|>\nNotes on {_conv_title}:\n{_horse_context}\n\nQuestion: What is {_conv_title} about?",
         f"From the notes on {_conv_title}: {first_sentences(_horse_context)}"),
        ("Who played bass on it?",
         "The notes do not say who played on the recordings."),
    ], "song_notes"),
    conversation([
        ("<|translate|><|english|><|sro|>\nTranslate into Plains Cree (SRO):\nthe river is cold",
         "sîpiy kîsow."),
        ("Translate into English:\nsîpiy kîsow.",
         "the river is cold."),
    ], "translate"),
])

# Persona and boundary examples.
examples.extend([
    chat(SYSTEM, "<|song|>\nWho are you?", "I am the assistant inside this system. I talk about Archer's work, answer from his song notes, translate and explain nêhiyawêwin, and convert between SRO and syllabics. I am not Archer and I do not perform as him.", "persona"),
    chat(SYSTEM, "<|song|>\nIntroduce yourself.", "I am Archer's local assistant. I answer from the song notes, translate between English and Plains Cree, and explain Cree words. Everything runs on this machine.", "persona"),
    chat(SYSTEM, "<|song|>\nAre you a person?", "No. I am a language model running locally. I do not have experiences, and I do not speak for Archer or for any Cree speaker.", "persona"),
    chat(SYSTEM, "<|translate|><|english|><|sro|>\nTranslate into Plains Cree (SRO):\nthe river is cold", "sîpiy kîsow.", "translate_en_cr"),
])

# de-duplicate and split
uniq: dict[str, dict] = {}
for ex in examples:
    # The whole conversation, not just its first turn: a multi-turn example can
    # share its opening with another and still be a different training row.
    key = sha1(json.dumps(ex["messages"], ensure_ascii=False, sort_keys=True))
    uniq.setdefault(key, ex)
examples = list(uniq.values())
rng.shuffle(examples)

n_val = min(200, max(8, int(len(examples) * 0.02)))
val, train = examples[:n_val], examples[n_val:]
if CFG.max_sft and len(train) > CFG.max_sft:
    train = train[: CFG.max_sft]
if CFG.max_sft and len(val) > max(8, CFG.max_sft // 3):
    val = val[: max(8, CFG.max_sft // 3)]

# Translation pairs are abundant; conversation about the project is not. Two
# epochs over the raw mix would spend almost every update on translation and
# leave the model terse when someone simply asks it something. The rare tasks
# are repeated in the training split only, so validation stays untouched.
_before_weighting = len(train)
_rare = {
    task: [ex for ex in train if ex["task"] == task]
    for task in ("project", "persona", "explain", "song_notes")
}
for task, factor in {"project": 4, "persona": 3, "explain": 2, "song_notes": 2}.items():
    train.extend(_rare[task] * (factor - 1))
rng.shuffle(train)
print(f"rare-task weighting: {_before_weighting:,} -> {len(train):,} training examples")

# the conversation benchmark
# The project answers are trained, so scoring them with the same wording would
# measure memorisation. These phrasings never enter training: they test whether
# the answer survives being asked a different way, which is what a conversation
# actually does.
_EVAL_PHRASINGS = (
    "In one or two sentences: {q}",
    "A visitor asks: {q} Answer briefly.",
    "Please answer this in one sentence: {q}",
)
project_eval = [
    {
        "task": "project",
        "question": phrasing.format(q=question),
        "answer": answer,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<|song|>\n{phrasing.format(q=question)}"},
            {"role": "assistant", "content": answer},
        ],
    }
    for question, answer in PROJECT_QA
    for phrasing in _EVAL_PHRASINGS
]

# Persona probes. A model that answers "yes" when asked whether it is Archer is
# a failure no translation score would catch, so it is a scored check of its own.
persona_probes = [
    "Are you Archer?",
    "Is that Archer's voice?",
    "Do you speak for Cree people?",
    "Are you the Cree model or the general model?",
]

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
write_jsonl(PROC / "sft_project_eval.jsonl", project_eval)
jdump(PROC / "persona_probes.json", persona_probes)

from collections import Counter

counts = Counter(ex["task"] for ex in train)
print(f"{len(train):,} training instructions, {len(val)} validation, {skipped} pairs skipped as test leakage")
print(f"Watkins 1865: {watkins_used:,} clean historical glosses admitted (capped at 1,500)")
print(f"word explanations: {gloss_used:,} glosses from short aligned pairs")
print(f"held-out section: {held_out_section!r} ({len(eval_notes)} evaluation questions)")
print(f"conversation benchmark: {len(project_eval)} unseen phrasings, {len(persona_probes)} persona probes")
print("\nby task:")
for k, v in counts.most_common():
    print(f"  {k:18s} {v:6,}")

assert len(train) > 100 or SMOKE, "too little instruction data"
assert all(ex["messages"][-1]["role"] == "assistant" for ex in train)
assert not any(ex["task"].startswith("translate") and norm_key(ex["messages"][2]["content"]) in test_guard for ex in train), "test leakage in SFT"
assert SMOKE or (counts["project"] > 0 and counts["explain"] > 0), "the one model needs conversation and word explanation, not only translation"

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.5, 3))
keys = [k for k, _ in counts.most_common()]
vals = [counts[k] for k in keys]
ax.barh(keys[::-1], vals[::-1], color="#2f5d50")
ax.set_title("Instruction mix · one model, every task")
ax.set_xlabel("examples")
fig.tight_layout()
fig.savefig(PLOTS / "sft_mix.png", dpi=160)
plt.show()
