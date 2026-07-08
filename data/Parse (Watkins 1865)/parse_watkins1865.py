import json
import re
import sys
from pathlib import Path

# abbreviation key in the front matter)
POS_ABBREVIATIONS = [
    "v. t. an.", "v. t. in.", "v. t.", "v. i.", "v. n.", "v. imp.", "v. ref.",
    "v. rec.", "adj. pref.", "adv. pref.", "adj.", "adv.", "n.", "pron.",
    "prep.", "conj.", "interj.", "interr.", "poss.", "pers.",
]
# Longest-first so "v. t. an." matches before the shorter "v. t."
POS_ABBREVIATIONS.sort(key=len, reverse=True)
POS_PATTERN = "|".join(re.escape(p) for p in POS_ABBREVIATIONS)

# Matches: "Headword, pos. rest-of-entry" at the start of a line
ENTRY_START = re.compile(
    rf"^([A-Z][A-Za-z'’\- ]{{1,40}}?),\s*({POS_PATTERN})\s*(.*)$"
)


def dehyphenate(raw_text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", raw_text)
    text = re.sub(r"(?<![.\n])\n(?!\n)", " ", text)
    return text


def split_cree_forms(rest: str) -> list:
    # Cut off at the first usage example / cross-reference, which
    # typically starts with 'e.g.', 'See ', or a capital 'He'/'It'
    rest = re.split(r"\be\.\s*g\.|\bSee\b|\. (?:He|It|They) ", rest, maxsplit=1)[0]
    rest = rest.strip().rstrip(".")
    if not rest:
        return []

    raw_forms = [f.strip() for f in rest.split(",") if f.strip()]
    expanded = []
    stem = None
    for form in raw_forms:
        if form.startswith("-"):
            if stem:
                expanded.append(stem + form[1:])
            continue
        # Track the stem before the first internal hyphen for suffix expansion
        if "-" in form:
            stem = form.split("-")[0]
            expanded.append(form.replace("-", ""))
        else:
            stem = form
            expanded.append(form)
    return expanded


def parse_entries(text: str):
    text = dehyphenate(text)
    lines = text.split("\n")
    entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = ENTRY_START.match(line)
        if match:
            headword, pos, rest = match.groups()
            # Entries can continue on the following lines until the next
            # entry start or a blank line
            j = i + 1
            while j < len(lines) and lines[j].strip() and not ENTRY_START.match(lines[j].strip()):
                rest += " " + lines[j].strip()
                j += 1
            forms = split_cree_forms(rest)
            if forms:
                entries.append({
                    "english": headword.strip().lower(),
                    "pos": pos.strip(),
                    "cree_forms": forms,
                })
            i = j
        else:
            i += 1
    return entries


def main():
    if len(sys.argv) != 3:
        print("Usage: python parse_watkins1865.py raw_ocr_text.txt output_entries.jsonl")
        sys.exit(1)

    in_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    text = in_path.read_text(encoding="utf-8", errors="replace")
    entries = parse_entries(text)

    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Parsed {len(entries)} entries -> {out_path}")
    print("Spot-check a random sample before trusting this data - OCR errors")
    print("are expected, especially in accented vowels.")


if __name__ == "__main__":
    main()
