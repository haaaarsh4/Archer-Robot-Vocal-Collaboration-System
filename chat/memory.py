"""Persistent memory for the assistant, stored on the user's own machine.

The model's weights do not change while the app runs. What changes is what it
is given: every turn is written to SQLite, every correction is kept beside the
answer it corrects, and a later question is answered with the relevant facts,
past answers and corrections placed in front of it. That is real-time learning
in the only sense a 7B model on a CPU can honestly offer, and it is also the
data a later training run needs to change the weights themselves.

The database is deliberately small and plain:

* ``turns``      one row per answered message, including the exact prompt
* ``feedback``   up/down votes and the visitor's correction, if they gave one
* ``facts``      things the visitor asked the assistant to remember
* ``turns_fts``  an FTS5 index over the turns, used for recall

Every public method opens its own connection and catches ``sqlite3.Error``:
memory is an enhancement, and a locked or unwritable database must never be the
reason a chat fails.
"""
from __future__ import annotations

import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "chat_memory.sqlite3"

# Function words are not evidence of similarity. Cree particles are short too,
# so only English function words are dropped, and only for scoring, never from
# what is stored or from what is put back into a prompt.
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "him", "his", "how", "i", "if", "in", "is", "it", "its", "me", "my", "no",
    "not", "of", "on", "or", "our", "she", "so", "that", "the", "their",
    "them", "there", "they", "this", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "you", "your",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL DEFAULT 'local',
    user TEXT NOT NULL,
    assistant TEXT NOT NULL,
    prompt TEXT NOT NULL DEFAULT '',
    task TEXT NOT NULL DEFAULT 'chat',
    answer_language TEXT NOT NULL DEFAULT 'en',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, id);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id INTEGER REFERENCES turns(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL DEFAULT 'local',
    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    correction TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_turn ON feedback(turn_id);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL DEFAULT 'user',
    created_at REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    user, assistant, content='turns', content_rowid='id',
    tokenize='unicode61 remove_diacritics 0'
);
CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
    INSERT INTO turns_fts(rowid, user, assistant)
    VALUES (new.id, new.user, new.assistant);
END;
CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, user, assistant)
    VALUES ('delete', old.id, old.user, old.assistant);
END;
CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, user, assistant)
    VALUES ('delete', old.id, old.user, old.assistant);
    INSERT INTO turns_fts(rowid, user, assistant)
    VALUES (new.id, new.user, new.assistant);
END;
"""


@dataclass(frozen=True)
class MemoryHit:
    """One piece of remembered context, ready to be put in a prompt."""

    kind: str          # "correction" | "fact" | "turn"
    text: str
    score: float
    created_at: float = 0.0


@dataclass(frozen=True)
class Turn:
    id: int
    session_id: str
    user: str
    assistant: str
    prompt: str
    task: str
    answer_language: str
    created_at: float


def _terms(text: str) -> list[str]:
    words = re.findall(r"[^\W\d_][^\W_]*", str(text).lower(), flags=re.UNICODE)
    return [w for w in words if len(w) > 1 and w not in _STOP]


def _overlap(query: list[str], text: str) -> float:
    """Share of the query's content words that appear in the text.

    Deliberately lexical: it is explainable, needs no model, works for Cree
    words that an English embedding model has never seen, and does not cost a
    second of CPU before every reply.
    """
    if not query:
        return 0.0
    have = set(_terms(text))
    return sum(1 for word in set(query) if word in have) / len(set(query))


class Memory:
    """The whole store. Construct with a path, or let it use the project's."""

    def __init__(self, path: str | Path | None = None,
                 max_turns: int | None = None):
        chosen = path or os.environ.get("ARCHER_MEMORY_DB") or DEFAULT_DB
        self.path = Path(chosen)
        self.max_turns = int(
            max_turns or os.environ.get("ARCHER_MEMORY_MAX_TURNS") or 50_000
        )

    # ---------------------------------------------------------------- storage
    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        return conn

    def record_turn(self, session_id: str, user: str, assistant: str,
                    prompt: str = "", task: str = "chat",
                    answer_language: str = "en") -> int:
        """Store one exchange and return its id, or 0 if it could not be saved."""
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO turns(session_id, user, assistant, prompt, task,"
                    " answer_language, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session_id or "local", str(user), str(assistant), str(prompt),
                     str(task), str(answer_language), time.time()),
                )
                turn_id = int(cur.lastrowid or 0)
                if turn_id and turn_id % 100 == 0:
                    conn.execute(
                        "DELETE FROM turns WHERE id <= ("
                        "SELECT id FROM turns ORDER BY id DESC LIMIT 1 OFFSET ?)",
                        (self.max_turns,),
                    )
                return turn_id
        except sqlite3.Error:
            return 0

    def record_feedback(self, session_id: str, turn_id: int | None, rating: str,
                        correction: str = "", note: str = "") -> int:
        if rating not in {"up", "down"}:
            return 0
        if turn_id is None:
            last = self.last_turn(session_id)
            turn_id = last.id if last else None
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO feedback(turn_id, session_id, rating, correction,"
                    " note, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (turn_id, session_id or "local", rating, str(correction or ""),
                     str(note or ""), time.time()),
                )
                return int(cur.lastrowid or 0)
        except sqlite3.Error:
            return 0

    def remember(self, text: str, source: str = "user") -> int:
        text = " ".join(str(text).split()).strip()
        if not text:
            return 0
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO facts(text, source, created_at)"
                    " VALUES (?, ?, ?)",
                    (text, source, time.time()),
                )
                if cur.rowcount:
                    return int(cur.lastrowid or 0)
                row = conn.execute(
                    "SELECT id FROM facts WHERE text = ?", (text,)
                ).fetchone()
                return int(row["id"]) if row else 0
        except sqlite3.Error:
            return 0

    # ---------------------------------------------------------------- reading
    def last_turn(self, session_id: str = "local") -> Turn | None:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM turns WHERE session_id = ?"
                    " ORDER BY id DESC LIMIT 1",
                    (session_id or "local",),
                ).fetchone()
                return _turn(row) if row else None
        except sqlite3.Error:
            return None

    def recent(self, session_id: str = "local", limit: int = 6) -> list[Turn]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM turns WHERE session_id = ?"
                    " ORDER BY id DESC LIMIT ?",
                    (session_id or "local", max(1, int(limit))),
                ).fetchall()
                return [_turn(row) for row in reversed(rows)]
        except sqlite3.Error:
            return []

    def recall(self, query: str, limit: int = 3) -> list[MemoryHit]:
        """The most relevant facts, corrections and past exchanges.

        Corrections rank above facts, and facts above ordinary turns, because a
        correction is the one kind of memory that exists specifically to stop a
        mistake being repeated.
        """
        terms = _terms(query)
        if not terms:
            return []
        hits: list[MemoryHit] = []
        try:
            with self._connect() as conn:
                # Corrections are few and precious: score all of the recent ones
                # directly instead of hoping FTS ranks a short correction well.
                rows = conn.execute(
                    "SELECT f.correction, f.created_at, t.user FROM feedback f"
                    " LEFT JOIN turns t ON t.id = f.turn_id"
                    " WHERE f.correction != '' ORDER BY f.id DESC LIMIT 500",
                ).fetchall()
                for row in rows:
                    score = _overlap(terms, f"{row['user'] or ''} {row['correction']}")
                    if score <= 0:
                        continue
                    asked = row["user"] or "an earlier question"
                    hits.append(MemoryHit(
                        "correction",
                        f'When asked "{asked}", the correct answer is:'
                        f" {row['correction']}",
                        2.0 + score,
                        float(row["created_at"] or 0.0),
                    ))

                for row in conn.execute(
                    "SELECT text, created_at FROM facts ORDER BY id DESC LIMIT 500"
                ).fetchall():
                    score = _overlap(terms, row["text"])
                    if score > 0:
                        hits.append(MemoryHit(
                            "fact", row["text"], 1.2 + score,
                            float(row["created_at"] or 0.0),
                        ))

                match = " OR ".join(
                    '"' + term.replace('"', '""') + '"' for term in terms
                )
                try:
                    rows = conn.execute(
                        "SELECT t.*, bm25(turns_fts) AS rank FROM turns_fts"
                        " JOIN turns t ON t.id = turns_fts.rowid"
                        " WHERE turns_fts MATCH ? ORDER BY rank LIMIT 20",
                        (match,),
                    ).fetchall()
                    source = "fts"
                except sqlite3.Error:
                    rows = conn.execute(
                        "SELECT * FROM turns ORDER BY id DESC LIMIT 500"
                    ).fetchall()
                    source = "scan"

                for row in rows:
                    text = f"Earlier, to \"{row['user']}\", you answered: {row['assistant']}"
                    score = _overlap(terms, f"{row['user']} {row['assistant']}")
                    if score <= 0:
                        continue
                    if source == "fts":
                        score += 1.0 / (1.0 + abs(float(row["rank"])))
                    hits.append(MemoryHit(
                        "turn", text, score, float(row["created_at"] or 0.0),
                    ))
        except sqlite3.Error:
            return []

        # One hit per distinct text, best score first. A correction and the turn
        # it corrects say the same thing, and repeating it crowds the prompt.
        best: dict[str, MemoryHit] = {}
        for hit in hits:
            old = best.get(hit.text)
            if old is None or hit.score > old.score:
                best[hit.text] = hit
        ordered = sorted(best.values(), key=lambda h: h.score, reverse=True)
        return ordered[: max(1, int(limit))]

    def feedback_rows(self, limit: int = 5000) -> list[dict]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT f.rating, f.correction, f.note, f.created_at,"
                    " t.user, t.assistant, t.prompt, t.task"
                    " FROM feedback f LEFT JOIN turns t ON t.id = f.turn_id"
                    " ORDER BY f.id DESC LIMIT ?",
                    (max(1, int(limit)),),
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error:
            return []

    def stats(self) -> dict:
        try:
            with self._connect() as conn:
                turns = conn.execute("SELECT COUNT(*) AS n FROM turns").fetchone()["n"]
                feedback = conn.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()["n"]
                corrections = conn.execute(
                    "SELECT COUNT(*) AS n FROM feedback WHERE correction != ''"
                ).fetchone()["n"]
                facts = conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()["n"]
                size = self.path.stat().st_size if self.path.exists() else 0
                return {
                    "db": str(self.path),
                    "bytes": size,
                    "turns": int(turns),
                    "feedback": int(feedback),
                    "corrections": int(corrections),
                    "facts": int(facts),
                }
        except (sqlite3.Error, OSError):
            return {"db": str(self.path), "bytes": 0, "turns": 0,
                    "feedback": 0, "corrections": 0, "facts": 0}

    def clear(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM feedback")
                conn.execute("DELETE FROM turns")
                conn.execute("DELETE FROM facts")
        except sqlite3.Error:
            pass


def _turn(row: sqlite3.Row) -> Turn:
    return Turn(
        id=int(row["id"]),
        session_id=row["session_id"],
        user=row["user"],
        assistant=row["assistant"],
        prompt=row["prompt"],
        task=row["task"],
        answer_language=row["answer_language"],
        created_at=float(row["created_at"]),
    )
