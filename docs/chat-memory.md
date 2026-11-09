# Memory, web search and the learning loop

The assistant now has three things it did not have before: it remembers every
conversation on this machine, it can search the web and answer from the source,
and it turns a correction into a permanent test and a training example.

This page says exactly what is real time and what is not.

## The fast loop: memory, active immediately

Every answered message is written to `data/chat_memory.sqlite3`:

| table | what it holds |
|---|---|
| `turns` | the question, the answer, the exact prompt, the task and the language |
| `feedback` | an up or down vote, and the answer the visitor says was right |
| `facts` | things the visitor explicitly asked the assistant to remember |
| `turns_fts` | an FTS5 index used to recall relevant turns |

Before a question is answered, the store is searched lexically. A correction
outranks a fact, and a fact outranks an ordinary past turn. The best few items
are placed in the prompt above the question, under a heading that says what
they are. A correction is phrased as:

```
When asked "what key was Horse recorded in?", the correct answer is:
The notes say Horse was recorded in D major.
```

That changes the next answer with no training run. It is the same mechanism a
chat product calls memory, and it is the only form of real time learning a 7B
model on a CPU can honestly provide.

The memory is also the conversation history when the client sends none. A
refreshed page, a curl call or a script still gets the previous turn.

### Corrections

A message that begins with "no", "wrong", "that's wrong", "not quite",
"actually", or that says "you said X but Y", is treated as a correction of the
previous answer. It is stored as a down vote with the message as the correction.
A correction can also be sent explicitly to `POST /api/chat/feedback`.

Only an explicit `remember that ...` is stored as a fact. Inferring facts from
ordinary sentences is how a memory becomes confidently wrong about its user.

## The slow loop: the weights

The weights do not change while the app runs. Doing that would need a GPU, a
training step that takes minutes to hours, and a re-export of the GGUF; this
machine has none of the three.

What the correction does produce, immediately, is the data for the next run:

```
data/training/feedback_sft.jsonl    chat-shaped SFT examples
data/training/feedback_dpo.jsonl    {prompt, chosen, rejected} preference pairs
data/eval/regressions.jsonl         one replayable case per correction
```

The SFT and DPO shapes are the ones `training/cree_llm/cells/08_sft_data.py` and
`cells/10_dpo.py` already consume. `chosen` is the visitor's correction and
`rejected` is what the model actually said, so the mistake is trained against
directly rather than described in a comment.

```bash
python tools/consolidate_feedback.py --stats
python tools/consolidate_feedback.py --export
python tools/run_regressions.py            # replay every correction
python tools/run_regressions.py --dry-run  # list them, no model calls
```

`run_regressions.py` exits 1 if any corrected answer has come back, so a new
model can be refused before it is installed. That is what stops the same mistake
from returning silently after a retrain.

## Web search

`POST /api/chat` searches before answering when the message asks for it
("search for ...", "look up ...") or asks about something that changes (news,
prices, weather, dates, this year). `chat.web_search.mode` controls it:

| mode | behaviour |
|---|---|
| `auto` | explicit searches, and fresh-information questions |
| `always` | every chat message |
| `off` | never |

Providers, in order of preference when a key exists:

1. Exa (`EXA_API_KEY`), semantic search built for assistants
2. Tavily (`TAVILY_API_KEY`)
3. Brave (`BRAVE_SEARCH_API_KEY`)
4. DuckDuckGo HTML, no key, the default
5. DuckDuckGo's instant-answer JSON API, when the HTML endpoint serves a bot challenge
6. Wikipedia's search API, as the last keyless resort

The HTML endpoint is currently answering automated requests with a 202
challenge page on this machine, so in practice a keyless search falls through
to the JSON API and Wikipedia. Those cover definitions, names, places and
encyclopedic questions. They do not cover today's news or a live price, which
is what Exa, Tavily or Brave is for. The response says `web_used: false` when
nothing was found, and the model is told to say so rather than guess.

Only result titles, URLs and snippets are used. Result pages are not fetched,
which keeps the feature from becoming an arbitrary URL fetcher. The prompt says
the results are the current evidence, that the source URL must be named, and
that the model must say when they do not contain the answer. The response's
`sources` list carries the URLs, and the panel renders them as links.

A blocked, offline or rate-limited search is an empty list, never an exception.
The model is told the search found nothing instead of being allowed to invent a
result.

## API

| method | path | purpose |
|---|---|---|
| `POST` | `/api/chat` | answer; now returns `turn_id`, `memory_used`, `web_used` |
| `POST` | `/api/chat/stream` | the same answer as server-sent events |
| `POST` | `/api/chat/feedback` | `{session_id, turn_id, rating, correction}` |
| `GET` | `/api/chat/memory` | counts for turns, feedback, corrections, facts |
| `DELETE` | `/api/chat/memory` | delete everything, on this machine only |
| `POST` | `/api/chat/remember` | `{text}` store an explicit fact |
| `GET` | `/api/chat/health` | now includes memory counts and the search mode |

`session_id` is a random id the browser keeps in `localStorage`. It is not a
login and it is not sent anywhere.

## Privacy and failure behaviour

The database is local, untracked, and deleted by `DELETE /api/chat/memory` or
`python tools/consolidate_feedback.py --clear`. Web search sends the query to
the configured provider; with no key that is DuckDuckGo. Nothing else leaves
the machine.

Memory and search are enhancements, so every call catches `sqlite3.Error` and
`requests` failures. A locked database or a dead network degrades the answer; it
does not produce a 500.

## What is still not true

- The model does not update its weights from a conversation. The correction is
  retrieved into the next prompt and exported for the next GPU run.
- Memory recall is lexical, not semantic. It is explainable and works for Cree
  words an English embedding has never seen, but a paraphrase with no shared
  content word will not match.
- A web answer is only as good as the snippets the provider returns. The model
  is instructed not to invent beyond them, and the source list is shown so the
  claim can be checked.
