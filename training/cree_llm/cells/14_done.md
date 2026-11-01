## What just happened

You now have a Cree model that was built from a ledgered corpus, tokenised for the language rather than for English, trained in three stages, and scored against its own base model on data it never saw.

The files that matter:

| File | What to do with it |
|---|---|
| `out/export/archer-cree-<size>.Q4_K_M.gguf` | copy to `assets/cree_llm/` and load with Ollama |
| `out/export/Modelfile` | `ollama create archer-cree:<size> -f Modelfile` |
| `out/report_card.md` | publish beside the model; it lists every source and licence |
| `out/creebench_report.json` | the evidence behind every number |
| `out/plots/` | the figures used in the report |

### A note on what this model is not

It is not a fluent speaker, and it should not be presented as one. It is a careful reader with a good memory for this corpus, a tokenizer that respects the spelling, and an instruction set that rewards saying "the notes do not say". Every translation that leaves the app should be reviewable by a fluent speaker, and the report card says so.

### The next run

The recipe is built to improve with data, not with parameter count. The highest-value additions are consented contemporary Cree text, more song notes, and corrections from fluent speakers. Drop them into the upload folder, rerun from the corpus cell, and the same evaluation will say whether they helped.
