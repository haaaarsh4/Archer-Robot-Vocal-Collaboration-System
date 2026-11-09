## 6. CreeBench

A single BLEU score hides everything that matters here.

Translation is scored with chrF++ and BLEU, then with three measures that are specific to a low-resource language. **Validity** asks whether each generated word exists in the 15,000-form lexicon or at least obeys Cree phonotactics; invented words are the failure a speaker notices first. **Code-switch** asks how often English leaks into a Cree answer. **Abstention** asks whether the model declines a question the notes cannot answer, instead of inventing a key, a player or a meaning.

The same prompts go to the base model with its adapters switched off and to the tuned model with them on, so the comparison is paired and the only thing that changes is the training. Perplexity is measured on held-out Cree, and the note-reading task uses the section that was held out of training.

The report also keeps the generated text, not just the scores. A number without an example is not evidence.

## 6b. The language-model checks

A translation score cannot see instability, invented confidence or a forgotten second language, so the cell after CreeBench measures those directly.

**Determinism** asks the model the same question twice with greedy decoding and counts the answers that come back identical. **A prompt ablation** removes the control tokens and scores the same sentences again, which is the only reliable way to say what those prefixes are worth. **Paraphrase consistency** asks three different ways for the same translation and compares the three replies. **Abstention** is measured in both directions, on questions the notes cannot answer and on questions they can, because a model that always declines is as useless as one that never does. **English perplexity** is compared with the adapters on and off, and that is the measurement the English replay during pretraining exists to protect. **Tokenizer fertility** reports what a Cree word costs in tokens. **A syllabics round trip** writes the script and converts it back to SRO. **Calibration** correlates the model's own token confidence with the sentence's chrF++, because a model that is confidently wrong is worse than one that is unsure.

Every check writes its numbers to `llm_tests.json` and merges them into `creebench_report.json`. The charts and a small JSON bundle for the project site are written to `out/web/`, in the site's own palette, so they can be used without restyling.
