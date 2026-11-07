# Automatic language identification, and what replaced the word list

The panel receives three kinds of message: English, Cree, and English with a Cree
sentence inside it. Each needs a different answer, and until this pass the app
guessed which was which by looking for one of about fifty hand-picked Cree words
or a circumflex. That is not language identification. It misses every Cree word
nobody typed into the list, fires on an English sentence containing one loanword,
and cannot express "this part is Cree and that part is English" at all.

This records what the literature says about doing it properly, what was applied,
what was rejected, and what the replacement measures.

---

## 1. What production systems do

| paper | finding | consequence here |
| --- | --- | --- |
| **fastText lid.176**, Joulin et al., 2016 ([link](https://fasttext.cc/docs/en/language-identification.html)) | Language identification is a linear classifier over character n-grams and word n-grams, trained on labelled text. The distributed model covers 176 languages in 917 kB and runs in microseconds per document. Character n-grams, not words, are what make it work on inflectional and polysynthetic languages. | This is the shape of the replacement: a linear model over character n-grams, with the weights exported so runtime scoring is a dot product. |
| **GlotLID**, Kargaran et al., Findings of EMNLP 2023 ([link](https://aclanthology.org/2023.findings-emnlp.410.pdf)) | A fastText model trained on 1,832 languages. The paper's real contribution for this project is the warning: low-resource LID fails through wrong corpus metadata, leakage from high-resource languages, and confusion between closely related languages (Plains Cree against English and against other Algonquian languages). Uniform accuracy hides this; skewed test sets expose it. | Two decisions follow. The classifier is trained on this project's own Cree, not on a downloaded model whose Cree slice cannot be checked. And the benchmark is reported per family, not as one average, because the average is what hides the failure. |
| **Understanding and Mitigating Language Confusion in LLMs**, EMNLP 2024 ([link](https://aclanthology.org/2024.emnlp-main.380.pdf)) | Language confusion is measured at line level and at word level separately. Off-the-shelf tools do not do word-level LID well, and LLMs themselves reach only 79 to 86 F1 on word-level code-switching, too low to use as an automatic judge. | The app needs word level anyway, for "this word is Cree and that sentence is English". It is done with a second, separately fitted classifier rather than by asking the chat model what language it is speaking. |
| **Multilingual Identification of English Code-Switching**, VarDial 2024 ([link](https://aclanthology.org/2024.vardial-1.14)) | Token-level code-switching identification trained across many corpora generalises to unseen language pairs better than pair-specific systems. | Word-level decisions are made by a general character model, not by a Cree-then-English rule chain. |
| **Improving LID for Code-Switched Utterances with Integer Linear Programming**, Findings of EMNLP 2026 ([link](https://aclanthology.org/2026.findings-emnlp.1429.pdf)) | MaskLID plus constraints improves code-switched identification by up to 100% on several benchmarks, and shows why naive fastText fails on mixed input: it labels a whole utterance from whichever language dominates it. | This is exactly the failure that made an English question quoting a Cree sentence read as Cree. The fix here is a span rule: a run of Cree words inside an English frame is mixed, one quoted word is a mention. |
| **Subword-Level LID for Intra-Word Code-Switching**, NAACL 2019 ([link](https://aclanthology.org/N19-1201.pdf)) | Switching can happen inside a word, and morpheme-level modelling beats word-level for those cases. | Not applied: intra-word switching is not a pattern this panel receives, and the morphology-aware tokenizer work is the place where sub-word Cree is handled. Recorded so it is a decision, not an oversight. |

The pattern across all of them: character n-grams, a linear classifier, labelled
text from the language itself, and word-level decisions for mixed input. None of
them use a vocabulary list, and none of them ask the generator what language it
is reading.

## 2. What was built

`chat/language_id.py` trains two classifiers and ships their weights as
`chat/language_id_model.json` (841 kB):

* a **sentence model** over character 2-grams to 5-grams, 60,000 features;
* a **word model** over character 1-grams to 4-grams, 30,000 features, with
  stronger regularisation, because a single word gives very few n-grams to fit.

The word model gets single characters on purpose. English uses b, d, f, g, j, l,
q, r, v, x and z; Cree does not. A four-letter word with a d in it is decided by
a feature the 2-gram model never saw, and adding 1-grams moved held-out word
accuracy from 0.727 to 0.771 and fixed "for", "does" and "poles" being read as
Cree.

`language_of(text)` returns a label (`cr`, `en`, `mixed`, `und`), the posterior,
the share of decided words that are Cree, and the per-word reading. `mixed`
requires at least two decided Cree words, at least two decided English words, and
an unbroken run of at least two Cree words. One Cree word in an English question
is a mention, not a mixed message: "what does maskwa mean" is English and is
answered in English.

The router uses it in three places:

* `looks_like_cree` and `script_of` delegate to the identifier, so a Cree word
  nobody listed is still read as Cree, and the Latin branch of `script_of` no
  longer depends on a vocabulary.
* a message the identifier reads as Cree is answered in Cree, through the
  English-then-translate path the notes answers already used.
* `_reliable_cree` now asks the identifier whether a generated reply is Cree
  instead of checking it against a list of English function words, so a reply
  that is half English is caught without a list of the words to look for.

Nothing in the app reads a Cree word list. The lists that remain are the
dictionary used to answer "what does this word mean", which is data and is the
answer itself, and the benchmark's English-marker filter, which exists to throw
mislabeled rows out of the test set.

## 3. Measured

Trained on 26,115 rows from this project: aligned pairs, corpus sentences,
Watkins headwords, and the project's English prose. Held out is a hash-based
tenth, so no row can appear in both halves.

| measure | value |
| --- | --- |
| held-out sentence accuracy | 0.996 |
| held-out sentences excluded as mislabeled residue | 618 of 1,542 |
| old diacritic rule on the same rows | 0.600 |
| first attempt, multinomial naive Bayes | 0.603 |
| held-out single-word accuracy | 0.771 |
| held-out marked Cree word accuracy | 1.000 |

The naive Bayes number is kept in the record because it is the useful one: the
n-gram features were identical and the score was 0.603, which is worse than
nothing. The scorer was the problem, and replacing it with the discriminative
linear model fastText uses took the same features to 0.996.

On the generated benchmark (`data/eval/language_tests.jsonl`, 4,000 rows):

| family | n | language | task | answer language |
| --- | --- | --- | --- | --- |
| cree_sentence | 500 | 0.98 | 1.00 | 1.00 |
| english_sentence | 500 | 1.00 | 1.00 | 1.00 |
| cree_word | 500 | 1.00 | 1.00 | 1.00 |
| english_word | 500 | 1.00 | 1.00 | 1.00 |
| note_question | 500 | 1.00 | 1.00 | 1.00 |
| translate_request | 500 | 1.00 | 1.00 | 1.00 |
| word_question | 500 | 1.00 | 1.00 | 1.00 |
| code_switch | 500 | 1.00 | 1.00 | 1.00 |
| **overall** | **4,000** | **0.998** | **1.000** | **1.000** |
| old diacritic rule, same rows | 4,000 | 0.750 | | |

The two remaining errors are the same row repeated: a Cree sentence containing a
personal name is read as mixed because the name is an English word. That is a
defensible reading, not a routing failure, and the answer language is still
correct.

The benchmark pools are small after filtering: 63 held-out Cree sentences, 72
Cree words, 115 English sentences and 26 English words, cycled to 500 rows each.
That is enough to test the rules and the identifier, and it is not enough to
claim a confidence interval. The next step is a larger held-out pool, which means
more Cree text, which is the same corpus gate the training pipeline already has.

## 4. What was rejected

* **Downloading GlotLID or fastText lid.176.** The app runs offline on the user's
  machine, the models are trained for a 1,832-language or 176-language decision
  and not for this distinction, and GlotLID's own paper documents contaminated
  low-resource slices. A 841 kB model trained on this corpus is checkable; a
  126 MB download whose Cree slice is unverified is not.
* **Asking the chat model what language the input is.** The EMNLP 2024 paper
  measures LLM word-level code-switching detection at 79 to 86 F1, and every
  call costs a generation on a CPU that already takes seconds per reply.
* **A CRF or neural token-level tagger.** It needs token-labelled code-switched
  training data, which this project does not have, and the linear word model
  already reaches 1.000 on marked Cree words.
* **Introducing the chat model's own output language check into the router
  loop.** The reply check stays where it was, as a guard on generated Cree, and
  does not feed back into detection.

## 5. Limits worth stating

* Unmarked Cree orthography (Watkins, 1865) is genuinely close to English
  spelling. The word model scores 0.771 across all held-out words and 1.000 on
  marked forms; the gap is the orthography, not the model. It is reported rather
  than averaged away.
* The sentence model is trained mostly on short sentences. A long paragraph with
  one Cree quote will be read as mixed, which is correct, but a long paragraph of
  unmarked Cree may be read as mixed too.
* Syllabics are identified by the Unicode block, which is a property of the
  writing system and not a word list. That decision has no learned component and
  does not need one: any character in that block is syllabic writing.
