## 2. Cleaning, deduplication and the split

Raw corpus text is not training data yet.

Three problems have to be solved before a single token is counted. The first is language: a file labelled Cree can contain English translations, headings, or a sentence that begins in Cree and ends in English. Each sentence is checked with function words and long-vowel marks, not just character ranges, because `the` is made entirely of letters Cree uses. The second is duplication: the same verse, refrain or dictionary example can appear in several collections, and a model that sees it in both train and test will report a score it did not earn. Exact hashes catch the obvious copies; a MinHash index catches the near-copies. The third is the split itself. Sentences are grouped by source document first, then divided, so a text cannot contribute half its paragraphs to training and half to the test set.

The split is 88% train, 7% development, 5% test. After dividing, a second pass checks every development and test sentence against the training set and moves any near-duplicate back into training. That is conservative: it is better to lose a test sentence than to publish a leaked score.

Bag-of-words files are detected by their shape and never used as prose. They still contribute word forms to the lexicon, which is used later to check whether generated Cree is a real word.
