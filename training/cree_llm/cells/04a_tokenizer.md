## 3. The tokenizer

This is the highest-leverage change in the whole notebook.

A tokenizer trained mostly on English has never seen a word like `nêhiyawêwin` often enough to give it a single piece. It splits it into fragments, and every fragment is a chance to lose a circumflex or to spend context on a boundary that carries no meaning. Measure it and the cost is obvious: several tokens per Cree word, where a language-aware tokenizer needs closer to one and a half.

The fix is a SentencePiece BPE model trained on the Cree corpus itself. Its pieces are filtered before they are added to the base vocabulary: no space markers, no control symbols, no bare punctuation, nothing shorter than two characters. The survivors are grafted onto the base tokenizer as ordinary tokens, and the next cell initialises their embeddings from the pieces they replace. A new token for `nêhiyawêwin` starts as the average of the embeddings for `n`, `ê`, `hi`, `yaw` and so on, so the model can already read the word on the first step of training.

The audit cell reports fertility (tokens per word), characters per token, and the 95th percentile, on held-out text only. Training text would flatter any tokenizer.
