## 1. The corpus

A language model is a mirror of its text, so the first job is deciding what goes in.

The notebook draws on six kinds of material. There is the EdTeKLA collection from the University of Alberta, which is the most carefully curated Plains Cree corpus available and also the most unevenly licensed: some subcollections are CC0, some are CC BY, and some are NonCommercial NoDerivatives. The loader reads the nearest README for each text and refuses anything whose licence forbids adapted works. There is the Plains Cree Figurative Language corpus, 10,847 sentence pairs from Leonard Bloomfield's 1934 texts with English translations, which is the backbone of the translation data.

There is the contemporary web: Cree Wikipedia, FineWeb-2 (192 documents), FinePDFs (733 documents and 1.19 million words taken from PDF text layers, the single largest source), FineTranslations (582 aligned chunks), GlotCC and GlotStoryBook. There is the public-domain past: 14 books from the Internet Archive, 612,000 words, including Bloomfield's Plains Cree Texts and Sacred Stories of the Sweet Grass Cree. Each 700-word window is checked against the GiellaLT lexicon, the English stopword rate and the Cree letter ratio, so a syllabic scan that OCR turned into noise is rejected instead of being counted as text.

There is reference material: Tatoeba's Cree sentences, 64 Plains Cree phrase recordings, and GiellaLT's public texts. There is Archer's own material: the project's aligned pairs, the song notes, and the 1865 Watkins dictionary, kept under its own historical tag. Finally there is a folder you can drop your own texts into.

The complete run accepts 12,743 documents and 2.02 million raw words, which cleaning reduces to 1,124,146 training words across 66,071 sentences, 10,934 training pairs and 203 held-out pairs. That is the largest honest Plains Cree corpus this notebook can assemble without scraping a copyrighted site or inventing sentences.

Every item is written with its source, licence, dialect and script. Nothing is silently merged into an anonymous blob, because a model card that cannot say where its Cree came from is not a model card.

The cell below downloads what it can and records what it could not reach. A missing source never stops the run and never gets replaced with invented text.
