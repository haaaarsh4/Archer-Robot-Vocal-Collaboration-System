## 5. Instruction data and tuning

The training set is built, not scraped from a chat log. There is one model, so every job the assistant does is in this one set.

**Translation** is the core language skill: English to Cree and Cree to English, with several natural phrasings of each instruction so the model does not learn one magic sentence. **Orthography** pairs SRO with syllabics in both directions, using the same conversion tables as the rest of the project, so the model can write the script rather than merely name it. **Word explanation** turns the short aligned pairs into a gloss, so "what does this word mean" is answered from the corpus instead of guessed. **Song notes** give the model a passage and a question and ask it to answer from the passage, naming the song, which is exactly the shape the app's retrieval layer produces. **Abstention** gives it a passage that does not contain the answer and rewards saying so. **Project** examples teach it what this system is, what it will not claim, and how it is installed. **Persona** examples establish that it is Archer's assistant, not Archer and not a performer.

The mix is the point. A model tuned only on translation becomes a translator with no conversation; a model tuned only on chat cannot spell Cree. Both are in the same adapter, and the task prefix selects the behaviour without selecting a different model.

The held-out section of the notes is never trained on. It is used later to test whether the model can read a passage it has not memorised, which is the only kind of note reading that generalises.

Only the assistant's reply contributes to the loss. Training on the whole conversation teaches a model to generate the user's question too, which shows up later as a model that answers by asking.
