## 5. Instruction data and tuning

The training set is built, not scraped from a chat log. Five kinds of example do five different jobs.

**Translation** is the core skill: English to Cree and Cree to English, with several natural phrasings of each instruction so the model does not learn one magic sentence. **Orthography** pairs SRO with syllabics in both directions, using the same conversion tables as the rest of the project, so the model can write the script rather than merely name it. **Song notes** give the model a passage and a question and ask it to answer from the passage with a citation, which is exactly the shape the app's retrieval layer produces. **Abstention** gives it a passage that does not contain the answer and rewards saying so. **Persona** examples establish that this is a language tool, not Archer and not a performer.

The held-out section of the notes is never trained on. It is used later to test whether the model can read a passage it has not memorised, which is the only kind of note reading that generalises.

Only the assistant's reply contributes to the loss. Training on the whole conversation teaches a model to generate the user's question too, which shows up later as a model that answers by asking.
