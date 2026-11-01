## 4. Continued pretraining

Instruction tuning teaches a model what to say. It does not teach it how a language works. That is what this stage is for: next-token prediction on running Cree, so the model absorbs spelling, suffix patterns and the rhythm of a Cree sentence before it is asked to follow an instruction.

Two details keep it from doing damage. The first is replay: one packed block in ten is English. A small LoRA trained only on Cree can drift far enough that it forgets how to follow an English instruction, which would make the model useless as an assistant. The second is packing: documents are joined with an end-of-sequence token and sliced into full windows, so every position in a batch carries a real token instead of padding.

The adapter is deliberately small. With a corpus of this size, a high-rank full fine-tune would memorise the text and lose the base model's general ability. A rank-64 LoRA over the attention and MLP projections can learn the language's shape while leaving the rest of the model intact.
