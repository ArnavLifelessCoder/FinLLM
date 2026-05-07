import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="financial_training_corpus_clean"
    ".txt",
    model_prefix="finance_tokenizer",
    vocab_size=32000,
    character_coverage=1.0,
    model_type="bpe"
)

import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("finance_tokenizer.model")

print(sp.encode("Apple revenue increased 12% year over year", out_type=str))