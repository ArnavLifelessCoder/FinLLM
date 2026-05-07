import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("finance_tokenizer.model")

tests = [
    "Apple revenue increased 12% year over year",
    "EBITDA margin expanded significantly",
    "Company filed its 10-K report with the SEC",
    "NASDAQ index declined due to macroeconomic pressure",
    "Operating cash flow reached $5.2 billion"
]

for t in tests:
    print("\nTEXT:", t)
    print("TOKENS:", sp.encode(t, out_type=str))