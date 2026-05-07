import os
import pandas as pd
from datasets import load_dataset

corpus = []

# ----------------------------
# 1. Financial PhraseBank
# ----------------------------
print("Loading Financial PhraseBank...")

phrasebank = load_dataset(
    "financial_phrasebank",
    "sentences_allagree",
    trust_remote_code=True
)

for row in phrasebank["train"]:
    corpus.append(row["sentence"])

print("After phrasebank:", len(corpus))


# ----------------------------
# 2. Financial News
# ----------------------------
print("Loading financial news...")

news1 = pd.read_csv("news/Combined_News_DJIA.csv")
news2 = pd.read_csv("news/RedditNews.csv")

for col in news1.columns[2:]:
    for text in news1[col].dropna():
        corpus.append(str(text))

for text in news2["News"].dropna():
    corpus.append(str(text))

print("After news:", len(corpus))


# ----------------------------
# 3. Function to load 10-K text
# ----------------------------
def load_10k_folder(folder):

    count = 0

    for root, dirs, files in os.walk(folder):
        for file in files:

            if file.endswith(".txt"):

                path = os.path.join(root, file)

                with open(path, "r", encoding="utf8", errors="ignore") as f:
                    corpus.append(f.read())

                count += 1

    print(f"Loaded {count} files from {folder}")


# ----------------------------
# 4. Load BOTH SEC folders
# ----------------------------
print("Loading SEC filings...")

load_10k_folder("sec_10k")
load_10k_folder("sec_10k_more")

print("After SEC filings:", len(corpus))


# ----------------------------
# 5. Save final corpus
# ----------------------------
print("Saving corpus...")

with open("financial_training_corpus.txt", "w", encoding="utf8") as f:

    for line in corpus:
        line = line.replace("\n", " ")
        f.write(line + "\n")

print("DONE")
print("Total samples:", len(corpus))