import re

input_file = "financial_training_corpus.txt"
output_file = "financial_training_corpus_clean.txt"

max_len = 1000   # characters per line

with open(input_file, encoding="utf8") as f:
    text = f.read()

# split by sentence-like punctuation
chunks = re.split(r'(?<=[\.\?\!])\s+', text)

final_lines = []

for chunk in chunks:
    chunk = chunk.strip()

    if len(chunk) == 0:
        continue

    # further split long chunks
    while len(chunk) > max_len:
        final_lines.append(chunk[:max_len])
        chunk = chunk[max_len:]

    final_lines.append(chunk)

with open(output_file, "w", encoding="utf8") as f:
    for line in final_lines:
        f.write(line + "\n")

print("Done cleaning corpus")
print("Total lines:", len(final_lines))