from datasets import load_dataset

ner = load_dataset("josephimperial/CebuaNER")
readability = load_dataset("filbench/cebuano-readability")
print(ner)          # check split names and columns!
print(readability)

lines = []

# CebuaNER: join tokens back into sentences, ignore the NER tags
for split in ner.keys():
    for row in ner[split]:
        lines.append(" ".join(row["text"]))   # change "tokens" if the column differs

# Readability: the text is already plain prose
for split in readability.keys():
    for row in readability[split]:
        lines.append(row["text"].replace("\n", " "))

with open("cebuano_corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Wrote {len(lines)} lines")
