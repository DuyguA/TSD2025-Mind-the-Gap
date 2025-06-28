import json

all_chunks_f = "puncted_chunks.json"
outfile = "train_puncted_chunks.jsonl"

all_chunks_f = "segmented_puncted_chunks.json"
outfile = "segmented_puncted_train_chunks.jsonl"

train_chunk_ids = "train_chunks.txt"

with open(all_chunks_f, "r") as allc:
    all_chunks = json.load(allc)


train_ids = open(train_chunk_ids, "r").read().split("\n")
train_ids = [ti for ti in train_ids if ti]

train_chunks = []

for t_id in train_ids:
    chunk = all_chunks[t_id]
    chunk["chunk_id"] = t_id
    train_chunks.append(chunk) 


with open(outfile, "w") as ofile:
  for chunk in train_chunks:
    chunkjs = json.dumps(chunk)
    ofile.write(chunkjs+"\n")

