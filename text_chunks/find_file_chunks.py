import json

fname="corrected_new.jsonl"
test_files = "test_files.txt"

trainfs = open(test_files, "r").read().split("\n")
trainfs = [trainf for trainf in trainfs if trainf]

with open(fname, "r") as infile, open("test_chunks.txt", "w") as ofile:
    for line in infile:
        linejs = json.loads(line)
        chunks = linejs["chunks"]
        filename = linejs["filename"]
        cids = [chunk["global_chunk_number"] for chunk in chunks]
        if filename in trainfs:
            for cid in cids:
                ofile.write(str(cid) + "\n")

