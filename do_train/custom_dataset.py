import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json, os
from transformers import WhisperTokenizerFast
import math

import math

def round_down_to_nearest_02(time_in_seconds):
    # Round down to the nearest 0.02 increment
    rounded_time = math.floor(time_in_seconds / 0.02) * 0.02
    return round(rounded_time, 2)




class PunctedDataset(Dataset):
  def __init__(self, phase="puncted"):  # puncted or segmented_puncted
    self.audio_path = "../featurize/feats/"
    self.chunk_file = "../text_chunks/segmented_puncted_train_chunks.jsonl"
    self.chunks = []
    with open(self.chunk_file, "r") as injs:
      for line in injs:
        chunk = json.loads(line)
        self.chunks.append(chunk)

    self.tokenizer_path = "../models/whisper_" + phase
    self.tokenizer = WhisperTokenizerFast.from_pretrained(self.tokenizer_path)

  def __len__(self):
    return len(self.chunks)

  def __getitem__(self, index):

    chunk = self.chunks[index]

    transcript = chunk["text"]
    start_time = chunk["start_time"]
    end_time = chunk["end_time"]
    m_stime = chunk["mstart_time"]
    r_stime = chunk["rstart_time"]
    chunk_id = chunk["chunk_id"]
    features_path = self.audio_path + "chunk" +  chunk_id + ".npz"

    if not os.path.isfile(features_path) or transcript.count("|") != 2:
        index = index // 3000
        chunk = self.chunks[index]

        transcript = chunk["text"]
        start_time = chunk["start_time"]
        end_time = chunk["end_time"]
        m_stime = chunk["mstart_time"]
        r_stime = chunk["rstart_time"]
        chunk_id = chunk["chunk_id"]
        features_path = self.audio_path + "chunk" +  chunk_id + ".npz"

    trans_length =  end_time - start_time 
    trans_length = trans_length / 1000
    end_time = round_down_to_nearest_02(trans_length)
    end_time_token = f"<|{end_time:.2f}|>"

    m_stime = m_stime - start_time
    m_stime = round_down_to_nearest_02(m_stime/1000)
    mst_token = f"<|{m_stime:.2f}|>"

    if r_stime:
      r_stime = r_stime - start_time
      r_stime = round_down_to_nearest_02(r_stime/1000)
      rst_token = f"<|{r_stime:.2f}|>"

    left, mid, right = transcript.split("|")
    left = left.lstrip()
    right = right.rstrip()

    if left and left.strip():
      mid = mst_token + mst_token + mid
    if r_stime:
      mid = mid + rst_token + rst_token

    transcript = left+ " " + mid+ " " + right
    transcript = transcript.strip()
    transcript = transcript.replace("  ", " ")

    transcript = "<|0.00|>" + transcript + end_time_token + end_time_token
    #print(transcript)
    input_ids = self.tokenizer(transcript, truncation=True, max_length=256).input_ids
    input_ids = input_ids[:1] + input_ids[2:]

    features = np.load(features_path)
    features = features["input_features"]

    item = {
        "labels": input_ids,
        "input_features": np.squeeze(features, axis=0)
    }
    return item




'''
pdataset = PunctedDataset("puncted")
for i in range(43300):
      print(i)
      i1 = pdataset[i]
      print(i1["labels"])
      print("==================")

i1 = pdataset[0]
print(i1["labels"])
i1 = pdataset[0]
print(i1["input_features"].shape)

maxv = 0
lens = []

print("==================")
lens.sort(reverse=True)

print(lens[:500])
print("==================")
print(lens[1000:1500])

print(i5["input_features"].shape)
'''


