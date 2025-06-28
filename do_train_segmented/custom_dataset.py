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
  def __init__(self, phase="segmented_puncted"):  # puncted or segmented_puncted
    self.audio_path = "../featurize/feats/"
    self.chunk_file = "../text_chunks/segmented_puncted_train_chunks.jsonl"
    self.chunks = []
    with open(self.chunk_file, "r") as injs:
      for line in injs:
        chunk = json.loads(line)
        self.chunks.append(chunk)

    self.tokenizer_path = "../models/whisper_puncted_segmented" 
    self.tokenizer = WhisperTokenizerFast.from_pretrained(self.tokenizer_path)
    #self.tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-base")

  def __len__(self):
    return len(self.chunks)

  def __getitem__(self, index):

    chunk = self.chunks[index]

    transcript = chunk["text"]
    start_time = chunk["start_time"]
    end_time = chunk["end_time"]
    chunk_id = chunk["chunk_id"]

    l_etime = chunk["lend_time"]

    m_stime = chunk["mstart_time"]
    m_etime = chunk["mend_time"]

    r_stime = chunk["rstart_time"]
    r_etime = chunk["rend_time"]

    features_path = self.audio_path + "chunk" +  chunk_id + ".npz"

    if not os.path.isfile(features_path) or transcript.count("|") != 2:
        index = index // 3000
        chunk = self.chunks[index]

        transcript = chunk["text"]
        start_time = chunk["start_time"]
        end_time = chunk["end_time"]
        chunk_id = chunk["chunk_id"]

        l_etime = chunk["lend_time"]

        m_stime = chunk["mstart_time"]
        m_etime = chunk["mend_time"]

        r_stime = chunk["rstart_time"]
        r_etime = chunk["rend_time"]

        features_path = self.audio_path + "chunk" +  chunk_id + ".npz"

    trans_length =  end_time - start_time 
    trans_length = trans_length / 1000
    end_time = round_down_to_nearest_02(trans_length)
    end_time_token = f"<|{end_time:.2f}|>"

    m_stime = m_stime - start_time
    m_stime = round_down_to_nearest_02(m_stime/1000)
    mst_token = f"<|{m_stime:.2f}|>"

    m_etime = m_etime - start_time
    m_etime = round_down_to_nearest_02(m_etime/1000)
    met_token = f"<|{m_etime:.2f}|>"

    if l_etime:
      l_etime = l_etime - start_time
      l_etime = round_down_to_nearest_02(l_etime/1000)
      let_token = f"<|{l_etime:.2f}|>"

    if r_stime:
      r_stime = r_stime - start_time
      r_stime = round_down_to_nearest_02(r_stime/1000)
      rst_token = f"<|{r_stime:.2f}|>"
    else:
      rst_token = met_token

    if r_etime:
      r_etime = r_etime - start_time
      r_etime = round_down_to_nearest_02(r_etime/1000)
      ret_token = f"<|{r_etime:.2f}|>"
    else:
      ret_token = met_token


    #print("TRANS")
    #print(transcript)
    left, mid, right = transcript.split("|")
    left = "<|left|>" + "<|0.00|>" +   left.strip() + " "
    if l_etime:
        left += let_token + let_token
    else:
        left += "<|0.00|><|0.00|>"  # left is empty

    mid = "<|mid|>" + "<|startoftranscript|>" + "<|notimestamps|>"  + mst_token + mst_token + mid.strip() + " " + met_token + met_token + "<|endoftext|>"
    right = "<|right|>" + rst_token + rst_token + right.strip() + ret_token + ret_token
    windowed_transcript = left+mid+right
    #windowed_transcript = windowed_transcript.replace("  ", " ")

    #transcript = "<|0.00|>" + windowed_transcript + end_time_token + end_time_token
    input_ids = self.tokenizer(windowed_transcript, truncation=True, max_length=256).input_ids
    input_ids = input_ids[2:-1] 
    #print(input_ids)
    #print(windowed_transcript)
    #print("------------")

    features = np.load(features_path)
    features = features["input_features"]

    item = {
        "labels": input_ids,
        "input_features": np.squeeze(features, axis=0)
    }
    return item




'''
pdataset = PunctedDataset("segmented_puncted")
i1 = pdataset[0]
for i in range(43300):
      print(i)
      i1 = pdataset[i]

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

