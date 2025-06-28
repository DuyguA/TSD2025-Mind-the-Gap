from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
import librosa
import os
from datasets import load_dataset

pipeline = FlaxWhisperPipline("../do_train/whisper-puncted-timed/89", dtype=jnp.bfloat16, batch_size=16)


from jax.experimental.compilation_cache import compilation_cache as cc

cc.initialize_cache("./jax_cache")


test_names_f = "../text_chunks/test_files.txt"

test_names = open(test_names_f, "r").read().split("\n")
test_names = [ti for ti in test_names if ti]


outdir = "../test_results/whisper-puncted/"
bad_files  = []
for test_name in test_names:
  try:
    audio_path = "../datasets/english/" + test_name + "/audio.ogg"
    audio, original_sr = librosa.load(audio_path, sr=16000)
    audio_input = {"array": audio, "sampling_rate": 16000}
    text = pipeline(audio, stride_length_s=5, chunk_length_s=30, return_boundary=True)
    outfile = outdir + "/" + test_name + ".txt"
    with open(outfile, "w") as ofile:
      ofile.write(text["text"])
  except:
    bad_files.append(test_name)

print("=======================")
for filen in bad_files:
    print(filen)


'''
for dset in ["21"]:
  outdir = "../test_results/whisper-base/whisper_output_" +  dset
  input_dir = "../testsets/earnings"+ dset + "/media/"
  files = os.listdir(input_dir)
  for filen in files:
    audio_path = input_dir + filen  # Replace with your file path
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)  # Resample to 16kHz for Whisper
    audio_input = {"array": audio, "sampling_rate": 16000}
    text = pipeline(audio, stride_length_s=5, chunk_length_s=30, return_boundary=True)
    outfile = outdir + "/" + filen[:-4] + ".txt"
    with open(outfile, "w") as ofile:
      ofile.write(text["text"])

test_dataset = load_dataset("distil-whisper/earnings22", "full", split="test")
outdir = "../test_results/whisper-base/whisper_output_22"
for datap in test_dataset:
  audio = datap["audio"]  # load the first sample (5 mins) and get the audio arr
  filen = datap["file_id"]
  text = pipeline(audio, stride_length_s=5, chunk_length_s=30, return_boundary=True)
  outfile = outdir + "/" + str(filen) + ".txt"
  with open(outfile, "w") as ofile:
    ofile.write(text["text"])
'''

