from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
import librosa

pipeline = FlaxWhisperPipline("openai/whisper-base", dtype=jnp.bfloat16, batch_size=16)


from jax.experimental.compilation_cache import compilation_cache as cc

cc.initialize_cache("./jax_cache")



from datasets import load_dataset

'''
test_dataset = load_dataset("sanchit-gandhi/whisper-jax-test-files", split="train")
audio = test_dataset[0]["audio"]  # load the first sample (5 mins) and get the audio array
'''

audio_path = "../testsets/earnings21/media/4320211.mp3"  # Replace with your file path
audio, sr = librosa.load(audio_path, sr=16000, mono=True)  # Resample to 16kHz for Whisper

audio_input = {"array": audio, "sampling_rate": 16000}



text = pipeline(audio, stride_length_s=5, chunk_length_s=30, return_boundary=True, return_timestamps=True)
print(text)
