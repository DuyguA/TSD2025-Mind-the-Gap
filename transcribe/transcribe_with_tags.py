#from whisper_jax import FlaxWhisperPipline
import jax
import jax.numpy as jnp
from transformers import FlaxWhisperForConditionalGeneration, WhisperProcessor
import librosa

## Initialize pipeline
#pipeline = FlaxWhisperPipline("../do_train/whisper-puncted", dtype=jnp.bfloat16, batch_size=16)

# Verify tokenizer
#tokenizer = pipeline.processor.tokenizer

# Transcribe audio
'''
output = pipeline(audio_path, generate_kwargs={"num_beams": 5, "max_length": 512})
print("Transcription:", output["text"])
print("===========================================")
'''

# Inspect raw output
audio_path = "../testsets/earnings21/media/4320211.mp3"
audio, sr = librosa.load(audio_path, sr=16000, mono=True)

duration_in_seconds = 30
num_samples = int(sr * duration_in_seconds)

# Slice the first 30 seconds
audio_30_sec = audio[:num_samples]


model = FlaxWhisperForConditionalGeneration.from_pretrained("../do_train/whisper-puncted-timed/24")
processor = WhisperProcessor.from_pretrained("../do_train/whisper-puncted-timed/24")

input_features = processor(audio_30_sec, sampling_rate=sr, return_tensors="np")
print(input_features)

outputs = model.generate(
    input_features.input_features,  # Preprocessed audio features
    max_length=100,  # Adjust as needed for your use case
    #num_beams=5,  # Optional: Use beam search for better results
    return_dict_in_generate=True,
    output_scores=True,  # Return logits/scores for debugging
    return_timestamps=True
)

# Extract predicted token IDs
predicted_token_ids = outputs.sequences[0]
print("Predicted Token IDs:", predicted_token_ids)

for token_id in predicted_token_ids:
    token = processor.tokenizer.decode([token_id])
    print(f"Token ID: {token_id}, Token: '{token}'")

decoded_output = processor.tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
print("Decoded Output:", decoded_output)
