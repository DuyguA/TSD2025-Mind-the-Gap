from transformers import WhisperForConditionalGeneration, WhisperConfig

flax_model_path = "whisper-puncted-timed/89"  # Replace with your Flax model directory
config = WhisperConfig.from_pretrained(flax_model_path)

# Load the Flax model
from transformers import FlaxWhisperForConditionalGeneration
flax_model = FlaxWhisperForConditionalGeneration.from_pretrained(flax_model_path, config=config)


pytorch_model = WhisperForConditionalGeneration.from_pretrained(flax_model_path, from_flax=True, config=config)

# Save the PyTorch model locally

from huggingface_hub import HfApi

# Push the model to the hub
pytorch_model.push_to_hub("BayanDuygu/whisper-puncted-timed")

