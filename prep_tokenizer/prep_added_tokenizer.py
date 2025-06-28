from transformers import WhisperTokenizer, WhisperTokenizerFast, WhisperFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoProcessor
import os

phase = "puncted"
tags_file = "all_tags_" + phase + ".txt"
output_dir = "whisper_" + phase

model_name = "openai/whisper-base"  

model = WhisperForConditionalGeneration.from_pretrained(model_name)
tokenizer = WhisperTokenizerFast.from_pretrained(model_name)
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
print(tokenizer.is_fast)
print(type(tokenizer))

#model_input_name = feature_extractor.model_input_names[0]
#print(model_input_name)


with open(tags_file, "r", encoding="utf-8") as f:
    new_tokens = [line.strip() for line in f if line.strip()]

print(new_tokens)



print(f"Adding {len(new_tokens)} new tokens to the tokenizer...")
tokenizer.add_tokens(new_tokens)

model.resize_token_embeddings(len(tokenizer))


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save tokenizer
tokenizer.save_pretrained(output_dir)

# Save feature extractor
feature_extractor.save_pretrained(output_dir)

# Save processor (combines tokenizer and feature extractor)
processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
print(processor.tokenizer.is_fast)
processor.save_pretrained(output_dir)

# Save the model
model.save_pretrained(output_dir)
