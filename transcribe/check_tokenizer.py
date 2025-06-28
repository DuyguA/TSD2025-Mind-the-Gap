from transformers import WhisperTokenizerFast, WhisperProcessor

# Load the tokenizer from your trained model
tokenizer = WhisperTokenizerFast.from_pretrained("../do_train/whisper-puncted")
print(tokenizer.is_fast)

# Verify the special tokens
print("Special Tokens:", tokenizer.additional_special_tokens)  # Should include "[PERCENT]", "[PERSON]", etc.
print("Vocabulary Size:", len(tokenizer))

test_sentence = "The inflation rate is [PERCENT]2.5%[/PERCENT]."
encoded = tokenizer(test_sentence, return_tensors="np")
print("Encoded Tokens:", encoded["input_ids"])
print("Decoded Text:", tokenizer.decode(encoded["input_ids"][0]))
print("Decoded Text:", tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True))

processor = WhisperProcessor.from_pretrained("../do_train/whisper-puncted")
tokenizer = processor.tokenizer
print("Special Tokens:", tokenizer.additional_special_tokens)  # Should include "[PERCENT]", "[PERSON]", etc.
print("Vocabulary Size:", len(tokenizer))
test_sentence = "The inflation rate is [PERCENT]2.5%[/PERCENT]."
encoded = tokenizer(test_sentence, return_tensors="np")
print("Encoded Tokens:", encoded["input_ids"])
print("Decoded Text:", tokenizer.decode(encoded["input_ids"][0]))
print("Decoded Text:", tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True))

print(tokenizer.is_fast)
