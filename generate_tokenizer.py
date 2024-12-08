from transformers import CLIPTokenizer

# Define the directory paths for the tokenizers using raw string literals
tokenizer1_directory = r'C:/Users/GAMER-03/Documents/RAVEN/DjrangoQwen2vl-Flux/qwen2-vl/tokenizer1'

# Load the slow tokenizer
tokenizer = CLIPTokenizer.from_pretrained(tokenizer1_directory)

# Save the tokenizer to generate tokenizer.json
tokenizer.save_pretrained(tokenizer1_directory)

print("Tokenizer saved successfully.")
