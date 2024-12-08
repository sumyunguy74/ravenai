import torch
from transformers import FluxPipeline

# Define the path to the model directory and file
model_dir = "c:/Users/GAMER-03/Documents/RAVEN/BLACKFORESTIMAGEMODEL"
model_file = "flux1-dev.safetensors"

# Check if the model file exists
model_path = f"{model_dir}/{model_file}"
print(f"Checking if model file exists at: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

# Attempt to load the model
try:
    pipe = FluxPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        filename=model_file
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
