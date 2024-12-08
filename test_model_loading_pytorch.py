import torch
import os

# Define the path to the model file
model_path = "c:/Users/GAMER-03/Documents/RAVEN/BLACKFORESTIMAGEMODEL/flux1-dev.safetensors"

# Check if the model file exists
print(f"Checking if model file exists at: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

# Attempt to load the model
try:
    model = torch.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
