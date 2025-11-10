# download_models.py
import os
import torch
import warnings

# --- IMPORTANT: Set Hub Caches BEFORE importing hub libraries ---
# Define a local directory to store the models within your project
CACHE_DIR = os.path.join(os.getcwd(), 'model_cache')
print(f"--- Model Download Script ---")
print(f"Cache directory set to: {CACHE_DIR}")

# Create the directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# 1. Setup cache directories for the libraries
TF_HUB_CACHE_DIR = os.path.join(CACHE_DIR, 'tf_hub')
os.environ['TFHUB_CACHE_DIR'] = TF_HUB_CACHE_DIR
os.makedirs(TF_HUB_CACHE_DIR, exist_ok=True)
print(f"TensorFlow Hub cache set to: {TF_HUB_CACHE_DIR}")

TORCH_HUB_CACHE_DIR = os.path.join(CACHE_DIR, 'torch_hub')
torch.hub.set_dir(TORCH_HUB_CACHE_DIR)
print(f"PyTorch Hub cache set to: {TORCH_HUB_CACHE_DIR}")

# Now, import the libraries
import tensorflow_hub as hub
from pytorchvideo.models.hub import x3d_s

warnings.filterwarnings('ignore')

# 2. Download the models by loading them
print("\nDownloading/Verifying YAMNet model...")
hub.load('https://tfhub.dev/google/yamnet/1')
print("✅ YAMNet download complete.")

print("\nDownloading/Verifying X3D model...")
x3d_s(pretrained=True)
print("✅ X3D download complete.")

print("\n--- Model Download Script Finished ---")