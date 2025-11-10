import os
import cv2
import numpy as np
import torch
import torch.nn as nn
# replaced import below with guarded import + diagnostics
import sys, traceback
try:
    from moviepy.editor import VideoFileClip
except Exception:
    try:
        # fallback path used by some installs
        from moviepy.video.io.VideoFileClip import VideoFileClip
        print("Imported VideoFileClip from moviepy.video.io.VideoFileClip (fallback).")
    except Exception:
        print("❌ Failed to import VideoFileClip from moviepy. Diagnostics:")
        try:
            import moviepy
            print("moviepy.__file__:", getattr(moviepy, "__file__", None))
        except Exception as ie:
            print("moviepy import failed:", ie)
        print("Python executable:", sys.executable)
        print("sys.path:")
        for p in sys.path:
            print(" ", p)
        traceback.print_exc()
        raise
import librosa
import warnings
import requests
import json
import time
from web3 import Web3

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Constants (can be used by the backend) ---
LABEL_NAMES = ['Fighting', 'Shooting', 'Riot', 'Abuse', 'Car Accident', 'Explosion']
CONFIDENCE_THRESHOLD = 0.50

# --- 1. Feature Extractors & Model Architecture ---

# Global variables to hold the large feature extraction models
yamnet_model = None
x3d_model = None
feature_extractor_device = None

# ==============================================================================
# --- MODIFIED FUNCTION ---
# This function is updated to load models from the local cache created
# by the download_models.py script during the build step on Render.
# ==============================================================================
def _load_feature_extractors():
    """
    Loads the YAMNet and X3D models from a local cache created during the build step.
    """
    global yamnet_model, x3d_model, feature_extractor_device
    
    if yamnet_model is not None and x3d_model is not None:
        return

    print("--- Loading models from local build cache ---")
    
    # Define the local cache path. Since this script is inside the 'backend' directory
    # on Render, os.getcwd() will be the 'backend' folder.
    CACHE_DIR = os.path.join(os.getcwd(), 'model_cache')
    print(f"Looking for cache in: {CACHE_DIR}")

    # Set environment variables BEFORE importing the hub libraries so they know where to look.
    os.environ['TFHUB_CACHE_DIR'] = os.path.join(CACHE_DIR, 'tf_hub')
    torch.hub.set_dir(os.path.join(CACHE_DIR, 'torch_hub'))
    
    import tensorflow as tf
    import tensorflow_hub as hub
    from pytorchvideo.models.hub import x3d_s

    # Render free tier is CPU-only, so we force CPU
    feature_extractor_device = torch.device("cpu")
    print(f"Using device for feature extraction: {feature_extractor_device}")

    # Load the models FROM THE CACHE. The libraries will find them automatically.
    with tf.device('/CPU:0'):
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

    x3d_model = x3d_s(pretrained=True).eval().to(feature_extractor_device)
    
    print("--- ✅ Models loaded successfully from cache ---")

def extract_audio_features(audio_path):
    """Extracts audio features using the YAMNet model."""
    try:
        audio_waveform, sr = librosa.load(audio_path, sr=16000)
        if audio_waveform.size == 0: return np.zeros(1024, dtype=np.float32)
        scores, embeddings, spectrogram = yamnet_model(audio_waveform)
        return np.mean(embeddings.numpy(), axis=0).astype(np.float32)
    except Exception: return np.zeros(1024, dtype=np.float32)

def extract_video_features(video_path, num_frames=16):
    """Extracts video features using the X3D model."""
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1: raise ValueError("No frames")
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame / 255.0)
        cap.release()
        if not frames: raise ValueError("Could not extract frames.")
        clip_tensor = torch.tensor(np.array(frames), dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0)
        clip_tensor = (clip_tensor - 0.45) / 0.225
        clip_tensor = clip_tensor.to(feature_extractor_device)
        with torch.no_grad():
            features = x3d_model(clip_tensor).view(1, -1)
        if features.shape[1] != 2048:
            pool = nn.AdaptiveAvgPool1d(2048)
            features = pool(features.unsqueeze(1)).squeeze(1)
        return features.squeeze(0).cpu().numpy()
    except Exception: return np.zeros(2048, dtype=np.float32)

class AttentionFusionClassifier(nn.Module):
    def __init__(self, audio_dim=1024, video_dim=2048, hidden_dim=512, output_dim=6):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, output_dim), nn.Sigmoid()
        )
    def forward(self, audio_feat, video_feat):
        audio_available, video_available = not torch.all(audio_feat == 0), not torch.all(video_feat == 0)
        device = next(self.parameters()).device
        fused = torch.zeros(audio_feat.size(0), self.audio_proj.out_features, device=device)
        if audio_available and video_available:
            a_proj, v_proj = self.audio_proj(audio_feat).unsqueeze(1), self.video_proj(video_feat).unsqueeze(1)
            kv, q = torch.cat([a_proj, v_proj], dim=1), a_proj
            fused = self.attention(q, kv, kv)[0].squeeze(1)
        elif audio_available:
            a_proj = self.audio_proj(audio_feat).unsqueeze(1)
            fused = self.attention(a_proj, a_proj, a_proj)[0].squeeze(1)
        elif video_available:
            v_proj = self.video_proj(video_feat).unsqueeze(1)
            fused = self.attention(v_proj, v_proj, v_proj)[0].squeeze(1)
        return self.classifier(fused)

# --- 2. Blockchain and IPFS Pipeline Components ---

def upload_to_pinata(file_path, pinata_jwt):
    """Uploads a file to IPFS via Pinata."""
    if not pinata_jwt:
        print("❌ Pinata JWT not found. Set PINATA_JWT environment variable.")
        return None
    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
    headers = {"Authorization": f"Bearer {pinata_jwt}"}
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(url, files={'file': f}, headers=headers)
        if response.status_code == 200:
            ipfs_hash = response.json().get('IpfsHash')
            print(f"✅ Successfully uploaded to Pinata. IPFS Hash: {ipfs_hash}")
            return ipfs_hash
        else:
            print(f"❌ Pinata upload failed: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception during Pinata upload: {e}")
        return None

def store_on_blockchain(contract, w3, signer_account, ipfs_hash, title):
    """Calls the smart contract to store the IPFS hash."""
    try:
        print("Building transaction to call 'uploadVideo'...")
        tx = contract.functions.uploadVideo(ipfs_hash, title).build_transaction({
            'from': signer_account.address,
            'nonce': w3.eth.get_transaction_count(signer_account.address),
        })
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=signer_account.key)
        
        # --- ROBUST VERSION CHECK (Typo Corrected) ---
        # This code now correctly checks for 'rawTransaction' (older web3.py)
        # or 'raw' (newer web3.py) to prevent version conflicts.
        raw_tx_bytes = None
        if hasattr(signed_tx, 'rawTransaction'):
            raw_tx_bytes = signed_tx.rawTransaction
        elif hasattr(signed_tx, 'raw'):
            raw_tx_bytes = signed_tx.raw
        else:
            raise AttributeError("Signed transaction object has neither 'rawTransaction' nor 'raw' attribute.")

        print("Sending transaction to the blockchain...")
        tx_hash = w3.eth.send_raw_transaction(raw_tx_bytes)
        
        print("Waiting for transaction to be mined...")
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        print(f"✅ Transaction successful! Block: {tx_receipt.blockNumber}")
        return tx_hash.hex()
    except Exception as e:
        print(f"❌ Exception during blockchain transaction: {e}")
        return None
    
def log_event_locally(log_file, event_data):
    """Appends event data to a local log file (JSON Lines format)."""
    try:
        with open(log_file, 'a') as f:
            f.write(json.dumps(event_data) + '\n')
        print(f"✅ Event successfully logged to {log_file}")
    except Exception as e:
        print(f"❌ Failed to write to local log: {e}")

# This function is now used by backend/app.py and should not be run directly.
def _detect_labels_from_probs(probs, label_names, threshold):
    out = []
    for i, name in enumerate(label_names):
        try:
            val = float(probs[i])
        except Exception:
            val = 0.0
        if val >= threshold:
            out.append({"label": name, "confidence": val})
    return out

# ==============================================================================
# --- REMOVED __main__ BLOCK ---
# This block is for direct script execution and contains secrets. It is unsafe for
# a deployed application module. The main execution logic is now correctly
# handled by your backend/app.py script.
# ==============================================================================