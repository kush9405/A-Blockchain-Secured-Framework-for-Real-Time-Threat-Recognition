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

# --- 1. Feature Extractors & Model Architecture (Unchanged) ---

# Global variables to hold the large feature extraction models
yamnet_model = None
x3d_model = None
feature_extractor_device = None

# In Pipeline.py

def _load_feature_extractors():
    """
    Loads the YAMNet and X3D models from a local cache created during the build step.
    """
    global yamnet_model, x3d_model, feature_extractor_device
    
    if yamnet_model is not None and x3d_model is not None:
        return

    print("--- Loading models from local build cache ---")
    
    # Define the SAME local cache path used in download_models.py
    CACHE_DIR = os.path.join(os.getcwd(), 'model_cache')
    
    # Set environment variables BEFORE importing the hub libraries
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
        
        # --- DEBUGGING LINES ---
        # These lines will tell us the exact structure of the object.
        # print("\n--- DEBUGGING SIGNED TRANSACTION OBJECT ---")
        # print(f"Type of signed_tx: {type(signed_tx)}")
        # print(f"Attributes of signed_tx: {dir(signed_tx)}")
        # print("--- END DEBUGGING ---")

        # Robust version check
        raw_tx_bytes = None
        if hasattr(signed_tx, 'raw_transaction'):
            raw_tx_bytes = signed_tx.raw_transaction
        elif hasattr(signed_tx, 'raw'):
            raw_tx_bytes = signed_tx.raw
        else:
            # This is the error you are getting. The printouts above will tell us why.
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

# --- 3. The Main Inference Pipeline ---

def run_inference_pipeline(video_path, model_path):
    _load_feature_extractors()
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return None
    audio_path = "temp_audio.wav"
    audio_extracted = False
    print("\nStep 1: Separating audio...")
    try:
        with VideoFileClip(video_path) as clip:
            if clip.audio:
                clip.audio.write_audiofile(audio_path, logger=None)
                audio_extracted = True
    except Exception: print("Warning: Could not extract audio.")
    print("Step 2: Extracting video features...")
    video_features = extract_video_features(video_path)
    print("Step 3: Extracting audio features...")
    audio_features = extract_audio_features(audio_path) if audio_extracted else np.zeros(1024, dtype=np.float32)
    if os.path.exists(audio_path): os.remove(audio_path)
    print("Step 4: Loading the trained fusion model...")
    try:
        model = AttentionFusionClassifier()
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ Fusion model loaded.")
    except Exception as e:
        print(f"❌ Error loading fusion model: {e}"); return None
    print("Step 5: Running inference...")
    try:
        video_tensor = torch.from_numpy(video_features).unsqueeze(0)
        audio_tensor = torch.from_numpy(audio_features).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        video_tensor, audio_tensor = video_tensor.to(device), audio_tensor.to(device)
        print(f"Running inference on device: {device}")
        with torch.no_grad():
            return model(audio_tensor, video_tensor)
    except Exception as e:
        print(f"❌ Error during model inference: {e}"); return None

if __name__ == '__main__':
    # --- Configuration ---
    # video_input_path = "/Users/kushagraagarwal/Downloads/riot1.mp4"
    video_input_path = "/Users/kushagraagarwal/Downloads/riot2.mp4"
    model_checkpoint_path = "checkpoint_epoch_49.pth"
    LABEL_NAMES = ['Fighting', 'Shooting', 'Riot', 'Abuse', 'Car Accident', 'Explosion']
    CONFIDENCE_THRESHOLD = 0.50
    LOCAL_LOG_FILE = "event_log.jsonl"
    
    # --- !! IMPORTANT !! PASTE YOUR DEPLOYED CONTRACT ADDRESS HERE ---
    CONTRACT_ADDRESS = Web3.to_checksum_address("0x3cc6c2523724a322795b59625ea3715a960c7a3c") 
    
    os.environ[ "ETHEREUM_NODE_URL"]="https://sepolia.infura.io/v3/76faca2e845c4fcfa3d3ef5a5a1a6b06"
    os.environ["PINATA_JWT"]="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySW5mb3JtYXRpb24iOnsiaWQiOiI3ODczNDU5Mi1kMmQ5LTQ5Y2ItYmRhYi02NWRmZjJiNGE4NDciLCJlbWFpbCI6Imt1c2hhZ3JhYWdhcndhbDIwMDNAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsInBpbl9wb2xpY3kiOnsicmVnaW9ucyI6W3siZGVzaXJlZFJlcGxpY2F0aW9uQ291bnQiOjEsImlkIjoiRlJBMSJ9LHsiZGVzaXJlZFJlcGxpY2F0aW9uQ291bnQiOjEsImlkIjoiTllDMSJ9XSwidmVyc2lvbiI6MX0sIm1mYV9lbmFibGVkIjpmYWxzZSwic3RhdHVzIjoiQUNUSVZFIn0sImF1dGhlbnRpY2F0aW9uVHlwZSI6InNjb3BlZEtleSIsInNjb3BlZEtleUtleSI6ImJmYzM1MzM5NTIyYzY0ZjRmNzMyIiwic2NvcGVkS2V5U2VjcmV0IjoiYTY5NTNmYjQwNWJiMTBhYWIzYjM3NzhjODJhMjBhOGUzZWIzNTg1Y2UyNTIzZWJjNWFmYzg1Njg1N2E0MThkMSIsImV4cCI6MTc5MzcyODcyNX0.qvjEkTy1nUXBRZZ4W2FvgykYSUtzXD4aIlgPhKMN9tM"
    os.environ["SIGNER_PRIVATE_KEY"]="9a03bd45d05f5d8d9b3cfe46a84f12f624a6d0caf4ca1c6a3412ab2087d488f9"
    # --- Load Environment Variables ---
    PINATA_JWT = os.getenv('PINATA_JWT')
    ETHEREUM_NODE_URL = os.getenv('ETHEREUM_NODE_URL')
    SIGNER_PRIVATE_KEY = os.getenv('SIGNER_PRIVATE_KEY')
    
    try:
        with open('VideoStorageABI.json', 'r') as f: CONTRACT_ABI = json.load(f)
    except FileNotFoundError:
        print("❌ CRITICAL: 'VideoStorageABI.json' not found."); CONTRACT_ABI = None

    # --- Run Pipeline ---
    probabilities = run_inference_pipeline(video_input_path, model_checkpoint_path)

    # --- Process Output & Trigger Next Steps ---
    if probabilities is not None:
        print("\n" + "="*35 + "\n      MODEL OUTPUT ANALYSIS\n" + "="*35)
        probs_np = probabilities.squeeze().cpu().numpy()
        detected_labels = []
        for i, name in enumerate(LABEL_NAMES):
            prob = probs_np[i]
            print(f"Probability of '{name}': {prob:.4f}")
            if prob >= CONFIDENCE_THRESHOLD:
                detected_labels.append({"label": name, "confidence": float(prob)})
        
        if detected_labels:
            print(f"\n🔥 Threat Detected! Triggering IPFS & Blockchain Pipeline...")
            
            # 1. Upload to IPFS
            ipfs_hash = upload_to_pinata(video_input_path, PINATA_JWT)
            
            # 2. Store on Blockchain
            tx_hash = None
            if ipfs_hash and CONTRACT_ABI and ETHEREUM_NODE_URL and SIGNER_PRIVATE_KEY and CONTRACT_ADDRESS :
                w3 = Web3(Web3.HTTPProvider(ETHEREUM_NODE_URL))
                if w3.is_connected():
                    signer = w3.eth.account.from_key(SIGNER_PRIVATE_KEY)
                    contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
                    title = f"Threat Event: {', '.join([d['label'] for d in detected_labels])}"
                    tx_hash = store_on_blockchain(contract, w3, signer, ipfs_hash, title)
                else: print("❌ Could not connect to Ethereum node.")
            else: print("\nSkipping blockchain transaction due to missing configuration.")

            # 3. Log event locally (Hybrid Approach)
            event_data = {
                "timestamp_utc": time.time(),
                "video_file": os.path.basename(video_input_path),
                "detections": detected_labels,
                "ipfs_hash": ipfs_hash,
                "transaction_hash": tx_hash,
                "ipfs_link": f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}" if ipfs_hash else None
            }
            log_event_locally(LOCAL_LOG_FILE, event_data)

            print("\n--- ✅ Permanent Evidence Record ---")
            print(f"IPFS Link: {event_data['ipfs_link']}")
            print(f"Proof of Record (Transaction Hash): {tx_hash}")
            print(f"Ethereum Explorer Link: https://sepolia.etherscan.io/tx/0x{tx_hash}" if tx_hash else "N/A")

        else: print("\n✅ No threats detected above the confidence threshold.")
        print("\nPipeline finished successfully!")
    else:
        print("\n❌ Pipeline failed to produce an output.")