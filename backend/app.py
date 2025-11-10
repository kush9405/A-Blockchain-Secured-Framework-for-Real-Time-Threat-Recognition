# backend/app.py
import asyncio
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import time
import shutil
import logging
from typing import Dict, Any

# make sure pipeline (project root) is importable
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import backend.Pipeline as pipeline  # project Pipeline.py

# --- NEW: Load environment variables at the top ---
from dotenv import load_dotenv
load_dotenv()

# --- NEW: Check for optional Web3 import ---
try:
    from web3 import Web3
except ImportError:
    Web3 = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Video Threat Pipeline Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity, can be restricted later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# --- NEW: Startup Event to Pre-load AI Models ---
# ==============================================================================
@app.on_event("startup")
async def startup_event():
    """
    This function will be executed once when the application starts.
    It pre-loads the heavy AI models into memory to avoid request timeouts.
    """
    print("--- Application Startup: Pre-loading AI models ---")
    # This calls the function in Pipeline.py to load models into its global variables
    pipeline._load_feature_extractors()
    print("--- AI models loaded. Application is ready. ---")
# ==============================================================================


@app.post("/infer")
async def infer(file: UploadFile = File(...), model_checkpoint: str = "checkpoint_epoch_49.pth"):
    start_time = time.time()
    job_id = str(uuid.uuid4())
    dest_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # --- MODIFIED: We no longer call the whole pipeline function. ---
        # Instead, we perform the steps here, using the pre-loaded models.
        
        print("Step 1: Separating audio...")
        audio_path = "temp_audio.wav"
        audio_extracted = False
        try:
            with pipeline.VideoFileClip(dest_path) as clip:
                if clip.audio:
                    clip.audio.write_audiofile(audio_path, logger=None)
                    audio_extracted = True
        except Exception: print("Warning: Could not extract audio.")

        print("Step 2: Extracting video features...")
        video_features = pipeline.extract_video_features(dest_path)
        
        print("Step 3: Extracting audio features...")
        audio_features = pipeline.extract_audio_features(audio_path) if audio_extracted else np.zeros(1024, dtype=np.float32)
        if os.path.exists(audio_path): os.remove(audio_path)
        
        # We need to load the fusion model here as it's small and quick
        print("Step 4: Loading the trained fusion model...")
        fusion_model = pipeline.AttentionFusionClassifier()
        checkpoint = torch.load(model_checkpoint, map_location='cpu', weights_only=False)
        fusion_model.load_state_dict(checkpoint['model_state_dict'])
        fusion_model.eval()

        print("Step 5: Running inference...")
        video_tensor = torch.from_numpy(video_features).unsqueeze(0)
        audio_tensor = torch.from_numpy(audio_features).unsqueeze(0)
        
        with torch.no_grad():
            probs = fusion_model(audio_tensor, video_tensor)

        if probs is None:
            raise HTTPException(status_code=500, detail="Inference returned no result")

        import numpy as np
        probs_np = np.asarray(probs).flatten().tolist()
        detections = pipeline._detect_labels_from_probs(probs_np, pipeline.LABEL_NAMES, pipeline.CONFIDENCE_THRESHOLD)

        response_data = {
            "video_name": file.filename,
            "duration_seconds": round(time.time() - start_time, 2),
            "probabilities": probs_np,
            "detections": detections,
            "ipfs_hash": None,
            "transaction_hash": None
        }

        if detections:
            print("🔥 Threat Detected! Triggering IPFS & Blockchain Pipeline...")
            pinata_jwt = os.getenv("PINATA_JWT")
            ipfs_hash = pipeline.upload_to_pinata(dest_path, pinata_jwt)
            response_data["ipfs_hash"] = ipfs_hash

            eth_node = os.getenv("ETHEREUM_NODE_URL")
            signer_key = os.getenv("SIGNER_PRIVATE_KEY")
            contract_addr = os.getenv("CONTRACT_ADDRESS")

            if ipfs_hash and Web3 and eth_node and signer_key and contract_addr:
                # ... (The rest of the blockchain logic is the same)
                w3 = Web3(Web3.HTTPProvider(eth_node))
                if w3.is_connected():
                    signer = w3.eth.account.from_key(signer_key)
                    abi_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "VideoStorageABI.json")
                    with open(abi_path, "r") as f: abi = json.load(f)
                    contract = w3.eth.contract(address=w3.to_checksum_address(contract_addr), abi=abi)
                    title = f"Threat Event: {', '.join([d['label'] for d in detections])}"
                    tx_hash = pipeline.store_on_blockchain(contract, w3, signer, ipfs_hash, title)
                    response_data["transaction_hash"] = tx_hash

        return response_data

    finally:
        # Clean up the uploaded video file in all cases
        if os.path.exists(dest_path):
            os.remove(dest_path)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Threat Detection API. Please use the /infer endpoint to upload a video."}

# NOTE: The other endpoints like /upload, /result, /infer-stream are removed for clarity.
# You can add them back if needed, but the primary fix is in the /infer endpoint and the startup event.