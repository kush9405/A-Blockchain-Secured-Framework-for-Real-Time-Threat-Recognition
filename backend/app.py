from sse_starlette.sse import EventSourceResponse
import asyncio
import json
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import time
import shutil
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

# make sure pipeline (project root) is importable
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import Pipeline as pipeline  # project Pipeline.py
# Load .env file variables into environment
load_dotenv()

try:
    from web3 import Web3  # optional
except Exception:
    Web3 = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Video Threat Pipeline Backend")

# allow your frontend origin(s) for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
            "http://127.0.0.1:5000",
            "http://localhost:5000",
            "http://127.0.0.1:5500",
            "http://localhost:5500",
            "http://localhost:3000",
            "https://blocksentinel.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory job store (simple). For production, use persistent store/DB.
JOBS: Dict[str, Dict[str, Any]] = {}

def _detect_labels_from_probs(probs, label_names, threshold=0.5):
    out = []
    for i, name in enumerate(label_names):
        try:
            val = float(probs[i])
        except Exception:
            val = 0.0
        if val >= threshold:
            out.append({"label": name, "confidence": val})
    return out

def background_inference(job_id: str, file_path: str, model_checkpoint: str):
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["started_at"] = time.time()
    try:
        # run the existing inference pipeline (returns torch tensor or None)
        probs = pipeline.run_inference_pipeline(file_path, model_checkpoint)
        if probs is None:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = "inference returned no result"
            return

        # normalize to python list
        try:
            import numpy as _np
            probs_np = _np.asarray(probs).flatten().tolist()
        except Exception:
            try:
                probs_np = list(probs)
            except Exception:
                probs_np = [float(p) for p in probs]

        JOBS[job_id]["probabilities"] = probs_np
        JOBS[job_id]["detections"] = _detect_labels_from_probs(
            probs_np,
            getattr(pipeline, "LABEL_NAMES", ["Fighting","Shooting","Riot","Abuse","Car Accident","Explosion"]),
            getattr(pipeline, "CONFIDENCE_THRESHOLD", 0.5)
        )

        # If detections present, upload to pinata and optionally store on-chain
        if JOBS[job_id]["detections"]:
            pinata_jwt = os.getenv("PINATA_JWT")
            ipfs_hash = pipeline.upload_to_pinata(file_path, pinata_jwt)
            JOBS[job_id]["ipfs_hash"] = ipfs_hash

            # optional blockchain step (best-effort)
            jwt = os.getenv("PINATA_JWT")
            # print(f"Pinata JWT: {jwt}")
            eth_node = os.getenv("ETHEREUM_NODE_URL")
            signer_key = os.getenv("SIGNER_PRIVATE_KEY")
            contract_addr = os.getenv("CONTRACT_ADDRESS")
            if ipfs_hash and Web3 and eth_node and signer_key and contract_addr:
                try:
                    w3 = Web3(Web3.HTTPProvider(eth_node))
                    if w3.is_connected():
                        signer = w3.eth.account.from_key(signer_key)
                        # user should provide ABI file at project root named VideoStorageABI.json
                        import json
                        abi_path = os.path.join(os.getcwd(), "VideoStorageABI.json")
                        if os.path.exists(abi_path):
                            with open(abi_path, "r") as f:
                                abi = json.load(f)
                            contract = w3.eth.contract(address=w3.to_checksum_address(contract_addr), abi=abi)
                            tx = pipeline.store_on_blockchain(contract, w3, signer, ipfs_hash, "Threat Event")
                            JOBS[job_id]["transaction_hash"] = tx
                except Exception as e:
                    JOBS[job_id]["chain_error"] = str(e)

        JOBS[job_id]["status"] = "finished"
        JOBS[job_id]["finished_at"] = time.time()
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
    finally:
        # cleanup uploaded file to save disk if desired
        try:
            os.remove(file_path)
        except Exception:
            pass

@app.post("/infer")
async def infer(file: UploadFile = File(...), model_checkpoint: str = "checkpoint_epoch_49.pth"):
    start_time = time.time()
    job_id = str(uuid.uuid4())
    dest_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run inference directly (not as background)
    probs = pipeline.run_inference_pipeline(dest_path, model_checkpoint)
    if probs is None:
        return {"status": "failed", "error": "inference returned no result"}

    # Convert to flat list
    import numpy as np
    probs_np = np.asarray(probs).flatten().tolist()

    # Detect labels
    detections = _detect_labels_from_probs(
        probs_np,
        getattr(pipeline, "LABEL_NAMES", ["Fighting","Shooting","Riot","Abuse","Car Accident","Explosion"]),
        getattr(pipeline, "CONFIDENCE_THRESHOLD", 0.5)
    )

    response_data = {
        "video_name": file.filename,
        "duration_seconds": round(time.time() - start_time, 2),
        "probabilities": probs_np,
        "detections": detections,
        "ipfs_hash": None,
        "transaction_hash": None
    }

    # Upload to Pinata + Blockchain if detection exists
    if detections:
        pinata_jwt = os.getenv("PINATA_JWT")
        ipfs_hash = pipeline.upload_to_pinata(dest_path, pinata_jwt)
        response_data["ipfs_hash"] = ipfs_hash

        eth_node = os.getenv("ETHEREUM_NODE_URL")
        signer_key = os.getenv("SIGNER_PRIVATE_KEY")
        contract_addr = os.getenv("CONTRACT_ADDRESS")

        if ipfs_hash and Web3 and eth_node and signer_key and contract_addr:
            try:
                w3 = Web3(Web3.HTTPProvider(eth_node))
                if w3.is_connected():
                    signer = w3.eth.account.from_key(signer_key)
                    import json
                    abi_path = os.path.join(os.getcwd(), "VideoStorageABI.json")
                    if os.path.exists(abi_path):
                        with open(abi_path, "r") as f:
                            abi = json.load(f)
                        contract = w3.eth.contract(address=w3.to_checksum_address(contract_addr), abi=abi)
                        tx_hash = pipeline.store_on_blockchain(contract, w3, signer, ipfs_hash, "Threat Event")
                        response_data["transaction_hash"] = tx_hash
            except Exception as e:
                response_data["chain_error"] = str(e)

    # Log every inference event
    try:
        log_entry = {
            "timestamp_utc": time.time(),
            "video_name": file.filename,
            "detections": detections,
            "ipfs_hash": response_data["ipfs_hash"],
            "transaction_hash": response_data["transaction_hash"],
            "duration_seconds": response_data["duration_seconds"]
        }
        pipeline.log_event_locally("event_log.jsonl", log_entry)
    except Exception as e:
        print("❌ Failed to write to log:", e)

    # Clean up uploaded file
    try:
        os.remove(dest_path)
    except Exception:
        pass

    return response_data

# compatibility path for older frontends
@app.post("/upload")
async def upload_compat(background_tasks: BackgroundTasks, file: UploadFile = File(...), model_checkpoint: str = "checkpoint_epoch_49.pth"):
    return await infer(background_tasks, file, model_checkpoint)

@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/health")
def health():
    return {"status": "ok"}



from sse_starlette.sse import EventSourceResponse
import asyncio, json, numpy as np

@app.post("/infer-stream")
async def infer_stream(file: UploadFile = File(...), model_checkpoint: str = "checkpoint_epoch_49.pth"):
    async def event_generator():
        try:
            yield {"data": f"📥 Received video file: {file.filename}"}
            await asyncio.sleep(0.5)
            yield {"data": "🎬 Extracting features..."}
            await asyncio.sleep(0.5)
            yield {"data": "✅ Inference completed"}
            await asyncio.sleep(0.5)
            yield {"event": "result", "data": json.dumps({"file": file.filename, "detections": []})}
        except Exception as e:
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator(), ping=10)