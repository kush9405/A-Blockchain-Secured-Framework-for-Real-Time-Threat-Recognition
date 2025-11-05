# A-Blockchain-Secured-Framework-for-Real-Time-Threat-Recognition


This repository contains a full-stack application that provides a blockchain-secured framework for real-time threat recognition. The system analyzes video clips, detects potential threats using a multi-modal deep learning model, and creates an immutable evidentiary record on the IPFS and Ethereum blockchain.

## Architecture

The project is composed of a frontend interface, a backend API, and a core processing pipeline.

1.  **Frontend**: A simple HTML/CSS/JavaScript interface for uploading video files. It sends the video to the backend and displays the analysis results, including links to the generated evidence on IPFS and the blockchain.

2.  **Backend (FastAPI)**: A Python-based server that exposes REST endpoints for video processing. It receives the uploaded video, invokes the inference pipeline, and returns a JSON response with the detection results, IPFS hash, and blockchain transaction hash.

3.  **Core Pipeline (`Pipeline.py`)**: The engine of the system. Its responsibilities include:
    *   **Feature Extraction**: Separates audio and video streams. It uses Google's **YAMNet** for audio feature extraction and Facebook's **X3D** model for video feature extraction.
    *   **Threat Classification**: Fuses the audio and video features using a custom **Attention-based Fusion Model** (PyTorch) to classify the content into one of several threat categories: `Fighting`, `Shooting`, `Riot`, `Abuse`, `Car Accident`, or `Explosion`.
    *   **Immutable Storage**: If a threat is detected above a confidence threshold, the pipeline:
        *   Uploads the original video file to **IPFS** via the Pinata service.
        *   Calls a smart contract on the **Ethereum (Sepolia) blockchain** to store the IPFS hash, creating a permanent, tamper-proof record.

## Key Features

-   **Multi-modal Analysis**: Leverages both audio and video data for more accurate threat detection.
-   **Attention-based Fusion**: Uses a state-of-the-art attention mechanism to intelligently combine features from different modalities.
-   **Decentralized Evidence Storage**: Creates immutable proof of an event by storing video evidence on IPFS and its corresponding hash on the Ethereum blockchain.
-   **RESTful API**: A modern FastAPI backend for easy integration and scalability.
-   **Standalone Pipeline Script**: The core logic can be run as a standalone script for direct inference without the web server.

## Getting Started

### Prerequisites

-   Python 3.8+
-   Git

### 1. Clone the Repository

```sh
git clone https://github.com/kush9405/A-Blockchain-Secured-Framework-for-Real-Time-Threat-Recognition.git
cd A-Blockchain-Secured-Framework-for-Real-Time-Threat-Recognition
```

### 2. Install Dependencies

Install all the required Python packages from the requirements file.

```sh
pip install -r requirements.txt
```
*Note: The first run may take some time as it will download pre-trained models (YAMNet, X3D) from TensorFlow Hub and PyTorch Hub.*

### 3. Configure Environment Variables

The blockchain and IPFS integration requires API keys and a private key. Create a `.env` file inside the `backend/` directory:

`backend/.env`
```
# Get this from your Pinata account (https://www.pinata.cloud/)
PINATA_JWT="YOUR_PINATA_JWT_TOKEN"

# Get this from a node provider like Infura or Alchemy
ETHEREUM_NODE_URL="https://sepolia.infura.io/v3/YOUR_INFURA_PROJECT_ID"

# The private key of the Ethereum account that will pay for transactions (must have Sepolia ETH)
SIGNER_PRIVATE_KEY="YOUR_ETHEREUM_PRIVATE_KEY"

# The deployed smart contract address on the Sepolia testnet
CONTRACT_ADDRESS="0x3cc6c2523724a322795b59625ea3715a960c7a3c"
```

### 4. Running the Application

You can use the system via the Web UI or the command-line script.

#### Option A: Run with the Web Interface (Recommended)

1.  **Start the Backend Server**:
    Navigate to the `backend` directory and start the Uvicorn server.

    ```sh
    cd backend
    uvicorn app:app --reload
    ```
    The API will be running at `http://127.0.0.1:8000`.

2.  **Modify and Launch the Frontend**:
    Open `frontend/index.html` in a text editor and change the `backendUrl` to point to your local server:

    ```javascript
    // In frontend/index.html
    const backendUrl = 'http://127.0.0.1:8000/infer';
    ```

    Now, open the `frontend/index.html` file directly in your web browser.

3.  **Upload and Analyze**:
    Use the web interface to select a video file and click "Upload Video" to start the analysis. The results, including IPFS and Etherscan links (if a threat is detected), will be displayed in an alert popup.

#### Option B: Run the Command-Line Pipeline

You can test the core pipeline directly on a local video file.

1.  **Configure `Pipeline.py`**:
    Open the `Pipeline.py` script in the root directory.
    -   Ensure your environment variables are configured correctly as described in Step 3 (the script also loads them).
    -   Update the `video_input_path` variable to point to your test video file.

    ```python
    # In Pipeline.py
    video_input_path = "/path/to/your/video.mp4"
    ```

2.  **Execute the Script**:
    Run the script from the root directory.

    ```sh
    python Pipeline.py
    ```
    The script will print the analysis progress, detection probabilities, and the final IPFS/blockchain record details to the console.

## Project Structure

```
.
├── backend/                  # FastAPI backend server
│   ├── app.py                # API endpoints and logic
│   ├── Pipeline.py           # Symlink/copy of the core pipeline
│   └── requirements.txt      # Backend dependencies
|   └── VideoStorageABI.json      # ABI of the Ethereum smart contract
├── frontend/                 # Web interface files
│   ├── index.html            # Main HTML page for the UI
│   ├── style.css             # Styles for the UI
│   └── app.py                # Optional minimal Flask server for frontend
├── checkpoint_epoch_49.pth   # Pre-trained weights for the fusion model
├── event_log.jsonl           # Local log file for all processed events
├── Pipeline.py               # Core inference and blockchain logic
├── requirements.txt          # Python dependencies for the project
└── VideoStorageABI.json      # ABI of the Ethereum smart contract
