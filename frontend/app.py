from flask import Flask, request, jsonify
import uuid

app = Flask(__name__)

@app.route("/")
def home():
    return app.send_static_file("index.html")

@app.route("/style.css")
def css():
    return app.send_static_file("style.css")

@app.route("/upload", methods=["POST"])
def upload_file():
    print("🟢 /upload endpoint called")
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Simulate processing & generate job ID
    job_id = str(uuid.uuid4())[:8]  # short random job id
    print(f"✅ Received file: {file.filename}, Job ID: {job_id}")

    # (You can pass the file to your processing pipeline here)

    return jsonify({
        "message": f"'{file.filename}' received and processed successfully!",
        "job_id": job_id
    })
if __name__ == "__main__":
    app.run(debug=True)
