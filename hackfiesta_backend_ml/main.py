from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import tempfile
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from urllib.parse import quote, unquote
from collections import defaultdict
import numpy as np
import time

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path(__file__).parent / "processed_videos"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = "model.pt"
model = YOLO(MODEL_PATH)

def get_unique_filename(directory: Path, filename: str) -> Path:
    """Return a unique file path in the directory preserving the original filename.
    If the file exists, append a counter before the file extension."""
    output_file = directory / filename
    if not output_file.exists():
        return output_file

    stem = output_file.stem
    suffix = output_file.suffix
    counter = 1
    while True:
        new_filename = f"{stem}_{counter}{suffix}"
        new_output_file = directory / new_filename
        if not new_output_file.exists():
            return new_output_file
        counter += 1

def process_video(input_path: str, output_path: str):
    """Process video using trained YOLO model and save output."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video")
        return False, None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Statistics tracking
    timestamp_counts = defaultdict(int)
    total_confidences = []
    frame_count = 0
    all_detections = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get timestamp in seconds
            timestamp = frame_count / fps
            frame_count += 1

            results = model(frame)
            people_count = 0

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())

                    # Assuming class_id 0 is for people/persons
                    if class_id == 0:
                        people_count += 1
                        total_confidences.append(confidence)

                    label = f"{model.names[class_id]}: {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Store counts for this timestamp
            if people_count > 0:
                timestamp_counts[round(timestamp, 2)] = people_count
                all_detections.append(people_count)

            out.write(frame)

    except Exception as e:
        print(f"❌ Error during video processing: {e}")
        return False, None
    finally:
        cap.release()
        out.release()

    # Calculate statistics
    stats = {
        "timestamp_counts": dict(timestamp_counts),
        "total_unique_people_estimate": int(np.mean(all_detections) * 1.5) if all_detections else 0,
        "average_people_per_frame": round(np.mean(all_detections), 2) if all_detections else 0,
        "max_people_in_frame": max(all_detections) if all_detections else 0,
        "model_confidence": {
            "average": round(np.mean(total_confidences), 3) if total_confidences else 0,
            "min": round(min(total_confidences), 3) if total_confidences else 0,
            "max": round(max(total_confidences), 3) if total_confidences else 0
        },
        "video_duration": round(frame_count / fps, 2),
        "frames_processed": frame_count
    }

    if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
        print(f"✅ Processed video saved at: {output_path}")
        return True, stats
    else:
        print("❌ Output file verification failed")
        return False, None

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Upload video and process it asynchronously using the trained model."""
    # Create a NamedTemporaryFile and close it immediately to release the handle
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video_name = temp_video.name
    temp_video.close()

    try:
        # Save uploaded file to temporary location
        with open(temp_video_name, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Use original filename for output while ensuring uniqueness
        output_path = get_unique_filename(OUTPUT_DIR, file.filename)
        print(f"Processing video: Input={temp_video_name}, Output={output_path}")

        # Process the video
        success, stats = process_video(temp_video_name, str(output_path))

        if not success:
            return JSONResponse(
                content={"error": "Failed to process video"},
                status_code=500
            )

        # Verify the output file exists and has content
        if output_path.exists():
            size = output_path.stat().st_size
            print(f"Output video created successfully: {output_path} (size: {size} bytes)")
        else:
            print(f"❌ Output video not created: {output_path}")
            return JSONResponse(
                content={"error": "Failed to create output video"},
                status_code=500
            )

        encoded_filename = quote(output_path.name)
        download_url = f"/download/{encoded_filename}"

        return {
            "message": f"Video {file.filename} has been processed.",
            "download_url": download_url,
            "file_size": size,
            "statistics": stats
        }
    finally:
        # Cleanup temp file
        try:
            Path(temp_video_name).unlink(missing_ok=True)
        except Exception as e:
            print(f"Error deleting temp file {temp_video_name}: {e}")

@app.get("/download/{video_name}")
async def download(video_name: str):
    """Download the processed video."""
    decoded_video_name = unquote(video_name)
    video_path = OUTPUT_DIR / decoded_video_name

    if not video_path.exists():
        return JSONResponse(content={"error": "File not found"}, status_code=404)

    # Attempt to open the file with retries to handle potential file locks
    retries = 3
    for i in range(retries):
        try:
            with open(video_path, 'rb') as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(0)
            break  # Successfully opened file, exit retry loop
        except PermissionError as e:
            print(f"❌ PermissionError on attempt {i+1}: {e}")
            time.sleep(1)
    else:
        return JSONResponse(content={"error": "File is locked by another process"}, status_code=500)

    if size == 0:
        return JSONResponse(content={"error": "File is empty"}, status_code=500)

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=decoded_video_name
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
