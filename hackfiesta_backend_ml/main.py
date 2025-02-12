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
        "timestamp_counts": dict(timestamp_counts),  # Convert defaultdict to regular dict
        "total_unique_people_estimate": int(np.mean(all_detections) * 1.5),  # Rough estimate accounting for people moving in/out
        "average_people_per_frame": round(np.mean(all_detections), 2),
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
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        # Save uploaded file
        with open(temp_video.name, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        output_path = OUTPUT_DIR / file.filename
        print(f"Processing video: Input={temp_video.name}, Output={output_path}")
        
        # Process the video
        success, stats = process_video(temp_video.name, str(output_path))
        
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

        encoded_filename = quote(file.filename)
        download_url = f"/download/{encoded_filename}"

        return {
            "message": f"Video {file.filename} has been processed.",
            "download_url": download_url,
            "file_size": size,
            "statistics": stats
        }
    finally:
        # Cleanup temp file
        Path(temp_video.name).unlink(missing_ok=True)

@app.get("/download/{video_name}")
async def download(video_name: str):
    """Download the processed video."""
    decoded_video_name = unquote(video_name)
    video_path = OUTPUT_DIR / decoded_video_name
    
    if not video_path.exists():
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    
    try:
        with open(video_path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(0)
            
        if size == 0:
            return JSONResponse(content={"error": "File is empty"}, status_code=500)
            
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=decoded_video_name
        )
    except Exception as e:
        print(f"❌ Error serving video: {e}")
        return JSONResponse(
            content={"error": f"Error accessing video: {str(e)}"},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)