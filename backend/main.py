import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(
    title="SIH Drone 3D Reconstruction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "SIH UAV 3D Reconstruction Backend is running!"}

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Accepts a new drone video from the frontend and saves it as the input."""
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)
    file_path = input_dir / "drone_video.mp4"
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        return {"filename": file.filename, "status": "Uploaded successfully. Ready for processing."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/run-pipeline")
def run_pipeline():
    """Triggers the Python frame extractor and mask generator scripts sequentially."""
    try:
        # 1. Run Frame Extractor
        extract_result = subprocess.run(["python", "reconstruction/frame_selection/extractor.py"], capture_output=True, text=True)
        if extract_result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Extractor failed: {extract_result.stderr}")
            
        # 2. Run Mask Generator
        mask_result = subprocess.run(["python", "reconstruction/detection/mask_generator.py"], capture_output=True, text=True)
        if mask_result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Mask generator failed: {mask_result.stderr}")
            
        return {"status": "success", "message": "Frames extracted and masks generated successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download-output")
def download_output():
    """Provides a zip of the processed assets for Person 1 or the frontend."""
    # We can point this to a final packaged zip or model file later
    return {"status": "ready", "message": "Endpoint configured for model hand-off."}