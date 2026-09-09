import os
import shutil
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="UAV 3D Reconstruction API")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Project directory paths based on data contract
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "input"
RECON_DIR = BASE_DIR / "reconstruction"
RESULTS_DIR = BASE_DIR / "results"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory execution state
pipeline_state = {
    "status": "idle",  # idle | running | completed | failed
    "message": "System ready",
    "step": 0
}


def execute_pipeline():
    global pipeline_state
    pipeline_state["status"] = "running"
    pipeline_state["message"] = "Executing reconstruction pipeline..."
    pipeline_state["step"] = 1

    script_path = RECON_DIR / "pipeline.py"
    if not script_path.exists():
        pipeline_state["status"] = "failed"
        pipeline_state["message"] = f"Pipeline script not found at {script_path}"
        return

    import sys
    import json
    video_path = INPUT_DIR / "video.mp4"
    config_path = BASE_DIR / "backend_config.json"
    
    # Write custom pipeline config override
    with open(config_path, "w") as f:
        json.dump({
            "input_path": str(video_path),
            "min_images": 2,
            "sim_threshold": 0.999,
            "blur_threshold": 10.0
        }, f, indent=2)

    try:
        # Run Person 1's end-to-end pipeline script using the same Python interpreter (venv)
        result = subprocess.run(
            [sys.executable, str(script_path), "--input", str(video_path), "--min-images", "2", "--config", str(config_path)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=True
        )
        pipeline_state["status"] = "completed"
        pipeline_state["message"] = "Reconstruction finished successfully"
        pipeline_state["step"] = 3
    except subprocess.CalledProcessError as e:
        pipeline_state["status"] = "failed"
        pipeline_state["message"] = f"Pipeline execution error: {e.stderr}"


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Accepts a new drone video from the frontend and saves it as the input."""
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)
    file_path = input_dir / "video.mp4"
    
    # Clear previous run outputs on new upload
    output_dir = BASE_DIR / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)

    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        return {"status": "success", "saved_to": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run-pipeline")
async def run_pipeline(background_tasks: BackgroundTasks):
    global pipeline_state
    video_path = INPUT_DIR / "video.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=400, detail="No video found in input/video.mp4. Upload a video first.")

    if pipeline_state["status"] == "running":
        return {"status": "already_running", "message": pipeline_state["message"]}

    background_tasks.add_task(execute_pipeline)
    return {"status": "started", "message": "Pipeline initiated in background"}


@app.get("/api/status")
async def get_status():
    report_file = BASE_DIR / "output" / "processing_report.json"
    if not report_file.exists():
        report_file = BASE_DIR / "output" / "results" / "processing_report.json"
        
    report_data = None
    if report_file.exists():
        try:
            import json
            with open(report_file, "r") as f:
                report_data = json.load(f)
        except Exception:
            pass

    return {
        "pipeline": pipeline_state,
        "report": report_data
    }


@app.get("/api/results/pointcloud")
async def get_pointcloud():
    ply_path = RECON_DIR / "pointcloud.ply"
    if not ply_path.exists():
        raise HTTPException(status_code=404, detail="Point cloud model not generated yet")
    return FileResponse(path=ply_path, filename="pointcloud.ply", media_type="application/octet-stream")


@app.get("/api/results/splat")
async def get_splat():
    splat_path = RECON_DIR / "splat.ply"
    if not splat_path.exists():
        raise HTTPException(status_code=404, detail="Gaussian splat not generated yet")
    return FileResponse(path=splat_path, filename="splat.ply", media_type="application/octet-stream")