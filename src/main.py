import os
import shutil
import uuid
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.automl.automl import AutoMLRunner

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent # Project root
TEMP_RUNS_DIR = BASE_DIR / "temp_runs"
TEMP_RUNS_DIR.mkdir(exist_ok=True) # Create directory if it doesn't exist

# --- FastAPI App ---
app = FastAPI(title="AutoML API")

# In-memory Job Status Store
job_status: Dict[str, Any] = {
    "job_id": None,
    "status": "idle", # idle, processing, completed, failed
    "message": "No job submitted yet.",
    "results": None # Will store paths to model, logs, plots, and metrics
}

# --- Background Task Function ---

def run_automl_processing(
    job_id: str,
    input_csv_path: Path,
    output_dir: Path,
    target_variable: str
):
    global job_status
    job_status["status"] = "processing"
    job_status["message"] = f"Processing data from {input_csv_path.name}..."
    job_status["results"] = None
    
    log_file_path = output_dir / f"{job_id}_log.txt"
    model_file_path = output_dir / f"{job_id}_model.pkl"
    
    try:
        runner = AutoMLRunner()
        
        # 1. Load Data
        print(f"[{job_id}] Loading data from {input_csv_path}...")
        df = runner.load_csv_data(str(input_csv_path))
        if df is None:
            raise ValueError("Failed to load or parse CSV data.")
        
        print(f"[{job_id}] Data loaded successfully. Shape: {df.shape}")
        
        # 2. Run Baseline Pipeline
        print(f"[{job_id}] Starting baseline pipeline for target '{target_variable}'...")
        
        # Placeholder implementation until AutoMLRunner is modified:
        print(f"[{job_id}] Simulating model training...")
        import time
        time.sleep(5) # Simulate work
        metrics = {"accuracy": 0.90, "precision": 0.88} # Dummy metrics
        # Simulate saving files
        log_file_path.write_text(f"[{job_id}] Processing started.\nData loaded.\nPipeline run.\nMetrics: {metrics}\nProcessing finished.")
        model_file_path.write_text("This is a placeholder model file.")
        print(f"[{job_id}] Pipeline finished.")
        # --- End Placeholder ---
        
        if metrics is None:
            raise RuntimeError("AutoML pipeline execution failed.")
        
        # 3. Update status - Success
        job_status["status"] = "completed"
        job_status["message"] = "AutoML processing completed successfully."
        job_status["results"] = {
            "metrics": metrics,
            "log_file": str(log_file_path.relative_to(BASE_DIR)),
            "model_file": str(model_file_path.relative_to(BASE_DIR)),
        }
        print(f"[{job_id}] Job completed successfully. Results stored.")
        
    except FileNotFoundError as e:
        print(f"[{job_id}] Error: {e}")
        job_status["status"] = "failed"
        job_status["message"] = f"Error: Input file not found or accessible - {e}"
        job_status["results"] = None
    except Exception as e:
        print(f"[{job_id}] Error during processing: {e}")
        job_status["status"] = "failed"
        job_status["message"] = f"An error occurred during processing: {str(e)}"
        job_status["results"] = None
        
        
# --- API Endpoints ---
@app.post("/run_automl", status_code=status.HTTP_202_ACCEPTED)
async def submit_automl_job(
    background_tasks: BackgroundTasks,
    target_variable: str = Form(...),
    file: UploadFile = File(...)
):
    global job_status
    if job_status["status"] == "processing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A job is already processing. Wait for it to complete."
        )
        
    job_id = str(uuid.uuid4())
    job_output_dir = TEMP_RUNS_DIR / job_id
    job_output_dir.mkdir(exist_ok=True)
    
    input_csv_path = job_output_dir / f"{job_id}_input_{file.filename}"
    
    # Save uploaded file
    try:
        with input_csv_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)  
    except Exception as e:
        job_output_dir.rmdir() 
    
        raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save uploaded file: {str(e)}"
            ) from e
    finally:
        await file.close()
        
    # Update global status immediately
    job_status = {
        "job_id": job_id,
        "status": "submitted",
        "message": "Job submitted, waiting to start processing...",
        "results": None
    }
    
    # Add the running task to the background
    background_tasks.add_task(
        run_automl_processing,
        job_id=job_id,
        input_csv_path=input_csv_path,
        output_dir=job_output_dir,
        target_variable=target_variable
    )
    
    return {"message": "Job submitted successfully.", "job_id": job_id}

@app.get("/status")
async def get_job_status():
    """
    Returns the status and results (if available) of the most recent job.
    """
    global job_status
    if not job_status["job_id"]:
         return JSONResponse(
             status_code=status.HTTP_404_NOT_FOUND,
             content={"message": "No job has been submitted yet."}
         )
    return job_status


    
    
