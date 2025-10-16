from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import torch
import uvicorn
import os
import uuid
import shutil
import asyncio
from datetime import datetime
from pathlib import Path
import logging
from queue import Queue
import threading
import json

from transformers.models.voxtral.modeling_voxtral import VoxtralForConditionalGeneration
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_DIR = Path("./server_uploads")
RECORDINGS_DIR = Path("./server_recordings")
RESULTS_DIR = Path("./server_results")

UPLOAD_DIR.mkdir(exist_ok=True)
RECORDINGS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Global variables
model = None
processor = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Enhanced job management
jobs = {}  # Completed/processing jobs
job_queue = Queue()  # Queue for pending jobs
active_websockets: Dict[str, WebSocket] = {}  # WebSocket connections
processing_lock = threading.Lock()
is_processing = False

# Statistics
stats = {
    "total_jobs": 0,
    "completed_jobs": 0,
    "failed_jobs": 0,
    "avg_processing_time": 0,
    "processing_times": []
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    try:
        logger.info("Loading Voxtral model...")
        repo_id = "/home/dlpda/vishnu/Voxtral-Mini-3B-2507"
        processor = AutoProcessor.from_pretrained(repo_id)
        model = VoxtralForConditionalGeneration.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16, device_map=device
        )
        logger.info(f"Model loaded successfully on {device}")
        
        # Start queue processor thread
        processor_thread = threading.Thread(target=queue_processor_thread, daemon=True)
        processor_thread.start()
        logger.info("Queue processor started")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    logger.info("Shutting down...")


app = FastAPI(title="Voxtral AI Server - Enhanced", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GlobalSummaryRequest(BaseModel):
    texts: List[str]


class SaveResultsRequest(BaseModel):
    meeting_date: str
    meeting_time: str
    venue: str
    agenda: str
    context: str
    results: List[dict]


# Serve HTML Client
@app.get("/", response_class=HTMLResponse)
async def serve_client():
    try:
        with open("client_enhanced.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: client_enhanced.html not found</h1>",
            status_code=404
        )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "queue_length": job_queue.qsize(),
        "is_processing": is_processing,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/queue_status")
async def get_queue_status():
    """Get current queue status and statistics"""
    queue_size = job_queue.qsize()
    estimated_wait = queue_size * 45  # 45 seconds per job average
    
    return {
        "queue_length": queue_size,
        "estimated_wait_seconds": estimated_wait,
        "estimated_wait_minutes": round(estimated_wait / 60, 1),
        "is_processing": is_processing,
        "total_jobs_today": stats["total_jobs"],
        "completed_jobs": stats["completed_jobs"],
        "failed_jobs": stats["failed_jobs"],
        "avg_processing_time": round(stats["avg_processing_time"], 1),
        "server_load": "high" if queue_size > 10 else "medium" if queue_size > 5 else "low"
    }


@app.post("/process_audio")
async def process_audio(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        mode: str = Form(...),
        summary_length: str = Form(default="Medium")
):
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    valid_modes = ["Transcription", "Summary", "Action Points"]
    if mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of {valid_modes}")

    # Generate job ID
    job_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = UPLOAD_DIR / f"{job_id}_{timestamp}_{file.filename}"

    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved audio file: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Get current queue position
    queue_position = job_queue.qsize() + 1
    estimated_wait = queue_position * 45

    # Create job entry
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "queue_position": queue_position,
        "mode": mode,
        "summary_length": summary_length if mode == "Summary" else None,
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "file_path": str(file_path),
        "estimated_wait_seconds": estimated_wait
    }

    # Add to queue
    job_queue.put({
        "job_id": job_id,
        "file_path": str(file_path),
        "mode": mode,
        "summary_length": summary_length
    })

    stats["total_jobs"] += 1

    logger.info(f"Job {job_id} queued - Position: {queue_position}, Mode: {mode}, Estimated wait: {estimated_wait}s")

    return {
        "job_id": job_id,
        "status": "queued",
        "queue_position": queue_position,
        "estimated_wait_seconds": estimated_wait,
        "estimated_wait_minutes": round(estimated_wait / 60, 1),
        "mode": mode,
        "message": f"Job queued. Position: {queue_position}. Estimated wait: {round(estimated_wait/60, 1)} minutes"
    }


def queue_processor_thread():
    """Background thread that processes jobs from queue"""
    global is_processing
    
    logger.info("Queue processor thread started")
    
    while True:
        try:
            # Get job from queue (blocks until available)
            job_data = job_queue.get()
            job_id = job_data["job_id"]
            
            with processing_lock:
                is_processing = True
            
            logger.info(f"Processing job {job_id} from queue")
            
            # Update job status
            if job_id in jobs:
                jobs[job_id]["status"] = "processing"
                jobs[job_id]["started_at"] = datetime.now().isoformat()
                jobs[job_id]["queue_position"] = 0
            
            # Notify via WebSocket
            asyncio.run(notify_websocket(job_id, {"status": "processing", "message": "Processing started"}))
            
            # Process the job
            process_audio_sync(
                job_id,
                job_data["file_path"],
                job_data["mode"],
                job_data["summary_length"]
            )
            
            job_queue.task_done()
            
            with processing_lock:
                is_processing = False
                
        except Exception as e:
            logger.error(f"Queue processor error: {str(e)}")
            with processing_lock:
                is_processing = False


def process_audio_sync(job_id: str, file_path: str, mode: str, summary_length: str):
    """Synchronous audio processing"""
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Starting processing for job {job_id} - Mode: {mode}")

        if mode == "Transcription":
            inputs = processor.apply_transcription_request(
                language="en",
                audio=file_path,
                model_id="/home/dlpda/vishnu/Voxtral-Mini-3B-2507"
            )
            inputs = inputs.to(device, dtype=torch.bfloat16)
            outputs = model.generate(**inputs, max_new_tokens=4096)
            decoded = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            result = f"[TRANSCRIPTION]\n\n{decoded[0]}"

        elif mode == "Summary":
            instructions = {
                "Short": "generate a short, concise summary",
                "Medium": "generate a medium-length, detailed summary",
                "Long": "generate a comprehensive, long-form summary"
            }
            instruction = instructions.get(summary_length, instructions["Medium"])

            conversation = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        f"You are a helpful assistant that works in summarizer mode. "
                        f"After listening to the audio, please understand the total audio, "
                        f"then {instruction} that should be organised topic-wise, with clear things, "
                        f"within the audio by utilizing entire input audio. "
                        f"Don't go for hallucinations and extended sentences which are not in the audio. "
                        f"Keep the summary within provided audio only."
                    )
                }]
            }, {
                "role": "user",
                "content": [{"type": "audio", "path": file_path}]
            }]

            inputs = processor.apply_chat_template(conversation)
            inputs = inputs.to(device, dtype=torch.bfloat16)
            outputs = model.generate(**inputs, max_new_tokens=2048)
            decoded = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            result = f"[SUMMARY - {summary_length}]\n\n{decoded[0]}"

        elif mode == "Action Points":
            conversation = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        "You are voxtral, a helpful assistant that works in action points extraction mode. "
                        "After listening the audio, extract all key action points. "
                        "Each action point should clearly state:\n"
                        "- The task or decision\n"
                        "- Who is responsible\n"
                        "- The deadline or timeline (if mentioned)\n"
                        "- Any dependencies or resources needed\n"
                        "Present the output in a numbered list.\n"
                        "If audio has no action points, then give the text you understood clearly."
                    )
                }]
            }, {
                "role": "user",
                "content": [{"type": "audio", "path": file_path}]
            }]

            inputs = processor.apply_chat_template(conversation)
            inputs = inputs.to(device, dtype=torch.bfloat16)
            outputs = model.generate(**inputs, max_new_tokens=2048)
            decoded = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            result = f"[ACTION POINTS]\n\n{decoded[0]}"

        # Calculate processing time
        processing_time = time.time() - start_time
        stats["processing_times"].append(processing_time)
        if len(stats["processing_times"]) > 100:
            stats["processing_times"] = stats["processing_times"][-100:]
        stats["avg_processing_time"] = sum(stats["processing_times"]) / len(stats["processing_times"])

        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["processing_time_seconds"] = round(processing_time, 2)

        stats["completed_jobs"] += 1

        # Notify via WebSocket
        asyncio.run(notify_websocket(job_id, {
            "status": "completed",
            "result": result,
            "processing_time": round(processing_time, 2)
        }))

        logger.info(f"Job {job_id} completed successfully in {processing_time:.2f}s - Mode: {mode}")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        stats["failed_jobs"] += 1
        
        # Notify via WebSocket
        asyncio.run(notify_websocket(job_id, {
            "status": "error",
            "error": str(e)
        }))

    finally:
        # Clean up temporary file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to remove file {file_path}: {str(e)}")


async def notify_websocket(job_id: str, data: dict):
    """Send update to WebSocket if connected"""
    if job_id in active_websockets:
        try:
            await active_websockets[job_id].send_json(data)
        except Exception as e:
            logger.error(f"WebSocket notification failed for {job_id}: {str(e)}")


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates"""
    await websocket.accept()
    active_websockets[job_id] = websocket
    logger.info(f"WebSocket connected for job {job_id}")
    
    try:
        # Send initial status
        if job_id in jobs:
            await websocket.send_json({
                "status": jobs[job_id]["status"],
                "queue_position": jobs[job_id].get("queue_position", 0)
            })
        
        # Keep connection alive and send updates
        while True:
            if job_id in jobs:
                job = jobs[job_id]
                
                if job["status"] in ["completed", "error"]:
                    await websocket.send_json(job)
                    break
                
                # Send periodic updates
                await websocket.send_json({
                    "status": job["status"],
                    "queue_position": job.get("queue_position", 0)
                })
            
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {str(e)}")
    finally:
        if job_id in active_websockets:
            del active_websockets[job_id]


@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job_data = jobs[job_id].copy()
    if "file_path" in job_data:
        del job_data["file_path"]

    return job_data


@app.post("/global_summary")
async def generate_global_summary(request: GlobalSummaryRequest):
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        filtered = [t for t in request.texts if not t.strip().startswith("[TRANSCRIPTION]")]
        if not filtered:
            raise HTTPException(status_code=400, detail="No valid summaries or action points to process")

        combined = "\n\n".join(filtered)

        conversation = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "You are a helpful assistant that works in summarizer mode.\n"
                    "Please read and understand the entire provided text, then "
                    "don't miss any topic in the entire provided text. "
                    "Provide at least short summary for all the topics which are in long text. "
                    "Generate a global summary that should be organised topic-wise in a short-way "
                    "& important key points with clear things. "
                    "Keep the summary within provided text only. "
                    "If you find any action points in the entire text please append it in the bottom, "
                    "because action points should be kept in the last.\n\n"
                    f"Here is the text to summarize:\n\n{combined}"
                )
            }]
        }]

        inputs = processor.apply_chat_template(conversation)
        inputs = inputs.to(device, dtype=torch.bfloat16)
        outputs = model.generate(**inputs, max_new_tokens=4096)
        decoded = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return {
            "status": "success",
            "global_summary": decoded[0]
        }

    except Exception as e:
        logger.error(f"Global summary generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save_results")
async def save_results(request: SaveResultsRequest):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = RESULTS_DIR / f"meeting_minutes_{timestamp}.md"

        content = f"""# Meeting Minutes

## Meeting Information
- **Date:** {request.meeting_date}
- **Time:** {request.meeting_time}
- **Venue:** {request.venue}
- **Agenda:** {request.agenda}

## Additional Context
{request.context}

## Results

"""

        for i, result in enumerate(request.results, 1):
            content += f"### Result #{i} - {result.get('mode', 'Unknown')}\n\n"
            content += f"{result.get('result', '')}\n\n"
            content += "---\n\n"

        content += f"\n*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Results saved to {filename}")

        return {
            "status": "success",
            "filename": str(filename),
            "message": f"Results saved to {filename}"
        }

    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download_results/{filename}")
async def download_results(filename: str):
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='text/markdown', filename=filename)


if __name__ == "__main__":
    import socket
    
    def find_free_port(start=8000):
        for port in range(start, start + 10):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('', port))
                sock.close()
                return port
            except OSError:
                continue
        return None
    
    port = find_free_port()
    if port:
        print(f"\n{'=' * 60}")
        print(f"🚀 Starting Enhanced Voxtral Server on http://localhost:{port}")
        print(f"🌐 Access the app at: http://localhost:{port}")
        print(f"📚 API Docs: http://localhost:{port}/docs")
        print(f"✨ Features: Queue Management, WebSocket Support, Statistics")
        print(f"{'=' * 60}\n")
        uvicorn.run(app, host="10.144.179.83", port=port)
    else:
        print("❌ No free ports available")
