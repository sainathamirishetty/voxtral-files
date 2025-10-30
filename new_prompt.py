from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import torch
import uvicorn
import os
import uuid
import shutil
import wave
from datetime import datetime
from pathlib import Path
import logging
import socket
import time
import subprocess

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

model = None
processor = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"
jobs = {}
all_results = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, processor
    try:
        logger.info("Loading AI model...")
        repo_id = "/Users/vishnukumarkudidela/Desktop/workspace/ASR/models/Voxtral-Mini-3B-2507"
        processor = AutoProcessor.from_pretrained(repo_id)
        model = VoxtralForConditionalGeneration.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16, device_map=device
        )
        logger.info(f"Model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(title="DEAIS AI Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/mom_icons", StaticFiles(directory="mom_icons"), name="mom_icons")


class GlobalSummaryRequest(BaseModel):
    texts: List[str]


class SaveResultsRequest(BaseModel):
    meeting_date: str
    meeting_time: str
    venue: str
    agenda: str
    context: str
    results: List[dict]


# ============================================================================
# HTML ROUTES - Serve different pages
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the index.html entry point"""
    try:
        with open("index_oct26.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: index.html not found</h1><p>Please ensure index.html is in the same directory as server.py</p>",
            status_code=404
        )


@app.get("/rms_zcr1.html", response_class=HTMLResponse)
async def serve_audio_assistant():
    """Serve the Audio Assistant (simple_transcription_page.html)"""
    try:
        with open("rms_zcr1.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: simple_transcription_page.html not found</h1>",
            status_code=404
        )


@app.get("/full_meeting_system_oct26_docx21.html", response_class=HTMLResponse)
async def serve_meeting_minutes():
    """Serve the Meeting Minutes Generator (full_meeting_system.html)"""
    try:
        with open("full_meeting_system_oct26_docx21.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: full_meeting_system.html not found</h1>",
            status_code=404
            )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/process_audio")
async def process_audio(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        mode: str = Form(...),
        summary_length: str = Form(default="Medium")
):
    """Process uploaded audio file for transcription, summary, or action points"""
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate mode
    valid_modes = ["Transcription", "Summary", "Action Points"]
    if mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of {valid_modes}")

    job_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # CHANGED: Save to RECORDINGS_DIR instead of UPLOAD_DIR
    file_path = RECORDINGS_DIR / f"{timestamp}_{mode.replace(' ', '_')}_{file.filename}"

    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved audio file: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Create job entry
    jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "mode": mode,
        "summary_length": summary_length if mode == "Summary" else None,
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "file_path": str(file_path)
    }

    # Process in background
    background_tasks.add_task(process_audio_task, job_id, str(file_path), mode, summary_length)

    logger.info(f"Job {job_id} created - Mode: {mode}, Length: {summary_length}")

    return {
        "job_id": job_id,
        "status": "processing",
        "mode": mode,
        "message": f"Processing audio with mode: {mode}"
    }


async def process_audio_task(job_id: str, file_path: str, mode: str, summary_length: str):
    """Background task to process audio"""
    try:
        logger.info(f"Starting processing for job {job_id} - Mode: {mode}")

        if mode == "Transcription":
            logger.info(f"Processing Transcription for {job_id}")
            inputs = processor.apply_transcription_request(
                language="en",
                audio=file_path,
                model_id="/Users/vishnukumarkudidela/Desktop/workspace/ASR/models/Voxtral-Mini-3B-2507"
            )
            inputs = inputs.to(device, dtype=torch.bfloat16)
            outputs = model.generate(**inputs, max_new_tokens=4096)
            decoded = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            result = f"[TRANSCRIPTION]\n\n{decoded[0]}"

        elif mode == "Summary":
            logger.info(f"Processing Summary ({summary_length}) for {job_id}")
            instructions = {
                "Short": "Write all information as point-wise bullet lines using ➤ symbol and Do NOT include any section headings. Keep each point concise and complete.",
                "Medium": "Generate a short, concise summary with proper indentation, donot include markdown syntax",
                "Long": "Generate a comprehensive, long-form summary."
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
                        f"generate the text within the audio by utilizing entire input audio. "
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
            logger.info(f"Processing Action Points for {job_id}")
            conversation = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        "You are a helpful assistant that works in action points extraction mode. "
                        "After listening the audio, extract all key action points. "
                        "Each action point should clearly state:\n"
                        "- The task or decision\n"
                        "- Who is responsible\n"
                        "- The deadline or timeline (if mentioned)\n"
                        "- Any dependencies or resources needed\n"
                        "Present the output using the ➤ symbol for each action point instead of numbers.\n"
                        "Format each action point exactly like this:\n"
                        "➤ Task/Decision: [description]\n"
                        "  • Responsible: [person]\n"
                        "  • Deadline: [date/timeline]\n\n"
                        "Do not use numbered lists (1., 2., 3., etc.). Use only the ➤ symbol for each action point.\n"
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

        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result
        jobs[job_id]["completed_at"] = datetime.now().isoformat()

        # Store in all results
        all_results.append({
            "mode": mode,
            "result": result,
            "timestamp": jobs[job_id]["completed_at"]
        })

        logger.info(f"Job {job_id} completed successfully - Mode: {mode}")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()


    finally:

        # CHANGED: Keep files in RECORDINGS_DIR, don't delete

        logger.info(f"Audio file saved permanently at: {file_path}")


@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    """Check the status of a processing job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job_data = jobs[job_id].copy()
    # Don't send file_path to client
    if "file_path" in job_data:
        del job_data["file_path"]

    return job_data


@app.post("/global_summary")
async def generate_global_summary(request: GlobalSummaryRequest):
    """Generate a global summary from multiple text results"""
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        # Filter out transcriptions, keep only summaries and action points
        filtered = [t for t in request.texts if not t.strip().startswith("[TRANSCRIPTION]")]

        if not filtered:
            raise HTTPException(status_code=400, detail="No valid summaries or action points to process")

        combined = "\n\n".join(filtered)

        conversation = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "You are an expert meeting summarizer. Your task is to create a comprehensive global summary "
                    "from multiple meeting segments.\n\n"

                    "INSTRUCTIONS:\n"
                    "1. Read and analyze all the provided text segments carefully\n"
                    "2. Identify and group related topics across all segments\n"
                    "3. Create a cohesive, well-organized summary that covers ALL topics discussed\n"
                    "4. Maintain chronological flow when relevant\n"
                    "5. Preserve important details, decisions, and discussions\n"
                    "6. Remove redundancies while keeping unique information from each segment\n\n"

                    "OUTPUT STRUCTURE:\n"
                    "- Start with an 'OVERVIEW' section (2-3 sentences about the overall meeting)\n"
                    "- Organize content by TOPICS with clear headings\n"
                    "- Under each topic, provide key points and discussions\n"
                    "- Include a 'KEY DECISIONS' section if any decisions were made\n"
                    "- End with 'ACTION ITEMS' section if action points exist\n\n"

                    "IMPORTANT GUIDELINES:\n"
                    "- Do NOT add information not present in the original text\n"
                    "- Do NOT miss any important topics, even if briefly mentioned\n"
                    "- Use clear, professional language\n"
                    "- Be concise but comprehensive\n"
                    "- If a topic appears in multiple segments, consolidate it intelligently\n\n"

                    f"TEXT TO SUMMARIZE:\n\n{combined}"
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
    """Save all results to markdown file"""
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
    """Download saved results file"""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type='text/markdown', filename=filename)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_free_port(start=8000):
    """Find an available port starting from the given port number"""
    for port in range(start, start + 10):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            continue
    return None


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    port = find_free_port()
    if port:
        print(f"\n{'=' * 70}")
        print(f"🚀 Starting Voxtral Server")
        print(f"{'=' * 70}")
        print(f"🏠 Main Entry:          http://localhost:{port}")
        print(f"🎤 Dictation:     http://localhost:{port}/simple_transcription_page.html")
        print(f"📝 Meeting Minutes:     http://localhost:{port}/full_meeting_system.html")
        print(f"{'=' * 70}")
        print(f"📚 API Documentation:   http://localhost:{port}/docs")
        print(f"💚 Health Check:        http://localhost:{port}/health")
        print(f"{'=' * 70}\n")
        uvicorn.run(app, host="127.0.0.1", port=port)
    else:
        print("❌ No free ports available")
