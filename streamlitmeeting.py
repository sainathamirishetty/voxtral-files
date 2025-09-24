import streamlit as st
import multiprocessing as mp
import pyaudio
import numpy as np
import whisper
from datetime import datetime
import queue
import os, time
import wave
from glob import glob
from markdown2 import markdown
import threading
import traceback
import html2docx
import transformers
from transformers.models.voxtral.modeling_voxtral import VoxtralForConditionalGeneration
from transformers import AutoProcessor
import torch
import tempfile
import io

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.meeting_active = False
    st.session_state.current_processing_mode = "Transcription"
    st.session_state.all_minutes = []
    st.session_state.all_transcript = []
    st.session_state.status_messages = []
    st.session_state.summary_cnt = 1
    st.session_state.recording_process = None
    st.session_state.transcription_process = None
    st.session_state.start_time = None
    st.session_state.running = False
    st.session_state.paused = False

# Directory setup
if not os.path.isdir('recordings'):
    os.mkdir('recordings')

recordings_cnt = len(glob('recordings/record_*'))
base_directory = os.path.abspath('')
write_record_to_dir = f'recordings/record_{recordings_cnt}'
if not os.path.isdir(write_record_to_dir):
    os.mkdir(write_record_to_dir)

# Global variables
CHANNELS = 1
RATE = 44100
CHUNK = 1024
FORMAT = pyaudio.paInt16
DTYPE = np.int16 if FORMAT == pyaudio.paInt16 else np.float32
MEETING_CHUNK_DURATION_MIN = 10

# Global queues and flags (now using manager for cross-process communication)
if not st.session_state.initialized:
    manager = mp.Manager()
    st.session_state.audio_queue_lst = manager.Queue()
    st.session_state.summary_queue_gui = manager.Queue()
    st.session_state.update_status_queue_gui = manager.Queue()
    st.session_state.exit_flag = manager.Event()
    st.session_state.pause_flag = manager.Event()
    st.session_state.stop_recording_flag = manager.Event()
    st.session_state.initialized = True

def update_status(message):
    """Add status message to session state"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    st.session_state.status_messages.append(f"{timestamp}: {message}")
    print(message, flush=True)

def get_summary_mode_config(mode):
    """Get summary mode configuration based on selection"""
    if mode == "Short":
        return "generate a short, concise summary"
    elif mode == "Medium":
        return "generate a medium-length, detailed summary"
    elif mode == "Long":
        return "generate a comprehensive, long-form summary"
    else:
        return "summarize the audio appropriately"

def record_audio(audio_queue_lst, exit_flag, stop_recording_flag, pause_flag, current_mode):
    """Record audio in chunks with mode tagging"""
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                            frames_per_buffer=CHUNK)

        duration_minutes = MEETING_CHUNK_DURATION_MIN
        record_seconds = duration_minutes * 60
        file_cnt = 0
        stop_recording_flag.clear()

        while not stop_recording_flag.is_set():
            # Capture the current mode at the START of recording this chunk
            chunk_mode = current_mode.value
            
            frames = []
            for i in range(int(RATE / CHUNK * record_seconds)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                if not pause_flag.is_set():
                    frames.append(data)
                if exit_flag.is_set() or stop_recording_flag.is_set():
                    break
                    
            if frames:
                file_cnt += 1
                wave_output_filename = f"{write_record_to_dir}/recording_{datetime.now().strftime('%H-%M-%S')}.wav"
                wave_file = wave.open(wave_output_filename, 'wb')
                wave_file.setnchannels(CHANNELS)
                wave_file.setsampwidth(audio.get_sample_size(FORMAT))
                wave_file.setframerate(RATE)
                wave_file.writeframes(b''.join(frames))
                wave_file.close()
                
                full_file_path = os.path.join(base_directory, wave_output_filename)
                audio_queue_lst.put((full_file_path, chunk_mode))
                
        stream.stop_stream()
        stream.close()
        audio.terminate()
    except Exception as e:
        print(f"Error in audio recording: {str(e)}")

def transcribe_audio(audio_queue_lst, exit_flag, summary_queue_gui):
    """Process audio files based on their tagged mode"""
    try:
        device = "cpu"
        repo_id = "/Users/vishnukumarkudidela/Desktop/workspace/ASR/models/Voxtral-Mini-3B-2507"

        processor = AutoProcessor.from_pretrained(repo_id)
        model = VoxtralForConditionalGeneration.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16, device_map=device
        )

        while not exit_flag.is_set():
            try:
                if audio_queue_lst.empty():
                    time.sleep(0.5)
                    continue
                
                file_path, chunk_mode = audio_queue_lst.get()
                print(f"Processing audio: {file_path} with mode: {chunk_mode}")

                decoded_outputs = []

                # Process based on chunk mode
                if chunk_mode == "Transcription":
                    inputs = processor.apply_transcription_request(language="en", audio=file_path, model_id=repo_id)
                    inputs = inputs.to(device, dtype=torch.bfloat16)
                    outputs = model.generate(**inputs)
                    decoded_outputs = processor.batch_decode(
                        outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
                    )
                    result = f"[TRANSCRIPTION]\n{decoded_outputs[0]}"
                    
                elif chunk_mode == "Summary":
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "you are voxtral, a helpful assistant that works in below mode:\n"
                                        "Summarizer: after listening the audio, please understand the total audio, "
                                        "then only generate a clear, concise summary that should be organised topic-wise. "
                                        "Stay strictly within the audio content, don't hallucinate."
                                    )
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "audio", "path": file_path}],
                        },
                    ]

                    inputs = processor.apply_chat_template(conversation)
                    inputs = inputs.to(device, dtype=torch.bfloat16)
                    outputs = model.generate(**inputs)
                    decoded_outputs = processor.batch_decode(
                        outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
                    )
                    result = f"[SUMMARY]\n{decoded_outputs[0]}"
                    
                elif chunk_mode == "Action Points":
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "you are voxtral, a helpful assistant that works in below mode:\n"
                                        "Action Points: after listening the audio, extract all key action points. "
                                        "Each action point should clearly state:\n"
                                        "- The task or decision\n"
                                        "- Who is responsible\n"
                                        "- The deadline or timeline (if mentioned)\n"
                                        "- Any dependencies or resources needed\n"
                                        "Present the output in a numbered list.\n"
                                        "If audio has no action points, then give the text you understood clearly."
                                    )
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "audio", "path": file_path}],
                        },
                    ]

                    inputs = processor.apply_chat_template(conversation)
                    inputs = inputs.to(device, dtype=torch.bfloat16)
                    outputs = model.generate(**inputs)
                    decoded_outputs = processor.batch_decode(
                        outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
                    )
                    result = f"[ACTION POINTS]\n{decoded_outputs[0]}"

                # Add to queue for UI display
                summary_queue_gui.put(result)

            except Exception as e:
                print(f"Error in processing audio: {str(e)}")
                
    except Exception as e:
        print(f"Error in transcribe_audio: {str(e)}")

def start_meeting():
    """Start the meeting recording and processing"""
    if st.session_state.meeting_active:
        return
        
    try:
        st.session_state.meeting_active = True
        st.session_state.start_time = time.time()
        st.session_state.running = True
        
        # Create manager for sharing current mode
        manager = mp.Manager()
        current_mode_shared = manager.Value('c', st.session_state.current_processing_mode.encode())
        st.session_state.current_mode_shared = current_mode_shared
        
        # Start recording process
        st.session_state.recording_process = mp.Process(
            target=record_audio,
            args=(st.session_state.audio_queue_lst, st.session_state.exit_flag, 
                 st.session_state.stop_recording_flag, st.session_state.pause_flag,
                 current_mode_shared),
            daemon=True
        )
        st.session_state.recording_process.start()
        
        # Start transcription process
        st.session_state.transcription_process = mp.Process(
            target=transcribe_audio,
            args=(st.session_state.audio_queue_lst, st.session_state.exit_flag,
                 st.session_state.summary_queue_gui),
            daemon=True
        )
        st.session_state.transcription_process.start()
        
        update_status("Meeting started successfully")
        
    except Exception as e:
        update_status(f"Error starting meeting: {str(e)}")
        st.session_state.meeting_active = False

def stop_meeting():
    """Stop the meeting and all processes"""
    try:
        st.session_state.stop_recording_flag.set()
        st.session_state.running = False
        st.session_state.meeting_active = False
        
        # Wait for processes to finish
        if st.session_state.recording_process and st.session_state.recording_process.is_alive():
            st.session_state.recording_process.join(timeout=5)
            if st.session_state.recording_process.is_alive():
                st.session_state.recording_process.terminate()
                
        if st.session_state.transcription_process and st.session_state.transcription_process.is_alive():
            st.session_state.transcription_process.join(timeout=5)
            if st.session_state.transcription_process.is_alive():
                st.session_state.transcription_process.terminate()
                
        update_status("Meeting ended successfully")
        
    except Exception as e:
        update_status(f"Error stopping meeting: {str(e)}")

def toggle_pause():
    """Toggle pause/resume meeting"""
    if st.session_state.paused:
        st.session_state.pause_flag.clear()
        st.session_state.paused = False
        update_status("Meeting resumed")
    else:
        st.session_state.pause_flag.set()
        st.session_state.paused = True
        update_status("Meeting paused")

def change_processing_mode(new_mode):
    """Change the current processing mode"""
    if hasattr(st.session_state, 'current_mode_shared'):
        old_mode = st.session_state.current_processing_mode
        st.session_state.current_processing_mode = new_mode
        st.session_state.current_mode_shared.value = new_mode.encode()
        update_status(f"Mode switched: {old_mode} → {new_mode}")

def update_outputs():
    """Update outputs from processing queues"""
    try:
        # Get new results from processing
        while not st.session_state.summary_queue_gui.empty():
            try:
                result = st.session_state.summary_queue_gui.get_nowait()
                st.session_state.all_minutes.append(result)
            except:
                break
    except:
        pass

# Streamlit App Layout
st.set_page_config(page_title="Meeting Minutes Generator", layout="wide")

st.title("🎙️ Meeting Minutes Generator")

# Sidebar for controls
with st.sidebar:
    st.header("📋 Meeting Details")
    
    # Meeting information
    meeting_date = st.date_input("Date", datetime.now().date())
    meeting_time = st.time_input("Time", datetime.now().time())
    venue = st.text_input("Venue")
    agenda = st.text_area("Agenda")
    context = st.text_area("Additional Context")
    
    st.divider()
    
    # Processing Mode Selection
    st.header("🔧 Processing Mode")
    
    # Radio buttons for mode selection
    processing_mode = st.radio(
        "Select Mode:",
        ["Transcription", "Summary", "Action Points"],
        index=["Transcription", "Summary", "Action Points"].index(st.session_state.current_processing_mode)
    )
    
    # Summary mode options (only show if Summary is selected)
    summary_mode = None
    if processing_mode == "Summary":
        summary_mode = st.selectbox(
            "Summary Length:",
            ["Short", "Medium", "Long"]
        )
    
    # Change mode button (only enabled during meeting)
    if st.button("🔄 Change Mode", disabled=not st.session_state.meeting_active):
        change_processing_mode(processing_mode)
        st.rerun()
    
    st.divider()
    
    # Meeting Controls
    st.header("🎛️ Meeting Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Start Meeting", disabled=st.session_state.meeting_active):
            if processing_mode == "Summary" and not summary_mode:
                st.error("Please select summary mode first!")
            else:
                st.session_state.current_processing_mode = processing_mode
                start_meeting()
                st.rerun()
        
        if st.button("⏸️ Pause/Resume", disabled=not st.session_state.meeting_active):
            toggle_pause()
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Meeting", disabled=not st.session_state.meeting_active):
            stop_meeting()
            st.rerun()
        
        # Audio file upload
        uploaded_files = st.file_uploader(
            "📁 Upload Audio Files", 
            type=['wav', 'mp3'], 
            accept_multiple_files=True
        )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 Status & Control")
    
    # Status indicator
    if st.session_state.meeting_active:
        if st.session_state.paused:
            st.warning(f"⏸️ **PAUSED** - Mode: {st.session_state.current_processing_mode}")
        else:
            st.success(f"🔴 **RECORDING** - Mode: {st.session_state.current_processing_mode}")
    else:
        st.info("⏹️ **STOPPED** - No active recording")
    
    # Meeting duration
    if st.session_state.running and st.session_state.start_time:
        elapsed = int(time.time() - st.session_state.start_time)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        st.metric("Meeting Duration", f"{hours:02d}:{minutes:02d}:{seconds:02d}")
    
    # Status messages
    st.subheader("📝 Status Log")
    status_container = st.container()
    with status_container:
        if st.session_state.status_messages:
            for message in st.session_state.status_messages[-10:]:  # Show last 10 messages
                st.text(message)
        else:
            st.text("No status messages yet...")

with col2:
    st.header("📄 Live Output")
    
    # Update outputs
    update_outputs()
    
    # Display results
    if st.session_state.all_minutes:
        for i, result in enumerate(st.session_state.all_minutes):
            with st.expander(f"Result {i+1}", expanded=True):
                st.text(result)
    else:
        st.info("No output yet. Start a meeting to see results here.")

# Bottom section - Save functionality
st.divider()
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("💾 Save Minutes") and st.session_state.all_minutes:
        # Combine all results
        content = f"""Meeting Date: {meeting_date}
Meeting Time: {meeting_time}
Venue: {venue}
Agenda: {agenda}
Additional Context: {context}

MINUTES OF DISCUSSION:
{'='*50}
"""
        for i, result in enumerate(st.session_state.all_minutes):
            content += f"\n{result}\n{'-'*30}\n"
        
        # Create download
        st.download_button(
            label="📥 Download Minutes (TXT)",
            data=content,
            file_name=f"meeting_minutes_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

with col2:
    if st.button("📊 Export Status Log"):
        log_content = "\n".join(st.session_state.status_messages)
        st.download_button(
            label="📥 Download Log",
            data=log_content,
            file_name=f"meeting_log_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

with col3:
    if st.button("🔄 Reset Session"):
        # Reset all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Auto-refresh for live updates
if st.session_state.meeting_active:
    time.sleep(2)
    st.rerun()