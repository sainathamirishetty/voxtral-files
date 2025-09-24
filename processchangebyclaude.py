import multiprocessing as mp
import pyaudio
import numpy as np
import whisper
import tkinter as tk
from tkinter import scrolledtext, filedialog
from datetime import datetime
from tkinter import ttk
import queue
from resampy import resample
import os, time
import wave
from glob import glob
from markdown2 import markdown
import threading
import time
import traceback
import html2docx
import transformers
from transformers.models.voxtral.modeling_voxtral import VoxtralForConditionalGeneration
from transformers import AutoProcessor
import torch
import json
from collections import deque

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
MEETING_CHUNK_DURATION_MIN = 10  # Hardcoded chunk duration

# Enhanced queues and flags for dynamic mode switching
audio_queue = mp.Queue()
audio_queue_lst = mp.Queue()
transcript_queue = mp.Queue()
summary_queue = mp.Queue()
transcript_queue_gui = mp.Queue()
improve_queue_gui = mp.Queue()
summary_queue_gui = mp.Queue()
update_status_queue_gui = mp.Queue()

# NEW: Mode switching queues
mode_switch_queue = mp.Queue()  # For sending mode change requests
current_mode_queue = mp.Queue()  # For broadcasting current mode
processing_queues = {
    'transcription': mp.Queue(),
    'summarization': mp.Queue(), 
    'action_points': mp.Queue()
}

exit_flag = mp.Event()
pause_flag = mp.Event()
stop_recording_flag = mp.Event()
mode_changed_flag = mp.Event()  # NEW: Flag for mode changes

recording_process = None
recorded_audio_process = None
transcription_process = None
summarization_process = None
load_audio_flag = False
start_time = None
running = False

# NEW: Mode switching state management
class ModeManager:
    def __init__(self):
        self.current_mode = None
        self.mode_history = []  # Track mode changes with timestamps
        self.audio_segments = deque()  # Store audio segments with mode info
        self.mode_start_time = None
        
    def switch_mode(self, new_mode, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
            
        if self.current_mode:
            # Record the previous mode duration
            duration = timestamp - self.mode_start_time
            self.mode_history.append({
                'mode': self.current_mode,
                'start_time': self.mode_start_time,
                'end_time': timestamp,
                'duration': duration
            })
        
        self.current_mode = new_mode
        self.mode_start_time = timestamp
        return True

# Global mode manager
mode_manager = ModeManager()

# Global widget dictionary
widgets = {}

root = tk.Tk()
root.geometry("1024x768+150+50")
root.resizable(False, False)
root.configure(bg='#F8EED9')
master = root

live_transcript_widget = {}
live_transcript = tk.Tk()
live_improve_widget = {}
live_improve = tk.Tk()
live_minutes_widget = {}
live_minutes = tk.Tk()

all_minutes = {'transcription': [], 'summarization': [], 'action_points': []}
all_improve = []
all_transcript = []

live_improve.withdraw()
live_minutes.withdraw()
live_transcript.withdraw()

default_window_font = ("Arial", 16)
window_font = ["Arial", 16]

summary_cnt = 1

# NEW: Dynamic mode switching functions
def enable_mode_switching():
    """Enable mode switching controls during live meeting"""
    widgets['mode_switch_frame'].grid()
    widgets['current_mode_label'].config(text=f"Current Mode: {mode_manager.current_mode}")
    
def disable_mode_switching():
    """Disable mode switching controls when not recording"""
    widgets['mode_switch_frame'].grid_remove()

def switch_processing_mode():
    """Handle mode switching button click"""
    new_mode = widgets['switch_mode_dropdown'].get().strip()
    
    if new_mode == "Select New Mode" or new_mode == mode_manager.current_mode:
        update_status("Please select a different processing mode")
        return
    
    # Send mode switch request to processing
    timestamp = time.time()
    mode_switch_request = {
        'old_mode': mode_manager.current_mode,
        'new_mode': new_mode,
        'timestamp': timestamp,
        'chunk_id': f"mode_switch_{int(timestamp)}"
    }
    
    mode_switch_queue.put(mode_switch_request)
    mode_changed_flag.set()
    
    # Update local mode manager
    mode_manager.switch_mode(new_mode, timestamp)
    
    # Update GUI
    widgets['current_mode_label'].config(text=f"Current Mode: {new_mode}")
    update_status(f"Switched to {new_mode} mode at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}")
    
    # Log mode history
    update_status(f"Mode History: {len(mode_manager.mode_history)} previous modes")

def process_mode_changed(event=None):
    """Handle initial process mode selection"""
    selected_mode = widgets['process_mode_dropdown'].get().strip()

    if selected_mode == "Summary":
        widgets['summary_mode_dropdown'].config(state="readonly")
    else:
        widgets['summary_mode_dropdown'].set("Select Summary Mode")
        widgets['summary_mode_dropdown'].config(state="disabled")

def update_status(message):
    update_status_queue_gui.put(message)
    print(message, flush=True)

def default_state():
    widgets['start_button'].config(state=tk.NORMAL)
    widgets['pause_resume_button'].config(state=tk.DISABLED)
    live_minutes_widget['save_button'].config(state=tk.NORMAL)
    disable_mode_switching()  # NEW: Disable mode switching
    update_status("Meeting stopped completely.")

def get_summary_mode_config():
    """Get summary mode configuration based on dropdown selection"""
    mode = widgets['summary_mode_dropdown'].get()

    if mode == " Short ":
        summary_instruction = "generate a short, concise summary"
    elif mode == " Medium ":
        summary_instruction = "generate a medium-length, detailed summary"
    elif mode == " Long ":
        summary_instruction = "generate a comprehensive, long-form summary"
    else:
        summary_instruction = "summarize the audio appropriately"

    return summary_instruction

def validate_inputs():
    # Validate Process Mode first
    process_mode = widgets['process_mode_dropdown'].get().strip()
    if not process_mode or process_mode == "Select Processing Mode":
        update_status("!!!Error: Please select a processing mode first!!!")
        widgets['process_mode_dropdown'].configure(style="Error.TCombobox")
        master.after(2000, lambda: widgets['process_mode_dropdown'].configure(style="TCombobox"))
        return False

    # If process mode is Summary, validate Summary Mode
    if process_mode == "Summary":
        summary_mode = widgets['summary_mode_dropdown'].get().strip()
        if not summary_mode or summary_mode == "Select Summary Mode":
            update_status("!!!Error: Please select a summary mode first!!!")
            widgets['summary_mode_dropdown'].configure(style="Error.TCombobox")
            master.after(2000, lambda: widgets['summary_mode_dropdown'].configure(style="TCombobox"))
            return False
        summary_instruction = get_summary_mode_config()
        widgets['summary_mode_dropdown'].configure(style="Success.TCombobox")
        master.after(1000, lambda: widgets['summary_mode_dropdown'].configure(style="TCombobox"))
        update_status(f"Validation granted - Using summary mode: {summary_mode} ({summary_instruction})")

    update_status(f"Validation granted - Using process mode: {process_mode}")
    return True

def update_gui():
    global summary_cnt
    if not exit_flag.is_set():
        try:
            while not transcript_queue_gui.empty():
                transcript_data = transcript_queue_gui.get()
                if isinstance(transcript_data, dict):
                    transcript = transcript_data.get('content', '')
                    mode = transcript_data.get('mode', 'unknown')
                    timestamp = transcript_data.get('timestamp', '')
                else:
                    transcript = transcript_data
                    mode = 'legacy'
                    timestamp = ''
                
                if len(transcript.strip()):
                    live_transcript_widget['transcript_display'].config(state=tk.NORMAL)
                    display_text = f"[{mode.upper()}] {timestamp}\n{transcript}\n" + "=" * 50 + "\n"
                    live_transcript_widget['transcript_display'].insert(tk.END, display_text)
                    live_transcript_widget['transcript_display'].see(tk.END)
                    live_transcript_widget['transcript_display'].config(state=tk.DISABLED)
                    all_transcript.append(transcript_data)
        except queue.Empty:
            pass
            
        try:
            while not update_status_queue_gui.empty():
                status_msg = update_status_queue_gui.get()
                if len(status_msg.strip()):
                    widgets['status_display'].config(state=tk.NORMAL)
                    widgets['status_display'].insert(
                        tk.END, f"{datetime.now().strftime('%H:%M:%S')}: {status_msg}\n")
                    widgets['status_display'].see(tk.END)
                    widgets['status_display'].config(state=tk.DISABLED)
        except queue.Empty:
            pass
            
        try:
            while not improve_queue_gui.empty():
                improved = improve_queue_gui.get()
                live_improve_widget['improve_display'].config(state=tk.NORMAL)
                live_improve_widget['improve_display'].insert(
                    tk.END, f"\n\n—- Improve - {summary_cnt} ———--\n\n" + improved)
                live_improve_widget['improve_display'].config(state=tk.DISABLED)
                all_improve.append(improved)
        except queue.Empty:
            pass
            
        try:
            while not summary_queue_gui.empty():
                summary_data = summary_queue_gui.get()
                if isinstance(summary_data, dict):
                    summary = summary_data.get('content', '')
                    mode = summary_data.get('mode', 'unknown')
                    timestamp = summary_data.get('timestamp', '')
                    display_text = f"-{mode.upper()} - {summary_cnt} - {timestamp}-\n\n{summary}"
                else:
                    summary = summary_data
                    mode = 'legacy'
                    display_text = f"-Summary - {summary_cnt} -\n\n{summary}"
                
                live_minutes_widget['minutes_display'].config(state=tk.NORMAL)
                live_minutes_widget['minutes_display'].insert(tk.END, display_text)
                live_minutes_widget['minutes_display'].config(state=tk.DISABLED)
                summary_cnt += 1
                
                # Store in appropriate mode category
                if mode in all_minutes:
                    all_minutes[mode].append(summary_data)
                else:
                    all_minutes['transcription'].append(summary_data)
        except queue.Empty:
            pass
        master.after(1000, update_gui)

# ENHANCED: Multi-mode audio processing with dynamic switching
def transcribe_audio_enhanced(audio_queue_lst, all_transcript, exit_flag, initial_mode):
    """Enhanced audio processing with dynamic mode switching support"""
    update_status("Enhanced audio processing started with mode switching support")
    
    try:
        device = "cpu"
        repo_id = "/Users/vishnukumarkudidela/Desktop/workspace/ASR/models/Voxtral-Mini-3B-2507"

        # Load model once at startup
        processor = AutoProcessor.from_pretrained(repo_id)
        model = VoxtralForConditionalGeneration.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16, device_map=device
        )
        update_status("Model successfully loaded for enhanced processing")

        # Initialize current mode
        current_processing_mode = initial_mode
        mode_manager.switch_mode(current_processing_mode)
        update_status(f"Starting with mode: {current_processing_mode}")

        while not exit_flag.is_set():
            try:
                # Check for mode switch requests
                if not mode_switch_queue.empty():
                    mode_request = mode_switch_queue.get()
                    old_mode = mode_request['old_mode']
                    new_mode = mode_request['new_mode']
                    switch_timestamp = mode_request['timestamp']
                    
                    update_status(f"Processing mode switch: {old_mode} → {new_mode}")
                    current_processing_mode = new_mode
                    mode_changed_flag.clear()  # Reset flag after handling

                # Process audio files
                if not audio_queue_lst.empty():
                    file_path = audio_queue_lst.get()
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    
                    update_status(f"Processing {file_path} with mode: {current_processing_mode}")
                    
                    result = process_audio_with_mode(
                        file_path, current_processing_mode, processor, model, repo_id, timestamp
                    )
                    
                    if result:
                        # Send to appropriate GUI queue with metadata
                        result_with_metadata = {
                            'content': result,
                            'mode': current_processing_mode,
                            'timestamp': timestamp,
                            'file_path': file_path
                        }
                        
                        if current_processing_mode == "Transcription":
                            transcript_queue_gui.put(result_with_metadata)
                        else:
                            summary_queue_gui.put(result_with_metadata)
                        
                        update_status(f"Completed processing with {current_processing_mode}")
                else:
                    time.sleep(0.5)
                    
            except Exception as e:
                update_status(f"Error in enhanced processing: {str(e)}")
                time.sleep(1)
                
    except Exception as e:
        update_status(f"Critical error in enhanced audio processing: {str(e)}")

def process_audio_with_mode(file_path, mode, processor, model, repo_id, timestamp):
    """Process audio file with specified mode"""
    try:
        decoded_outputs = []
        
        if mode == "Transcription":
            inputs = processor.apply_transcription_request(language="en", audio=file_path, model_id=repo_id)
            inputs = inputs.to("cpu", dtype=torch.bfloat16)
            outputs = model.generate(**inputs)
            decoded_outputs = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            
        elif mode == "Summarization":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are voxtral, a helpful assistant that works in summarization mode. "
                                "After listening to the audio, understand the content and generate a clear, "
                                "concise summary organized topic-wise. Stay strictly within the audio content."
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
            inputs = inputs.to("cpu", dtype=torch.bfloat16)
            outputs = model.generate(**inputs)
            decoded_outputs = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            
        elif mode == "Action Points":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are voxtral, a helpful assistant that extracts action points. "
                                "After listening to the audio, extract all key action points. "
                                "Each action point should clearly state:\n"
                                "- The task or decision\n"
                                "- Who is responsible\n"
                                "- The deadline or timeline (if mentioned)\n"
                                "Present the output in a numbered list."
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
            inputs = inputs.to("cpu", dtype=torch.bfloat16)
            outputs = model.generate(**inputs)
            decoded_outputs = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            
        return decoded_outputs[0] if decoded_outputs else None
        
    except Exception as e:
        update_status(f"Error processing audio with {mode}: {str(e)}")
        return None

def record_audio(audio_queue, exit_flag, stop_recording_flag, pause_flag):
    """Enhanced recording with mode switching awareness"""
    update_status("Recording audio started with mode switching support")
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                            frames_per_buffer=CHUNK)

        duration_minutes = MEETING_CHUNK_DURATION_MIN
        record_seconds = duration_minutes * 60
        update_status(f"Recording chunks of {duration_minutes} minutes each")
        
        file_cnt = 0
        stop_recording_flag.clear()

        while not stop_recording_flag.is_set():
            frames = []
            chunk_start_time = time.time()
            
            for i in range(int(RATE / CHUNK * record_seconds)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                if not pause_flag.is_set():
                    frames.append(data)
                if exit_flag.is_set() or stop_recording_flag.is_set():
                    break
                    
            if frames:
                file_cnt += 1
                chunk_timestamp = datetime.now().strftime('%H:%M:%S')
                wave_output_filename = f"{write_record_to_dir}/recording_{chunk_timestamp.replace(':', '-')}.wav"
                
                # Save audio file
                wave_file = wave.open(wave_output_filename, 'wb')
                wave_file.setnchannels(CHANNELS)
                wave_file.setsampwidth(audio.get_sample_size(FORMAT))
                wave_file.setframerate(RATE)
                wave_file.writeframes(b''.join(frames))
                wave_file.close()
                
                full_file_path = os.path.join(base_directory, wave_output_filename)
                audio_queue_lst.put(full_file_path)
                
                update_status(f"Audio Chunk {file_cnt} created at {chunk_timestamp}")
                
        stream.stop_stream()
        stream.close()
        audio.terminate()
        update_status("Recording completed")
        
    except Exception as e:
        update_status(f"Error in audio recording: {str(e)}")

def setup_gui():
    master.title("Meeting Minutes Generator - Enhanced with Mode Switching")
    text_font = ("Arial", 14)
    scrolledText_font = ('Arial', 14, 'bold', 'underline')

    # Styles for dropdown
    style = ttk.Style()
    style.configure("TCombobox", fieldbackground="white", background="white")
    style.configure("Error.TCombobox", fieldbackground="#ffcccc", background="#ffcccc")
    style.configure("Success.TCombobox", fieldbackground="#ccffcc", background="#ccffcc")

    master.grid_columnconfigure(1, weight=1)
    for i in range(16):  # Increased for new mode switching section
        master.grid_rowconfigure(i, weight=1)

    # Original input frame
    legend_frame_input = tk.Frame(master, borderwidth=2, padx=50, pady=20, relief='groove')
    legend_frame_input.grid(row=1, columnspan=3)

    tk.Label(legend_frame_input, text="Date:", font=text_font).grid(row=0, column=0, sticky="e")
    widgets['date_entry'] = tk.Entry(legend_frame_input, font=text_font)
    widgets['date_entry'].grid(row=0, column=1, sticky="we")
    widgets['date_entry'].insert(0, datetime.now().strftime("%Y-%m-%d"))

    tk.Label(legend_frame_input, text="Time:", font=text_font).grid(row=1, column=0, sticky="e")
    widgets['time_entry'] = tk.Entry(legend_frame_input, font=text_font)
    widgets['time_entry'].grid(row=1, column=1, sticky="we")
    widgets['time_entry'].insert(0, datetime.now().strftime("%H:%M"))

    tk.Label(legend_frame_input, text="Venue:", font=text_font).grid(row=2, column=0, sticky="e")
    widgets['venue_entry'] = tk.Entry(legend_frame_input, font=text_font)
    widgets['venue_entry'].grid(row=2, column=1, sticky="we")

    tk.Label(legend_frame_input, text="Agenda:", font=text_font).grid(row=3, column=0, sticky="e")
    widgets['agenda_entry'] = tk.Entry(legend_frame_input, font=text_font, width=60)
    widgets['agenda_entry'].grid(row=3, column=1, sticky="we")

    tk.Label(legend_frame_input, text="Initial Mode:", font=text_font).grid(row=4, column=0, sticky="e")
    process_modes = ["Select Processing Mode", " Transcription ", " Summarization ", " Action Points "]
    widgets['process_mode_dropdown'] = ttk.Combobox(
        legend_frame_input, values=process_modes,
        font=text_font, state="readonly", width=25
    )
    widgets['process_mode_dropdown'].grid(row=4, column=1, sticky="we")
    widgets['process_mode_dropdown'].set("Select Processing Mode")
    widgets['process_mode_dropdown'].bind("<<ComboboxSelected>>", process_mode_changed)

    tk.Label(legend_frame_input, text="Summary Mode:", font=text_font).grid(row=5, column=0, sticky="e")
    summary_modes = ["Select Summary Mode", " Short ", " Medium ", " Long "]
    widgets['summary_mode_dropdown'] = ttk.Combobox(legend_frame_input, values=summary_modes,
                                                    font=text_font, state="readonly", width=25)
    widgets['summary_mode_dropdown'].grid(row=5, column=1, sticky="we")
    widgets['summary_mode_dropdown'].set("Select Summary Mode")

    # NEW: Mode switching frame (hidden initially)
    widgets['mode_switch_frame'] = tk.Frame(master, borderwidth=2, padx=50, pady=15, relief='groove', bg='#E8F4FF')
    widgets['mode_switch_frame'].grid(row=2, columnspan=3, sticky="ew")
    widgets['mode_switch_frame'].grid_remove()  # Hidden initially

    tk.Label(widgets['mode_switch_frame'], text="🔄 LIVE MODE SWITCHING", 
             font=('Arial', 12, 'bold'), bg='#E8F4FF', fg='#0066CC').grid(row=0, columnspan=3, pady=(0,10))
    
    widgets['current_mode_label'] = tk.Label(widgets['mode_switch_frame'], 
                                           text="Current Mode: None", 
                                           font=('Arial', 11, 'bold'), bg='#E8F4FF')
    widgets['current_mode_label'].grid(row=1, column=0, sticky="w", padx=(0,20))

    tk.Label(widgets['mode_switch_frame'], text="Switch to:", font=text_font, bg='#E8F4FF').grid(row=1, column=1, sticky="e")
    
    switch_modes = ["Select New Mode", " Transcription ", " Summarization ", " Action Points "]
    widgets['switch_mode_dropdown'] = ttk.Combobox(
        widgets['mode_switch_frame'], values=switch_modes,
        font=text_font, state="readonly", width=20
    )
    widgets['switch_mode_dropdown'].grid(row=1, column=2, sticky="w", padx=(5,0))
    widgets['switch_mode_dropdown'].set("Select New Mode")

    widgets['switch_mode_button'] = tk.Button(
        widgets['mode_switch_frame'], text="Switch Mode", command=switch_processing_mode,
        font=text_font, bg='#4CAF50', fg='white', width=15
    )
    widgets['switch_mode_button'].grid(row=2, column=0, columnspan=3, pady=(10,0))

    # Continue with rest of GUI... (context text, buttons, etc.)
    tk.Label(master, text="Additional Context:", font=scrolledText_font, justify="left").grid(row=6, column=0, columnspan=1)
    widgets['context_text'] = scrolledtext.ScrolledText(master, height=2, width=40, font=text_font)
    widgets['context_text'].grid(row=7, column=0, columnspan=3, sticky="nswe")

    legend_frame = tk.Frame(master, borderwidth=2, padx=25, pady=15, relief='groove')
    legend_frame.grid(row=8, columnspan=3)

    widgets['start_button'] = tk.Button(
        legend_frame, text="Start Meeting", command=lambda: start_meeting(), width=20, font=text_font)
    widgets['start_button'].grid(row=8, column=0, padx=20, pady=10)

    widgets['pause_resume_button'] = tk.Button(
        legend_frame, text="Pause Meeting", command=lambda: pause_resume_meeting(),
        width=20, state=tk.DISABLED, font=text_font)
    widgets['pause_resume_button'].grid(row=8, column=1, padx=20, pady=10)

    widgets['stop_button'] = tk.Button(
        legend_frame, text="End Meeting", command=lambda: stop_audio_recording(),
        width=20, state=tk.DISABLED, font=text_font)
    widgets['stop_button'].grid(row=8, column=2, padx=20, pady=10)

    widgets['load_button'] = tk.Button(
        legend_frame, text="Load Audio", command=lambda: load_audio_files(),
        width=20, font=text_font)
    widgets['load_button'].grid(row=9, column=0, padx=20, pady=10)

    widgets['toggle'] = tk.Button(
        legend_frame, text="View Minutes", command=lambda: toggle_window(live_minutes),
        font=text_font, padx=20, pady=10, width=20)
    widgets['toggle'].grid(row=9, column=1, padx=20, pady=10)

    tk.Label(master, text="Command Status:", font=scrolledText_font, justify="left").grid(row=10, column=0, columnspan=1)
    widgets['status_display'] = scrolledtext.ScrolledText(master, height=5, background="black", fg="white", width=50)
    widgets['status_display'].grid(row=11, column=0, columnspan=3, padx=0, sticky="nswe")

    legend_frame_output = tk.Frame(master, borderwidth=2, padx=100, pady=10, relief='groove')
    legend_frame_output.grid(row=12, columnspan=3)

    widgets['quit_button'] = tk.Button(
        legend_frame_output, text="Quit Application", command=lambda: quit_application(),
        width=20, font=text_font, background="orange")
    widgets['quit_button'].grid(row=12, column=2)

    # Setup live windows
    setup_live_windows()

def setup_live_windows():
    """Setup live transcript and minutes windows"""
    live_transcript.title("Live Transcripts - Enhanced")
    live_transcript_widget['transcript_display'] = scrolledtext.ScrolledText(
        live_transcript, height=22, width=80, font=tuple(window_font))
    live_transcript_widget['transcript_display'].grid(row=0, column=0, columnspan=1, sticky="nswe")
    live_transcript.grid_columnconfigure(1, weight=1)
    live_transcript.grid_rowconfigure(0, weight=1)

    live_transcript_widget['save_button'] = tk.Button(
        live_transcript, text="Save Transcription", command=lambda: save_live_transcription(),
        font=("Arial", 14))
    live_transcript_widget['save_button'].grid(row=1, column=0, padx=10, pady=5)

    live_minutes.title("Live Minutes - Mode Aware")
    live_minutes_widget['zoom_plus'] = tk.Button(
        live_minutes, text="Zoom +", command=zoom_plus, font=("Arial", 14))
    live_minutes_widget['zoom_plus'].place(x=10, y=1)

    live_minutes_widget['zoom_minus'] = tk.Button(
        live_minutes, text="Zoom -", command=zoom_minus, font=("Arial", 14))
    live_minutes_widget['zoom_minus'].place(x=120, y=1)

    live_minutes_widget['minutes_display'] = scrolledtext.ScrolledText(
        live_minutes, height=22, width=80, font=tuple(window_font))
    live_minutes_widget['minutes_display'].place(x=10, y=35)
    
    live_minutes_widget['save_button'] = tk.Button(
        live_minutes, text="Save Minutes", command=save_minutes,
        state=tk.DISABLED, font=("Arial", 14))
    live_minutes_widget['save_button'].place(x=240, y=1)

def start_meeting():
    """Enhanced start meeting with mode switching support"""
    global recording_process, transcription_process

    # Validate inputs
    if not validate_inputs():
        return

    selected_mode = widgets['process_mode_dropdown'].get().strip()
    update_status(f"Starting meeting with initial mode: {selected_mode}")

    # Initialize mode manager
    mode_manager.switch_mode(selected_mode)
    
    start_timer()
    widgets['start_button'].config(state=tk.DISABLED)
    widgets['pause_resume_button'].config(state=tk.NORMAL)
    widgets['stop_button'].config(state=tk.NORMAL)
    
    # Enable mode switching controls
    enable_mode_switching()

    try:
        # Clear all flags and queues
        exit_flag.clear()
        pause_flag.clear()
        stop_recording_flag.clear()
        mode_changed_flag.clear()
        
        # Clear mode switching queue
        while not mode_switch_queue.empty():
            mode_switch_queue.get()

        # Start recording process
        recording_process = mp.Process(
            target=record_audio,
            args=(audio_queue, exit_flag, stop_recording_flag, pause_flag),
            daemon=True
        )
        recording_process.start()

        # Start enhanced transcription process with mode switching
        transcription_process = mp.Process(
            target=transcribe_audio_enhanced,
            args=(audio_queue_lst, all_transcript, exit_flag, selected_mode),
            daemon=True
        )
        transcription_process.start()

        update_status("Meeting started successfully with dynamic mode switching enabled")
        update_gui()
        
    except Exception as e:
        update_status(f"Error starting meeting: {str(e)}")
        widgets['start_button'].config(state=tk.NORMAL)
        disable_mode_switching()

def load_audio_files():
    """Enhanced load audio with mode switching (for pre-recorded files)"""
    global recorded_audio_process, transcription_process

    if not validate_inputs():
        return

    selected_mode = widgets['process_mode_dropdown'].get().strip()
    
    audio_dir = filedialog.askdirectory()
    if not audio_dir:
        return

    # Initialize mode manager for loaded audio
    mode_manager.switch_mode(selected_mode)
    
    update_status(f"Loading audio from: {audio_dir}")
    update_status(f"Using initial mode: {selected_mode}")

    widgets['start_button'].config(state=tk.DISABLED)
    widgets['pause_resume_button'].config(state=tk.NORMAL)
    widgets['stop_button'].config(state=tk.NORMAL)
    
    # Enable mode switching for loaded audio too
    enable_mode_switching()

    try:
        exit_flag.clear()
        pause_flag.clear()
        stop_recording_flag.clear()

        # Process to load audio files
        recorded_audio_process = mp.Process(
            target=loaded_audio_process,
            args=(audio_dir, audio_queue, exit_flag, stop_recording_flag),
            daemon=True
        )
        recorded_audio_process.start()

        # Enhanced processing for loaded audio
        transcription_process = mp.Process(
            target=transcribe_audio_enhanced,
            args=(audio_queue_lst, all_transcript, exit_flag, selected_mode),
            daemon=True
        )
        transcription_process.start()

        update_status("Audio processing started with mode switching enabled")
        update_gui()
        
    except Exception as e:
        update_status(f"Error loading audio files: {str(e)}")
        widgets['start_button'].config(state=tk.NORMAL)
        disable_mode_switching()

def pause_resume_meeting():
    """Enhanced pause/resume with mode switching awareness"""
    if pause_flag.is_set():
        # Currently paused → resume
        pause_flag.clear()
        widgets['pause_resume_button'].config(text="Pause Meeting")
        update_status("Meeting resumed - mode switching remains active")
    else:
        # Currently running → pause
        pause_flag.set()
        widgets['pause_resume_button'].config(text="Resume Meeting")
        update_status("Meeting paused - you can still switch modes")

def stop_audio_recording():
    """Enhanced stop with mode switching cleanup"""
    update_status("Stopping meeting and cleaning up processes...")
    stop_timer()
    stop_recording_flag.set()
    widgets['stop_button'].config(state=tk.DISABLED)
    
    # Disable mode switching
    disable_mode_switching()

    # Wait for processes to finish
    try:
        if recording_process and recording_process.is_alive():
            recording_process.join(timeout=5)
            if recording_process.is_alive():
                recording_process.terminate()
                update_status("Recording process terminated")

        if recorded_audio_process and recorded_audio_process.is_alive():
            recorded_audio_process.join(timeout=5)
            if recorded_audio_process.is_alive():
                recorded_audio_process.terminate()
                update_status("Audio loading process terminated")

        if transcription_process and transcription_process.is_alive():
            transcription_process.join(timeout=5)
            if transcription_process.is_alive():
                transcription_process.terminate()
                update_status("Transcription process terminated")

    except Exception as e:
        update_status(f"Error during process cleanup: {str(e)}")

    # Clear all queues
    clear_all_queues()
    
    # Save mode history
    save_mode_history()

    # Reset GUI buttons
    default_state()

def clear_all_queues():
    """Clear all processing queues"""
    queues_to_clear = [
        audio_queue, audio_queue_lst, transcript_queue, summary_queue,
        mode_switch_queue, current_mode_queue
    ]
    
    for q in queues_to_clear:
        while not q.empty():
            try:
                q.get_nowait()
            except:
                break
    
    update_status("All processing queues cleared")

def save_mode_history():
    """Save mode switching history to file"""
    if mode_manager.mode_history:
        history_file = f"{write_record_to_dir}/mode_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        history_data = {
            'session_info': {
                'date': widgets['date_entry'].get(),
                'time': widgets['time_entry'].get(),
                'venue': widgets['venue_entry'].get(),
                'agenda': widgets['agenda_entry'].get()
            },
            'mode_switches': mode_manager.mode_history,
            'final_mode': mode_manager.current_mode,
            'total_switches': len(mode_manager.mode_history)
        }
        
        try:
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
            update_status(f"Mode history saved: {len(mode_manager.mode_history)} switches recorded")
        except Exception as e:
            update_status(f"Error saving mode history: {str(e)}")

def loaded_audio_process(audio_dir, audio_queue, exit_flag, stop_recording_flag):
    """Enhanced loaded audio process"""
    update_status("Loading audio files with mode switching support")
    audio_files = sorted(glob(audio_dir + "/*.wav"))
    files_cnt = len(audio_files)
    
    try:
        for i, audio_file in enumerate(audio_files):
            if exit_flag.is_set() or stop_recording_flag.is_set():
                break
                
            if audio_file.endswith('.wav'):
                try:
                    frames = []
                    with wave.open(audio_file, 'rb') as wf:
                        num_frames = wf.getnframes()
                        frames.append(wf.readframes(num_frames))
                    
                    if frames:
                        full_path = os.path.abspath(audio_file)
                        audio_queue_lst.put(full_path)
                        update_status(f"Loaded audio file {i+1}/{files_cnt}: {os.path.basename(audio_file)}")
                    else:
                        update_status(f"Invalid audio file: {audio_file}")
                        
                except Exception as e:
                    update_status(f"Error loading {audio_file}: {str(e)}")
        
        update_status(f"All {files_cnt} audio files loaded successfully")
        
    except Exception as e:
        update_status(f"Error in load audio process: {str(e)}")

def save_minutes():
    """Enhanced save minutes with mode-aware content"""
    file_path = filedialog.asksaveasfilename(
        defaultextension=".docx",
        filetypes=[("Word documents", "*.docx"), ("Text files", "*.txt"), ("All files", "*.*")]
    )
    
    if file_path:
        try:
            meeting_date = widgets['date_entry'].get()
            meeting_time = widgets['time_entry'].get()
            meeting_venue = widgets['venue_entry'].get()
            meeting_agenda = widgets['agenda_entry'].get()
            additional_context = widgets['context_text'].get("1.0", tk.END)

            # Get content from display
            save_minutes_text = live_minutes_widget['minutes_display'].get("1.0", tk.END)

            # Create enhanced content with mode history
            content_parts = [
                f"# Meeting Minutes - Enhanced with Dynamic Mode Switching",
                f"",
                f"**Date:** {meeting_date}",
                f"**Time:** {meeting_time}",
                f"**Venue:** {meeting_venue}",
                f"**Agenda:** {meeting_agenda}",
                f"",
                f"**Additional Context:**",
                f"{additional_context}",
                f"",
            ]
            
            # Add mode switching summary
            if mode_manager.mode_history:
                content_parts.extend([
                    f"## Mode Switching Summary",
                    f"Total mode switches during meeting: {len(mode_manager.mode_history)}",
                    f""
                ])
                
                for i, switch in enumerate(mode_manager.mode_history, 1):
                    start_time = datetime.fromtimestamp(switch['start_time']).strftime('%H:%M:%S')
                    end_time = datetime.fromtimestamp(switch['end_time']).strftime('%H:%M:%S')
                    duration = int(switch['duration'] / 60)  # Convert to minutes
                    content_parts.append(f"{i}. **{switch['mode']}** ({start_time} - {end_time}, {duration}min)")
                
                content_parts.append("")
            
            content_parts.extend([
                f"## Meeting Content",
                f"",
                save_minutes_text
            ])

            full_content = "\n".join(content_parts)

            if file_path.endswith('.docx'):
                content_html = markdown(full_content)
                with open(file_path, "wb") as fp:
                    doc_content = html2docx.html2docx(content_html, title="Enhanced Meeting Minutes")
                    fp.write(doc_content.getvalue())
            else:
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write(full_content)

            update_status(f"Enhanced minutes saved to {file_path}")
            
        except Exception as e:
            update_status(f"Error saving minutes: {str(e)}")
            print(f"Error saving minutes: {traceback.format_exc()}")

def save_live_transcription():
    """Enhanced save transcription with mode information"""
    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*")]
    )
    
    if file_path:
        try:
            if file_path.endswith('.json'):
                # Save structured data with mode information
                structured_data = {
                    'meeting_info': {
                        'date': widgets['date_entry'].get(),
                        'time': widgets['time_entry'].get(),
                        'venue': widgets['venue_entry'].get(),
                        'agenda': widgets['agenda_entry'].get()
                    },
                    'transcripts': all_transcript,
                    'mode_history': mode_manager.mode_history
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(structured_data, f, indent=2, default=str, ensure_ascii=False)
            else:
                # Save as plain text
                transcript_text = live_transcript_widget['transcript_display'].get("1.0", tk.END)
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write("# Enhanced Live Transcription with Mode Switching\n\n")
                    f.write(transcript_text)
            
            update_status(f"Enhanced transcription saved to {file_path}")
            
        except Exception as e:
            update_status(f"Error saving transcription: {str(e)}")

def quit_application():
    """Enhanced quit with comprehensive backup"""
    update_status("Quitting application and saving comprehensive backups...")
    
    # Save minutes backup by mode
    for mode, minutes_list in all_minutes.items():
        if minutes_list:
            file_name = f"{write_record_to_dir}/{mode}_minutes_backup_{datetime.now().strftime('%d-%m-%Y_%H-%M')}.json"
            try:
                with open(file_name, "w", encoding='utf-8') as file:
                    json.dump(minutes_list, file, indent=2, default=str, ensure_ascii=False)
                update_status(f"{mode} minutes backup saved: {len(minutes_list)} entries")
            except Exception as e:
                update_status(f"Error saving {mode} backup: {str(e)}")

    # Save transcript backup
    if all_transcript:
        file_name = f"{write_record_to_dir}/transcripts_backup_{datetime.now().strftime('%d-%m-%Y_%H-%M')}.json"
        try:
            with open(file_name, "w", encoding='utf-8') as file:
                json.dump(all_transcript, file, indent=2, default=str, ensure_ascii=False)
            update_status(f"Transcripts backup saved: {len(all_transcript)} entries")
        except Exception as e:
            update_status(f"Error saving transcripts backup: {str(e)}")

    # Save final mode history
    save_mode_history()

    # Cleanup processes
    exit_flag.set()
    stop_recording_flag.set()
    
    # Close windows
    live_transcript.quit()
    live_improve.quit() 
    live_minutes.quit()
    master.quit()
    
    print("Enhanced application closed with comprehensive backups saved")
    exit(0)

# Additional utility functions
def toggle_window(window):
    """Toggle window visibility"""
    if window.winfo_viewable():
        window.withdraw()
    else:
        window.deiconify()

def zoom_plus():
    """Increase font size"""
    window_font[1] += 2
    if window_font[1] > 29:
        window_font[1] = 29
    live_minutes_widget['minutes_display'].config(font=tuple(window_font))

def zoom_minus():
    """Decrease font size"""
    window_font[1] -= 2
    if window_font[1] < 4:
        window_font[1] = 4
    live_minutes_widget['minutes_display'].config(font=tuple(window_font))

def update_duration():
    """Update meeting duration display"""
    global start_time, running
    while running:
        elapsed_time = int(time.time() - start_time)
        hours = elapsed_time // 3600
        minutes = (elapsed_time % 3600) // 60
        seconds = elapsed_time % 60
        time.sleep(1)

def start_timer():
    """Start meeting timer"""
    global start_time, running
    start_time = time.time()
    running = True
    threading.Thread(target=update_duration, daemon=True).start()

def stop_timer():
    """Stop meeting timer"""
    global running
    running = False

def do_not_close():
    """Prevent window closing"""
    pass

def get_meeting_context():
    """Get meeting context information"""
    context = ""
    context_fields = [
        ("Meeting date", widgets['date_entry'].get().strip()),
        ("Time", widgets['time_entry'].get().strip()),
        ("Venue", widgets['venue_entry'].get().strip()),
        ("Agenda", widgets['agenda_entry'].get().strip())
    ]
    
    for label, value in context_fields:
        if value:
            context += f"\n{label}: {value}"
    
    additional_context = widgets['context_text'].get("1.0", tk.END).strip()
    if additional_context:
        context += f"\n{additional_context}\n\n"
    
    return context

if __name__ == "__main__":
    # Setup window protocols
    live_transcript.protocol("WM_DELETE_WINDOW", lambda: toggle_window(live_transcript))
    live_improve.protocol("WM_DELETE_WINDOW", lambda: toggle_window(live_improve))
    live_minutes.protocol("WM_DELETE_WINDOW", lambda: toggle_window(live_minutes))
    
    # Setup and start GUI
    setup_gui()
    update_status("Enhanced Meeting Minutes Generator with Dynamic Mode Switching ready!")
    root.mainloop()