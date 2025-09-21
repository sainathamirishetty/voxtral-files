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

# Global queues and flags
audio_queue = mp.Queue()
audio_queue_lst = mp.Queue()
transcript_queue = mp.Queue()
summary_queue = mp.Queue()
transcript_queue_gui = mp.Queue()
improve_queue_gui = mp.Queue()
summary_queue_gui = mp.Queue()
update_status_queue_gui = mp.Queue()
exit_flag = mp.Event()
pause_flag = mp.Event()
stop_recording_flag = mp.Event()
recording_process = None
recorded_audio_process = None
transcription_process = None
summarization_process = None
load_audio_flag = False
start_time = None
running = False

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

all_minutes = []
all_improve = []
all_transcript = []

live_improve.withdraw()
live_minutes.withdraw()
live_transcript.withdraw()

default_window_font = ("Arial", 16)
window_font = ["Arial", 16]

summary_cnt = 1

def update_status(message):
    update_status_queue_gui.put(message)
    print(message, flush=True)

def default_state():
    widgets['start_button'].config(state=tk.NORMAL)
    widgets['pause_resume_button'].config(state=tk.DISABLED)
    live_minutes_widget['save_button'].config(state=tk.NORMAL)
    update_status("Meeting stopped completely.")

# Summary Modes
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
        # Default fallback
        summary_instruction = "summarize the audio appropriately"
    
    return summary_instruction
 
#Meeting Duration    
def get_meeting_duration_config():
    """Get meeting chunk duration configuration from GUI input"""
    try:
        duration_str = widgets['duration_entry'].get().strip()
        if not duration_str:
            return None, "Duration field is empty"
        
        duration = float(duration_str)
        if duration <= 0:
            return None, "Duration must be greater than 0"
        if duration > 25:  # Maximum 25 mins per chunk
            return None, "Duration cannot exceed  25 minutes"
        
        return duration, None
    except ValueError:
        return None, "Duration must be a valid number"

#Validate required inputs before starting meeting"""
def validate_inputs():
    
    mode = widgets['summary_mode_dropdown'].get()
    
    if not mode or mode == "Select Summary Mode":
        update_status("!!!Error: Please select a summary mode first!!!")
        widgets['summary_mode_dropdown'].configure(style="Error.TCombobox")
        master.after(2000, lambda: widgets['summary_mode_dropdown'].configure(style="TCombobox"))
        return False
    
    summary_instruction = get_summary_mode_config()
    
    # Visual feedback to show the selection is recognized
    widgets['summary_mode_dropdown'].configure(style="Success.TCombobox")
    master.after(1000, lambda: widgets['summary_mode_dropdown'].configure(style="TCombobox"))
    
    update_status(f"Validation granted - Using mode: {mode} ({summary_instruction} )")
    
    # Validate meeting duration
    duration, duration_error = get_meeting_duration_config()
    if duration_error:
        update_status(f"Error: {duration_error}")
        widgets['duration_entry'].configure(bg="#ffcccc")
        master.after(2000, lambda: widgets['duration_entry'].configure(bg="white"))
        return False
    
    # Visual feedback for duration validation
    widgets['duration_entry'].configure(bg="#ccffcc")
    master.after(1000, lambda: widgets['duration_entry'].configure(bg="white"))
    
    update_status(f"Meeting chunk duration set to: {duration} minutes")
    
    return True

def update_gui():
    global summary_cnt
    if not exit_flag.is_set():
        try:
            while not transcript_queue_gui.empty():
                transcript = transcript_queue_gui.get()
                if len(transcript.strip()):
                    live_transcript_widget['transcript_display'].config(state=tk.NORMAL)
                    live_transcript_widget['transcript_display'].insert(
                        tk.END, transcript + "\n" + "=" * 20 + "\n")
                    live_transcript_widget['transcript_display'].see(tk.END)
                    live_transcript_widget['transcript_display'].config(state=tk.DISABLED)
                    all_transcript.append(transcript)
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
                summary = summary_queue_gui.get()
                live_minutes_widget['minutes_display'].config(state=tk.NORMAL)
                live_minutes_widget['minutes_display'].insert(
                    tk.END, f"-Summary - {summary_cnt} -\n\n" + summary)
                live_minutes_widget['minutes_display'].config(state=tk.DISABLED)
                summary_cnt += 1
                all_minutes.append(summary)
        except queue.Empty:
            pass
        master.after(1000, update_gui)

#for audio loading from folders
def loaded_audio_process(audio_dir, audio_queue, exit_flag, stop_recording_flag):
    update_status("Loading audio started")
    print("hello")
    audio_files = sorted(glob(audio_dir + "/*.wav"))
    print(audio_files)
    files_cnt = len(audio_files)
    try:
        i = 0
        while i < files_cnt:
            print(audio_files[i])
            if audio_files[i].split('.')[-1] == 'wav':
                frames = []
                with wave.open(audio_files[i], 'rb') as wf:
                    num_frames = wf.getnframes()
                    frames.append(wf.readframes(num_frames))
                if exit_flag.is_set() or stop_recording_flag.is_set():
                    break
                if frames:
                    audio_queue.put(frames)
                    audio_queue_lst.put(audio_files[i])
                else:
                    update_status("Invalid Audio File: " + str(audio_files[i]))
            i += 1
        update_status("All audio files Loaded")
    except Exception as e:
        update_status(f"Error in load audio: {str(e)}")

#recording audio
def record_audio(audio_queue, exit_flag, stop_recording_flag, pause_flag):
    update_status("Recording audio started")
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        
        # INPUT-dynamic duration from gui
        duration_minutes, _ = get_meeting_duration_config()
        if duration_minutes is None:
            duration_minutes = MEETING_CHUNK_DURATION_MIN  #fall back to default
        
        record_seconds = duration_minutes * 60
        update_status(f"Recording chunks of {duration_minutes} minutes each")
        update_status("Audio Stream initialized")
        file_cnt = 0
        stop_recording_flag.clear()

        while not stop_recording_flag.is_set():
            frames = []
            for i in range(int(RATE / CHUNK * record_seconds)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                if not pause_flag.is_set():
                    frames.append(data)
                if exit_flag.is_set() or stop_recording_flag.is_set():
                    break
            if frames:
                file_cnt += 1
                wave_output_filename = f"{write_record_to_dir}/recording_{datetime.now().strftime('%H:%M:%S')}.wav"
                wave_file = wave.open(wave_output_filename, 'wb')
                wave_file.setnchannels(CHANNELS)
                wave_file.setsampwidth(audio.get_sample_size(FORMAT))
                wave_file.setframerate(RATE)
                wave_file.writeframes(b''.join(frames))
                wave_file.close()
                full_file_path = os.path.join(base_directory, wave_output_filename)
                audio_queue_lst.put(full_file_path)
                update_status(f"{file_cnt} Audio Chunk created")
        stream.stop_stream()
        stream.close()
        audio.terminate()
    except Exception as e:
        print(f"Error in audio recording: {str(e)}")
        update_status(f"Error in audio recording: {str(e)}")

#Transcription process
def transcribe_audio(audio_queue_lst, all_transcript, exit_flag, summary_instruction):
    update_status("Transcription audio started")
    try:
        device = "cuda:0"
        repo_id = "/home/advaita/Documents/Voxtral-Mini-3B-2507"
        
        processor = AutoProcessor.from_pretrained(repo_id)
        model = VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)
        
        update_status("model successfully loaded")
      
        while not exit_flag.is_set():
            try:
                if audio_queue_lst.empty():
                    continue
                else:
                    file_path = audio_queue_lst.get()

                    print("////  AUDIO_LOADED  ////")
                    
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (f"you are voxtral, a helpful assistant that works in below mode:\n"
                                    f"Summarizer: after listening the audio, please understand the total audio, then only {summary_instruction} within that should be organised topic-wise, with clear things, within the audio by utilizing entire input audio, don't go for hallucinations and extended sentences which are not in the audio. Keep the summary within provided audio only."
                                    )
                                },
                            ],
                        },                        
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio",
                                    "path": file_path,
                                },
                            ],
                        },
                    ]
                    
                    update_status("audio loaded")

                    inputs = processor.apply_chat_template(conversation)
                    inputs = inputs.to(device, dtype=torch.bfloat16)
                    
                    update_status("generating summary")
                    
                    outputs = model.generate(**inputs, max_new_tokens=2048)
                    decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    print("\nGenerated_response:")
                    print("=" * 80)
                    print(decoded_outputs[0])
                    print("=" * 80)
                    summary_queue_gui.put(decoded_outputs[0])
                    all_transcript += decoded_outputs[0]
                    
                    update_status("Summary Generated")
                    
                    
                    update_status("audio loaded")
                    
                    inputs = processor.apply_transcription_request(language="en", audio=file_path, model_id=repo_id)
                    inputs = inputs.to(device, dtype=torch.bfloat16)
                    
                    update_status("generating transcription")
                    
                    outputs = model.generate(**inputs, max_new_tokens=4000)
                    decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

                    print("\nGenerated_response:")
                    print("=" * 80)
                    print(decoded_outputs[0])
                    print("=" * 80)
                    transcript_queue_gui.put(decoded_outputs[0])
                    
                    update_status("transcription generated.")
                    
                    
            except Exception as e:
                print(f"Error in transcribe_audio: {str(e)}", flush=True)
                update_status(f"Error in transcribe_audio: {str(e)}")
    except Exception as e:
        print(f"Error in transcribe_audio: {str(e)}")
        update_status(f"Error in transcribe_audio: {str(e)}")

def do_not_close():
    pass

def toggle_window(window):
    if window.winfo_viewable():
        window.withdraw()  # Hide the window
    else:
        window.deiconify()  # Show the window

def zoom_plus():
    window_font[1] += 2
    if window_font[1] > 29:
        window_font[1] = 29
    live_minutes_widget['minutes_display'].config(font=tuple(window_font))

def zoom_minus():
    window_font[1] -= 2
    if window_font[1] < 4:
        window_font[1] = 4
    live_minutes_widget['minutes_display'].config(font=tuple(window_font))

#Complete setup_gui
def setup_gui():
    master.title("Meeting Minutes Generator")
    text_font = ("Arial", 14)
    scrolledText_font = ('Arial', 14, 'bold', 'underline')
    text_bg = "#f0f0f0"
    text_fg = "black"

    #custom styles for dropdown
    style = ttk.Style()
    style.configure("TCombobox", fieldbackground="white", background="white")
    style.configure("Error.TCombobox", fieldbackground="#ffcccc", background="#ffcccc")
    style.configure("Success.TCombobox", fieldbackground="#ccffcc", background="#ccffcc")

    master.grid_columnconfigure(1, weight=1)
    for i in range(14):
        master.grid_rowconfigure(i, weight=1)

    #legend_frame
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

    #Max Tokens entry with Summary Mode dropdown
    tk.Label(legend_frame_input, text="Summary Mode:", font=text_font).grid(row=4, column=0, sticky="e")
    summary_modes = ["Select Summary Mode", " Short ", " Medium ", " Long "]
    widgets['summary_mode_dropdown'] = ttk.Combobox(legend_frame_input, values=summary_modes, 
                                                   font=text_font, state="readonly", width=25)
    widgets['summary_mode_dropdown'].grid(row=4, column=1, sticky="we")
    widgets['summary_mode_dropdown'].set("Select Summary Mode")  # Default selection

    # Meeting Duration input field
    tk.Label(legend_frame_input, text="Meeting Duration (min):", font=text_font).grid(row=5, column=0, sticky="e")
    widgets['duration_entry'] = tk.Entry(legend_frame_input, font=text_font, width=10)
    widgets['duration_entry'].grid(row=5, column=1, sticky="w", padx=(0, 10))
    widgets['duration_entry'].insert(0, str(MEETING_CHUNK_DURATION_MIN))  # Default value
    
    # Added a small help label
    tk.Label(legend_frame_input, text="(1-25 min)", font=("Arial", 10), fg="gray").grid(row=5, column=1, sticky="e")

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
    
    widgets['duration_label'] = tk.Label(legend_frame, text="Duration: 00:00:00", font=("Arial", 14))
    widgets['duration_label'].grid(row=9, column=2, padx=20, pady=10, sticky='e') 

    tk.Label(master, text="Command Status:", font=scrolledText_font, justify="left").grid(row=10, column=0, columnspan=1)
    widgets['status_display'] = scrolledtext.ScrolledText(master, height=5, background="black", fg="white", width=50)
    widgets['status_display'].grid(row=11, column=0, columnspan=3, padx=0, sticky="nswe")

    legend_frame_output = tk.Frame(master, borderwidth=2, padx=100, pady=10, relief='groove')
    legend_frame_output.grid(row=12, columnspan=3)
    
    widgets['toggle'] = tk.Button(legend_frame_output, text="View Transcript", command=lambda: toggle_window(live_transcript), font=text_font, width=20)
    widgets['toggle'].grid(row=12, column=0)  # , sticky="we")

    widgets['quit_button'] = tk.Button(
        legend_frame_output, text="Quit Application", command=lambda: quit_application(),
        width=20, font=text_font, background="orange")
    widgets['quit_button'].grid(row=12, column=2)

    live_transcript.title("Live Transcripts")
    live_transcript_widget['transcript_display'] = scrolledtext.ScrolledText(
        live_transcript, height=22, width=80, font=tuple(window_font))
    live_transcript_widget['transcript_display'].grid(row=0, column=0, columnspan=1, sticky="nswe")
    live_transcript.grid_columnconfigure(1, weight=1)
    live_transcript.grid_rowconfigure(0, weight=1)

    live_improve.title("Live Improve")
    live_improve_widget['improve_display'] = scrolledtext.ScrolledText(
        live_improve, height=22, width=80, font=tuple(window_font))
    live_improve_widget['improve_display'].grid(row=0, column=1, columnspan=1, sticky="nswe")
    live_improve.grid_columnconfigure(1, weight=1)
    live_improve.grid_rowconfigure(0, weight=1)

    live_minutes.title("Live Summary")
    live_minutes_widget['zoom_plus'] = tk.Button(
        live_minutes, text="Zoom +", command=zoom_plus, font=text_font)
    live_minutes_widget['zoom_plus'].place(x=10, y=1)

    live_minutes_widget['zoom_minus'] = tk.Button(
        live_minutes, text="Zoom -", command=zoom_minus, font=text_font)
    live_minutes_widget['zoom_minus'].place(x=120, y=1)

    live_minutes_widget['minutes_display'] = scrolledtext.ScrolledText(
        live_minutes, height=22, width=80, font=tuple(window_font))
    live_minutes_widget['minutes_display'].place(x=10, y=35)
    live_minutes_widget['save_button'] = tk.Button(
        live_minutes, text="Save Minutes", command=save_minutes,
        state=tk.DISABLED, font=text_font)
    live_minutes_widget['save_button'].place(x=240, y=1)

def update_duration():
    global start_time, running
    while running:
        elapsed_time = int(time.time() - start_time)
        hours = elapsed_time // 3600
        minutes = (elapsed_time % 3600) // 60
        seconds = elapsed_time % 60
        widgets['duration_label'].config(
            text=f"Duration: {hours:02}:{minutes:02}:{seconds:02}")
        time.sleep(1)

def start_timer():
    global start_time, running
    start_time = time.time()
    running = True
    threading.Thread(target=update_duration, daemon=True).start()

def stop_timer():
    global running
    running = False

#start_meeting function
def start_meeting():
    global recording_process, transcription_process, summarization_process
    
    # Validate inputs before starting
    if not validate_inputs():
        return
    
    summary_instruction = get_summary_mode_config()
    mode = widgets['summary_mode_dropdown'].get()
    update_status(f"Starting meeting with mode: {mode}")
    
    start_timer()
    widgets['start_button'].config(state=tk.DISABLED)
    widgets['pause_resume_button'].config(state=tk.NORMAL)
    widgets['stop_button'].config(state=tk.NORMAL)
    try:
        recording_process = mp.Process(
            target=record_audio,
            args=(audio_queue, exit_flag, stop_recording_flag, pause_flag,),
            daemon=True)
        transcription_process = mp.Process(
            target=transcribe_audio,
            args=(audio_queue, transcript_queue, exit_flag, summary_instruction),
            daemon=True)
        summarization_process = mp.Process(
            target=transcribe_audio,
            args=(audio_queue_lst, all_transcript, exit_flag, summary_instruction),
            daemon=True)
        recording_process.start()
        # transcription_process.start()
        summarization_process.start()
        update_status("Meeting started successfully")
        update_gui()
    except Exception as e:
        update_status(f"Error starting meeting: {str(e)}")
        widgets['start_button'].config(state=tk.NORMAL)

#load_audio_files function
def load_audio_files():
    global recorded_audio_process, transcription_process, summarization_process, load_audio_flag
    
    # Validate inputs before starting
    if not validate_inputs():
        return
    
    summary_instruction = get_summary_mode_config()
    mode = widgets['summary_mode_dropdown'].get()
    
    load_audio_flag = True
    audio_dir = filedialog.askdirectory()
    if not audio_dir:  # User cancelled directory selection
        widgets['start_button'].config(state=tk.NORMAL)
        return
        
    update_status(f"Loaded Audio directory: {audio_dir}")
    update_status(f"Using mode: {mode}")
    
    widgets['start_button'].config(state=tk.DISABLED)
    widgets['pause_resume_button'].config(state=tk.NORMAL)
    widgets['stop_button'].config(state=tk.NORMAL)
    print("inside load audio")
    try:
        exit_flag.clear()
        pause_flag.clear()
        stop_recording_flag.clear()
        recorded_audio_process = mp.Process(
            target=loaded_audio_process,
            args=(audio_dir, audio_queue, exit_flag, stop_recording_flag,),
            daemon=True)
        summarization_process = mp.Process(
            target=transcribe_audio,
            args=(audio_queue_lst, all_transcript, exit_flag, summary_instruction),
            daemon=True)
        recorded_audio_process.start()
        summarization_process.start()
        print("Meeting started")
        update_status("Audio processing started successfully")
        update_gui()
    except Exception as e:
        update_status(f"Error starting meeting: {str(e)}")
        widgets['start_button'].config(state=tk.NORMAL)

def pause_resume_meeting():
    button_text = widgets['pause_resume_button']['text']
    widgets['pause_resume_button'].config(text=button_text)
    if pause_flag.is_set():
        update_status("Meeting paused")
    else:
        update_status("Meeting resumed")

def stop_audio_recording():
    update_status("Stopping audio recording...")
    stop_timer()
    stop_recording_flag.set()
    widgets['stop_button'].config(state=tk.DISABLED)
    if load_audio_flag:
        if not recorded_audio_process.is_alive():
            recorded_audio_process.join(timeout=5)
    if not load_audio_flag:
        while not recording_process.is_alive():
            time.sleep(1)
            pass
    update_status("Audio queue cleared")
    while not transcript_queue.empty():
        time.sleep(1)
        pass
    update_status("Transcript queue cleared")
    while not summary_queue.empty():
        time.sleep(1)
        pass
    update_status("Summary queue cleared")
    widgets['start_button'].config(state=tk.NORMAL)
    widgets['pause_resume_button'].config(state=tk.DISABLED)
    live_minutes_widget['save_button'].config(state=tk.NORMAL)
    update_status("Meeting Ended.")

def regenerate_summary():
    update_status("Regenerating summary...")
    # Implement summary regeneration logic here
    update_status("Summary regenerated")

def save_minutes():
    file_path = filedialog.asksaveasfilename(defaultextension=".docx",
                                             filetypes=[("Text files", "*.docx"), ("All files", "*.*")])
    if file_path:
        try:
            # Retrieve all meeting details
            meeting_date = widgets['date_entry'].get()
            meeting_time = widgets['time_entry'].get()
            meeting_venue = widgets['venue_entry'].get()
            meeting_agenda = widgets['agenda_entry'].get()
            additional_context = widgets['context_text'].get("1.0", tk.END)

            save_minutes_text = live_minutes_widget['minutes_display'].get("1.0", tk.END)

            # Compose content to save
            content = f"Meeting Date: {meeting_date}\n\nMeeting Time: {meeting_time}\n\nVenue: {meeting_venue}\n\nAgenda: {meeting_agenda}\n\nAdditional Context:\n\n{additional_context}\n\nMINUTES OF DISCUSSION:\n\n{save_minutes_text}"

            save_minutes_text_md = markdown(content)

            with open(file_path, "wb") as fp:
                doc_content = html2docx.html2docx(save_minutes_text_md, title="Minutes")
                fp.write(doc_content.getvalue())
            update_status(f"Minutes saved to {file_path}")
        except Exception as e:
            update_status(f"Error saving minutes: {str(e)}")
            print(f"error saving minutes: {traceback.format_exc()}{e}")

#for quitting_application
def quit_application():
    update_status("Quitting application...")
    file_name = f"{write_record_to_dir}/Minutes_backup_{datetime.now().strftime('%d-%m-%Y_%H-%M')}.md"
    with open(file_name, "w+") as file:
        for i, minutes in enumerate(all_minutes):
            file.write(f"# -- Summary {i} \n" + minutes + "\n\n")
    update_status(f"minutes backup saved to : {file_name}")          
            
    file_name = f"{write_record_to_dir}/live_trans_backup_{datetime.now().strftime('%d-%m-%Y_%H:%M')}.md"
    with open(file_name, "w+") as file:
        for i, trans_text in enumerate(all_transcript):
            file.write(f"# ------------ transcript {i} -------------- \n" + trans_text + "\n\n")
    update_status(f"transcript backup saved to : {file_name}")  
      
    update_status(f"Backups saved to : {file_name}")
    exit_flag.set()
    stop_recording_flag.set()
    live_transcript.quit()
    live_improve.quit()
    live_minutes.quit()
    master.quit()
    print("All processes completed. closing application")
    exit(0)

def get_meeting_context():
    context = ""
    text = widgets['date_entry'].get().strip()
    if len(text):
        context += f"\n Meeting date: {text}"
    text = widgets['time_entry'].get().strip()
    if len(text):
        context += f"\n Time: {text}"
    text = widgets['venue_entry'].get().strip()
    if len(text):
        context += f"\n Venue: {text}"
    text = widgets['agenda_entry'].get().strip()
    if len(text):
        context += f"\n Agenda: {text}"
    acronyms_text = widgets['context_text'].get("1.0", tk.END).strip()
    text = acronyms_text.strip()
    if len(text):
        context += f"\n {text} \n\n"
    return context

if __name__ == "__main__":
    live_transcript.protocol("WM_DELETE_WINDOW", lambda: toggle_window(live_transcript))
    live_improve.protocol("WM_DELETE_WINDOW", lambda: toggle_window(live_improve))
    live_minutes.protocol("WM_DELETE_WINDOW", lambda: toggle_window(live_minutes))
    setup_gui()
    root.mainloop()
