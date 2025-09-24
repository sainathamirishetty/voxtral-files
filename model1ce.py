import multiprocessing as mp
import pyaudio
import numpy as np
import tkinter as tk
from tkinter import scrolledtext, filedialog, ttk
from datetime import datetime
import os, time, wave, queue, threading
import torch
from transformers import AutoProcessor
from transformers.models.voxtral.modeling_voxtral import VoxtralForConditionalGeneration

# ------------------ Setup directories ------------------
if not os.path.isdir('recordings'):
    os.mkdir('recordings')
recordings_cnt = 0

# ------------------ Load Voxtral model once ------------------
print("Loading Voxtral model... Please wait")
processor = AutoProcessor.from_pretrained("vikhyatk/moondream2")   # change if using other checkpoint
voxtral_model = VoxtralForConditionalGeneration.from_pretrained("vikhyatk/moondream2")
print("✅ Voxtral model loaded")

# ------------------ Helper functions ------------------
def get_summary_mode_config():
    mode = widgets['summary_mode_dropdown'].get()
    if mode == "Bullet":
        return "Provide a concise bullet-point summary of the meeting."
    elif mode == "Paragraph":
        return "Provide a detailed paragraph-style summary of the meeting."
    elif mode == "Crisp":
        return "Provide a very short and crisp summary highlighting only key points."
    else:
        return "Provide a summary of the meeting."

# ------------------ Audio Recording ------------------
def record_audio(stop_flag, audio_queue, status_queue):
    global recordings_cnt
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5

    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)
    except Exception as e:
        status_queue.put(f"❌ Error initializing microphone: {e}")
        return

    status_queue.put("🎙️ Recording started")
    while not stop_flag.is_set():
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            if stop_flag.is_set():
                break
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except Exception as e:
                status_queue.put(f"⚠️ Error while recording: {e}")
                continue

        if frames:
            recordings_cnt += 1
            filename = f"recordings/meeting_{recordings_cnt}.wav"
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            # enqueue with current mode
            audio_queue.put((filename, current_processing_mode))

    stream.stop_stream()
    stream.close()
    p.terminate()
    status_queue.put("🛑 Recording stopped")

# ------------------ Transcription (thread, reuses model) ------------------
def transcribe_audio_thread(audio_queue, status_queue, live_queue, stop_flag):
    while not stop_flag.is_set():
        try:
            file_path, mode = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        status_queue.put(f"⏳ Processing {file_path} in mode {mode}")
        try:
            with open(file_path, "rb") as f:
                inputs = processor(f.read(), sampling_rate=16000, return_tensors="pt")
            generated_ids = voxtral_model.generate(**inputs, max_new_tokens=200)
            transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if mode == "Summary":
                summary_instruction = get_summary_mode_config()
                messages = [{"role": "user",
                             "content": f"{summary_instruction}\n\nMeeting transcript:\n{transcript}"}]
            elif mode == "Action Points":
                messages = [{"role": "user",
                             "content": f"Extract action points from this meeting transcript: {transcript}"}]
            else:
                messages = [{"role": "user", "content": transcript}]

            live_queue.put((mode, transcript))
            status_queue.put(f"✅ Finished {file_path} ({mode})")

        except Exception as e:
            status_queue.put(f"❌ Error processing {file_path}: {e}")

# ------------------ GUI Setup ------------------
root = tk.Tk()
root.title("Meeting Minutes Generator")
widgets = {}

status_queue = mp.Queue()
audio_queue = queue.Queue()  # use threading queue here
live_queue = queue.Queue()

stop_recording_flag = mp.Event()
stop_transcription_flag = threading.Event()

recording_process = None
transcription_thread = None

current_processing_mode = "Transcription"

# --- GUI Layout ---
widgets['mode_label'] = tk.Label(root, text="Select Processing Mode:")
widgets['mode_label'].pack(pady=5)

processing_modes = ["Transcription", "Summary", "Action Points"]
processing_mode_var = tk.StringVar(value=processing_modes[0])
for mode in processing_modes:
    rb = tk.Radiobutton(root, text=mode, variable=processing_mode_var,
                        value=mode, command=lambda m=mode: update_processing_mode(m))
    rb.pack(anchor="w")

widgets['summary_mode_dropdown'] = ttk.Combobox(root, values=["Bullet", "Paragraph", "Crisp"], state="disabled")
widgets['summary_mode_dropdown'].set("Select Summary Mode")
widgets['summary_mode_dropdown'].pack(pady=5)

widgets['start_button'] = tk.Button(root, text="Start Meeting", command=start_meeting)
widgets['start_button'].pack(pady=5)

widgets['stop_button'] = tk.Button(root, text="End Meeting", command=stop_meeting, state=tk.DISABLED)
widgets['stop_button'].pack(pady=5)

widgets['status_label'] = tk.Label(root, text="Status: Ready", fg="green")
widgets['status_label'].pack(pady=5)

widgets['log'] = scrolledtext.ScrolledText(root, width=70, height=20, wrap=tk.WORD)
widgets['log'].pack(padx=10, pady=10, fill="both", expand=True)

# Separate windows
live_transcript = tk.Toplevel(root)
live_transcript.title("Live Transcript")
widgets['live_transcript'] = scrolledtext.ScrolledText(live_transcript, width=60, height=10, wrap=tk.WORD)
widgets['live_transcript'].pack(padx=10, pady=10, fill="both", expand=True)

live_improve = tk.Toplevel(root)
live_improve.title("Live Improve")
widgets['live_improve'] = scrolledtext.ScrolledText(live_improve, width=60, height=10, wrap=tk.WORD)
widgets['live_improve'].pack(padx=10, pady=10, fill="both", expand=True)

live_minutes = tk.Toplevel(root)
live_minutes.title("Live Minutes")
widgets['live_minutes'] = scrolledtext.ScrolledText(live_minutes, width=60, height=10, wrap=tk.WORD)
widgets['live_minutes'].pack(padx=10, pady=10, fill="both", expand=True)

# ------------------ GUI Functions ------------------
def update_status(msg, error=False):
    widgets['status_label'].config(text=f"Status: {msg}", fg="red" if error else "green")
    widgets['log'].insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n")
    widgets['log'].see(tk.END)

def update_processing_mode(mode):
    global current_processing_mode
    current_processing_mode = mode
    if mode == "Summary":
        widgets['summary_mode_dropdown'].config(state="readonly")
    else:
        widgets['summary_mode_dropdown'].set("Select Summary Mode")
        widgets['summary_mode_dropdown'].config(state="disabled")
    update_status(f"🔄 Processing mode switched to: {mode}")

def start_meeting():
    global recording_process, transcription_thread
    if processing_mode_var.get() == "Summary" and widgets['summary_mode_dropdown'].get() == "Select Summary Mode":
        update_status("⚠️ Please select a summary mode", error=True)
        return
    stop_recording_flag.clear()
    stop_transcription_flag.clear()
    recording_process = mp.Process(target=record_audio,
                                   args=(stop_recording_flag, audio_queue, status_queue))
    recording_process.start()
    transcription_thread = threading.Thread(target=transcribe_audio_thread,
                                            args=(audio_queue, status_queue, live_queue, stop_transcription_flag),
                                            daemon=True)
    transcription_thread.start()
    widgets['start_button'].config(state=tk.DISABLED)
    widgets['stop_button'].config(state=tk.NORMAL)
    update_status("✅ Meeting started")

def stop_meeting():
    stop_recording_flag.set()
    stop_transcription_flag.set()
    if recording_process and recording_process.is_alive():
        recording_process.terminate()
    widgets['start_button'].config(state=tk.NORMAL)
    widgets['stop_button'].config(state=tk.DISABLED)
    update_status("🛑 Meeting ended")

def update_gui():
    while not status_queue.empty():
        update_status(status_queue.get())
    while not live_queue.empty():
        mode, text = live_queue.get()
        if mode == "Transcription":
            widgets['live_transcript'].insert(tk.END, text + "\n")
            widgets['live_transcript'].see(tk.END)
        elif mode == "Summary":
            widgets['live_minutes'].insert(tk.END, text + "\n")
            widgets['live_minutes'].see(tk.END)
        elif mode == "Action Points":
            widgets['live_improve'].insert(tk.END, text + "\n")
            widgets['live_improve'].see(tk.END)
    root.after(500, update_gui)

update_gui()
root.mainloop()
