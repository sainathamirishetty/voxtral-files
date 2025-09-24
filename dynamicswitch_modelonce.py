import multiprocessing as mp
import pyaudio
import numpy as np
import tkinter as tk
from tkinter import scrolledtext, filedialog, ttk
from datetime import datetime
import os, time, wave, queue
import torch
from transformers import AutoProcessor
from transformers.models.voxtral.modeling_voxtral import VoxtralForConditionalGeneration

# ------------------ Setup directories ------------------
if not os.path.isdir('recordings'):
    os.mkdir('recordings')
recordings_cnt = len(os.listdir('recordings'))
write_record_to_dir = f'recordings/record_{recordings_cnt}'
if not os.path.isdir(write_record_to_dir):
    os.mkdir(write_record_to_dir)

# ------------------ Global Variables ------------------
CHANNELS = 1
RATE = 44100
CHUNK = 1024
FORMAT = pyaudio.paInt16
MEETING_CHUNK_DURATION_MIN = 10

audio_queue_lst = mp.Queue()
transcript_queue_gui = mp.Queue()
summary_queue_gui = mp.Queue()
improve_queue_gui = mp.Queue()
update_status_queue_gui = mp.Queue()
exit_flag = mp.Event()
pause_flag = mp.Event()
stop_recording_flag = mp.Event()

# Mode Queue for dynamic switching
mode_queue = mp.Queue()
current_mode = mp.Value('u', 'Transcription')  # Default mode

# Widget dictionary
widgets = {}

root = tk.Tk()
root.geometry("1024x768+150+50")
root.resizable(False, False)
root.configure(bg='#F8EED9')
master = root

# Live windows
live_transcript_widget = {}
live_transcript = tk.Toplevel(master)
live_improve_widget = {}
live_improve = tk.Toplevel(master)
live_minutes_widget = {}
live_minutes = tk.Toplevel(master)

live_transcript.withdraw()
live_improve.withdraw()
live_minutes.withdraw()

all_transcript, all_improve, all_minutes = [], [], []
summary_cnt = 1

# ------------------ Model Loading ------------------
def load_voxtral_model():
    device = "cpu"
    repo_id = "/Users/vishnukumarkudidela/Desktop/workspace/ASR/models/Voxtral-Mini-3B-2507"
    processor = AutoProcessor.from_pretrained(repo_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        repo_id, torch_dtype=torch.bfloat16, device_map=device
    )
    return processor, model

global_processor, global_model = load_voxtral_model()

# ------------------ GUI Update Function ------------------
def update_status(message):
    update_status_queue_gui.put(message)
    print(message, flush=True)

def update_gui():
    global summary_cnt
    if not exit_flag.is_set():
        try:
            while not transcript_queue_gui.empty():
                transcript = transcript_queue_gui.get()
                if transcript.strip():
                    live_transcript_widget['transcript_display'].config(state=tk.NORMAL)
                    live_transcript_widget['transcript_display'].insert(
                        tk.END, transcript + "\n" + "="*20 + "\n")
                    live_transcript_widget['transcript_display'].see(tk.END)
                    live_transcript_widget['transcript_display'].config(state=tk.DISABLED)
                    all_transcript.append(transcript)
        except queue.Empty:
            pass
        try:
            while not update_status_queue_gui.empty():
                status_msg = update_status_queue_gui.get()
                if status_msg.strip():
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

# ------------------ Record Audio ------------------
def record_audio(audio_queue_lst, exit_flag, stop_recording_flag, pause_flag, current_mode):
    update_status("Recording audio started")
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        record_seconds = MEETING_CHUNK_DURATION_MIN * 60
        file_cnt = 0
        stop_recording_flag.clear()

        while not stop_recording_flag.is_set():
            frames = []
            for _ in range(int(RATE / CHUNK * record_seconds)):
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
                # Send audio + mode at chunk time
                audio_queue_lst.put((wave_output_filename, current_mode.value))
                update_status(f"{file_cnt} Audio Chunk created: {wave_output_filename}")
        stream.stop_stream()
        stream.close()
        audio.terminate()
    except Exception as e:
        update_status(f"Error in audio recording: {str(e)}")

# ------------------ Process Audio ------------------
def transcribe_audio(audio_queue_lst, exit_flag, processor, model, transcript_queue_gui, summary_queue_gui):
    update_status("Audio processing started")
    while not exit_flag.is_set():
        try:
            # Check for mode switch
            if not mode_queue.empty():
                new_mode = mode_queue.get().strip()
                with current_mode.get_lock():
                    current_mode.value = new_mode
                update_status(f"Transcription mode switched to: {new_mode}")

            if audio_queue_lst.empty():
                time.sleep(0.2)
                continue

            file_path, chunk_mode = audio_queue_lst.get()
            update_status(f"Processing chunk: {file_path} in mode: {chunk_mode}")

            # ------------------ Transcription ------------------
            if chunk_mode == "Transcription":
                transcript_queue_gui.put(f"Transcribed: {file_path}")
            elif chunk_mode == "Summary":
                summary_queue_gui.put(f"Summary: {file_path}")
            elif chunk_mode == "Action Points":
                summary_queue_gui.put(f"Action Points: {file_path}")
            else:
                update_status(f"Unknown mode: {chunk_mode}")
        except Exception as e:
            update_status(f"Error processing audio: {str(e)}")

# ------------------ Start Meeting ------------------
def start_meeting():
    global recording_process, transcription_process

    selected_mode = widgets['process_mode_dropdown'].get().strip()
    with current_mode.get_lock():
        current_mode.value = selected_mode

    mode_queue.put(selected_mode)
    stop_recording_flag.clear()
    exit_flag.clear()

    recording_process = mp.Process(
        target=record_audio,
        args=(audio_queue_lst, exit_flag, stop_recording_flag, pause_flag, current_mode),
        daemon=True
    )
    recording_process.start()

    transcription_process = mp.Process(
        target=transcribe_audio,
        args=(audio_queue_lst, exit_flag, global_processor, global_model,
              transcript_queue_gui, summary_queue_gui),
        daemon=True
    )
    transcription_process.start()

    update_gui()
    update_status("Meeting started successfully")

# ------------------ Pause/Resume ------------------
def pause_meeting():
    if not pause_flag.is_set():
        pause_flag.set()
        update_status("Recording paused")
    else:
        pause_flag.clear()
        update_status("Recording resumed")

# ------------------ Stop Meeting ------------------
def stop_meeting():
    stop_recording_flag.set()
    exit_flag.set()
    update_status("Meeting ended")

# ------------------ Switch Mode ------------------
def switch_mode(event=None):
    new_mode = widgets['process_mode_dropdown'].get().strip()
    mode_queue.put(new_mode)
    update_status(f"Requested mode switch to: {new_mode}")

# ------------------ GUI Setup ------------------
def setup_gui():
    master.title("Meeting Minutes Generator")
    text_font = ("Arial", 14)

    process_modes = ["Transcription", "Summary", "Action Points"]
    widgets['process_mode_dropdown'] = ttk.Combobox(master, values=process_modes, font=text_font, state="readonly", width=25)
    widgets['process_mode_dropdown'].grid(row=0, column=0, padx=10, pady=10)
    widgets['process_mode_dropdown'].set("Transcription")
    widgets['process_mode_dropdown'].bind("<<ComboboxSelected>>", switch_mode)

    widgets['start_button'] = tk.Button(master, text="Start Meeting", command=start_meeting)
    widgets['start_button'].grid(row=0, column=1, padx=10, pady=10)

    widgets['pause_button'] = tk.Button(master, text="Pause/Resume", command=pause_meeting)
    widgets['pause_button'].grid(row=0, column=2, padx=10, pady=10)

    widgets['stop_button'] = tk.Button(master, text="Stop Meeting", command=stop_meeting)
    widgets['stop_button'].grid(row=0, column=3, padx=10, pady=10)

    widgets['status_display'] = scrolledtext.ScrolledText(master, height=10, width=120, state=tk.DISABLED)
    widgets['status_display'].grid(row=1, column=0, columnspan=4, padx=10, pady=10)

# ------------------ Main ------------------
if __name__ == "__main__":
    setup_gui()
    root.mainloop()
