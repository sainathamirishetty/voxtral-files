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
MEETING_CHUNK_DURATION_MIN = 10  # Hardcoded chunk duration

# Global queues and flags
audio_queue = mp.Queue()
audio_queue_lst = mp.Queue()
unified_output_queue = mp.Queue()  # NEW: Single queue for all outputs
update_status_queue_gui = mp.Queue()
mode_switch_queue = mp.Queue()
exit_flag = mp.Event()
pause_flag = mp.Event()
stop_recording_flag = mp.Event()
mode_switch_flag = mp.Event()
recording_process = None
recorded_audio_process = None
transcription_process = None
load_audio_flag = False
start_time = None
running = False
meeting_active = False

# Global widget dictionary
widgets = {}

root = tk.Tk()
root.geometry("1024x768+150+50")
root.resizable(False, False)
root.configure(bg='#F8EED9')
master = root

live_minutes_widget = {}
live_minutes = tk.Tk()

# Sequential storage for all outputs
all_sequential_outputs = []  # NEW: Single list for all outputs in chronological order

# Mode counters for numbering
mode_counters = {
    'Transcription': 0,
    'Summary': 0,
    'Action Points': 0
}

live_minutes.withdraw()

default_window_font = ("Arial", 16)
window_font = ["Arial", 16]

current_processing_mode = None


# ----------------- Function to handle Process Mode change -----------------
def process_mode_changed(event=None):
    global meeting_active, current_processing_mode
    selected_mode = widgets['process_mode_dropdown'].get().strip()

    if selected_mode == "Summary":
        widgets['summary_mode_dropdown'].config(state="readonly")
    else:
        widgets['summary_mode_dropdown'].set("Select Summary Mode")
        widgets['summary_mode_dropdown'].config(state="disabled")
    
    # Handle mode switching during active meeting
    if meeting_active and selected_mode != current_processing_mode:
        switch_processing_mode(selected_mode)


def switch_processing_mode(new_mode):
    """Handle switching processing mode during an active meeting"""
    global current_processing_mode
    
    if new_mode == "Select Processing Mode":
        return
    
    old_mode = current_processing_mode
    current_processing_mode = new_mode
    
    # Signal mode switch to processing thread
    mode_switch_flag.set()
    mode_switch_queue.put({
        'new_mode': new_mode,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    
    update_status(f"Mode switched from '{old_mode}' to '{new_mode}' at {datetime.now().strftime('%H:%M:%S')}")
    
    # Update GUI to reflect mode switch
    if new_mode == "Summary":
        widgets['summary_mode_dropdown'].config(state="readonly")
    else:
        widgets['summary_mode_dropdown'].set("Select Summary Mode")
        widgets['summary_mode_dropdown'].config(state="disabled")


def update_status(message):
    update_status_queue_gui.put(message)
    print(message, flush=True)


def default_state():
    global meeting_active
    meeting_active = False
    widgets['start_button'].config(state=tk.NORMAL)
    widgets['pause_resume_button'].config(state=tk.DISABLED)
    widgets['process_mode_dropdown'].config(state="readonly")
    live_minutes_widget['save_button'].config(state=tk.NORMAL)
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
    if not exit_flag.is_set():
        # Handle unified outputs
        try:
            while not unified_output_queue.empty():
                output_data = unified_output_queue.get()
                if output_data and len(output_data['content'].strip()):
                    # Add to sequential storage
                    all_sequential_outputs.append(output_data)
                    
                    # Display in live minutes window
                    live_minutes_widget['minutes_display'].config(state=tk.NORMAL)
                    live_minutes_widget['minutes_display'].insert(
                        tk.END, f"\n{output_data['header']}\n{'-'*50}\n{output_data['content']}\n{'='*50}\n")
                    live_minutes_widget['minutes_display'].see(tk.END)
                    live_minutes_widget['minutes_display'].config(state=tk.DISABLED)
        except queue.Empty:
            pass
            
        # Handle status updates
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
            
        master.after(1000, update_gui)


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
                    audio_queue_lst.put({
                        'file_path': audio_files[i],
                        'is_partial': False,
                        'duration_mins': 10.0,
                        'use_previous_mode': False
                    })
                else:
                    update_status("Invalid Audio File: " + str(audio_files[i]))
            i += 1
        update_status("All audio files Loaded")
    except Exception as e:
        update_status(f"Error in load audio: {str(e)}")


def record_audio(audio_queue, exit_flag, stop_recording_flag, pause_flag):
    update_status("Recording audio started")
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                            frames_per_buffer=CHUNK)

        # Always use hardcoded chunk duration:
        duration_minutes = MEETING_CHUNK_DURATION_MIN
        record_seconds = duration_minutes * 60
        update_status(f"Recording chunks of {duration_minutes} minutes each")
        update_status("Audio Stream initialized")
        file_cnt = 0
        stop_recording_flag.clear()

        while not stop_recording_flag.is_set():
            frames = []
            chunk_start_time = time.time()
            
            for i in range(int(RATE / CHUNK * record_seconds)):
                # Check for mode switch during recording
                if mode_switch_flag.is_set() and frames:
                    # Mode switch detected - save partial chunk immediately
                    elapsed_time = time.time() - chunk_start_time
                    elapsed_minutes = elapsed_time / 60
                    
                    update_status(f"Mode switch detected - saving partial chunk ({elapsed_minutes:.1f} minutes)")
                    
                    # Save the partial chunk
                    file_cnt += 1
                    wave_output_filename = f"{write_record_to_dir}/recording_partial_{datetime.now().strftime('%H:%M:%S')}.wav"
                    wave_file = wave.open(wave_output_filename, 'wb')
                    wave_file.setnchannels(CHANNELS)
                    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
                    wave_file.setframerate(RATE)
                    wave_file.writeframes(b''.join(frames))
                    wave_file.close()
                    full_file_path = os.path.join(base_directory, wave_output_filename)
                    
                    # Add mode info to the queue for processing with old mode
                    audio_queue_lst.put({
                        'file_path': full_file_path,
                        'is_partial': True,
                        'duration_mins': elapsed_minutes,
                        'use_previous_mode': True
                    })
                    
                    update_status(f"{file_cnt} Partial Audio Chunk created ({elapsed_minutes:.1f} mins)")
                    
                    # Reset for new chunk with new mode
                    frames = []
                    chunk_start_time = time.time()
                
                data = stream.read(CHUNK, exception_on_overflow=False)
                if not pause_flag.is_set():
                    frames.append(data)
                if exit_flag.is_set() or stop_recording_flag.is_set():
                    break
            
            # Save complete chunk (normal 10-minute chunk or final chunk)
            if frames:
                file_cnt += 1
                elapsed_time = time.time() - chunk_start_time
                elapsed_minutes = elapsed_time / 60
                
                wave_output_filename = f"{write_record_to_dir}/recording_{datetime.now().strftime('%H:%M:%S')}.wav"
                wave_file = wave.open(wave_output_filename, 'wb')
                wave_file.setnchannels(CHANNELS)
                wave_file.setsampwidth(audio.get_sample_size(FORMAT))
                wave_file.setframerate(RATE)
                wave_file.writeframes(b''.join(frames))
                wave_file.close()
                full_file_path = os.path.join(base_directory, wave_output_filename)
                
                # Add to queue - this will use current mode
                audio_queue_lst.put({
                    'file_path': full_file_path,
                    'is_partial': False,
                    'duration_mins': elapsed_minutes,
                    'use_previous_mode': False
                })
                
                update_status(f"{file_cnt} Audio Chunk created ({elapsed_minutes:.1f} mins)")
                
        stream.stop_stream()
        stream.close()
        audio.terminate()
    except Exception as e:
        print(f"Error in audio recording: {str(e)}")
        update_status(f"Error in audio recording: {str(e)}")


def transcribe_audio(audio_queue_lst, exit_flag, initial_mode):
    """
    Process audio files with dynamic mode switching and unified sequential output.
    """
    update_status("Audio processing started")
    current_mode = initial_mode
    previous_mode = initial_mode
    
    try:
        device = "cpu"
        repo_id = "/Users/vishnukumarkudidela/Desktop/workspace/ASR/models/Voxtral-Mini-3B-2507"

        processor = AutoProcessor.from_pretrained(repo_id)
        model = VoxtralForConditionalGeneration.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16, device_map=device
        )

        update_status("Model successfully loaded")

        while not exit_flag.is_set():
            # Check for mode switch
            if mode_switch_flag.is_set():
                try:
                    mode_info = mode_switch_queue.get_nowait()
                    previous_mode = current_mode
                    current_mode = mode_info['new_mode']
                    timestamp = mode_info['timestamp']
                    update_status(f"Processing mode switched from '{previous_mode}' to '{current_mode}' at {timestamp}")
                    mode_switch_flag.clear()
                except queue.Empty:
                    mode_switch_flag.clear()
            
            try:
                if audio_queue_lst.empty():
                    time.sleep(0.5)
                    continue
                else:
                    audio_info = audio_queue_lst.get()
                    
                    # Handle both old format (string) and new format (dict)
                    if isinstance(audio_info, str):
                        file_path = audio_info
                        is_partial = False
                        use_previous_mode = False
                        duration_mins = 10.0
                    else:
                        file_path = audio_info['file_path']
                        is_partial = audio_info.get('is_partial', False)
                        use_previous_mode = audio_info.get('use_previous_mode', False)
                        duration_mins = audio_info.get('duration_mins', 10.0)
                
                # Determine which mode to use for this chunk
                processing_mode = previous_mode if use_previous_mode else current_mode
                
                # Increment counter for this mode
                mode_counters[processing_mode] += 1
                mode_number = mode_counters[processing_mode]
                
                chunk_type = "PARTIAL" if is_partial else "FULL"
                print(f"Processing {chunk_type} audio: {file_path} ({duration_mins:.1f} mins) with mode: {processing_mode}")
                update_status(f"Processing {chunk_type} chunk ({duration_mins:.1f} mins) with {processing_mode} mode")

                decoded_outputs = []
                output_header = ""
                
                # ------------------ Transcription ------------------
                if processing_mode == "Transcription":
                    inputs = processor.apply_transcription_request(language="en", audio=file_path, model_id=repo_id)
                    inputs = inputs.to(device, dtype=torch.bfloat16)
                    outputs = model.generate(**inputs, max_new_tokens=4000)
                    decoded_outputs = processor.batch_decode(
                        outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
                    )
                    print("=== Transcription Output ===")
                    print(decoded_outputs[0])

                    chunk_info = f" [{chunk_type} - {duration_mins:.1f} mins]" if is_partial else ""
                    output_header = f"Transcription{mode_number}{chunk_info} - {datetime.now().strftime('%H:%M:%S')}"

                # ------------------ Summarization ------------------
                elif processing_mode == "Summary":
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
                    outputs = model.generate(**inputs, max_new_tokens=2048)
                    decoded_outputs = processor.batch_decode(
                        outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
                    )
                    print("=== Summarization Output ===")
                    print(decoded_outputs[0])

                    chunk_info = f" [{chunk_type} - {duration_mins:.1f} mins]" if is_partial else ""
                    output_header = f"Summary{mode_number}{chunk_info} - {datetime.now().strftime('%H:%M:%S')}"

                # ------------------ Action Points ------------------
                elif processing_mode == "Action Points":
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
                    outputs = model.generate(**inputs, max_new_tokens=2048)
                    decoded_outputs = processor.batch_decode(
                        outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
                    )
                    print("=== Action Points Output ===")
                    print(decoded_outputs[0])

                    chunk_info = f" [{chunk_type} - {duration_mins:.1f} mins]" if is_partial else ""
                    output_header = f"Action Points{mode_number}{chunk_info} - {datetime.now().strftime('%H:%M:%S')}"

                else:
                    update_status(f"Warning: Unknown processing mode '{processing_mode}', skipping audio chunk")
                    continue

                # Send to unified output queue
                if decoded_outputs and len(decoded_outputs[0].strip()):
                    unified_output_queue.put({
                        'header': output_header,
                        'content': decoded_outputs[0],
                        'mode': processing_mode,
                        'number': mode_number,
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'is_partial': is_partial,
                        'duration': duration_mins
                    })

                update_status(f"Successfully processed {chunk_type} chunk with {processing_mode}{mode_number}")

            except Exception as e:
                print(f"Error in processing audio: {str(e)}", flush=True)
                update_status(f"Error in processing audio: {str(e)}")
    except Exception as e:
        print(f"Error in transcribe_audio: {str(e)}")
        update_status(f"Error in transcribe_audio: {str(e)}")


def do_not_close():
    pass


def toggle_window(window):
    if window.winfo_viewable():
        window.withdraw()
    else:
        window.deiconify()


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


def setup_gui():
    master.title("Meeting Minutes Generator")
    text_font = ("Arial", 14)
    scrolledText_font = ('Arial', 14, 'bold', 'underline')
    text_bg = "#f0f0f0"
    text_fg = "black"

    # Styles for dropdown
    style = ttk.Style()
    style.configure("TCombobox", fieldbackground="white", background="white")
    style.configure("Error.TCombobox", fieldbackground="#ffcccc", background="#ffcccc")
    style.configure("Success.TCombobox", fieldbackground="#ccffcc", background="#ccffcc")

    master.grid_columnconfigure(1, weight=1)
    for i in range(14):
        master.grid_rowconfigure(i, weight=1)

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

    tk.Label(legend_frame_input, text="Processing Mode:", font=text_font).grid(row=4, column=0, sticky="e")
    process_modes = ["Select Processing Mode", " Transcription ", " Summary ", " Action Points "]
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

    # Current mode indicator
    tk.Label(legend_frame_input, text="Current Mode:", font=text_font).grid(row=6, column=0, sticky="e")
    widgets['current_mode_label'] = tk.Label(legend_frame_input, text="None", font=text_font, fg="blue")
    widgets['current_mode_label'].grid(row=6, column=1, sticky="w")

    tk.Label(master, text="Additional Context:", font=scrolledText_font, justify="left").grid(row=6, column=0,
                                                                                              columnspan=1)
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

    tk.Label(master, text="Command Status:", font=scrolledText_font, justify="left").grid(row=10, column=0,
                                                                                          columnspan=1)
    widgets['status_display'] = scrolledtext.ScrolledText(master, height=5, background="black", fg="white", width=50)
    widgets['status_display'].grid(row=11, column=0, columnspan=3, padx=0, sticky="nswe")

    legend_frame_output = tk.Frame(master, borderwidth=2, padx=100, pady=10, relief='groove')
    legend_frame_output.grid(row=12, columnspan=3)

    widgets['quit_button'] = tk.Button(
        legend_frame_output, text="Quit Application", command=lambda: quit_application(),
        width=20, font=text_font, background="orange")
    widgets['quit_button'].grid(row=12, column=2)

    # Setup unified minutes window
    live_minutes.title("Live Minutes - Sequential View")
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
        time.sleep(1)


def start_timer():
    global start_time, running
    start_time = time.time()
    running = True
    threading.Thread(target=update_duration, daemon=True).start()


def stop_timer():
    global running
    running = False


def start_meeting():
    global recording_process, transcription_process, meeting_active, current_processing_mode

    # Validate inputs
    if not validate_inputs():
        return

    selected_mode = widgets['process_mode_dropdown'].get().strip()
    current_processing_mode = selected_mode
    meeting_active = True
    
    # Update current mode display
    widgets['current_mode_label'].config(text=selected_mode, fg="green")
    
    update_status(f"Starting meeting with mode: {selected_mode}")

    start_timer()
    widgets['start_button'].config(state=tk.DISABLED)
    widgets['pause_resume_button'].config(state=tk.NORMAL)
    widgets['stop_button'].config(state=tk.NORMAL)

    try:
        # Clear flags and reset counters
        exit_flag.clear()
        pause_flag.clear()
        stop_recording_flag.clear()
        mode_switch_flag.clear()
        
        # Reset mode counters
        global mode_counters
        mode_counters = {'Transcription': 0, 'Summary': 0, 'Action Points': 0}

        # Start recording process
        recording_process = mp.Process(
            target=record_audio,
            args=(audio_queue, exit_flag, stop_recording_flag, pause_flag),
            daemon=True
        )
        recording_process.start()

        # Start transcription/summarization/action-points process with initial mode
        transcription_process = mp.Process(
            target=transcribe_audio,
            args=(audio_queue_lst, exit_flag, selected_mode),
            daemon=True
        )
        transcription_process.start()

        update_status("Meeting started successfully")
        update_gui()
    except Exception as e:
        update_status(f"Error starting meeting: {str(e)}")
        widgets['start_button'].config(state=tk.NORMAL)
        meeting_active = False


def load_audio_files():
    global recorded_audio_process, transcription_process, load_audio_flag, meeting_active, current_processing_mode

    if not validate_inputs():
        return

    selected_mode = widgets['process_mode_dropdown'].get().strip()
    current_processing_mode = selected_mode
    meeting_active = True

    # Update current mode display
    widgets['current_mode_label'].config(text=selected_mode, fg="green")

    load_audio_flag = True
    audio_dir = filedialog.askdirectory()
    if not audio_dir:
        widgets['start_button'].config(state=tk.NORMAL)
        meeting_active = False
        return

    update_status(f"Loaded Audio directory: {audio_dir}")
    update_status(f"Using mode: {selected_mode}")

    widgets['start_button'].config(state=tk.DISABLED)
    widgets['pause_resume_button'].config(state=tk.NORMAL)
    widgets['stop_button'].config(state=tk.NORMAL)

    try:
        exit_flag.clear()
        pause_flag.clear()
        stop_recording_flag.clear()
        mode_switch_flag.clear()
        
        # Reset mode counters
        global mode_counters
        mode_counters = {'Transcription': 0, 'Summary': 0, 'Action Points': 0}

        # Process to load audio files
        recorded_audio_process = mp.Process(
            target=loaded_audio_process,
            args=(audio_dir, audio_queue, exit_flag, stop_recording_flag),
            daemon=True
        )
        recorded_audio_process.start()

        # Process to handle transcription/summarization/action points with initial mode
        transcription_process = mp.Process(
            target=transcribe_audio,
            args=(audio_queue_lst, exit_flag, selected_mode),
            daemon=True
        )
        transcription_process.start()

        update_status("Audio processing started successfully")
        update_gui()
    except Exception as e:
        update_status(f"Error loading audio files: {str(e)}")
        widgets['start_button'].config(state=tk.NORMAL)
        meeting_active = False


def pause_resume_meeting():
    if pause_flag.is_set():
        # Currently paused → resume
        pause_flag.clear()
        widgets['pause_resume_button'].config(text="Pause Meeting")
        update_status("Meeting resumed")
    else:
        # Currently running → pause
        pause_flag.set()
        widgets['pause_resume_button'].config(text="Resume Meeting")
        update_status("Meeting paused")


def stop_audio_recording():
    global meeting_active, current_processing_mode
    update_status("Stopping audio recording...")
    stop_timer()
    meeting_active = False
    current_processing_mode = None
    
    # Update current mode display
    widgets['current_mode_label'].config(text="None", fg="black")
    
    stop_recording_flag.set()
    widgets['stop_button'].config(state=tk.DISABLED)

    # Wait for processes to finish or terminate
    if load_audio_flag and recorded_audio_process and recorded_audio_process.is_alive():
        recorded_audio_process.join(timeout=5)
        if recorded_audio_process.is_alive():
            recorded_audio_process.terminate()

    if not load_audio_flag and recording_process and recording_process.is_alive():
        recording_process.join(timeout=5)
        if recording_process.is_alive():
            recording_process.terminate()

    if transcription_process and transcription_process.is_alive():
        transcription_process.join(timeout=5)
        if transcription_process.is_alive():
            transcription_process.terminate()

    # Clear queues
    while not audio_queue.empty():
        audio_queue.get()
    update_status("Audio queue cleared")

    while not unified_output_queue.empty():
        unified_output_queue.get()
    update_status("Output queue cleared")

    while not mode_switch_queue.empty():
        mode_switch_queue.get()
    update_status("Mode switch queue cleared")

    # Reset GUI buttons
    widgets['start_button'].config(state=tk.NORMAL)
    widgets['pause_resume_button'].config(state=tk.DISABLED)
    widgets['process_mode_dropdown'].config(state="readonly")
    live_minutes_widget['save_button'].config(state=tk.NORMAL)

    update_status("Meeting Ended.")


def save_minutes():
    file_path = filedialog.asksaveasfilename(defaultextension=".docx",
                                             filetypes=[("Text files", "*.docx"), ("All files", "*.*")])
    if file_path:
        try:
            meeting_date = widgets['date_entry'].get()
            meeting_time = widgets['time_entry'].get()
            meeting_venue = widgets['venue_entry'].get()
            meeting_agenda = widgets['agenda_entry'].get()
            additional_context = widgets['context_text'].get("1.0", tk.END)

            # Get the sequential content from the display
            save_minutes_text = live_minutes_widget['minutes_display'].get("1.0", tk.END)

            content = f"Meeting Date: {meeting_date}\n\nMeeting Time: {meeting_time}\n\nVenue: {meeting_venue}\n\nAgenda: {meeting_agenda}\n\nAdditional Context:\n\n{additional_context}\n\nSEQUENTIAL MEETING MINUTES:\n\n{save_minutes_text}"
            save_minutes_text_md = markdown(content)

            with open(file_path, "wb") as fp:
                doc_content = html2docx.html2docx(save_minutes_text_md, title="Sequential Meeting Minutes")
                fp.write(doc_content.getvalue())
            update_status(f"Sequential minutes saved to {file_path}")
        except Exception as e:
            update_status(f"Error saving minutes: {str(e)}")
            print(f"error saving minutes: {traceback.format_exc()}{e}")


def quit_application():
    global meeting_active
    update_status("Quitting application and saving all backups...")
    meeting_active = False
    
    # Save unified sequential backup
    file_name = f"{write_record_to_dir}/Sequential_Minutes_backup_{datetime.now().strftime('%d-%m-%Y_%H-%M')}.md"
    with open(file_name, "w+") as file:
        meeting_date = widgets['date_entry'].get()
        meeting_time = widgets['time_entry'].get()
        meeting_venue = widgets['venue_entry'].get()
        meeting_agenda = widgets['agenda_entry'].get()
        additional_context = widgets['context_text'].get("1.0", tk.END)
        
        # Write meeting header
        file.write(f"# Meeting Minutes - Sequential View\n\n")
        file.write(f"**Date:** {meeting_date}\n")
        file.write(f"**Time:** {meeting_time}\n")
        file.write(f"**Venue:** {meeting_venue}\n")
        file.write(f"**Agenda:** {meeting_agenda}\n\n")
        file.write(f"**Additional Context:**\n{additional_context}\n\n")
        file.write("---\n\n")
        
        # Write all sequential outputs
        for i, output in enumerate(all_sequential_outputs):
            file.write(f"## {output['header']}\n\n")
            file.write(f"{output['content']}\n\n")
            file.write("---\n\n")
    
    update_status(f"Sequential minutes backup saved to: {file_name}")

    # Save individual mode backups for reference
    transcription_outputs = [o for o in all_sequential_outputs if o['mode'] == 'Transcription']
    if transcription_outputs:
        file_name = f"{write_record_to_dir}/Transcription_backup_{datetime.now().strftime('%d-%m-%Y_%H-%M')}.md"
        with open(file_name, "w+") as file:
            file.write("# Transcription Outputs\n\n")
            for output in transcription_outputs:
                file.write(f"## {output['header']}\n\n{output['content']}\n\n---\n\n")
        update_status(f"Transcription backup saved to: {file_name}")

    summary_outputs = [o for o in all_sequential_outputs if o['mode'] == 'Summary']
    if summary_outputs:
        file_name = f"{write_record_to_dir}/Summary_backup_{datetime.now().strftime('%d-%m-%Y_%H-%M')}.md"
        with open(file_name, "w+") as file:
            file.write("# Summary Outputs\n\n")
            for output in summary_outputs:
                file.write(f"## {output['header']}\n\n{output['content']}\n\n---\n\n")
        update_status(f"Summary backup saved to: {file_name}")

    action_outputs = [o for o in all_sequential_outputs if o['mode'] == 'Action Points']
    if action_outputs:
        file_name = f"{write_record_to_dir}/ActionPoints_backup_{datetime.now().strftime('%d-%m-%Y_%H-%M')}.md"
        with open(file_name, "w+") as file:
            file.write("# Action Points Outputs\n\n")
            for output in action_outputs:
                file.write(f"## {output['header']}\n\n{output['content']}\n\n---\n\n")
        update_status(f"Action Points backup saved to: {file_name}")

    update_status(f"All backups saved to directory: {write_record_to_dir}")
    
    # Terminate processes and exit
    exit_flag.set()
    stop_recording_flag.set()
    live_minutes.quit()
    master.quit()
    print("All processes completed. Closing application with complete backups saved.")
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
    live_minutes.protocol("WM_DELETE_WINDOW", lambda: toggle_window(live_minutes))
    setup_gui()
    root.mainloop()