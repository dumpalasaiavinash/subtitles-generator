import soundcard as sc
import numpy as np
from vosk import Model, KaldiRecognizer, SetLogLevel
import tkinter as tk
from threading import Thread
from queue import Queue
import sys
import os
import json
import configparser

SetLogLevel(-1)


class SmoothCaptions:
    def __init__(self):
        # Load configurations
        self.config = self.load_config()

        # Initialize Vosk model
        model_path = self.config.get("Model", "model_path")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at '{model_path}'. Check your config.ini.")
        self.model = Model(model_path)
        self.sample_rate = self.config.getint("Audio", "sample_rate")
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(False)

        # Audio setup
        self.mic = self.get_loopback_mic()
        self.frame_size = self.config.getint("Audio", "frame_size")

        # GUI Configuration
        self.root = tk.Tk()
        self.root.title(self.config.get("Window", "title"))
        self.root.attributes('-topmost', self.config.getboolean("Window", "topmost"))

        # Window properties
        self.default_width = self.config.getint("Window", "default_width")
        self.default_height = self.config.getint("Window", "default_height")
        self.min_width = self.config.getint("Window", "min_width")
        self.min_height = self.config.getint("Window", "min_height")
        self.dragging = False
        self.resizing = False
        self.resize_type = None

        # Color scheme
        self.bg_color = self.config.get("Colors", "background")
        self.text_color = self.config.get("Colors", "text")
        self.handle_color = self.config.get("Colors", "handle")
        self.border_color = self.config.get("Colors", "border")
        self.close_button_color = self.config.get("Window", "close_button_color")

        # Initialize window
        self.setup_window()
        self.create_widgets()
        self.setup_bindings()

        # Text handling
        self.text_buffer = ""
        self.transcript_queue = Queue()
        self.running = True

        # Start threads
        self.audio_thread = Thread(target=self.capture_audio)
        self.audio_thread.start()

        self.update_gui()
        self.root.mainloop()
        self.running = False
        self.audio_thread.join()

    def load_config(self):
        """Load configurations from config.ini file."""
        config = configparser.ConfigParser()
        if not os.path.exists("config.ini"):
            raise FileNotFoundError("config.ini file not found. Please create one.")
        config.read("config.ini")
        return config

    def get_loopback_mic(self):
        """Get system audio output as input (loopback)."""
        try:
            return sc.get_microphone(
                id=str(sc.default_speaker().name),
                include_loopback=True
            )
        except Exception as e:
            print(f"Audio Device Error: {str(e)}")
            print("Available microphones:")
            for mic in sc.all_microphones():
                print(f"- {mic.name}")
            sys.exit(1)

    def setup_window(self):
        """Configure window appearance and initial position."""
        self.root.configure(bg=self.bg_color)
        self.root.attributes('-alpha', self.config.getfloat("Window", "window_alpha"))
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(
            f"{self.default_width}x{self.default_height}"+
            f"+{screen_width // 2 - self.default_width // 2}"+
            f"+{screen_height - self.default_height - 20}"
        )

    def create_widgets(self):
        """Create GUI elements."""
        # Main text container
        self.text_frame = tk.Frame(self.root, bg=self.bg_color)
        self.text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Text label
        self.label = tk.Label(
            self.text_frame,
            text="",
            font=(self.config.get("Font", "family"), self.config.getint("Font", "size")),
            fg=self.text_color,
            bg=self.bg_color,
            wraplength=self.default_width - 40,
            justify='center'
        )
        self.label.pack(expand=True)

        # Create resize handles
        self.create_resize_handles()

        # Create close button
        self.create_close_button()

    def create_resize_handles(self):
        """Create resizable handles with modern design."""
        handle_size = 8
        # Right handle
        self.right_handle = tk.Frame(
            self.root,
            bg=self.handle_color,
            width=2,
            height=30
        )
        self.right_handle.place(
            relx=1.0,
            rely=0.5,
            anchor='e',
            x=-1
        )

        # Bottom handle
        self.bottom_handle = tk.Frame(
            self.root,
            bg=self.handle_color,
            width=30,
            height=2
        )
        self.bottom_handle.place(
            relx=0.5,
            rely=1.0,
            anchor='s',
            y=-1
        )

    def create_close_button(self):
        """Add a close button to the overlay."""
        self.close_button = tk.Button(
            self.root,
            text="X",
            bg=self.close_button_color,
            fg="white",
            command=self.terminate_program,
            bd=0,
            font=("Arial", 12, "bold")
        )
        # self.close_button.place(relx=1.0, rely=0.0, anchor="ne")  # Top right corner close button comment it to use default window close button

    def setup_bindings(self):
        """Set up mouse bindings for moving and resizing."""
        # Drag bindings
        self.text_frame.bind("<ButtonPress-1>", self.start_drag)
        self.text_frame.bind("<B1-Motion>", self.on_drag)
        self.text_frame.bind("<ButtonRelease-1>", self.stop_drag)

        # Resize bindings
        self.right_handle.bind("<ButtonPress-1>", lambda e: self.start_resize('right'))
        self.right_handle.bind("<B1-Motion>", lambda e: self.on_resize('right'))
        self.bottom_handle.bind("<ButtonPress-1>", lambda e: self.start_resize('bottom'))
        self.bottom_handle.bind("<B1-Motion>", lambda e: self.on_resize('bottom'))

    def terminate_program(self):
        """Terminate the program gracefully."""
        self.running = False
        self.root.destroy()

    # Remaining methods remain unchanged (capture_audio, update_gui, etc.)


    def start_drag(self, event):
        self.dragging = True
        self.offset_x = event.x_root - self.root.winfo_x()
        self.offset_y = event.y_root - self.root.winfo_y()

    def on_drag(self, event):
        if self.dragging:
            x = event.x_root - self.offset_x
            y = event.y_root - self.offset_y
            self.root.geometry(f"+{x}+{y}")

    def stop_drag(self, event):
        self.dragging = False

    def start_resize(self, resize_type):
        self.resizing = True
        self.resize_type = resize_type
        self.initial_x = self.root.winfo_pointerx()
        self.initial_y = self.root.winfo_pointery()
        self.initial_width = self.root.winfo_width()
        self.initial_height = self.root.winfo_height()

    def on_resize(self, event):
        if self.resizing:
            current_x = self.root.winfo_pointerx()
            current_y = self.root.winfo_pointery()

            new_width = self.initial_width
            new_height = self.initial_height

            if self.resize_type in ('right', 'corner'):
                new_width = max(self.min_width, self.initial_width + (current_x - self.initial_x))

            if self.resize_type in ('bottom', 'corner'):
                new_height = max(self.min_height, self.initial_height + (current_y - self.initial_y))

            self.root.geometry(f"{new_width}x{new_height}")
            self.label.config(wraplength=new_width - 40)
            self.update_canvas_border(new_width, new_height)

    def update_canvas_border(self, width, height):
        """Update the border rectangle when resizing."""
        self.canvas.delete("all")
        self.canvas.create_rectangle(
            1, 1,
            width - 1, height - 1,
            outline=self.border_color,
            width=2
        )

    def stop_resize(self, event):
        self.resizing = False

    def capture_audio(self):
        """Capture audio and process for captions."""
        try:
            with self.mic.recorder(
                    samplerate=self.sample_rate,
                    channels=1,
                    blocksize=self.frame_size) as mic:

                while self.running:
                    data = mic.record(numframes=self.frame_size)
                    data = np.clip(data * self.config.getfloat("Audio", "audio_boost"), -1.0, 1.0)  # Boost audio levels
                    data = (data * 32767).astype(np.int16)

                    if self.recognizer.AcceptWaveform(data.tobytes()):
                        result = json.loads(self.recognizer.Result())
                        text = result.get('text', '').strip()
                        if text:
                            self.transcript_queue.put(text.upper())
                    else:
                        partial = json.loads(self.recognizer.PartialResult())
                        text = partial.get('partial', '').strip()
                        if text:
                            self.transcript_queue.put(text.upper() + "...")
        except Exception as e:
            print(f"Audio Capture Error: {str(e)}")
            self.running = False

    def update_gui(self):
        """Update text display without flickering."""
        try:
            if not self.transcript_queue.empty():
                new_text = self.transcript_queue.get()
                self.text_buffer = new_text if "..." in new_text else f"{self.text_buffer} {new_text}"
                display_text = self.text_buffer[-self.config.getint("Text", "max_buffer_length"):].replace("...", "")
                self.label.config(text=display_text)

        except Exception as e:
            pass

        if self.running:
            self.root.after(self.config.getint("Text", "update_delay_ms"), self.update_gui)


if __name__ == "__main__":
    if not os.path.exists("model"):
        print("Download and extract the Vosk model:")
        print("https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip")
        sys.exit(1)
    SmoothCaptions()