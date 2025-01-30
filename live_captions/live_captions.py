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

        # Color scheme
        self.bg_color = self.config.get("Colors", "background")
        self.text_color = self.config.get("Colors", "text")

        # Initialize window
        self.setup_window()
        self.create_widgets()

        # Text handling
        self.text_buffer = ""
        self.transcript_queue = Queue()
        self.running = True

        # Start audio thread
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
            f"{self.default_width}x{self.default_height}"
            f"+{screen_width // 2 - self.default_width // 2}"
            f"+{screen_height - self.default_height - 20}"
        )
        # Allow normal OS window resizing (default style)
        self.root.minsize(self.min_width, self.min_height)

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
                # If it's a partial (with "..."), just show partial
                # Otherwise, append to main buffer
                if "..." in new_text:
                    # Show partial but don't override full buffer
                    display_text = self.text_buffer + " " + new_text
                else:
                    self.text_buffer = f"{self.text_buffer} {new_text}"
                    display_text = self.text_buffer

                # Trim display text to max buffer length (remove trailing partials if needed)
                max_length = self.config.getint("Text", "max_buffer_length")
                display_text = display_text[-max_length:]
                self.label.config(text=display_text)
        except Exception:
            pass

        if self.running:
            self.root.after(self.config.getint("Text", "update_delay_ms"), self.update_gui)


if __name__ == "__main__":
    if not os.path.exists("model"):
        print("Download and extract the Vosk model:")
        print("https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip")
        sys.exit(1)
    SmoothCaptions()
