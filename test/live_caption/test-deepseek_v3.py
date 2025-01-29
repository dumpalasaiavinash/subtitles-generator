# working with movable bar

import soundcard as sc
import numpy as np
from vosk import Model, KaldiRecognizer, SetLogLevel
import tkinter as tk
from threading import Thread
from queue import Queue
import sys
import os
import json

SetLogLevel(-1)

class EnhancedVisualCaptions:
    def __init__(self):
        self.model = Model("model")
        self.sample_rate = 16000
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(False)
        
        # Audio setup
        self.mic = self.get_loopback_mic()
        self.frame_size = 2048
        
        # GUI Configuration
        self.root = tk.Tk()
        self.root.title("Live Captions")
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)
        
        # Transparent background setup
        self.root.configure(bg='gray1')
        self.root.wm_attributes('-transparentcolor', 'gray1')
        self.root.attributes('-alpha', 0.85)  # Overall window transparency
        
        # Window dimensions and positioning
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.bar_height = 100
        self.root.geometry(f"{self.screen_width}x{self.bar_height}+0+{self.screen_height - self.bar_height}")
        
        # Visual elements
        self.create_gradient_background()
        self.create_text_labels()
        
        # Dragging functionality
        self.dragging = False
        self.offset_x = 0
        self.offset_y = 0
        self.bind_drag_events()
        
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

    def create_gradient_background(self):
        """Create a semi-transparent gradient background"""
        self.canvas = tk.Canvas(self.root, bg='gray1', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create gradient from black to slight blue tint
        for i in range(self.bar_height):
            alpha = i / self.bar_height
            color = "#%02x%02x%02x" % (
                int(0 * (1 - alpha) + 20 * alpha),
                int(0 * (1 - alpha) + 30 * alpha),
                int(0 * (1 - alpha) + 40 * alpha)
            )
            self.canvas.create_rectangle(
                0, i, 
                self.screen_width, i+1, 
                fill=color, 
                outline=color
            )

    def create_text_labels(self):
        """Create text labels with shadow effect"""
        # Shadow label
        self.shadow_label = tk.Label(
            self.canvas,
            text="",
            font=("Arial", 24, "bold"),
            fg="#000080",
            bg='gray1'
        )
        self.shadow_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER, x=2, y=2)
        
        # Main text label
        self.main_label = tk.Label(
            self.canvas,
            text="",
            font=("Arial", 24, "bold"),
            fg="#00ffff",  # Cyan color for better visibility
            bg='gray1'
        )
        self.main_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def bind_drag_events(self):
        """Enable window dragging functionality"""
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drag)

    def start_drag(self, event):
        """Handle drag start"""
        self.dragging = True
        self.offset_x = event.x
        self.offset_y = event.y

    def on_drag(self, event):
        """Handle dragging motion"""
        if self.dragging:
            x = self.root.winfo_x() + (event.x - self.offset_x)
            y = self.root.winfo_y() + (event.y - self.offset_y)
            self.root.geometry(f"+{x}+{y}")

    def stop_drag(self, event):
        """Handle drag release"""
        self.dragging = False

    def get_loopback_mic(self):
        """Audio device setup (unchanged from previous version)"""
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

    def capture_audio(self):
        """Audio capture (unchanged from previous version)"""
        try:
            with self.mic.recorder(
                    samplerate=self.sample_rate,
                    channels=1,
                    blocksize=self.frame_size) as mic:
                
                while self.running:
                    data = mic.record(numframes=self.frame_size)
                    data = np.clip(data * 1.5, -1.0, 1.0)
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
        """Update text with animation"""
        try:
            if not self.transcript_queue.empty():
                new_text = self.transcript_queue.get()
                self.text_buffer = new_text if "..." in new_text else f"{self.text_buffer} {new_text}"
                display_text = self.text_buffer[-120:].replace("...", "")
                
                # Update both labels for shadow effect
                self.main_label.config(text=display_text)
                self.shadow_label.config(text=display_text)
                
                # Fade animation
                self.canvas.configure(bg='gray1')
                self.canvas.after(50, lambda: self.canvas.configure(bg='gray1'))
                
        except Exception as e:
            pass
        
        if self.running:
            self.root.after(50, self.update_gui)

if __name__ == "__main__":
    if not os.path.exists("model"):
        print("Download and extract the Vosk model:")
        print("https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip")
        sys.exit(1)
    EnhancedVisualCaptions()