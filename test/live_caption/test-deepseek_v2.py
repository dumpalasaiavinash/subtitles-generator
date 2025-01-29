# accuracy is good and it is working 


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

class FixedLiveCaptions:
    def __init__(self):
        self.model = Model("model")
        self.sample_rate = 16000
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(False)
        
        # Get loopback microphone for system audio capture
        self.mic = self.get_loopback_mic()
        self.frame_size = 2048
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Live Captions")
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)
        self.root.configure(bg='black')
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width//2}x80+{screen_width//4}+{screen_height-120}")
        
        self.text_buffer = ""
        self.label = tk.Label(self.root, 
                            text="", 
                            font=("Arial", 20, "bold"), 
                            fg="#00FF00",
                            bg="black",
                            wraplength=screen_width//2 - 20,
                            justify='center')
        self.label.pack(pady=10)
        
        self.transcript_queue = Queue()
        self.running = True
        
        self.audio_thread = Thread(target=self.capture_audio)
        self.audio_thread.start()
        
        self.update_gui()
        self.root.mainloop()
        self.running = False
        self.audio_thread.join()

    def get_loopback_mic(self):
        """Find and return the loopback microphone"""
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
        """Fixed audio capture using proper loopback device"""
        try:
            with self.mic.recorder(
                    samplerate=self.sample_rate,
                    channels=1,
                    blocksize=self.frame_size) as mic:
                
                while self.running:
                    data = mic.record(numframes=self.frame_size)
                    # Apply audio processing
                    data = np.clip(data * 1.5, -1.0, 1.0)  # Increased gain
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
        """Improved text rendering with buffer"""
        try:
            if not self.transcript_queue.empty():
                new_text = self.transcript_queue.get()
                self.text_buffer = new_text if "..." in new_text else f"{self.text_buffer} {new_text}"
                self.label.config(text=self.text_buffer[-120:].replace("...", ""))
        except:
            pass
        
        if self.running:
            self.root.after(50, self.update_gui)

if __name__ == "__main__":
    if not os.path.exists("model"):
        print("Download and extract the Vosk model:")
        print("https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip")
        sys.exit(1)
    FixedLiveCaptions()