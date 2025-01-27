# code is working but issue with accuracy



import soundcard as sc
import numpy as np
from vosk import Model, KaldiRecognizer
import tkinter as tk
from threading import Thread
from queue import Queue
import sys
import os

class LiveCaptions:
    def __init__(self):
        self.model = Model("model")
        self.sample_rate = 16000
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(False)
        
        self.root = tk.Tk()
        self.root.title("Live Captions")
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)
        self.root.configure(bg='black')
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"800x60+{screen_width//2-400}+{screen_height-100}")
        
        self.label = tk.Label(self.root, 
                            text="", 
                            font=("Arial", 24), 
                            fg="white", 
                            bg="black",
                            wraplength=780,
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

    def capture_audio(self):
        with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(
            samplerate=self.sample_rate) as mic:
            while self.running:
                data = mic.record(numframes=1024)
                data = (data * 32767).astype(np.int16)
                if self.recognizer.AcceptWaveform(data.tobytes()):
                    result = self.recognizer.Result()[18:-3]
                    if result:
                        self.transcript_queue.put(result)
                else:
                    partial = self.recognizer.PartialResult()[17:-3]
                    if partial:
                        self.transcript_queue.put(partial)

    def update_gui(self):
        try:
            while not self.transcript_queue.empty():
                text = self.transcript_queue.get()
                self.label.config(text=text)
        except:
            pass
        if self.running:
            self.root.after(100, self.update_gui)

if __name__ == "__main__":
    if not os.path.exists("model"):
        print("Please download and extract the Vosk small English model to 'model' folder")
        print("Download from: https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
        sys.exit(1)
    LiveCaptions()