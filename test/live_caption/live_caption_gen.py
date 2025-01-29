import os
import wave
import pyaudio
from faster_whisper import WhisperModel

# Function to record a chunk of audio and save it to a file
def record_chunk(p, stream, file_path, chunk_length=2):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):  # 16000 Hz sample rate, 1024 buffer size
        data = stream.read(1024)
        frames.append(data)
    
    # Save the audio chunk to a file
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

# Function to transcribe the recorded audio chunk
def transcribe_chunk(model, file_path):
    segments, _ = model.transcribe(file_path)
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    return transcription.strip()

# Main function for real-time transcription
def main():
    # Model settings
    model_size = "base"  # Change to "base","medium.en", "large-v2","large-v3", etc., as needed
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")  # Use "cuda" for GPU, "cpu" for CPU
    model = WhisperModel(model_size, device="cpu", compute_type="int8")  # Use "cuda" for GPU, "cpu" for CPU

    # PyAudio setup
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    
    accumulated_transcription = ""  # To store the accumulated transcription

    try:
        print("Listening... Press Ctrl+C to stop.")
        while True:
            chunk_file = "temp_chunk.wav"
            
            # Record a chunk of audio
            record_chunk(p, stream, chunk_file)

            # Transcribe the recorded chunk
            transcription = transcribe_chunk(model, chunk_file)
            print(f"Transcription: {transcription}")

            # Append the transcription to the accumulated text
            accumulated_transcription += transcription + " "

            # Remove the temporary chunk file
            os.remove(chunk_file)

    except KeyboardInterrupt:
        print("Stopping transcription...")

        # Save the accumulated transcription to a log file
        with open("transcription_log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)
        print("Transcription saved to 'transcription_log.txt'.")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# Run the main function
if __name__ == "__main__":
    main()
