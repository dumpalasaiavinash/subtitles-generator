import soundcard as sc
import numpy as np
import tkinter as tk
from threading import Thread
from queue import Queue
import sys
import os
import json
import configparser

# Vosk for speech recognition
from vosk import Model, KaldiRecognizer, SetLogLevel

# Argos Translate (2.x or newer)
import argostranslate.package
import argostranslate.translate

SetLogLevel(-1)


def load_config(config_file="config.ini"):
    """
    Load configurations from config.ini.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found. Please create or provide one.")
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def load_vosk_model(model_path):
    """
    Load Vosk model from the specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Vosk model folder not found at '{model_path}'")
    return Model(model_path)


def setup_recognizer(model, sample_rate):
    """
    Create a KaldiRecognizer from the Vosk model and sample rate.
    """
    recognizer = KaldiRecognizer(model, sample_rate)
    recognizer.SetWords(False)  # set True for word-level info if desired
    return recognizer


def get_loopback_mic():
    """
    Attempt to get a loopback microphone device from the default speaker.
    """
    try:
        mic = sc.get_microphone(
            id=str(sc.default_speaker().name),
            include_loopback=True
        )
        return mic
    except Exception as e:
        print(f"Audio Device Error: {str(e)}")
        print("Available microphones:")
        for m in sc.all_microphones():
            print(" -", m.name)
        sys.exit(1)


def setup_translator_auto(from_code, to_code):
    """
    Automatically downloads/installs the Argos Translate language package for `from_code`->`to_code`
    if not already installed.

    Returns a function: translator_fn(text) -> str
    that translates `text` from `from_code` to `to_code`.

    If no package is found, returns a function that returns the input text unchanged.
    """

    # 1. Update Argos package index (requires internet if not cached).
    argostranslate.package.update_package_index()

    # 2. Check if the from_code->to_code package is already installed.
    installed_languages = argostranslate.translate.get_installed_languages()
    already_installed = False

    for lang in installed_languages:
        if lang.code == from_code:
            # ✅ Filter out IdentityTranslation objects
            valid_translations = [t for t in lang.translations_to if hasattr(t, 'code')]

            # ✅ Now check if target language exists in valid translations
            for to_lang_obj in valid_translations:
                if to_lang_obj.code == to_code:
                    already_installed = True
                    break
            if already_installed:
                break

    # 3. If not installed, try to download & install
    if not already_installed:
        print(f"Installing Argos Translate package for {from_code} -> {to_code}...")
        available_packages = argostranslate.package.get_available_packages()
        try:
            package_to_install = next(
                p for p in available_packages
                if p.from_code == from_code and p.to_code == to_code
            )
            argostranslate.package.install_from_path(package_to_install.download())
            print("Installation complete.")
        except StopIteration:
            print(f"ERROR: No Argos package found for {from_code}->{to_code}.")
            print("Translation will be skipped (identity function).")
            return lambda text: text  # Return a function that does nothing

    # 4. Define a translator function that uses Argos' main translate() entry point.
    def translator_fn(text):
        return argostranslate.translate.translate(text, from_code, to_code)

    return translator_fn



def capture_audio_loop(state):
    """
    Continuously record system audio from the loopback device, pass it to the Vosk recognizer,
    translate recognized text, then place it in the transcript queue.
    """
    mic = state["mic"]
    recognizer = state["recognizer"]
    sample_rate = state["sample_rate"]
    frame_size = state["frame_size"]
    boost = state["audio_boost"]
    translator = state["translator"]
    queue = state["transcript_queue"]

    try:
        with mic.recorder(samplerate=sample_rate, channels=1, blocksize=frame_size) as m:
            while state["running"]:
                audio_data = m.record(numframes=frame_size)
                # Boost and convert to int16 for Vosk
                audio_data = np.clip(audio_data * boost, -1.0, 1.0)
                audio_data = (audio_data * 32767).astype(np.int16)

                if recognizer.AcceptWaveform(audio_data.tobytes()):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        # Translate final recognized text
                        translated = translator(text)
                        queue.put(translated)
                else:
                    partial = json.loads(recognizer.PartialResult())
                    text = partial.get("partial", "").strip()
                    if text:
                        # Translate partial
                        translated = translator(text)
                        # Indicate partial with ...
                        queue.put(translated + "...")
    except Exception as e:
        print(f"Audio capture error: {e}")
        state["running"] = False


def setup_gui(state):
    """
    Create the main Tkinter window and label for displaying the (translated) captions.
    Returns (root, label).
    """
    config = state["config"]
    root = tk.Tk()
    root.title(config.get("Window", "title"))
    root.attributes('-topmost', config.getboolean("Window", "topmost"))

    # Window geometry
    default_w = config.getint("Window", "default_width")
    default_h = config.getint("Window", "default_height")
    min_w = config.getint("Window", "min_width")
    min_h = config.getint("Window", "min_height")
    alpha = config.getfloat("Window", "window_alpha")

    bg_color = config.get("Colors", "background")
    text_color = config.get("Colors", "text")

    font_family = config.get("Font", "family")
    font_size = config.getint("Font", "size")

    root.configure(bg=bg_color)
    root.attributes("-alpha", alpha)

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x_pos = (screen_w // 2) - (default_w // 2)
    y_pos = screen_h - default_h - 20
    root.geometry(f"{default_w}x{default_h}+{x_pos}+{y_pos}")
    root.minsize(min_w, min_h)

    # Main frame
    frame = tk.Frame(root, bg=bg_color)
    frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Label for captions
    label = tk.Label(
        frame,
        text="",
        font=(font_family, font_size),
        fg=text_color,
        bg=bg_color,
        wraplength=default_w - 40,
        justify="center"
    )
    label.pack(expand=True)

    return root, label


def update_gui(state):
    """
    Periodically poll the transcript queue for new text, update the label display.
    """
    queue = state["transcript_queue"]
    label = state["label"]
    config = state["config"]
    buffer_obj = state["text_buffer"]

    if not queue.empty():
        new_text = queue.get()

        # If partial (ends with "..."), just show it appended (but don't store permanently).
        if new_text.endswith("..."):
            display_text = buffer_obj["value"] + " " + new_text
        else:
            # Final text: append to buffer
            buffer_obj["value"] = buffer_obj["value"] + " " + new_text
            display_text = buffer_obj["value"]

        # Trim if too long
        max_len = config.getint("Text", "max_buffer_length")
        if len(display_text) > max_len:
            display_text = display_text[-max_len:]
            buffer_obj["value"] = buffer_obj["value"][-max_len:]

        label.config(text=display_text)

    if state["running"]:
        root = state["root"]
        delay_ms = config.getint("Text", "update_delay_ms")
        root.after(delay_ms, update_gui, state)


def main():
    # 1. Load config
    config = load_config("config.ini")

    # 2. Prepare translation (auto-install if missing)
    src_lang = config.get("Translation", "source_language", fallback="en")
    tgt_lang = config.get("Translation", "target_language", fallback="hi")
    translator_fn = setup_translator_auto(src_lang, tgt_lang)

    # 3. Load Vosk model & recognizer
    model_path = config.get("Model", "model_path", fallback="model")
    vosk_model = load_vosk_model(model_path)
    sample_rate = config.getint("Audio", "sample_rate", fallback=16000)
    recognizer = setup_recognizer(vosk_model, sample_rate)

    # 4. Setup audio
    mic = get_loopback_mic()
    frame_size = config.getint("Audio", "frame_size", fallback=2048)
    audio_boost = config.getfloat("Audio", "audio_boost", fallback=1.0)

    # 5. Shared state
    state = {
        "config": config,
        "running": True,
        "recognizer": recognizer,
        "mic": mic,
        "sample_rate": sample_rate,
        "frame_size": frame_size,
        "audio_boost": audio_boost,
        "translator": translator_fn,
        "transcript_queue": Queue(),
        "text_buffer": {"value": ""},
    }

    # 6. GUI
    root, label = setup_gui(state)
    state["root"] = root
    state["label"] = label

    # 7. Start audio thread
    audio_thread = Thread(target=capture_audio_loop, args=(state,))
    audio_thread.start()

    # 8. Start GUI update loop
    update_gui(state)

    # 9. Main TK loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        state["running"] = False
        audio_thread.join()


if __name__ == "__main__":
    if not os.path.exists("model"):
        print("Please download and extract a Vosk model into the 'model/' folder")
        print("or update 'model_path' in config.ini to point to your model location.")
        sys.exit(1)

    main()
