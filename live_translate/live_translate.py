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

# Argos Translate
import argostranslate.package
import argostranslate.translate

SetLogLevel(-1)


def load_config(config_file="config.ini"):
    """Load configuration from config.ini."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found.")
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def load_vosk_model(model_path):
    """Load local Vosk model from 'model_path' folder."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Vosk model not found at '{model_path}'")
    return Model(model_path)


def setup_recognizer(vosk_model, sample_rate):
    """Create and return a KaldiRecognizer."""
    recognizer = KaldiRecognizer(vosk_model, sample_rate)
    recognizer.SetWords(False)
    return recognizer


def get_loopback_mic():
    """Get a loopback microphone device for system audio."""
    try:
        return sc.get_microphone(
            id=str(sc.default_speaker().name),
            include_loopback=True
        )
    except Exception as e:
        print(f"Audio Device Error: {str(e)}")
        print("Available microphones:")
        for m in sc.all_microphones():
            print(" -", m.name)
        sys.exit(1)


def setup_translator_auto(from_code, to_code):
    """
    Auto-download Argos Translate package for `from_code` -> `to_code` if not present.
    Returns a function: translator_fn(text) -> str
    """
    # Update Argos package list (requires internet if not cached)
    argostranslate.package.update_package_index()

    installed_langs = argostranslate.translate.get_installed_languages()
    already_installed = any(
        lang.code == from_code and any(
            hasattr(t, 'code') and t.code == to_code
            for t in lang.translations_to
        )
        for lang in installed_langs
    )

    if not already_installed:
        print(f"Installing Argos Translate package for {from_code} -> {to_code}...")
        available_pkgs = argostranslate.package.get_available_packages()
        try:
            pkg = next(
                p for p in available_pkgs
                if p.from_code == from_code and p.to_code == to_code
            )
            argostranslate.package.install_from_path(pkg.download())
            print("Installation complete.")
        except StopIteration:
            print(f"ERROR: No Argos package found for {from_code}->{to_code}.")
            print("Falling back to identity function (no translation).")
            return lambda text: text

    # Return the actual translator
    def translator_fn(text):
        return argostranslate.translate.translate(text, from_code, to_code)

    return translator_fn


def capture_audio_loop(state):
    """
    Continuously captures system audio, performs speech recognition with Vosk.
    - Partial results => recognized_queue (is_final=False)
    - Final results => recognized_queue (is_final=True)
    """
    mic = state["mic"]
    recognizer = state["recognizer"]
    frame_size = state["frame_size"]
    sample_rate = state["sample_rate"]
    audio_boost = state["audio_boost"]
    recognized_queue = state["recognized_queue"]

    try:
        with mic.recorder(samplerate=sample_rate, channels=1, blocksize=frame_size) as m:
            while state["running"]:
                audio_data = m.record(numframes=frame_size)
                # Boost
                audio_data = np.clip(audio_data * audio_boost, -1.0, 1.0)
                audio_data = (audio_data * 32767).astype(np.int16)

                if recognizer.AcceptWaveform(audio_data.tobytes()):
                    # We got a final result
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        recognized_queue.put({"text": text, "is_final": True})
                else:
                    # We got a partial result
                    partial = json.loads(recognizer.PartialResult())
                    text = partial.get("partial", "").strip()
                    if text:
                        recognized_queue.put({"text": text, "is_final": False})
    except Exception as e:
        print(f"Audio capture error: {e}")
        state["running"] = False


def translate_loop(state):
    """
    Pulls items from recognized_queue, translates them with Argos, pushes them to translated_queue.
    Each item is dict: {"text": <english text>, "is_final": bool}
    We add "translated_text": <target text>.
    """
    recognized_queue = state["recognized_queue"]
    translated_queue = state["translated_queue"]
    translator = state["translator"]

    while state["running"]:
        if not recognized_queue.empty():
            item = recognized_queue.get()
            text = item["text"]
            # Translate partial or final text
            translated = translator(text)
            item["translated_text"] = translated
            translated_queue.put(item)


def setup_gui(state):
    """
    Create main Tkinter window and label for displaying captions.
    """
    config = state["config"]
    root = tk.Tk()
    root.title(config.get("Window", "title"))
    root.attributes('-topmost', config.getboolean("Window", "topmost"))

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

    frame = tk.Frame(root, bg=bg_color)
    frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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
    Periodically read from translated_queue:
     - if is_final=False => partial_text = item["translated_text"] + "..."
     - if is_final=True  => append to text_buffer, clear partial_text
    Then display text_buffer + partial_text (if any).
    """
    translated_queue = state["translated_queue"]
    config = state["config"]

    while not translated_queue.empty():
        item = translated_queue.get()
        translated_text = item["translated_text"]
        if item["is_final"]:
            # Append final text to buffer
            state["text_buffer"] += " " + translated_text
            # Clear partial
            state["partial_text"] = ""
        else:
            # Partial text
            state["partial_text"] = translated_text + "..."

    # Compose display
    display_text = state["text_buffer"].strip()
    if state["partial_text"]:
        display_text += "\n" + state["partial_text"].strip()

    # Trim if too long
    max_len = config.getint("Text", "max_buffer_length")
    if len(display_text) > max_len:
        display_text = display_text[-max_len:]
        # We also want to shrink text_buffer if it's too long
        if len(state["text_buffer"]) > max_len:
            state["text_buffer"] = state["text_buffer"][-max_len:]

    # Update label
    state["label"].config(text=display_text)

    if state["running"]:
        state["root"].after(config.getint("Text", "update_delay_ms"), update_gui, state)


def main():
    # 1. Read config
    config = load_config("config.ini")

    # 2. Setup translator
    src_lang = config.get("Translation", "source_language", fallback="en")
    tgt_lang = config.get("Translation", "target_language", fallback="hi")
    translator_fn = setup_translator_auto(src_lang, tgt_lang)

    # 3. Load Vosk model
    model_path = config.get("Model", "model_path", fallback="model")
    vosk_model = load_vosk_model(model_path)
    sample_rate = config.getint("Audio", "sample_rate", fallback=16000)
    recognizer = setup_recognizer(vosk_model, sample_rate)

    # 4. Audio device
    mic = get_loopback_mic()
    frame_size = config.getint("Audio", "frame_size", fallback=1024)
    audio_boost = config.getfloat("Audio", "audio_boost", fallback=1.0)

    # 5. Shared state
    state = {
        "config": config,
        "running": True,
        "mic": mic,
        "recognizer": recognizer,
        "frame_size": frame_size,
        "sample_rate": sample_rate,
        "audio_boost": audio_boost,
        "translator": translator_fn,

        "recognized_queue": Queue(),    # partial/final recognized text (EN)
        "translated_queue": Queue(),    # partial/final translated text (HI)

        # Text buffers
        "text_buffer": "",    # Accumulated final translations
        "partial_text": "",   # Latest partial translation
    }

    # 6. GUI
    root, label = setup_gui(state)
    state["root"] = root
    state["label"] = label

    # 7. Threads
    t_capture = Thread(target=capture_audio_loop, args=(state,))
    t_translate = Thread(target=translate_loop, args=(state,))

    t_capture.start()
    t_translate.start()

    # 8. Start periodic GUI update
    update_gui(state)

    # 9. Main loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        state["running"] = False
        t_capture.join()
        t_translate.join()


if __name__ == "__main__":
    if not os.path.exists("model"):
        print("Please place a Vosk model in `model/` folder or update your config.ini path.")
        sys.exit(1)

    main()
