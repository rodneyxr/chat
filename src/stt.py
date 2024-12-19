#!/usr/bin/env python3
import logging
import queue
import sys
import threading
import warnings

import click
import keyboard
import numpy as np
import pyperclip
import sounddevice as sd
import torch
import whisper
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama

warnings.filterwarnings("ignore")


SYSTEM_PROMPT = """
You are a voice dictation engine. You will be given input and you must dictate the text as accurately as possible. You should not correct grammar or punctuation errors. If the input is a single word, you should spell it out. If the input is a sentence, you should capitalize the first word and add punctuation at the end.
Do not add any additional information or instructions in your output; it must simply be what the user is trying to type with their voice.

If you come across "space" or "period" or any other special character, you should dictate the corresponding character.

Translate the following input: {text}
"""

JARVIS_PROMPT = """
Your name is Jarvis, an advanced voice dictation engine. You will be given input in the form of transcribed text from voice. You must determine what the user is trying to type and only output the text that the user would type directly.
Do not add any additional information or instructions in your output; it must simply be what the user is trying to type with their voice.

Translate the following input: {text}
"""

prompt = PromptTemplate(template=SYSTEM_PROMPT, input_variables=["text"])


class VoiceDictation:
    def __init__(
        self, model_name: str, hotkey: str, sample_rate: int, channels: int, device, post_processing: bool
    ):
        self.hotkey = hotkey
        self.sample_rate = sample_rate
        self.channels = channels
        self.transcribing = threading.Event()
        self.recording = False
        self.audio_queue = queue.Queue()
        self.post_processing = post_processing

        logging.info(f"Loading whisper model '{model_name}' on {device.type}...")
        self.whisper_model = whisper.load_model(model_name, device=device)
        self.llm = Ollama(model="llama3.2:3b-instruct-q5_K_M", temperature=0)

    def audio_callback(self, indata, frames, time, status):
        if status:
            logging.error(f"Error: {status}")
        self.audio_queue.put(indata.copy())

    def process_with_ollama(self, text):
        try:
            processed_text = self.llm.invoke(prompt.format(text=text))
            logging.info(f"Ollama processed text: {processed_text}")
            return processed_text.lstrip(" ")
        except Exception as e:
            logging.error(f"Ollama error: {e}")
            return text

    def handle_commands(self, transcribed_text):
        """
        Check if the transcribed text represents a command and handle it.
        Returns a tuple (processed_text, was_command).
        """
        text_lower = transcribed_text.lower().strip()
        args = text_lower.split()

        def arg(i):
            return args[i].strip(".!,?") if len(args) > i else None

        # Special multi-word commands
        if len(args) > 1 and arg(0) == "hey" and arg(1) == "jarvis":
            logging.info("COMMAND: hey jarvis")
            new_text = " ".join(transcribed_text.split(" ")[2:])
            jarvis_prompt = PromptTemplate(
                template=JARVIS_PROMPT, input_variables=["text"]
            )
            return self.llm.invoke(jarvis_prompt.format(text=new_text)), True

        # Simple commands mapping
        command_map = {
            "paste": {
                "desc": "paste",
                "action": lambda: pyperclip.paste()
                if isinstance(pyperclip.paste(), str)
                else "",
            },
            "undo": {
                "desc": "undo",
                "action": lambda: keyboard.press_and_release("ctrl+z"),
            },
            "redo": {
                "desc": "redo",
                "action": lambda: keyboard.press_and_release("ctrl+y"),
            },
            "delete": {
                "desc": "delete",
                "action": lambda: keyboard.press_and_release("ctrl+backspace"),
            },
            "backspace": {
                "desc": "backspace",
                "action": lambda: keyboard.press_and_release("ctrl+backspace"),
            },
            "space": {
                "desc": "space",
                "action": lambda: keyboard.press_and_release("space"),
            },
            "comma": {
                "desc": "comma",
                "action": lambda: keyboard.press_and_release("comma"),
            },
            "help": {
                "desc": "help",
                "action": lambda: logging.info(
                    "\nAvailable commands:"
                    "\n  - hey jarvis"
                    "\n  - paste"
                    "\n  - undo"
                    "\n  - redo"
                    "\n  - delete/backspace"
                    "\n  - replace line"
                ),
            },
        }

        # Handle one-word commands
        if arg(0) in command_map and len(args) == 1:
            cmd_info = command_map[arg(0)]
            logging.info(f"COMMAND: {cmd_info['desc']}")
            result = cmd_info["action"]()
            return (result if result else "", True)

        # Handle "replace line"
        if arg(0) == "replace" and arg(1) in ["line", "lime"]:
            logging.info("COMMAND: replace line (WIP)")
            keyboard.press_and_release("end")
            keyboard.send("shift+home shift+home")
            return " ".join(transcribed_text.split(" ")[2:]), True

        # No command recognized
        return transcribed_text, False

    def transcribe_audio(self):
        logging.info(f"Press and hold {self.hotkey} to speak")

        while True:
            self.transcribing.wait()
            buffer = []
            self.audio_queue.queue.clear()

            if not self.recording:
                self.recording = True
                logging.info("Recording...")

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
            ):
                while self.transcribing.is_set():
                    try:
                        data = self.audio_queue.get(timeout=0.1)
                        buffer.extend(data[:, 0])
                    except queue.Empty:
                        pass

            self.recording = False
            if buffer:
                logging.info("Transcribing...")
                audio_data = np.array(buffer, dtype=np.float32)
                result = self.whisper_model.transcribe(audio_data, language="en")
                transcribed_text = result["text"].rstrip()
                logging.info(f"Raw transcription: {transcribed_text}")

                processed_text, was_command = self.handle_commands(transcribed_text)
                if was_command and not processed_text:
                    # Command executed, no text to type
                    logging.info(f"Press and hold {self.hotkey} to speak")
                    continue

                # Optional post-processing
                if self.post_processing and not was_command:
                    logging.info("Analyzing...")
                    processed_text = self.process_with_ollama(processed_text)

                # Type the resulting text
                if "\n" in processed_text:
                    for line in processed_text.split("\n"):
                        keyboard.write(line, delay=0.01)
                        keyboard.press_and_release("shift+enter")
                else:
                    keyboard.write(processed_text, delay=0.01)

                logging.info(f"Press and hold {self.hotkey} to speak")

    def on_press(self):
        self.transcribing.set()

    def on_release(self):
        self.transcribing.clear()


@click.command(help="A friendly CLI for voice dictation using Whisper and Ollama.")
@click.option("--model", default="turbo", help="Whisper model name")
@click.option("--hotkey", default="F24", help="Hotkey to hold while speaking")
@click.option("--samplerate", default=16000, help="Audio sample rate")
@click.option("--channels", default=1, help="Number of audio channels")
@click.option("--verbose", is_flag=True, help="Enable debug output")
@click.option("--post-processing", is_flag=True, default=False, help="Enable LLM post-processing of transcribed text")
def main(model, hotkey, samplerate, channels, verbose, post_processing):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logging.info("GPU detected. Using GPU for Whisper.")
    else:
        logging.info("No GPU detected. Using CPU for Whisper.")

    vd = VoiceDictation(model, hotkey, samplerate, channels, device, post_processing)

    threading.Thread(target=vd.transcribe_audio, daemon=True).start()
    keyboard.on_press_key(hotkey, lambda _: vd.on_press())
    keyboard.on_release_key(hotkey, lambda _: vd.on_release())

    try:
        keyboard.wait()
    except KeyboardInterrupt:
        logging.info("Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
