#!/usr/bin/env python3
import importlib
import logging
import os
import queue
import sys
import threading
import warnings

import click
import keyboard
import numpy as np
import sounddevice as sd
import torch
import whisper
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama

from commands import CommandInput
from config import JARVIS_PROMPT, SYSTEM_PROMPT

warnings.filterwarnings("ignore")

prompt = PromptTemplate(template=SYSTEM_PROMPT, input_variables=["text"])
llm = Ollama(model="llama3.2:3b-instruct-q5_K_M", temperature=0)


class VoiceDictation:
    def __init__(
        self,
        model_name: str,
        hotkey: str,
        sample_rate: int,
        channels: int,
        device,
        post_processing: bool,
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

    def audio_callback(self, indata, frames, time, status):
        if status:
            logging.error(f"Error: {status}")
        self.audio_queue.put(indata.copy())

    def process_with_ollama(self, text):
        try:
            processed_text = llm.invoke(prompt.format(text=text))
            logging.info(f"Ollama processed text: {processed_text}")
            return processed_text.lstrip(" ")
        except Exception as e:
            logging.error(f"Ollama error: {e}")
            return text

    def load_commands(self):
        command_map = {}
        commands_dir = os.path.join(os.path.dirname(__file__), "commands")
        for filename in os.listdir(commands_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                command_name = filename[:-3]
                module = importlib.import_module(f"commands.{command_name}")
                command_map[command_name] = {
                    "desc": command_name,
                    "action": module.action,
                    "args": module.args if hasattr(module, "args") else [command_name],
                }
        return command_map

    def handle_commands(self, transcribed_text):
        """
        Check if the transcribed text represents a command and handle it.
        Returns a tuple (processed_text, was_command).
        """
        text_lower = transcribed_text.lower().strip()
        args = text_lower.split()

        def arg(i):
            return args[i].strip(".!,?") if len(args) > i else None

        # Load commands dynamically
        command_map = self.load_commands()

        # Handle commands
        for cmd_name, cmd_info in command_map.items():
            for cmd_arg in cmd_info["args"]:
                if text_lower.startswith(cmd_arg):
                    logging.info(f"COMMAND: {cmd_info['desc']}")
                    args: list[str] = cmd_arg.split()
                    nargs = len(args)
                    logging.debug(f"Args: {args}")
                    tmp = transcribed_text.split(maxsplit=nargs)
                    new_text = " ".join(tmp[nargs:]) if len(tmp) > nargs else ""
                    cmd_input = CommandInput(args, new_text)
                    result = cmd_info["action"](cmd_input)
                    return (result if result else "", True)

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
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.option(
    "--post-processing",
    is_flag=True,
    default=False,
    help="Enable LLM post-processing of transcribed text",
)
def main(model, hotkey, samplerate, channels, debug, post_processing):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
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
