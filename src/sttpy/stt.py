import importlib
import logging
import os
import queue
import threading
import time
import warnings
from collections import defaultdict

import keyboard
import numpy as np
import pyperclip
import sounddevice as sd
import torch
import whisper
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama

from sttpy.commands import CommandInput
from sttpy.config import SYSTEM_PROMPT

warnings.filterwarnings("ignore")

prompt = PromptTemplate(template=SYSTEM_PROMPT, input_variables=["text"])
llm = Ollama(model="llama3.2:3b-instruct-q5_K_M", temperature=0)


class VoiceDictation:
    # FIXME: Short recording + lengthy transcription text length should be ignored. (quick click of hotkey should not be transcribed)
    def __init__(
        self,
        hotkey: str = "F1",
        model_name: str = "base.en",
        post_processing: bool = False,
        sample_rate: int = 16000,
        channels: int = 1,
        device: str | None = None,
        hot_reload: bool = False,
        type_mode: bool = False,
        paste_delay: float = 0.1,  # Delay between copying to clipboard and pasting
    ):
        self.hotkey = hotkey
        self.sample_rate = sample_rate
        self.channels = channels
        self.transcribing = threading.Event()
        self.recording = False
        self.audio_queue = queue.Queue()
        self.post_processing = post_processing
        self.command_map = self.load_commands(reload=hot_reload)
        self.hot_reload = hot_reload
        self.type_mode = type_mode
        self.paste_delay = paste_delay

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def load_commands(self, reload=False):
        command_map = {}
        commands_dir = os.path.join(os.path.dirname(__file__), "commands")
        for filename in os.listdir(commands_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                command_name = filename[:-3]
                try:
                    module = importlib.import_module(f"sttpy.commands.{command_name}")
                    if reload:
                        importlib.reload(module)
                    command_map[command_name] = {
                        "desc": command_name,
                        "action": module.action,
                        "args": module.args
                        if hasattr(module, "args")
                        else [command_name],
                    }
                except Exception as e:
                    logging.warning(f"Failed to load command '{command_name}': {e}")
                    continue
        # Create a dictionary to map command arguments to a list of command information
        arg_to_command = {}
        for cmd_info in command_map.values():
            for cmd_arg in cmd_info["args"]:
                arg_to_command[cmd_arg] = cmd_info
        self.arg_to_command = arg_to_command
        return command_map

    def handle_commands(self, transcribed_text: str):
        """
        Check if the transcribed text represents a command and handle it.
        Returns a tuple (processed_text, was_command).
        """
        # if hot_reload mode is enabled, reload the commands on every invocation
        self.command_map = self.load_commands(reload=self.hot_reload)

        arg_search = (
            transcribed_text.lower().strip().translate(str.maketrans("", "", ".,!?"))
        )

        if arg_search == "help":
            command_groups = defaultdict(list)
            for command, value in self.arg_to_command.items():
                command_groups[str(value)].append(command)

            log_message = "\nAvailable commands:\n"
            for value, commands in command_groups.items():
                log_message += f"  - {' / '.join(commands)}\n"

            logging.info(log_message)
            return None, True

        for command in self.arg_to_command:
            if arg_search.startswith(command):
                cmd_info = self.arg_to_command[command]
                logging.info(f"COMMAND: {cmd_info['desc']}")
                args: list[str] = command.split()
                nargs = len(args)
                logging.debug(f"Args: {args}")
                tmp = transcribed_text.split(maxsplit=nargs)
                new_text = " ".join(tmp[nargs:]) if len(tmp) > nargs else ""
                cmd_input = CommandInput(args, new_text)
                try:
                    result = cmd_info["action"](cmd_input)
                    return result, True
                except Exception as e:
                    logging.error(f"Error executing command: {e}")
                    return None, True
        # No command recognized
        return None, False

    def process_text(self, text: str):
        # Process command (if it is one)
        cmd_text, was_command = self.handle_commands(text)
        if was_command:
            if not cmd_text:
                # Command executed, no text to type
                return
            else:
                text = cmd_text

        # Optional post-processing
        if self.post_processing:
            logging.info("Analyzing...")
            processed_text = self.process_with_ollama(text)
        else:
            processed_text = text

        if self.type_mode:
            # Type the resulting text
            if "\n" in processed_text:
                for line in processed_text.split("\n"):
                    keyboard.write(line, delay=0.01)
                    keyboard.press_and_release("shift+enter")
            else:
                keyboard.write(processed_text, delay=0.01)
        else:
            # Save the original clipboard content to restore it after pasting (only works for text; not media)
            original_clipboard = pyperclip.paste()

            # Load the processed text to the clipboard
            pyperclip.copy(processed_text)
            time.sleep(self.paste_delay)

            # Paste the processed text
            keyboard.press_and_release("ctrl+v")
            time.sleep(self.paste_delay)

            # Restore the original clipboard content
            pyperclip.copy(original_clipboard)

    def transcribe_audio(self):
        logging.info(f"Press and hold '{self.hotkey}' to speak")

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
                transcribed_text = result["text"].strip()
                logging.info(f"Raw transcription: {transcribed_text}")
                self.process_text(transcribed_text)
                logging.info(f"Press and hold '{self.hotkey}' to speak")

    def on_press(self):
        self.transcribing.set()

    def on_release(self):
        self.transcribing.clear()
