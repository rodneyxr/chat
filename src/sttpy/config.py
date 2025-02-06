import os

import yaml

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

CONFIG_FILE = "config.yaml"
DEFAULT_CONFIG = {
    "hotkey": "ctrl+space",
    # "prompt": SYSTEM_PROMPT,
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)
    else:
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(config, f)
