import sys
import time

import keyboard
import pygetwindow

from commands import CommandInput

args = ["chatgpt", "chat gpt", "ciao gbt", "chat"]


def action(cmd: CommandInput):
    keyboard.press_and_release("alt+space")
    # Wait for the window named 'ChatGPT' to focus with a timeout of 3 seconds
    start_time = time.time()
    while True:
        window = pygetwindow.getActiveWindowTitle()
        if window == "ChatGPT":
            break
        if time.time() - start_time > 3:
            print("Timeout: 'ChatGPT' window did not come into focus within 3 seconds.", file=sys.stderr)
            return
        time.sleep(0.1)

    # New Chat
    keyboard.press_and_release("ctrl+n")

    # Search mode
    keyboard.write("/search ", delay=0.01)

    # Write the prompt
    keyboard.write(cmd.text, delay=0.01)

    # Submit the prompt
    keyboard.press_and_release("enter")