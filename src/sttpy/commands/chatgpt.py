import sys
import time

import keyboard
import pygetwindow

from sttpy.commands import CommandInput
from pygetwindow import Win32Window
import logging

args = ["chatgpt", "chat gpt", "chat"]


def action(cmd: CommandInput):
    already_active = False

    window = pygetwindow.getActiveWindow()
    logging.debug(
        f"{window=}, {window.title=}, {window.isMinimized=}, {window.isMaximized=}, {window.isActive=}, {window.size=}, {window.area=}, {window.visible=}"
    )
    if window.title == "ChatGPT" and window.visible:
        print("ChatGPT window is already active.")
        window.show()
        already_active = True

    if not already_active:
        window = get_window_with_exact_title("ChatGPT")
        logging.debug(f"{window=}")

        if window:
            logging.debug(f"Found window: {window.title}")
            # FIXME: Obtaining focus via activate() doesn't work.
            # Workaround is to hide and restore with application shortcut.
            window.hide()

        # Open ChatGPT
        keyboard.press_and_release("alt+space")

        # Wait for the window named 'ChatGPT' to focus with a timeout of 2 seconds
        start_time = time.time()
        while True:
            window = get_window_with_exact_title("ChatGPT")
            if window and window.visible:
                break
            if time.time() - start_time > 2:
                print(
                    "Timeout: 'ChatGPT' window did not come into focus within 2 seconds.",
                    file=sys.stderr,
                )
                return
            time.sleep(0.1)

    # TODO: support modes: continue, new, search, canvas

    # Continue existing chat
    # To reset chatgpt window (i.e. disable search mode), we can type '/canvas' + space then ctrl+backspace
    keyboard.press_and_release("ctrl+A")
    keyboard.write("/canvas ", delay=0.01)
    keyboard.press_and_release("ctrl+backspace")

    # New Chat
    # keyboard.press_and_release("ctrl+n")

    # Search mode
    # keyboard.write("/search ", delay=0.01)

    # Write the prompt
    keyboard.write(cmd.text, delay=0.01)

    # Submit the prompt
    return  # FIXME: re-enable this
    keyboard.press_and_release("enter")


def get_window_with_exact_title(title: str) -> Win32Window | None:
    potential_windows = pygetwindow.getWindowsWithTitle(title)
    # Window title much match exactly
    window = [w for w in potential_windows if w.title == title]
    if window:
        return window[0]
    return None


def center_window(window: pygetwindow.Win32Window):
    import ctypes

    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)  # SM_CXSCREEN
    screen_height = user32.GetSystemMetrics(1)  # SM_CYSCREEN
    # center_x = screen_width // 2
    # center_y = screen_height // 2
    window_width, window_height = window.size
    new_x = (screen_width - window_width) // 2
    new_y = (screen_height - window_height) // 2
    window.moveTo(new_x, new_y)
    return new_x, new_y
