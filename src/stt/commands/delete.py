import keyboard

from stt.commands import CommandInput


def action(cmd: CommandInput):
    keyboard.press_and_release("ctrl+backspace")
