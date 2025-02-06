import keyboard

from sttpy.commands import CommandInput


def action(cmd: CommandInput):
    keyboard.press_and_release("ctrl+z")
