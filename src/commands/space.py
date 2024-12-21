import keyboard

from commands import CommandInput


def action(cmd: CommandInput):
    keyboard.press_and_release("space")
