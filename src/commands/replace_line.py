from click import Command
import keyboard

from commands import CommandInput

args = ["replace line", "replace lime"]


def action(cmd: CommandInput):
    return cmd.text
