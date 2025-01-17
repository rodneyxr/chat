import pyperclip

from sttpy.commands import CommandInput


def action(cmd: CommandInput):
    return pyperclip.paste() if isinstance(pyperclip.paste(), str) else ""
