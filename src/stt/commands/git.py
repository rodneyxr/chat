import keyboard

from stt.commands import CommandInput

# TODO: find a better way to handle mispronunciations/misunderstandings
args = [
    "git status",
    "git pull",
    "git add",
    "get status",
    "get pull",
    "get add",
    "get ad",
    "get at",
]


def action(cmd: CommandInput):
    if cmd.args[1] == "status":
        keyboard.write("git status", delay=0.01)
        keyboard.press_and_release("enter")
    elif cmd.args[1] == "pull":
        keyboard.write("git pull", delay=0.01)
        keyboard.press_and_release("enter")
    elif cmd.args[1] in ["add", "ad", "at"]:
        keyboard.write("git add -u", delay=0.01)
        keyboard.press_and_release("enter")
        keyboard.write("git status", delay=0.01)
        keyboard.press_and_release("enter")
