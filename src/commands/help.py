import logging


def action(args: list[str], text: str):
    logging.info(
        "\nAvailable commands:"
        "\n  - hey jarvis"
        "\n  - paste"
        "\n  - undo"
        "\n  - redo"
        "\n  - delete/backspace"
        "\n  - replace line"
    )
