import logging
import sys
import threading

import click
import keyboard

from sttpy.config import load_config, save_config
from sttpy.stt import VoiceDictation
from sttpy.utils.hotkey import prompt_for_hotkey


@click.command(help="Voice dictation (speech-to-text) completely on device.")
@click.option(
    "--stt",
    default="tiny.en",
    help="Whisper model name (tiny.en, base.en, turbo, ...)",
)
@click.option("--hotkey", default=None, help="Hotkey to hold while speaking")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option(
    "--post-processing",
    is_flag=True,
    default=False,
    help="Enable LLM post-processing of transcribed text",
)
@click.option(
    "--type-mode", is_flag=True, help="Use keystrokes instead of pasting text"
)
@click.option(
    "--paste-delay", default=0.1, help="Delay between copying to clipboard and pasting"
)
def main(
    stt: str,
    hotkey: str | None,
    debug: bool,
    post_processing: bool,
    type_mode: bool,
    paste_delay: float,
):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config()

    # Prompt for hotkey if 'prompt' is passed, otherwise use the hotkey flag if specified, otherwise use the config
    # --hotkey always takes precedence over the config value
    if hotkey:
        if hotkey.lower() == "prompt":
            hotkey = prompt_for_hotkey()
            print(f"Hotkey confirmed: {hotkey}")

            config["hotkey"] = hotkey
            save_config(config)
    else:
        hotkey = config["hotkey"]
    logging.debug(f"Hotkey: {hotkey}")

    vd = VoiceDictation(
        model_name=stt,
        hotkey=hotkey,
        post_processing=post_processing,
        hot_reload=True,
        type_mode=type_mode,
        paste_delay=paste_delay,
    )

    threading.Thread(target=vd.transcribe_audio, daemon=True).start()

    # Register the hotkeys
    keyboard.add_hotkey(hotkey, lambda: vd.on_press(), suppress=True)
    keyboard.add_hotkey(
        hotkey, lambda: vd.on_release(), trigger_on_release=True, suppress=True
    )

    try:
        keyboard.wait()
    except KeyboardInterrupt:
        logging.info("Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
