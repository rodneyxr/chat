import logging
import sys
import threading

import click
import keyboard

from stt.stt import VoiceDictation


@click.command(help="Voice dictation (speech-to-text) completely on device.")
@click.option(
    "--stt",
    default="tiny.en",
    help="Whisper model name (tiny.en, base.en, small.en, medium.en, large, turbo)",
)
@click.option("--hotkey", default="F24", help="Hotkey to hold while speaking")
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
def main(stt: str, hotkey: str, debug: bool, post_processing: bool, type_mode: bool):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if hotkey.lower() == "auto":
        print("Press the key or combination of keys you want to use as the hotkey:")
        pressed_keys = set()
        while True:
            event = keyboard.read_event(suppress=False)
            if event.event_type == keyboard.KEY_DOWN:
                if event.name not in pressed_keys:
                    pressed_keys.add(event.name)
                    print(f"Key pressed: {event.name}")
            elif event.event_type == keyboard.KEY_UP:
                if event.name in pressed_keys:
                    pressed_keys.remove(event.name)
                    print(f"Key released: {event.name}")
                if event.name == "esc":
                    print("Hotkey selection confirmed.")
                    break
        hotkey = "+".join(pressed_keys)
        print(f"Hotkey '{hotkey}' confirmed.")
        # TODO: Test this, save the response to a config
        return

    vd = VoiceDictation(
        model_name=stt,
        hotkey=hotkey,
        post_processing=post_processing,
        hot_reload=True,
        type_mode=type_mode,
    )

    threading.Thread(target=vd.transcribe_audio, daemon=True).start()
    keyboard.on_press_key(hotkey, lambda _: vd.on_press())
    keyboard.on_release_key(hotkey, lambda _: vd.on_release())

    try:
        keyboard.wait()
    except KeyboardInterrupt:
        logging.info("Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
