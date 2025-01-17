import keyboard


def prompt_for_hotkey():
    print("Enter the hotkey you want to use followed by 'escape':")
    pressed_keys = set()
    hotkeys = set()
    while True:
        event = keyboard.read_event(suppress=False)
        key_name = event.name.lower()
        if event.event_type == keyboard.KEY_DOWN:
            if key_name == "esc":
                continue
            if key_name not in pressed_keys:
                pressed_keys.add(key_name)
                hotkeys = set()
        elif event.event_type == keyboard.KEY_UP:
            if key_name in pressed_keys:
                pressed_keys.remove(key_name)
                hotkeys.add(key_name)
                sorted_hotkeys = sorted(
                    hotkeys,
                    key=lambda k: (k != "ctrl", k != "alt", k != "shift", k),
                )
                print(f"Hotkey: {'+'.join(sorted_hotkeys)}. Press escape to confirm.")
            if key_name == "esc":
                if not hotkeys:
                    print("No hotkey selected.")
                    print("Enter the hotkey you want to use followed by 'escape':")
                    continue
                break
    sorted_hotkeys = sorted(
        hotkeys, key=lambda k: (k != "ctrl", k != "alt", k != "shift", k)
    )
    hotkey = "+".join(sorted_hotkeys)
    return hotkey
