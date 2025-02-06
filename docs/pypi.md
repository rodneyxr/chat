# Quickstart

This project is a simple voice dictation (speech-to-text) tool that runs completely on device. It uses the openai-whisper models for speech recognition and optionally uses the local LLMs for post-processing of transcribed text (currently supporting all ollama models).

Simply install, run and hold the hotkey to speak. The transcribed text will be pasted into the active window. Say 'help' to view voice commands.

## Installation

```sh
# CPU only
pip install sttpy

# If you have a GPU and want to use the CUDA version
pip install sttpy[cuda] --extra-index-url https://download.pytorch.org/whl/cu124

# AMD ROCm (linux only)
pip install sttpy[rocm] --extra-index-url https://download.pytorch.org/whl/rocm6.2
```

## Usage

```sh
Usage: stt [OPTIONS]

  Voice dictation (speech-to-text) completely on device.

Options:
  --stt TEXT           Whisper model name (tiny.en, base.en, turbo, ...)
  --hotkey TEXT        Hotkey to hold while speaking
  --debug              Enable debug mode
  --post-processing    Enable LLM post-processing of transcribed text
  --type-mode          Use keystrokes instead of pasting text
  --paste-delay FLOAT  Delay between copying to clipboard and pasting
  --help               Show this message and exit.
```

## Examples

Run with debug mode enabled, using the openai-whisper turbo model, and the hold-to-speak hotkey `f12`:
```sh
stt --debug --stt turbo --hotkey f12
```

Prompt for a hotkey to hold while speaking:

```sh
stt --hotkey prompt
Enter the hotkey you want to use followed by 'escape':
Hotkey: space. Press escape to confirm.
Hotkey: ctrl+space. Press escape to confirm.
Hotkey confirmed: ctrl+space
Hotkey: ctrl+space
2025-01-16 22:29:00 - INFO - Loading whisper model 'tiny.en' on cuda...
2025-01-16 22:29:00 - INFO - Press and hold 'ctrl+space' to speak
```

## Commands

There are a few commands built-in to the voice dictation interface:

Just hold the hotkey and say 'help' to view commands.

## Post-processing

Note: Post-processing is not enabled by default since there is latency and its still under development. To enable post-processing, use the `--post-processing` flag. You will need to have an local ollama server running and the model `llama3.2:3b-instruct-q5_K_M` available. 