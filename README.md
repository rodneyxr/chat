# Quickstart

This project requires [uv](https://docs.astral.sh/uv/).

```sh
pip install -U uv
```

Windows installation:

```powershell
git clone https://github.com/rodneyxr/chat.git
cd chat
uv run --python 3.11 pip install -U -e . --extra-index-url https://download.pytorch.org/whl/cu124
uv run stt
```

Install from git repository:

```sh
uv run --python 3.11 pip install -U git+https://github.com/rodneyxr/chat.git@dev --extra-index-url https://download.pytorch.org/whl/cu124
```

With SSH:

```sh
uv run --python 3.11 pip install -U git+ssh://github.com/rodneyxr/chat.git@dev --extra-index-url https://download.pytorch.org/whl/cu124
```

## Before commit

```sh
uv tool run ruff check --fix
uv rool run ruff format
```