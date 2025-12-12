## Requirements

- Python 3.12+
- vLLM-compatible environment (CUDA / GPU setup as needed by your model)

## Install

If you use `uv`:

```bash
uv sync
```

Or with pip:

```bash
python -m pip install -e .
```

## Run server

Start FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Generate (example):

```bash
curl -X POST http://127.0.0.1:8000/generate \
	-H 'Content-Type: application/json' \
	-d '{"prompt":"Hello!","max_tokens":128,"temperature":0.7,"top_p":0.9,"top_k":50}'
```

## Terminal chat client

Run terminal chatbot (redirects your messages to the server):

```bash
python cli_chat.py --url http://127.0.0.1:8000/generate
```

Useful options:

```bash
python cli_chat.py --url http://127.0.0.1:8000/generate --timeout 600 --mode chat
```

Interactive commands:

- `/help` show commands
- `/exit` quit
- `/reset` clear local history (client-side)
- `/params` show current sampling params
- `/set temperature=0.7` (also: `top_p`, `top_k`, `max_tokens`)
- `/raw on|off` print raw JSON response from server

