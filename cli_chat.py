from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx


def _print_help() -> None:
    print(
        "Commands:\n"
        "  /exit           quit\n"
        "  /reset          clear local history\n"
        "  /raw on|off     print raw JSON response\n"
        "  /params         show current sampling params\n"
        "  /set key=value  set param (temperature, top_p, top_k, max_tokens)\n"
    )


def _build_prompt(history: list[dict[str, str]], user_text: str, mode: str) -> str:
    if mode == "raw":
        return user_text

    # Minimal chat-style prompt wrapper.
    # Works reasonably for instruction-tuned/chat models; adjust if your model expects a different template.
    lines: list[str] = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            lines.append(f"User: {content}")
        else:
            lines.append(f"Assistant: {content}")

    lines.append(f"User: {user_text}")
    lines.append("Assistant:")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple terminal chat client for /generate")
    parser.add_argument("--url", default="http://127.0.0.1:8000/generate", help="Server /generate URL")
    parser.add_argument("--mode", choices=["chat", "raw"], default="chat", help="Prompt mode")
    parser.add_argument("--timeout", type=float, default=600.0, help="Request timeout seconds")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    params: dict[str, Any] = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    history: list[dict[str, str]] = []
    raw = False

    print("Terminal chat -> vLLM server")
    print(f"POST {args.url}")
    _print_help()

    with httpx.Client(timeout=args.timeout) as client:
        while True:
            try:
                user_text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            if not user_text:
                continue

            if user_text == "/exit":
                return 0
            if user_text == "/reset":
                history.clear()
                print("(history cleared)")
                continue
            if user_text.startswith("/raw "):
                value = user_text.split(" ", 1)[1].strip().lower()
                if value in {"on", "1", "true", "yes"}:
                    raw = True
                    print("(raw on)")
                elif value in {"off", "0", "false", "no"}:
                    raw = False
                    print("(raw off)")
                else:
                    print("Usage: /raw on|off")
                continue
            if user_text == "/params":
                print(json.dumps(params, ensure_ascii=False, indent=2))
                continue
            if user_text.startswith("/set "):
                kv = user_text.split(" ", 1)[1].strip()
                if "=" not in kv:
                    print("Usage: /set key=value")
                    continue
                key, value = kv.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key not in params:
                    print(f"Unknown key: {key}. Allowed: {', '.join(params.keys())}")
                    continue
                try:
                    if key in {"temperature", "top_p"}:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                    print(f"({key}={params[key]})")
                except ValueError:
                    print("Invalid value")
                continue
            if user_text == "/help":
                _print_help()
                continue

            prompt = _build_prompt(history, user_text, args.mode)

            payload = {"prompt": prompt, **params}
            try:
                resp = client.post(args.url, json=payload)
            except httpx.HTTPError as e:
                print(f"error> {e}")
                continue

            if resp.status_code != 200:
                print(f"error> HTTP {resp.status_code}: {resp.text}")
                continue

            data = resp.json()
            if raw:
                print("server>", json.dumps(data, ensure_ascii=False, indent=2))

            answer = data.get("answer")
            if not isinstance(answer, str):
                print("error> Unexpected response: missing 'answer'")
                continue

            print("bot>", answer)

            if args.mode == "chat":
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    raise SystemExit(main())
