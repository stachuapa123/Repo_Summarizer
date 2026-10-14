"""
Chat with a local LLM (Qwen3-8B by default) through AMD Lemonade Server.

Lemonade Server exposes an OpenAI-compatible API, so we just use the
official `openai` Python client pointed at localhost.

On a Ryzen AI 9 HX 370, the "-GGUF" models run through llama.cpp and are
accelerated on the Radeon 890M iGPU (Vulkan backend).

Setup:
    1. Install Lemonade Server and start it.
       Fedora: sudo dnf install ./lemonade-server-*-fc44.x86_64.rpm
               sudo systemctl enable --now lemond
       Windows: launch Lemonade from the Start Menu.
    2. pip install openai requests
    3. (optional) lemonade pull Qwen3-8B-GGUF   # the script can also pull it for you
    4. python lemonade_chat.py

Usage:
    python lemonade_chat.py                       # interactive chat with Qwen3-8B
    python lemonade_chat.py --prompt "Hi there"   # single prompt, then exit
    python lemonade_chat.py --no-think            # turn off Qwen3's reasoning mode
    python lemonade_chat.py --model Qwen3-4B-GGUF # use a different model
    python lemonade_chat.py --hide-thinking       # reason, but don't print the thinking
"""

import argparse
import sys
import time

import requests
from openai import OpenAI

DEFAULT_MODEL = "Qwen3-8B-GGUF"

# Newer Lemonade releases listen on port 13305; older ones used 8000.
CANDIDATE_BASE_URLS = [
    "http://localhost:13305/api/v1",
    "http://localhost:13305/v1",
    "http://localhost:8000/api/v1",
    "http://localhost:8000/v1",
]

DIM = "\033[2m"
RESET = "\033[0m"


def find_server(base_url: str | None) -> tuple[str, list[str]]:
    """Find a running Lemonade Server and return (base_url, installed model ids)."""
    candidates = [base_url.rstrip("/")] if base_url else CANDIDATE_BASE_URLS
    for url in candidates:
        try:
            r = requests.get(f"{url}/models", timeout=3)
            if r.ok and "data" in r.json():
                return url, [m["id"] for m in r.json()["data"]]
        except (requests.RequestException, ValueError):
            continue
    sys.exit(
        "Can't reach Lemonade Server. Tried:\n  "
        + "\n  ".join(candidates)
        + "\n\nCheck that it's running:"
        "\n  Linux:   sudo systemctl status lemond   (start with: sudo systemctl enable --now lemond)"
        "\n  Windows: look for the lemon icon in the tray, or launch it from the Start Menu"
        "\nThen open http://localhost:13305 in a browser to confirm the address."
        "\nIf it uses another port, pass it with --base-url http://localhost:PORT/api/v1"
    )


def ensure_model(base_url: str, model: str, installed: list[str]) -> None:
    """Download the model through the server if it isn't installed yet."""
    if model in installed:
        return
    print(f"'{model}' is not installed. Asking Lemonade to download it (several GB, be patient)...")
    try:
        r = requests.post(f"{base_url}/pull", json={"model_name": model}, timeout=None)
        r.raise_for_status()
    except requests.RequestException as e:
        sys.exit(
            f"Automatic download failed ({e}).\n"
            f"Pull it manually with:  lemonade pull {model}"
        )
    print("Download finished.\n")


def stream_reply(client: OpenAI, model: str, messages: list[dict], show_thinking: bool) -> str:
    """Stream one assistant reply to the terminal and return the visible answer text."""
    start = time.time()
    first_token_at = None
    n_chunks = 0
    answer = []
    in_think = False  # tracks <think>...</think> blocks embedded in the content

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0.6,  # Qwen's recommended setting for thinking mode
        top_p=0.95,
        max_tokens=4096,
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if first_token_at is None:
            first_token_at = time.time()
        n_chunks += 1

        # Some server versions return reasoning in a separate field.
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning and show_thinking:
            print(f"{DIM}{reasoning}{RESET}", end="", flush=True)

        text = delta.content or ""
        # Others inline it as <think>...</think> inside the normal content.
        while text:
            if in_think:
                end = text.find("</think>")
                part, text = (text, "") if end == -1 else (text[:end], text[end + 8:])
                if show_thinking:
                    print(f"{DIM}{part}{RESET}", end="", flush=True)
                if end != -1:
                    in_think = False
                    if show_thinking:
                        print()
            else:
                start_tag = text.find("<think>")
                part, text = (text, "") if start_tag == -1 else (text[:start_tag], text[start_tag + 7:])
                print(part, end="", flush=True)
                answer.append(part)
                if start_tag != -1:
                    in_think = True

    total = time.time() - start
    ttft = (first_token_at - start) if first_token_at else total
    gen_time = max(total - ttft, 1e-6)
    print(f"\n{DIM}[first token {ttft:.2f}s | ~{n_chunks / gen_time:.1f} tok/s]{RESET}")
    return "".join(answer).strip()


def main() -> None:
    p = argparse.ArgumentParser(description="Chat with an LLM via AMD Lemonade Server")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default=None, help="e.g. http://localhost:13305/api/v1 (auto-detected if omitted)")
    p.add_argument("--prompt", help="Send a single prompt and exit")
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--no-think", action="store_true", help="Disable Qwen3 reasoning (faster)")
    p.add_argument("--hide-thinking", action="store_true", help="Don't print the reasoning trace")
    args = p.parse_args()

    base_url, installed = find_server(args.base_url)
    print(f"{DIM}Connected to Lemonade Server at {base_url}{RESET}")
    ensure_model(base_url, args.model, installed)

    # API key is required by the client but ignored by Lemonade.
    client = OpenAI(base_url=base_url, api_key="lemonade")
    messages = [{"role": "system", "content": args.system}]
    suffix = " /no_think" if args.no_think else ""  # Qwen3's soft switch
    show_thinking = not args.hide_thinking

    def ask(user_text: str) -> None:
        messages.append({"role": "user", "content": user_text + suffix})
        reply = stream_reply(client, args.model, messages, show_thinking)
        messages.append({"role": "assistant", "content": reply})

    if args.prompt:
        ask(args.prompt)
        return

    print(f"Chatting with {args.model}. The first message loads the model, so it's slower.")
    print("Commands: /reset clears history, /exit quits.\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text in ("/exit", "/quit"):
            break
        if user_text == "/reset":
            messages = messages[:1]
            print("History cleared.\n")
            continue
        print("Qwen: ", end="", flush=True)
        ask(user_text)
        print()


if __name__ == "__main__":
    main()