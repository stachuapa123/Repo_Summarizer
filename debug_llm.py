#!/usr/bin/env python3
"""
Step 1: try the LLM providers on their own (no MCP yet).

Put this file next to the `providers/` folder, plus a `.env` file.

    python try_provider.py                  # chat with the provider set in .env
    python try_provider.py --provider ollama
    python try_provider.py --check-all      # quick test: can I reach all three?
    python try_provider.py --tools          # also give the model a toy tool to call

After each reply it prints the token count and speed, e.g.
    [212 tokens · 16.8 tok/s · first token 0.41s · total 13.0s · prompt 35 tokens]
A "~" means the count is estimated because the server didn't report it.

In the chat, /reset clears the conversation and /exit quits.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

from providers import PROVIDERS, ProviderError, ToolResult, ToolSpec, create_provider

DIM, RESET, GREEN, RED, YELLOW = "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m"

load_dotenv(Path(__file__).resolve().parent / ".env")


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


# ---------------------------------------------------------------------------
# Settings: the providers read these attributes. In a later step this can
# move into its own config.py.
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    provider: str = env("LLM_PROVIDER", "lemonade").lower()
    temperature: float = float(env("TEMPERATURE", "0.6"))
    show_thinking: bool = env("SHOW_THINKING", "true").lower() in ("1", "true", "yes")
    # Lemonade
    lemonade_base_url: str = env("LEMONADE_BASE_URL", "http://localhost:13305/api/v1")
    lemonade_model: str = env("LEMONADE_MODEL", "Qwen3-8B-GGUF")
    # Ollama
    ollama_host: str = env("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = env("OLLAMA_MODEL", "qwen3:8b")
    ollama_num_ctx: int = int(env("OLLAMA_NUM_CTX", "16384"))
    ollama_think: bool | None = None if not env("OLLAMA_THINK") else env("OLLAMA_THINK").lower() in ("1", "true", "yes")
    # Gemini
    gemini_api_key: str = env("GEMINI_API_KEY") or env("GOOGLE_API_KEY")
    gemini_model: str = env("GEMINI_MODEL", "gemini-3.8-flash")
    gemini_temperature: float | None = float(env("GEMINI_TEMPERATURE")) if env("GEMINI_TEMPERATURE") else None
    gemini_base_url: str = env("GEMINI_BASE_URL")


SYSTEM_PROMPT = "You are a helpful assistant. Keep answers short."


# ---------------------------------------------------------------------------
# A toy tool, to see how tool calling works before adding MCP.
# The model gets the name, description and a JSON Schema of the arguments;
# when it "calls" the tool, *we* run the function and send back the result.
# ---------------------------------------------------------------------------
TIME_TOOL = ToolSpec(
    name="get_current_time",
    description="Get the current date and time in a time zone.",
    parameters={
        "type": "object",
        "properties": {
            "timezone": {"type": "string", "description": "IANA time zone, e.g. Europe/Warsaw or Asia/Tokyo"},
        },
        "required": ["timezone"],
    },
)


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    """Run a tool the model asked for. Returns (result text, is_error)."""
    if name != "get_current_time":
        return f"Unknown tool: {name}", True
    try:
        now = datetime.now(ZoneInfo(args.get("timezone", "UTC")))
    except (ZoneInfoNotFoundError, ValueError):
        return f"Unknown time zone: {args.get('timezone')}", True
    return now.strftime("%Y-%m-%d %H:%M:%S %Z"), False


# ---------------------------------------------------------------------------
# Printing streamed output
# ---------------------------------------------------------------------------
class Printer:
    def __init__(self, show_thinking: bool):
        self.show_thinking = show_thinking
        self.mode = None

    def on_thinking(self, text: str) -> None:
        if not self.show_thinking:
            return
        if self.mode != "thinking":
            text = text.lstrip()
            if not text:
                return
            print(f"{DIM}[thinking] ", end="")
            self.mode = "thinking"
        print(f"{DIM}{text}{RESET}", end="", flush=True)

    def on_text(self, text: str) -> None:
        if self.mode != "text":
            text = text.lstrip()
            if not text:
                return
            if self.mode == "thinking":
                print(RESET)
            self.mode = "text"
        print(text, end="", flush=True)

    def end(self) -> None:
        if self.mode:
            print(RESET)
        self.mode = None


def speed_line(usage, start: float, first: float | None, end: float) -> str:
    """Format token counts and speed for one reply.

    Speed counts only generation time (from the first token to the last),
    so the wait for the model to read the prompt doesn't drag it down.
    If the server measured the speed itself (Ollama, llama.cpp), we use that.
    """
    parts = []
    n = usage.output_tokens
    approx = "" if usage.exact else "~"
    if n:
        parts.append(f"{approx}{n} token{'s' if n != 1 else ''}")
        if usage.server_tokens_per_second:
            parts.append(f"{usage.server_tokens_per_second:.1f} tok/s")
        elif first is not None and end - first > 0.05 and n > 1:
            parts.append(f"{approx}{(n - 1) / (end - first):.1f} tok/s")
    if first is not None:
        parts.append(f"first token {first - start:.2f}s")
    total = end - start
    parts.append(f"total {total:.2f}s" if total < 10 else f"total {total:.1f}s")
    if usage.prompt_tokens:
        parts.append(f"prompt {usage.prompt_tokens} tokens")
    return "[" + " · ".join(parts) + "]"


async def ask(provider, printer: Printer, text: str, max_steps: int = 5) -> None:
    """Send one message. If the model calls tools, run them and let it continue."""
    provider.add_user_message(text)
    for _ in range(max_steps):
        start = time.perf_counter()
        first = None

        # Wrap the printer callbacks to note when the first token arrives.
        def on_text(t: str) -> None:
            nonlocal first
            first = first or time.perf_counter()
            printer.on_text(t)

        def on_thinking(t: str) -> None:
            nonlocal first
            first = first or time.perf_counter()
            printer.on_thinking(t)

        turn = await provider.generate(on_text, on_thinking)
        end = time.perf_counter()
        printer.end()
        print(f"{DIM}{speed_line(turn.usage, start, first, end)}{RESET}")
        if not turn.tool_calls:
            return
        results = []
        for call in turn.tool_calls:
            output, is_error = run_tool(call.name, call.arguments or {})
            mark = f"{RED}✗" if is_error else f"{GREEN}✓"
            print(f"  {mark} {call.name}({call.arguments}) -> {output}{RESET}")
            results.append(ToolResult(call, output, is_error))
        provider.add_tool_results(results)
    print(f"{YELLOW}(stopped after {max_steps} tool rounds){RESET}")


async def check_all() -> None:
    """Try each provider: connect, then ask for a one-word reply."""
    for name in PROVIDERS:
        settings = Settings(provider=name, show_thinking=False)
        print(f"{name:9} ", end="", flush=True)
        try:
            provider = create_provider(settings, SYSTEM_PROMPT)
            await provider.check()
            provider.add_user_message("Reply with just the word: ready /no_think")
            turn = await provider.generate(lambda t: None, lambda t: None)
            print(f"{GREEN}OK{RESET}  {provider.model}: {turn.text[:60]!r}")
        except ProviderError as e:
            print(f"{RED}FAILED{RESET}\n          " + str(e).replace("\n", "\n          "))
        except Exception as e:  # anything unexpected: show it rather than crash
            print(f"{RED}ERROR{RESET} {type(e).__name__}: {e}")


async def main() -> int:
    ap = argparse.ArgumentParser(description="Try the LLM providers")
    ap.add_argument("--provider", choices=PROVIDERS, help="override LLM_PROVIDER from .env")
    ap.add_argument("--check-all", action="store_true", help="test all three providers and exit")
    ap.add_argument("--tools", action="store_true", help="give the model a toy get_current_time tool")
    ap.add_argument("--prompt", "-p", help="send one message and exit")
    args = ap.parse_args()

    if args.check_all:
        await check_all()
        return 0

    settings = Settings()
    if args.provider:
        settings.provider = args.provider
    try:
        provider = create_provider(settings, SYSTEM_PROMPT)
        await provider.check()
    except ProviderError as e:
        print(f"{RED}{e}{RESET}")
        return 1

    if args.tools:
        provider.set_tools([TIME_TOOL])
    printer = Printer(settings.show_thinking)
    print(f"Connected: {settings.provider} · {provider.model}"
          f"{' · tool: get_current_time' if args.tools else ''}")

    if args.prompt:
        await ask(provider, printer, args.prompt)
        return 0

    tip = " Try: what time is it in Tokyo?" if args.tools else ""
    print(f"{DIM}/reset clears the chat, /exit quits.{tip}{RESET}\n")
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line in ("/exit", "/quit"):
            return 0
        if line == "/reset":
            provider.reset()
            print(f"{DIM}cleared{RESET}")
            continue
        if line:
            try:
                await ask(provider, printer, line)
            except ProviderError as e:
                printer.end()
                print(f"{RED}{e}{RESET}")
            print()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))