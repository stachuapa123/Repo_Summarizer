#!/usr/bin/env python3
"""
Step 2: chat with the model, which can look around a project folder
using the Workspace tools (list_files, read_file).

    python agent.py
    python agent.py --provider gemini
    python agent.py --workspace ~/Desktop/other_project
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from debug_llm import DIM, GREEN, RED, RESET, YELLOW, Printer, Settings, speed_line  # reuse, don't copy
from file_tools import Workspace
from providers import PROVIDERS, ProviderError, ToolResult, ToolSpec, create_provider

DEFAULT_WORKSPACE = "~/Desktop/MCP_Project"   # NOT the folder with your .env!

SYSTEM_PROMPT = """You are a helpful coding assistant.
You can explore the user's project with tools: list_files shows which files exist,
read_file shows a file's contents. Use them instead of guessing; never invent file names.
All paths are relative to the project root. Keep answers short and concrete."""


# ---------------------------------------------------------------------------
# 1. Tool descriptions: what the model sees
# ---------------------------------------------------------------------------
LIST_FILES_TOOL = ToolSpec(
    name="list_files",
    description=(
        "List files in the project, skipping .git, node_modules and virtualenvs. "
        "Returns one path per line, relative to the project root."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Folder to list, relative to the project root. Use '.' for the whole project.",
            },
            "pattern": {
                "type": "string",
                "description": "Only list files whose name matches this pattern, e.g. '*.py'. Default: all files.",
            },
        },
        "required": [],
    },
)

READ_FILE_TOOL = ToolSpec(
    name="read_file",
    description="Read the full text of a file in the project. Use list_files first to find the right path.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the project root, e.g. 'calculator.py' or 'src/main.py'.",
            },
        },
        "required": ["path"],
    },
)

TOOL_SPECS = [LIST_FILES_TOOL, READ_FILE_TOOL]


# ---------------------------------------------------------------------------
# 2. Running the tools: what actually happens on your computer
# ---------------------------------------------------------------------------
def run_tool(tool_functions: dict, name: str, args: dict) -> tuple[str, bool]:
    """Run the tool the model asked for. Returns (result text, is_error)."""
    func = tool_functions.get(name)
    if func is None:
        return f"Unknown tool: {name}", True
    try:
        return func(**args), False
    except (ValueError, TypeError) as e:   # TypeError = the model used wrong argument names
        return str(e), True


# ---------------------------------------------------------------------------
# 3. The conversation loop: model answers, or asks for tools, until done
# ---------------------------------------------------------------------------
async def ask(provider, printer: Printer, tool_functions: dict, text: str, max_steps: int = 10) -> None:
    provider.add_user_message(text)
    for _ in range(max_steps):
        start, first = time.perf_counter(), None

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

        if not turn.tool_calls:          # no tools requested: the answer is final
            return

        results = []
        for call in turn.tool_calls:
            output, is_error = run_tool(tool_functions, call.name, call.arguments or {})
            mark = f"{RED}✗" if is_error else f"{GREEN}✓"
            first_line = output.splitlines()[0] if output else ""
            print(f"  {mark} {call.name}({call.arguments}){RESET} {DIM}{first_line[:80]}{RESET}")
            results.append(ToolResult(call, output, is_error))
        provider.add_tool_results(results)   # send results back; the loop asks the model again
    print(f"{YELLOW}(stopped after {max_steps} steps){RESET}")


# ---------------------------------------------------------------------------
# 4. Start-up and the chat
# ---------------------------------------------------------------------------
async def main() -> int:
    ap = argparse.ArgumentParser(description="Chat with an LLM that can read your project")
    ap.add_argument("--provider", choices=PROVIDERS, help="override LLM_PROVIDER from .env")
    ap.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="project folder the model may read")
    args = ap.parse_args()

    try:
        ws = Workspace(args.workspace)
    except ValueError as e:
        print(f"{RED}{e}{RESET}")
        return 1
    tool_functions = {"list_files": ws.list_files, "read_file": ws.read_file}

    settings = Settings()
    if args.provider:
        settings.provider = args.provider
    try:
        provider = create_provider(settings, SYSTEM_PROMPT)
        await provider.check()
    except ProviderError as e:
        print(f"{RED}{e}{RESET}")
        return 1
    provider.set_tools(TOOL_SPECS)

    printer = Printer(settings.show_thinking)
    print(f"Connected: {settings.provider} · {provider.model}")
    print(f"{DIM}Project: {ws.root}\n/reset clears the chat, /exit quits. "
          f"Try: what files are in my project?{RESET}\n")

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
                await ask(provider, printer, tool_functions, line)
            except ProviderError as e:
                printer.end()
                print(f"{RED}{e}{RESET}")
            print()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))