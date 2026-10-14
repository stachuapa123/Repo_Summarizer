"""
The interface every LLM provider implements, plus small shared helpers.

Each provider keeps the conversation in its own native format. That matters
for Gemini, whose replies carry "thought signatures" that must be sent back
unchanged, and it keeps every provider free to use its SDK's own types.

The agent only ever does this:

    provider.set_tools(specs)
    provider.add_user_message("fix the failing test")
    turn = await provider.generate(on_text, on_thinking)   # records the reply
    provider.add_tool_results([...])                      # if turn.tool_calls
    turn = await provider.generate(...)                    # and so on
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

TextCallback = Callable[[str], None]


@dataclass
class ToolSpec:
    """A tool the model may call. `parameters` is a JSON Schema object."""
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] | None  # None when the model sent arguments that aren't valid JSON
    raw_arguments: str = ""


@dataclass
class ToolResult:
    call: ToolCall
    content: str
    is_error: bool = False


@dataclass
class Usage:
    """Token counts for one reply, as far as the server reports them."""
    output_tokens: int | None = None      # answer + thinking tokens
    prompt_tokens: int | None = None      # everything sent in (history, tools, ...)
    server_tokens_per_second: float | None = None  # when the server measures speed itself
    exact: bool = True                    # False when output_tokens is only an estimate


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class Provider(ABC):
    name = "base"

    def __init__(self, settings, system_prompt: str):
        self.settings = settings
        self.system_prompt = system_prompt
        self.tools: list[ToolSpec] = []

    @property
    @abstractmethod
    def model(self) -> str: ...

    def set_tools(self, tools: list[ToolSpec]) -> None:
        self.tools = tools

    @abstractmethod
    def reset(self) -> None:
        """Forget the conversation (keeps the system prompt and tools)."""

    @abstractmethod
    def add_user_message(self, text: str) -> None: ...

    @abstractmethod
    async def generate(self, on_text: TextCallback, on_thinking: TextCallback) -> Turn:
        """Stream one model reply, append it to the history and return it."""

    @abstractmethod
    def add_tool_results(self, results: list[ToolResult]) -> None: ...

    async def check(self) -> None:
        """Optional connectivity check; raise ProviderError with a helpful message."""

    @property
    def history_length(self) -> int:
        """Number of stored messages (shown by /status)."""
        return 0


class ProviderError(RuntimeError):
    """A problem the user can fix (server not running, missing key, unknown model)."""


class ThinkFilter:
    """Splits streamed text into thinking and answer parts.

    Qwen3 and other reasoning models served through llama.cpp often put their
    reasoning inline as <think>...</think>. Tags can be split across chunks,
    so partial tags are held back until the next chunk arrives.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.inside = False
        self.buf = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        self.buf += text
        out: list[tuple[str, str]] = []
        while self.buf:
            tag = self.CLOSE if self.inside else self.OPEN
            i = self.buf.find(tag)
            if i >= 0:
                if i:
                    out.append(("thinking" if self.inside else "text", self.buf[:i]))
                self.buf = self.buf[i + len(tag):]
                self.inside = not self.inside
                continue
            # Keep back a possible partial tag at the end of the buffer.
            keep = next((k for k in range(len(tag) - 1, 0, -1) if self.buf.endswith(tag[:k])), 0)
            emit = self.buf[: len(self.buf) - keep]
            if emit:
                out.append(("thinking" if self.inside else "text", emit))
            self.buf = self.buf[len(self.buf) - keep:]
            break
        return out

    def flush(self) -> list[tuple[str, str]]:
        rest, self.buf = self.buf, ""
        return [("thinking" if self.inside else "text", rest)] if rest else []