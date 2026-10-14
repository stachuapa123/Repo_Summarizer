"""
Ollama provider.

Uses the official `ollama` Python library (native API rather than Ollama's
OpenAI-compatible endpoint) because only the native API lets us set the
context window per request. Ollama's default context is small, and a coding
agent's tool definitions plus file contents overflow it quickly.

.env:
    LLM_PROVIDER=ollama
    OLLAMA_HOST=http://localhost:11434
    OLLAMA_MODEL=qwen3:8b
    OLLAMA_NUM_CTX=16384
    OLLAMA_THINK=            # true / false / empty = model default
"""

from __future__ import annotations

import httpx
import ollama
from ollama import AsyncClient

from .base import Provider, ProviderError, ThinkFilter, ToolCall, ToolResult, Turn, Usage


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, settings, system_prompt: str):
        super().__init__(settings, system_prompt)
        self.client = AsyncClient(host=settings.ollama_host)
        self.messages: list[dict] = []
        self._n = 0
        self.reset()

    @property
    def model(self) -> str:
        return self.settings.ollama_model

    @property
    def history_length(self) -> int:
        return len(self.messages)

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]

    async def check(self) -> None:
        try:
            listing = await self.client.list()
        except (ConnectionError, httpx.HTTPError):
            raise ProviderError(
                f"Can't reach Ollama at {self.settings.ollama_host}.\n"
                "Start it with: sudo systemctl start ollama   (or run: ollama serve)"
            ) from None
        names = {m.model for m in listing.models}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names and self.model not in names:
            raise ProviderError(
                f"Model '{self.model}' isn't downloaded in Ollama.\n"
                f"Download it with: ollama pull {self.model}\n"
                f"Installed: {', '.join(sorted(names)) or '(none)'}"
            )

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    async def generate(self, on_text, on_thinking) -> Turn:
        kwargs = dict(
            model=self.model,
            messages=self.messages,
            stream=True,
            options={"num_ctx": self.settings.ollama_num_ctx, "temperature": self.settings.temperature},
        )
        if self.tools:
            kwargs["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in self.tools
            ]
        if self.settings.ollama_think is not None:
            kwargs["think"] = self.settings.ollama_think

        think = ThinkFilter()
        text: list[str] = []
        calls: list[ToolCall] = []
        usage = Usage()

        def emit(pieces):
            for kind, piece in pieces:
                if kind == "thinking":
                    on_thinking(piece)
                else:
                    text.append(piece)
                    on_text(piece)

        try:
            async for part in await self.client.chat(**kwargs):
                if part.done:
                    # Ollama's last chunk has exact counts and timings
                    # (durations are in nanoseconds).
                    usage = Usage(output_tokens=part.eval_count, prompt_tokens=part.prompt_eval_count)
                    if part.eval_count and part.eval_duration:
                        usage.server_tokens_per_second = part.eval_count / (part.eval_duration / 1e9)
                msg = part.message
                if getattr(msg, "thinking", None):
                    on_thinking(msg.thinking)
                if msg.content:
                    emit(think.feed(msg.content))
                for tc in msg.tool_calls or []:
                    self._n += 1
                    args = tc.function.arguments
                    calls.append(ToolCall(
                        id=f"call_{self._n}",
                        name=tc.function.name,
                        arguments=dict(args) if isinstance(args, dict) or hasattr(args, "items") else None,
                        raw_arguments=str(args),
                    ))
        except (ConnectionError, httpx.HTTPError):
            raise ProviderError(f"Lost connection to Ollama at {self.settings.ollama_host}.") from None
        except ollama.ResponseError as e:
            hint = f"\nDownload it with: ollama pull {self.model}" if e.status_code == 404 else ""
            raise ProviderError(f"Ollama returned an error ({e.status_code}): {e.error}{hint}") from None
        emit(think.flush())

        turn = Turn(text="".join(text).strip(), tool_calls=calls, usage=usage)
        msg: dict = {"role": "assistant", "content": turn.text}
        if calls:
            msg["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments or {}}} for c in calls]
        self.messages.append(msg)
        return turn

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self.messages.append({"role": "tool", "content": r.content, "tool_name": r.call.name})