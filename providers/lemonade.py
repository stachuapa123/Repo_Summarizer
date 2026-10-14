"""
Lemonade Server provider (AMD's local AI server).

Lemonade exposes an OpenAI-compatible API, so this uses the official
`openai` client pointed at your local server. GGUF models run through
llama.cpp, on the Radeon iGPU via Vulkan.

.env:
    LLM_PROVIDER=lemonade
    LEMONADE_BASE_URL=http://localhost:13305/api/v1
    LEMONADE_MODEL=Qwen3-8B-GGUF
"""

from __future__ import annotations

import json

import openai
from openai import AsyncOpenAI

from .base import Provider, ProviderError, ThinkFilter, ToolCall, ToolResult, Turn


class LemonadeProvider(Provider):
    name = "lemonade"

    def __init__(self, settings, system_prompt: str):
        super().__init__(settings, system_prompt)
        # The API key is required by the client library but ignored by Lemonade.
        self.client = AsyncOpenAI(base_url=settings.lemonade_base_url, api_key="lemonade", timeout=600)
        self.messages: list[dict] = []
        self.reset()

    @property
    def model(self) -> str:
        return self.settings.lemonade_model

    @property
    def history_length(self) -> int:
        return len(self.messages)

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]

    async def check(self) -> None:
        try:
            models = [m.id async for m in self.client.models.list()]
        except openai.APIConnectionError:
            raise ProviderError(
                f"Can't reach Lemonade Server at {self.settings.lemonade_base_url}.\n"
                "Start it with: sudo systemctl enable --now lemond   (check with: lemonade status)"
            ) from None
        if self.model not in models:
            raise ProviderError(
                f"Model '{self.model}' isn't downloaded in Lemonade.\n"
                f"Download it with: lemonade pull {self.model}\n"
                f"Installed: {', '.join(models) or '(none)'}"
            )

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def _tool_payload(self) -> list[dict]:
        return [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in self.tools
        ]

    async def generate(self, on_text, on_thinking) -> Turn:
        kwargs = dict(model=self.model, messages=self.messages, stream=True, temperature=self.settings.temperature)
        if self.tools:
            kwargs["tools"] = self._tool_payload()
        try:
            stream = await self.client.chat.completions.create(**kwargs)
        except openai.APIConnectionError:
            raise ProviderError(f"Lost connection to Lemonade at {self.settings.lemonade_base_url}.") from None
        except openai.APIStatusError as e:
            raise ProviderError(f"Lemonade returned an error ({e.status_code}): {e.message}") from None

        think = ThinkFilter()
        text: list[str] = []
        slots: dict[int, dict] = {}  # tool calls arrive in pieces, keyed by index

        def emit(pieces):
            for kind, piece in pieces:
                if kind == "thinking":
                    on_thinking(piece)
                else:
                    text.append(piece)
                    on_text(piece)

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                on_thinking(reasoning)
            if delta.content:
                emit(think.feed(delta.content))
            for tc in delta.tool_calls or []:
                idx = tc.index if tc.index is not None else (len(slots) if tc.id else max(slots, default=0))
                slot = slots.setdefault(idx, {"id": None, "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name and not slot["name"]:
                        slot["name"] = tc.function.name
                    if tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
        emit(think.flush())

        turn = Turn(text="".join(text).strip())
        for n, idx in enumerate(sorted(slots)):
            s = slots[idx]
            raw = s["arguments"].strip() or "{}"
            try:
                args = json.loads(raw)
                args = args if isinstance(args, dict) else None
            except json.JSONDecodeError:
                args = None
            turn.tool_calls.append(ToolCall(id=s["id"] or f"call_{len(self.messages)}_{n}", name=s["name"], arguments=args, raw_arguments=raw))

        # Thinking is left out of the history on purpose: Qwen3 is trained to
        # see only the final answers of earlier turns.
        msg: dict = {"role": "assistant", "content": turn.text or None}
        if turn.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments}}
                for c in turn.tool_calls
            ]
        self.messages.append(msg)
        return turn

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self.messages.append({"role": "tool", "tool_call_id": r.call.id, "content": r.content})
