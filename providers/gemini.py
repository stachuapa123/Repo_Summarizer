"""
Google Gemini provider (cloud, needs an API key).

Uses Google's official `google-genai` SDK. The conversation is kept as the
SDK's own Content objects and model replies are stored exactly as received,
which preserves the "thought signatures" Gemini 3 models attach to function
calls. Gemini rejects follow-up requests if those are missing or altered.

.env:
    LLM_PROVIDER=gemini
    GEMINI_API_KEY=your-key        # https://aistudio.google.com/apikey
    GEMINI_MODEL=gemini-3.8-flash
    GEMINI_TEMPERATURE=            # leave empty: Google recommends the default (1.0) for Gemini 3
"""

from __future__ import annotations

import json

from google import genai
from google.genai import errors, types

from .base import Provider, ProviderError, ToolCall, ToolResult, Turn


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, settings, system_prompt: str):
        super().__init__(settings, system_prompt)
        if not settings.gemini_api_key:
            raise ProviderError(
                "GEMINI_API_KEY is empty. Create a key at https://aistudio.google.com/apikey "
                "and put it in your .env file."
            )
        http_options = types.HttpOptions(base_url=settings.gemini_base_url) if settings.gemini_base_url else None
        self.client = genai.Client(api_key=settings.gemini_api_key, http_options=http_options)
        self.contents: list[types.Content] = []
        self._generated_ids: set[str] = set()
        self._n = 0

    @property
    def model(self) -> str:
        return self.settings.gemini_model

    @property
    def history_length(self) -> int:
        return len(self.contents)

    def reset(self) -> None:
        self.contents = []

    async def check(self) -> None:
        try:
            await self.client.aio.models.get(model=self.model)
        except errors.APIError as e:
            raise ProviderError(self._explain(e)) from None

    def _explain(self, e: errors.APIError) -> str:
        if e.code in (401, 403) or "API key" in str(e.message):
            return f"Gemini rejected the API key ({e.code}): {e.message}\nCheck GEMINI_API_KEY in .env."
        if e.code == 404:
            return f"Gemini model '{self.model}' not found. Set GEMINI_MODEL in .env (e.g. gemini-3.8-flash)."
        if e.code == 429:
            return f"Gemini rate limit or quota reached (429): {e.message}"
        return f"Gemini API error ({e.code}): {e.message}"

    def _config(self) -> types.GenerateContentConfig:
        cfg: dict = {"system_instruction": self.system_prompt}
        if self.tools:
            cfg["tools"] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
                for t in self.tools
            ])]
        if self.settings.gemini_temperature is not None:
            cfg["temperature"] = self.settings.gemini_temperature
        if self.settings.show_thinking:
            cfg["thinking_config"] = types.ThinkingConfig(include_thoughts=True)
        return types.GenerateContentConfig(**cfg)

    def add_user_message(self, text: str) -> None:
        self.contents.append(types.Content(role="user", parts=[types.Part(text=text)]))

    async def generate(self, on_text, on_thinking) -> Turn:
        parts: list[types.Part] = []
        text: list[str] = []
        calls: list[ToolCall] = []
        finish = None
        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=self.model, contents=self.contents, config=self._config()
            )
            async for chunk in stream:
                if not chunk.candidates:
                    continue
                cand = chunk.candidates[0]
                finish = cand.finish_reason or finish
                for part in (cand.content.parts if cand.content and cand.content.parts else []):
                    parts.append(part)  # keep every part as-is, signatures included
                    if part.function_call:
                        fc = part.function_call
                        call_id = fc.id
                        if not call_id:
                            self._n += 1
                            call_id = f"gemini_call_{self._n}"
                            self._generated_ids.add(call_id)
                        args = dict(fc.args or {})
                        calls.append(ToolCall(id=call_id, name=fc.name, arguments=args, raw_arguments=json.dumps(args)))
                    elif part.text:
                        if part.thought:
                            on_thinking(part.text)
                        else:
                            text.append(part.text)
                            on_text(part.text)
        except errors.APIError as e:
            raise ProviderError(self._explain(e)) from None

        if parts:
            self.contents.append(types.Content(role="model", parts=parts))
        else:
            note = f"[Gemini returned no content (finish reason: {finish})]"
            on_text(note)
            text.append(note)
        return Turn(text="".join(text).strip(), tool_calls=calls)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        parts = []
        for r in results:
            response = {"error": r.content} if r.is_error else {"result": r.content}
            call_id = None if r.call.id in self._generated_ids else r.call.id
            parts.append(types.Part(function_response=types.FunctionResponse(id=call_id, name=r.call.name, response=response)))
        self.contents.append(types.Content(role="user", parts=parts))
