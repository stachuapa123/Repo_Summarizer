"""Pick the LLM provider named by LLM_PROVIDER in .env."""

from __future__ import annotations

from .base import Provider, ProviderError, ToolCall, ToolResult, ToolSpec, Turn

PROVIDERS = ("lemonade", "ollama", "gemini")


def create_provider(settings, system_prompt: str) -> Provider:
    # Imports are lazy so you only need the SDK for the provider you use.
    name = settings.provider
    if name == "lemonade":
        from .lemonade import LemonadeProvider
        return LemonadeProvider(settings, system_prompt)
    if name == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider(settings, system_prompt)
    if name == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(settings, system_prompt)
    raise ProviderError(f"Unknown LLM_PROVIDER '{name}'. Use one of: {', '.join(PROVIDERS)}.")


__all__ = ["Provider", "ProviderError", "ToolCall", "ToolResult", "ToolSpec", "Turn", "create_provider", "PROVIDERS"]
