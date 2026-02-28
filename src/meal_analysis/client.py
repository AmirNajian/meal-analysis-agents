"""Async OpenAI API client (httpx) for chat/vision with token usage capture."""

from __future__ import annotations

import base64
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from meal_analysis.config import get_config


class ChatCompletionResult(BaseModel):
    """Result of a chat completion call with content and token usage."""

    model_config = ConfigDict(frozen=True)

    content: str
    input_tokens: int = Field(ge=0, description="Prompt tokens used")
    output_tokens: int = Field(ge=0, description="Completion tokens used")


def image_bytes_to_data_url(image_bytes: bytes, media_type: str = "image/jpeg") -> str:
    """Build a data URL for inline image in chat messages (e.g. vision)."""
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    return f"data:{media_type};base64,{b64}"


class OpenAIClient:
    """Async OpenAI API client using httpx. Reuse one instance (e.g. app state)."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        config = get_config()
        self._base_url = (base_url or config.openai_api_base).rstrip("/")
        self._api_key = api_key or config.openai_api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout,
            )
        return self._client

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_format: dict[str, str] | None = None,
    ) -> ChatCompletionResult:
        """Call POST /chat/completions and return content plus token usage."""
        body: dict[str, Any] = {"model": model, "messages": messages}
        if response_format is not None:
            body["response_format"] = response_format

        client = self._get_client()
        response = await client.post("/chat/completions", json=body)
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("OpenAI response has no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)

        usage = data.get("usage") or {}
        input_tokens = usage.get("prompt_tokens", 0) or 0
        output_tokens = usage.get("completion_tokens", 0) or 0

        return ChatCompletionResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def aclose(self) -> None:
        """Close the underlying httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> OpenAIClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()


__all__ = [
    "ChatCompletionResult",
    "OpenAIClient",
    "image_bytes_to_data_url",
]
