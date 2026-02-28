"""Unit tests for meal_analysis.client (OpenAI async client)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from pydantic import ValidationError

from meal_analysis.client import (
    ChatCompletionResult,
    OpenAIClient,
    image_bytes_to_data_url,
)


# ---- ChatCompletionResult ----
def test_chat_completion_result_parses_valid() -> None:
    r = ChatCompletionResult(content="hello", input_tokens=5, output_tokens=3)
    assert r.content == "hello"
    assert r.input_tokens == 5
    assert r.output_tokens == 3


def test_chat_completion_result_rejects_negative_tokens() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionResult(content="x", input_tokens=-1, output_tokens=0)


# ---- image_bytes_to_data_url ----
def test_image_bytes_to_data_url_default_media_type() -> None:
    url = image_bytes_to_data_url(b"\xff\xd8\xff")
    assert url.startswith("data:image/jpeg;base64,")
    assert len(url) > len("data:image/jpeg;base64,")


def test_image_bytes_to_data_url_custom_media_type() -> None:
    url = image_bytes_to_data_url(b"abc", media_type="image/png")
    assert url.startswith("data:image/png;base64,")


def test_image_bytes_to_data_url_roundtrip() -> None:
    raw = b"binary image data"
    url = image_bytes_to_data_url(raw)
    import base64
    prefix = "data:image/jpeg;base64,"
    assert url.startswith(prefix)
    b64 = url[len(prefix):]
    assert base64.standard_b64decode(b64) == raw


# ---- OpenAIClient (async, mocked) ----
@pytest.fixture
def mock_get_config() -> MagicMock:
    config = MagicMock()
    config.openai_api_base = "https://api.openai.com/v1"
    config.openai_api_key = "test-key"
    return config


def _make_response_mock(json_return: dict) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value=json_return)
    return response


@pytest.mark.asyncio
async def test_chat_completion_returns_result(
    mock_get_config: MagicMock,
    openai_chat_response: dict,
) -> None:
    with (
        patch("meal_analysis.client.get_config", return_value=mock_get_config),
        patch("meal_analysis.client.httpx.AsyncClient") as mock_client_class,
    ):
        mock_http = MagicMock()
        mock_http.post = AsyncMock(return_value=_make_response_mock(openai_chat_response))
        mock_client_class.return_value = mock_http

        client = OpenAIClient()
        result = await client.chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert result.content == openai_chat_response["choices"][0]["message"]["content"]
    assert result.input_tokens == 10
    assert result.output_tokens == 20
    mock_http.post.assert_called_once()
    call_kwargs = mock_http.post.call_args.kwargs
    assert call_kwargs["json"]["model"] == "gpt-4o"
    assert call_kwargs["json"]["messages"] == [{"role": "user", "content": "Hello"}]
    assert "response_format" not in call_kwargs["json"]


@pytest.mark.asyncio
async def test_chat_completion_includes_response_format_when_given(
    mock_get_config: MagicMock,
    openai_chat_response: dict,
) -> None:
    with (
        patch("meal_analysis.client.get_config", return_value=mock_get_config),
        patch("meal_analysis.client.httpx.AsyncClient") as mock_client_class,
    ):
        mock_http = MagicMock()
        mock_http.post = AsyncMock(return_value=_make_response_mock(openai_chat_response))
        mock_client_class.return_value = mock_http

        client = OpenAIClient()
        await client.chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            response_format={"type": "json_object"},
        )

    call_kwargs = mock_http.post.call_args.kwargs
    assert call_kwargs["json"]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_chat_completion_no_usage_defaults_tokens_to_zero(
    mock_get_config: MagicMock,
    openai_chat_response_no_usage: dict,
) -> None:
    with (
        patch("meal_analysis.client.get_config", return_value=mock_get_config),
        patch("meal_analysis.client.httpx.AsyncClient") as mock_client_class,
    ):
        mock_http = MagicMock()
        mock_http.post = AsyncMock(
            return_value=_make_response_mock(openai_chat_response_no_usage)
        )
        mock_client_class.return_value = mock_http

        client = OpenAIClient()
        result = await client.chat_completion(
            model="gpt-4o",
            messages=[],
        )

    assert result.input_tokens == 0
    assert result.output_tokens == 0


@pytest.mark.asyncio
async def test_chat_completion_empty_choices_raises(
    mock_get_config: MagicMock,
    openai_chat_response_empty_choices: dict,
) -> None:
    with (
        patch("meal_analysis.client.get_config", return_value=mock_get_config),
        patch("meal_analysis.client.httpx.AsyncClient") as mock_client_class,
    ):
        mock_http = MagicMock()
        mock_http.post = AsyncMock(
            return_value=_make_response_mock(openai_chat_response_empty_choices)
        )
        mock_client_class.return_value = mock_http

        client = OpenAIClient()
        with pytest.raises(ValueError, match="no choices"):
            await client.chat_completion(model="gpt-4o", messages=[])


@pytest.mark.asyncio
async def test_chat_completion_http_error_propagates(
    mock_get_config: MagicMock,
    openai_chat_response: dict,
) -> None:
    with (
        patch("meal_analysis.client.get_config", return_value=mock_get_config),
        patch("meal_analysis.client.httpx.AsyncClient") as mock_client_class,
    ):
        mock_http = MagicMock()
        response = _make_response_mock(openai_chat_response)
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429",
            request=MagicMock(),
            response=MagicMock(),
        )
        mock_http.post = AsyncMock(return_value=response)
        mock_client_class.return_value = mock_http

        client = OpenAIClient()
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat_completion(model="gpt-4o", messages=[])


@pytest.mark.asyncio
async def test_aclose_closes_underlying_client(
    mock_get_config: MagicMock,
) -> None:
    with (
        patch("meal_analysis.client.get_config", return_value=mock_get_config),
        patch("meal_analysis.client.httpx.AsyncClient") as mock_client_class,
    ):
        mock_http = MagicMock()
        mock_http.aclose = AsyncMock()
        mock_client_class.return_value = mock_http

        client = OpenAIClient()
        _ = client._get_client()
        await client.aclose()

    mock_http.aclose.assert_called_once()
    assert client._client is None


@pytest.mark.asyncio
async def test_context_manager_closes_on_exit(
    mock_get_config: MagicMock,
) -> None:
    with (
        patch("meal_analysis.client.get_config", return_value=mock_get_config),
        patch("meal_analysis.client.httpx.AsyncClient") as mock_client_class,
    ):
        mock_http = MagicMock()
        mock_http.aclose = AsyncMock()
        mock_client_class.return_value = mock_http

        async with OpenAIClient() as client:
            client._get_client()

    mock_http.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_init_uses_injected_base_url_and_api_key() -> None:
    """When base_url and api_key are passed, get_config is not used for them."""
    with patch("meal_analysis.client.get_config") as mock_config:
        client = OpenAIClient(base_url="https://custom/v1", api_key="custom-key")
        mock_config.assert_called_once()
        # Client should use injected values when building the HTTP client
        client._client = MagicMock()
        client._client.aclose = AsyncMock()
        await client.aclose()
    # After aclose, _client is None; we just need to verify init didn't fail
    assert client._base_url == "https://custom/v1"
    assert client._api_key == "custom-key"
