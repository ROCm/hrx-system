# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A small OpenAI-shaped client with exact client-side timing.

The launcher's chat and benchmark paths need the same primitive: send one
completion, stream the deltas as they arrive, and know when the first token
arrived. This client is deliberately thin, standard-library only, and tolerant
of response fields a given package does not send.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field

from loom.launch.errors import LaunchError
from loom.launch.values import dict_value, list_value, str_value


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None

    @classmethod
    def from_document(cls, document: object) -> Usage | None:
        usage = dict_value(document)
        if not usage:
            return None
        details = dict_value(usage.get("prompt_tokens_details"))

        def count(value: object) -> int | None:
            return value if isinstance(value, int) else None

        return cls(
            prompt_tokens=count(usage.get("prompt_tokens")),
            completion_tokens=count(usage.get("completion_tokens")),
            total_tokens=count(usage.get("total_tokens")),
            cached_tokens=count(details.get("cached_tokens")),
        )

    def describe(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
        }


@dataclass
class ChatResult:
    """One completion as the client observed it."""

    status: int = 0
    text: str = ""
    reasoning: str = ""
    tool_calls: list = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage | None = None
    ttft_seconds: float | None = None
    duration_seconds: float | None = None
    chunks: int = 0
    error: str | None = None
    document: dict = field(default_factory=dict)

    @property
    def decode_tokens_per_second(self) -> float | None:
        if self.usage is None or self.ttft_seconds is None:
            return None
        if not self.usage.completion_tokens or self.usage.completion_tokens < 2:
            return None
        if self.duration_seconds is None:
            return None
        interval = self.duration_seconds - self.ttft_seconds
        if interval <= 0:
            return None
        return (self.usage.completion_tokens - 1) / interval

    def describe(self) -> dict:
        return {
            "status": self.status,
            "text": self.text,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
            "usage": self.usage.describe() if self.usage else None,
            "ttft_seconds": _round(self.ttft_seconds),
            "duration_seconds": _round(self.duration_seconds),
            "decode_tokens_per_second": _round(self.decode_tokens_per_second),
            "chunks": self.chunks,
            "error": self.error,
        }


class Client:
    """An OpenAI-compatible endpoint client bound to one base URL."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 900.0,
    ):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in ("http", ""):
            raise LaunchError(f"unsupported endpoint scheme in {base_url}")
        self.base_url = base_url.rstrip("/")
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or 80
        self._prefix = parsed.path.rstrip("/")
        self._token = token
        self._timeout = timeout

    # -- plumbing -----------------------------------------------------------
    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if extra:
            headers.update(extra)
        return headers

    def _connection(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)

    def request(
        self,
        method: str,
        path: str,
        *,
        document: dict | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> tuple[int, bytes]:
        body = None
        outgoing = self._headers(headers)
        if document is not None:
            body = json.dumps(document).encode("utf-8")
            outgoing["Content-Type"] = "application/json"
        if timeout is not None:
            connection = http.client.HTTPConnection(
                self._host, self._port, timeout=timeout
            )
        else:
            connection = self._connection()
        try:
            connection.request(method, self._prefix + path, body=body, headers=outgoing)
            response = connection.getresponse()
            return response.status, response.read()
        except (OSError, http.client.HTTPException) as exc:
            raise LaunchError(f"{method} {path} failed: {exc}") from exc
        finally:
            connection.close()

    def get_json(self, path: str, *, timeout: float | None = None) -> tuple[int, dict]:
        status, body = self.request("GET", path, timeout=timeout)
        return status, _document(body)

    def post_json(
        self, path: str, document: dict | None = None, **kwargs: object
    ) -> tuple[int, dict]:
        status, body = self.request("POST", path, document=document, **kwargs)
        return status, _document(body)

    # -- discovery ----------------------------------------------------------
    def healthy(self, timeout: float = 2.0) -> bool:
        try:
            status, _ = self.get_json("/healthz", timeout=timeout)
        except LaunchError:
            return False
        return status == 200

    def models(self, timeout: float | None = None) -> list[dict]:
        status, document = self.get_json("/v1/models", timeout=timeout)
        if status != 200:
            return []
        return [
            entry
            for entry in list_value(document.get("data"))
            if isinstance(entry, dict)
        ]

    def model_names(self, timeout: float | None = None) -> list[str]:
        names = []
        for entry in self.models(timeout=timeout):
            name = str_value(entry.get("id"))
            if name:
                names.append(name)
        return names

    def server_status(self, timeout: float | None = None) -> dict:
        status, document = self.get_json("/v1/status", timeout=timeout)
        if status != 200:
            return {}
        return document

    # -- completion ---------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        stream: bool = True,
        on_delta: Callable[[dict], None] | None = None,
        headers: dict[str, str] | None = None,
        **fields: object,
    ) -> ChatResult:
        payload: dict = {"messages": messages, "stream": stream}
        if model:
            payload["model"] = model
        for name, value in fields.items():
            if value is not None:
                payload[name] = value
        if stream:
            payload.setdefault("stream_options", {"include_usage": True})
        return self.complete(payload, headers=headers, on_delta=on_delta)

    def complete(
        self,
        payload: dict,
        *,
        headers: dict[str, str] | None = None,
        on_delta: Callable[[dict], None] | None = None,
    ) -> ChatResult:
        """Sends one completion request and measures it from the client side."""
        started = time.monotonic()
        result = ChatResult()
        outgoing = self._headers(headers)
        outgoing["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
        connection = self._connection()
        try:
            connection.request(
                "POST",
                self._prefix + "/v1/chat/completions",
                body=body,
                headers=outgoing,
            )
            response = connection.getresponse()
            result.status = response.status
            content_type = response.getheader("Content-Type") or ""
            streaming = content_type.startswith("text/event-stream")
            if not streaming:
                data = response.read()
                document = _document(data)
                result.document = document
                message = _first_message(document)
                result.text = str_value(message.get("content")) or ""
                result.reasoning = str_value(message.get("reasoning_content")) or ""
                result.tool_calls = [
                    entry
                    for entry in list_value(message.get("tool_calls"))
                    if isinstance(entry, dict)
                ]
                result.finish_reason = _finish_reason(document)
                result.usage = Usage.from_document(document.get("usage"))
                result.duration_seconds = time.monotonic() - started
                result.ttft_seconds = result.duration_seconds
                if result.status >= 400:
                    result.error = _error_message(document) or f"HTTP {result.status}"
                return result
            while True:
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:") :].strip()
                if not data:
                    continue
                if data == b"[DONE]":
                    break
                try:
                    event = json.loads(data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(event, dict):
                    continue
                result.chunks += 1
                choice_text, choice_reasoning, calls = _delta_text(event)
                if result.ttft_seconds is None and (
                    choice_text or choice_reasoning or calls
                ):
                    result.ttft_seconds = time.monotonic() - started
                result.text += choice_text
                result.reasoning += choice_reasoning
                result.tool_calls.extend(calls)
                finish = _finish_reason(event)
                if finish:
                    result.finish_reason = finish
                usage = Usage.from_document(event.get("usage"))
                if usage is not None:
                    result.usage = usage
                if on_delta is not None:
                    on_delta(event)
            result.duration_seconds = time.monotonic() - started
            if result.ttft_seconds is None:
                result.ttft_seconds = result.duration_seconds
            return result
        finally:
            connection.close()


def _document(data: bytes) -> dict:
    try:
        document = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return document if isinstance(document, dict) else {}


def _first_message(document: dict) -> dict:
    for choice in list_value(document.get("choices")):
        message = dict_value(dict_value(choice).get("message"))
        if message:
            return message
    return {}


def _finish_reason(document: dict) -> str | None:
    for choice in list_value(document.get("choices")):
        reason = str_value(dict_value(choice).get("finish_reason"))
        if reason:
            return reason
    return None


def _delta_text(event: dict) -> tuple[str, str, list]:
    text = ""
    reasoning = ""
    calls: list = []
    for choice in list_value(event.get("choices")):
        choice = dict_value(choice)
        delta = dict_value(choice.get("delta"))
        if not delta:
            delta = dict_value(choice.get("message"))
        text += str_value(delta.get("content")) or ""
        reasoning += str_value(delta.get("reasoning_content")) or ""
        for call in list_value(delta.get("tool_calls")):
            if isinstance(call, dict):
                calls.append(call)
    return text, reasoning, calls


def _error_message(document: dict) -> str | None:
    error = dict_value(document.get("error"))
    return str_value(error.get("message"))


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)
