# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A terminal conversation against a launched profile.

The chat client is a normal OpenAI client with two launcher additions: it knows
the profile's template policy (so it can explain and toggle thinking, preview a
rendered prompt and report how much of the configured context a turn used) and
it keeps the full conversation history, which is what a retained-session server
expects for exact continuation.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from loom.launch.client import ChatResult, Client
from loom.launch.profile import Profile


@dataclass
class ChatSession:
    """One conversation bound to an endpoint and a model name."""

    client: Client
    profile: Profile
    model: str | None = None
    thinking: bool | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    system: str | None = None
    session: str | None = None
    #: The served context, when it is narrower than the package's own limit.
    context_length: int | None = None
    #: Extra controls passed to the package's chat policy, unchanged.
    template_kwargs: dict = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    results: list[ChatResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = self.profile.model_id
        if self.thinking is None and self.profile.chat is not None:
            self.thinking = self.profile.chat.default_thinking
        if self.system is not None:
            self.messages.append({"role": "system", "content": self.system})
        self._request_id = 0

    # -- conversation -------------------------------------------------------
    def reset(self) -> None:
        self.messages = [
            message for message in self.messages if message.get("role") == "system"
        ]
        self.results = []

    def send(
        self,
        text: str,
        *,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> ChatResult:
        self.messages.append({"role": "user", "content": text})
        self._request_id += 1
        fields: dict = {}
        if self.max_tokens is not None:
            fields["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            fields["temperature"] = self.temperature
        if self.thinking is not None:
            fields["enable_thinking"] = self.thinking
        if self.template_kwargs:
            fields["chat_template_kwargs"] = dict(self.template_kwargs)
        headers = {"X-Loom-Session": self.session} if self.session else None

        def on_delta(event: dict) -> None:
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content") or ""
                if reasoning and on_reasoning is not None:
                    on_reasoning(reasoning)
                content = delta.get("content") or ""
                if content and on_text is not None:
                    on_text(content)

        result = self.client.chat(
            self.messages,
            model=self.model,
            stream=True,
            headers=headers,
            on_delta=on_delta,
            **fields,
        )
        if result.error:
            # The user message stays in history only if the turn completed.
            self.messages.pop()
            raise ChatFailure(result.error)
        assistant: dict = {"role": "assistant", "content": result.text}
        if result.tool_calls:
            assistant["tool_calls"] = result.tool_calls
        self.messages.append(assistant)
        self.results.append(result)
        return result

    # -- observation --------------------------------------------------------
    @property
    def context_budget(self) -> int | None:
        """The context a request is actually held to."""
        return self.context_length or self.profile.context_length

    def last_prompt_tokens(self) -> int | None:
        if not self.results:
            return None
        usage = self.results[-1].usage
        return usage.prompt_tokens if usage else None

    def transcript(self) -> dict:
        return {
            "model": self.model,
            "profile": self.profile.repo,
            "thinking": self.thinking,
            "messages": self.messages,
            "usage": [
                result.usage.describe() if result.usage else None
                for result in self.results
            ],
        }


class ChatFailure(RuntimeError):
    """The endpoint rejected or failed one turn."""


HELP = """commands
  /help                 show this help
  /reset                forget the conversation, keep the system prompt
  /think on|off         request or suppress the package's reasoning block
  /system TEXT          replace the system prompt and reset
  /prompt               print the rendered prompt for the next turn
  /usage                show the last turn's token usage and rate
  /save PATH            write the transcript as JSON
  /exit                 leave (Ctrl-D works too)"""


def run_repl(
    session: ChatSession,
    *,
    input_fn: Callable[[str], str] = input,
    write: Callable[[str], None] = lambda text: print(text, end="", flush=True),
    prompt: str = "you> ",
) -> int:
    """A small interactive loop; returns a process exit code."""
    while True:
        try:
            line = input_fn(prompt)
        except (EOFError, KeyboardInterrupt):
            write("\n")
            return 0
        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            if _command(session, line, write):
                return 0
            continue
        _turn(session, line, write)


def _turn(session: ChatSession, text: str, write: Callable[[str], None]) -> None:
    write("\nassistant> ")
    shown_reasoning = False

    def on_reasoning(chunk: str) -> None:
        nonlocal shown_reasoning
        if not shown_reasoning:
            write("\n[reasoning] ")
            shown_reasoning = True
        write(chunk)

    def on_text(chunk: str) -> None:
        nonlocal shown_reasoning
        if shown_reasoning:
            write("\n\n")
            shown_reasoning = False
        write(chunk)

    try:
        result = session.send(text, on_text=on_text, on_reasoning=on_reasoning)
    except ChatFailure as exc:
        write(f"\nerror: {exc}\n")
        return
    write("\n")
    if result.tool_calls:
        write(f"[{len(result.tool_calls)} tool call(s) requested]\n")
        write(json.dumps(result.tool_calls, indent=1) + "\n")
    _report_usage(session, write, result)


def _report_usage(
    session: ChatSession, write: Callable[[str], None], result: ChatResult
) -> None:
    usage = result.usage
    if usage is None:
        return
    parts = []
    if usage.prompt_tokens:
        parts.append(f"{usage.prompt_tokens} prompt")
    if usage.completion_tokens:
        parts.append(f"{usage.completion_tokens} completion")
    rate = result.decode_tokens_per_second
    if rate:
        parts.append(f"{rate:,.1f} tok/s")
    if result.ttft_seconds is not None:
        parts.append(f"ttft {result.ttft_seconds * 1000:.0f} ms")
    if usage.cached_tokens:
        parts.append(f"{usage.cached_tokens} cached")
    if not parts:
        return
    suffix = ""
    context = session.context_budget
    if usage.prompt_tokens and context:
        suffix = f"  ({usage.prompt_tokens * 100 // context}% of {context})"
    write("[" + ", ".join(parts) + "]" + suffix + "\n")


def _command(session: ChatSession, line: str, write: Callable[[str], None]) -> bool:
    command, _, argument = line.partition(" ")
    argument = argument.strip()
    if command in ("/exit", "/quit"):
        return True
    if command == "/help":
        write(HELP + "\n")
        return False
    if command == "/reset":
        session.reset()
        write("conversation reset\n")
        return False
    if command == "/think":
        if argument in ("on", "off"):
            session.thinking = argument == "on"
            write(f"thinking {argument}\n")
        else:
            write(f"thinking is {session.thinking}\n")
        return False
    if command == "/system":
        session.messages = [{"role": "system", "content": argument}]
        session.results = []
        write("system prompt replaced; conversation reset\n")
        return False
    if command in ("/prompt", "/prompt-preview"):
        try:
            rendered = session.profile.render(
                session.messages + [{"role": "user", "content": "<next turn>"}],
                thinking=session.thinking,
            )
        except Exception as exc:  # A preview failing must not end the session.
            write(f"cannot render a prompt: {exc}\n")
            return False
        write(rendered + "\n")
        return False
    if command == "/usage":
        if not session.results:
            write("no turns yet\n")
            return False
        result = session.results[-1]
        write(json.dumps(result.describe(), indent=1) + "\n")
        return False
    if command == "/save":
        if not argument:
            write("usage: /save PATH\n")
            return False
        path = argument
        with open(path, "w", encoding="utf-8") as file:
            json.dump(session.transcript(), file, indent=1)
        write(f"wrote {path}\n")
        return False
    write(f"unknown command {command}; /help lists them\n")
    return False


def one_shot(
    session: ChatSession,
    text: str,
    *,
    write: Callable[[str], None] = lambda value: print(value, end="", flush=True),
) -> ChatResult:
    """Sends one prompt, streaming its answer, and reports usage on stderr."""

    def on_reasoning(chunk: str) -> None:
        sys.stderr.write(chunk)
        sys.stderr.flush()

    result = session.send(text, on_text=write, on_reasoning=on_reasoning)
    write("\n")
    if result.tool_calls:
        sys.stderr.write(json.dumps(result.tool_calls, indent=1) + "\n")
    _report_usage(session, lambda value: sys.stderr.write(value), result)
    sys.stderr.flush()
    return result


def describe_turn(result: ChatResult) -> dict:
    """The JSON form `chat --json` prints for one turn."""
    document = result.describe()
    document["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return document
