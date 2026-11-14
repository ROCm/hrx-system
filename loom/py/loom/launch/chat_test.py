# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json

import pytest

from loom.launch.chat import (
    ChatFailure,
    ChatSession,
    one_shot,
    run_repl,
)
from loom.launch.client import ChatResult, Usage
from loom.launch.profile import Profile


class StubClient:
    """A client that answers from a script instead of a network."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(
        self,
        messages,
        *,
        model=None,
        stream=True,
        on_delta=None,
        headers=None,
        **fields,
    ):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "fields": dict(fields),
                "headers": headers,
                "model": model,
            }
        )
        reply = self.replies.pop(0)
        if isinstance(reply, ChatResult):
            return reply
        if on_delta is not None:
            for chunk in reply.get("chunks", []):
                on_delta({"choices": [{"delta": {"content": chunk}}]})
        text = reply.get("text", "".join(reply.get("chunks", [])))
        return ChatResult(
            status=200,
            text=text,
            usage=Usage(
                prompt_tokens=reply.get("prompt_tokens", 12),
                completion_tokens=reply.get("completion_tokens", 4),
                total_tokens=16,
            ),
            ttft_seconds=0.05,
            duration_seconds=0.25,
        )


def profile():
    return Profile.from_documents(
        "loom-fun/test-loom",
        {"model": {"id": "test/model", "name": "Test"}},
        llm_config={
            "context_length": 4096,
            "chat": {
                "system_prefix": "<|im_start|>system\n",
                "user_prefix": "<|im_start|>user\n",
                "assistant_prefix": "<|im_start|>assistant\n",
                "suffix": "<|im_end|>\n",
                "generation_prompt": "<|im_start|>assistant\n",
                "generation_prompt_plain": "<|im_start|>assistant\n",
                "thinking": "toggle",
                "thinking_default": True,
                "terminal": "<|im_end|>",
            },
        },
    )


def session(client, **overrides):
    arguments = {"client": client, "profile": profile()}
    arguments.update(overrides)
    return ChatSession(**arguments)


def test_a_turn_appends_history_and_reports_usage():
    client = StubClient([{"chunks": ["he", "llo"]}])
    chat = session(client)
    result = chat.send("hi")
    assert result.text == "hello"
    assert [message["role"] for message in chat.messages] == ["user", "assistant"]
    assert chat.messages[-1]["content"] == "hello"
    assert client.calls[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert chat.last_prompt_tokens() == 12
    assert chat.context_budget == 4096
    assert session(client, context_length=1024).context_budget == 1024


def test_template_controls_and_sessions_are_sent():
    client = StubClient([{"text": "ok"}, {"text": "again"}])
    chat = session(
        client,
        thinking=False,
        session="history",
        max_tokens=64,
        template_kwargs={"preserve_thinking": True},
    )
    chat.send("hi")
    fields = client.calls[0]["fields"]
    assert fields["enable_thinking"] is False
    assert fields["max_tokens"] == 64
    assert fields["chat_template_kwargs"] == {"preserve_thinking": True}
    assert client.calls[0]["headers"] == {"X-Loom-Session": "history"}
    chat.send("more")
    assert len(client.calls[1]["messages"]) == 3


def test_a_failed_turn_leaves_history_unchanged():
    client = StubClient([ChatResult(status=500, error="model failed")])
    chat = session(client)
    with pytest.raises(ChatFailure) as error:
        chat.send("hi")
    assert "model failed" in str(error.value)
    assert chat.messages == []
    assert chat.results == []


def test_reset_keeps_the_system_prompt():
    client = StubClient([{"text": "ok"}])
    chat = session(client, system="be terse")
    chat.send("hi")
    chat.reset()
    assert chat.messages == [{"role": "system", "content": "be terse"}]
    assert chat.results == []


def test_tool_calls_are_kept_in_history():
    result = ChatResult(
        status=200,
        text="",
        tool_calls=[{"id": "1", "function": {"name": "read", "arguments": "{}"}}],
    )
    chat = session(StubClient([result]))
    chat.send("read a file")
    assert chat.messages[-1]["tool_calls"][0]["function"]["name"] == "read"


def feed(lines):
    remaining = list(lines)

    def read(_prompt):
        if not remaining:
            raise EOFError
        return remaining.pop(0)

    return read


def test_repl_commands_change_session_state():
    written = []
    client = StubClient([{"text": "ok"}])
    chat = session(client)
    code = run_repl(
        chat,
        input_fn=feed(
            ["/help", "/think off", "/system be nice", "/usage", "/unknown", "/exit"]
        ),
        write=written.append,
    )
    assert code == 0
    text = "".join(written)
    assert "commands" in text
    assert "thinking off" in text
    assert "system prompt replaced" in text
    assert "no turns yet" in text
    assert "unknown command" in text
    assert chat.thinking is False
    assert chat.messages == [{"role": "system", "content": "be nice"}]


def test_repl_ends_at_end_of_input():
    written = []
    code = run_repl(session(StubClient([])), input_fn=feed([]), write=written.append)
    assert code == 0


def test_repl_prints_answers_and_usage():
    written = []
    chat = session(StubClient([{"chunks": ["answer"]}]))
    run_repl(chat, input_fn=feed(["hello", "/exit"]), write=written.append)
    text = "".join(written)
    assert "answer" in text
    assert "12 prompt" in text
    assert "4 completion" in text
    assert "tok/s" in text
    assert "% of 4096" in text


def test_repl_saves_the_transcript(tmp_path):
    written = []
    target = tmp_path / "transcript.json"
    chat = session(StubClient([{"text": "ok"}]))
    run_repl(
        chat,
        input_fn=feed(["hi", f"/save {target}", "/exit"]),
        write=written.append,
    )
    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["profile"] == "loom-fun/test-loom"
    assert document["messages"][-1]["content"] == "ok"


def test_one_shot_streams_the_answer_and_reports_usage():
    written = []
    chat = session(StubClient([{"chunks": ["a", "b"]}]))
    result = one_shot(chat, "hi", write=written.append)
    assert result.text == "ab"
    assert "".join(written).startswith("ab")
