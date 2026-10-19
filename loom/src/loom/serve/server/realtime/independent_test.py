# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent native response ownership through the actual WebSocket endpoint."""

import base64
import struct

from catalog_test_util import Client as HttpClient
from catalog_test_util import check_pcm, complete
from client_test_util import Client


def message(text, role="user"):
    return {
        "type": "message",
        "role": role,
        "content": [
            {"type": "input_text" if role != "assistant" else "text", "text": text}
        ],
    }


def reference(identity):
    return {"type": "item_reference", "id": identity}


def create(client, events, **settings):
    client.send("response.create", response=settings)
    while True:
        event = client.receive()
        events.append(event)
        if event["type"] == "response.created":
            return event["response"]["id"]


def collect(client, events, identities):
    responses = {
        e["response"]["id"]: e["response"]
        for e in events
        if e["type"] == "response.done"
    }
    while not identities <= responses.keys():
        event = client.receive()
        events.append(event)
        if event["type"] == "response.done":
            responses[event["response"]["id"]] = event["response"]
    return responses


def check(events, response, expected, voice="positive", independent=True):
    identity = response["id"]
    selected = [
        e
        for e in events
        if e.get("response_id", e.get("response", {}).get("id")) == identity
    ]
    assert [
        e["type"]
        for e in selected
        if e["type"] in ("response.created", "response.done")
    ] == ["response.created", "response.done"], selected
    assert response["status"] in ("completed", "incomplete"), response
    assert (response["conversation_id"] is None) == independent, response
    assert all(
        e["response"]["conversation_id"] == response["conversation_id"]
        for e in selected
        if "response" in e
    )
    text = "".join(
        e["delta"]
        for e in selected
        if e["type"] == "response.output_audio_transcript.delta"
    )
    assert text == expected, (text, expected, selected)
    audio = b"".join(
        base64.b64decode(e["delta"], validate=True)
        for e in selected
        if e["type"] == "response.output_audio.delta"
    )
    if expected:
        check_pcm(audio, expected, 42, voice)
    else:
        assert not audio, selected
    if independent:
        items = {item["id"] for item in response["output"]}
        assert not any(
            e["type"].startswith("conversation.item.")
            and e.get("item", {}).get("id") in items
            for e in events
        ), events
    return text


def close(client):
    client.send_frame(8, struct.pack(">H", 1000))
    assert client.read_frame() == (8, struct.pack(">H", 1000))


def independent_contexts(address, trace):
    client, admin = Client(address, trace), HttpClient(address)
    drained = False
    try:
        client.receive()
        client.message("original", "a")
        previous = client.respond([{"role": "user", "content": "a"}])
        prior_item = previous["output"][0]
        prior_text = prior_item["content"][0]["transcript"]
        original = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": prior_text},
        ]
        admin.request("/v1/models/drain", {"model": "fixture-text"})
        drained = True
        events = []
        first = create(
            client,
            events,
            conversation="none",
            max_output_tokens=3,
            metadata={"purpose": "snapshot"},
        )
        mixed = [
            {"role": "user", "content": "a"},
            {"role": "system", "content": "b"},
            {"role": "assistant", "content": prior_text},
        ]
        second = create(
            client,
            events,
            conversation="none",
            max_output_tokens=4,
            input=[
                reference("original"),
                message("b", "system"),
                reference(prior_item["id"]),
            ],
            audio={"output": {"voice": "negative"}},
        )
        client.send("conversation.item.delete", item_id="original")
        events += client.until("conversation.item.deleted")
        client.send(
            "conversation.item.truncate",
            item_id=prior_item["id"],
            content_index=0,
            audio_end_ms=0,
        )
        events += client.until("conversation.item.truncated")
        default = create(client, events, input=[message("x")], max_output_tokens=2)
        empty = create(
            client,
            events,
            conversation="none",
            input=[],
            instructions="a",
            max_output_tokens=2,
        )
        client.send(
            "response.create",
            response={"conversation": "none", "input": []},
            event_id="full",
        )
        error = client.receive(allow_error=True)
        assert error["type"] == "error" and error["error"]["event_id"] == "full", error
        admin.request("/v1/models/undrain", {"model": "fixture-text"})
        drained = False
        responses = collect(client, events, {first, second, default, empty})
        check(events, responses[first], complete(original, 3)[0])
        check(events, responses[second], complete(mixed, 4)[0], "negative")
        default_text = check(
            events,
            responses[default],
            complete([{"role": "user", "content": "x"}], 2)[0],
            independent=False,
        )
        check(
            events,
            responses[empty],
            complete([{"role": "system", "content": "a"}], 2)[0],
        )
        # Completed independent output has a session identity but no default-history position.
        detached = responses[first]["output"][0]
        client.respond(
            [{"role": "assistant", "content": detached["content"][0]["transcript"]}],
            overrides={"conversation": "none", "input": [reference(detached["id"])]},
        )
        client.send(
            "conversation.item.create",
            previous_item_id=detached["id"],
            item=message("a"),
            event_id="detached-position",
        )
        error = client.receive(allow_error=True)
        assert (
            error["type"] == "error"
            and error["error"]["event_id"] == "detached-position"
        ), error
        client.send("conversation.item.delete", item_id=detached["id"])
        client.until("conversation.item.deleted")
        client.respond([{"role": "assistant", "content": default_text}])
        close(client)
    finally:
        if drained:
            admin.request("/v1/models/undrain", {"model": "fixture-text"})
        client.close()


def independent_pending_input(address, trace):
    client, admin = Client(address, trace), HttpClient(address)
    drained = False
    try:
        client.receive()
        admin.request("/v1/models/drain", {"model": "recognition"})
        drained = True
        client.send(
            "input_audio_buffer.append", audio=base64.b64encode(bytes(480)).decode()
        )
        client.send("input_audio_buffer.commit")
        item = client.until("conversation.item.added")[-1]["item"]["id"]
        events = []
        kept = create(
            client,
            events,
            conversation="none",
            input=[reference(item)],
            max_output_tokens=3,
        )
        cancelled = create(
            client,
            events,
            conversation="none",
            input=[reference(item)],
            max_output_tokens=4,
        )
        omitted = create(
            client,
            events,
            conversation="none",
            max_output_tokens=2,
            audio={"output": {"voice": "negative"}},
        )
        # Failure after acquiring a pending reference must release that interest.
        client.send(
            "response.create",
            response={
                "conversation": "none",
                "input": [reference(item), reference("missing")],
            },
            event_id="failed-retain",
        )
        error = client.receive(allow_error=True)
        assert (
            error["type"] == "error" and error["error"]["event_id"] == "failed-retain"
        ), error
        client.send("conversation.item.delete", item_id=item)
        events += client.until("conversation.item.deleted")
        client.send("response.cancel", response_id=cancelled)
        responses = collect(client, events, {cancelled})
        assert responses[cancelled]["status"] == "cancelled", responses[cancelled]
        # Neither a pending transcript nor its cancelled peer blocks unrelated input.
        default = client.respond(
            [{"role": "user", "content": "a"}], overrides={"input": [message("a")]}
        )
        admin.request("/v1/models/undrain", {"model": "recognition"})
        drained = False
        responses = collect(client, events, {kept, omitted})
        check(
            events, responses[kept], complete([{"role": "user", "content": "A"}], 3)[0]
        )
        check(
            events,
            responses[omitted],
            complete([{"role": "user", "content": "A"}], 2)[0],
            "negative",
        )
        assert not any(
            e["type"] == "conversation.item.input_audio_transcription.completed"
            and e["item_id"] == item
            for e in events
        ), events
        client.respond(
            [
                {
                    "role": "assistant",
                    "content": default["output"][0]["content"][0]["transcript"],
                }
            ]
        )
        close(client)
    finally:
        if drained:
            admin.request("/v1/models/undrain", {"model": "recognition"})
        client.close()


def independent_function_input(address, trace):
    client, admin = Client(address, trace), HttpClient(address)
    try:
        client.receive()
        client.message("context", "b")
        raw_call = {
            "type": "function_call",
            "call_id": "selected-call",
            "name": "lookup",
            "arguments": '{"value":"a"}',
        }
        raw_result = {
            "type": "function_call_output",
            "call_id": "selected-call",
            "output": "a",
        }
        messages = [
            {"role": "system", "content": "a"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "selected-call",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"value":"a"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "selected-call", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        native = admin.request(
            "/v1/chat/completions",
            {
                "model": "fixture-text",
                "messages": messages,
                "max_tokens": 2,
                "seed": 42,
            },
        )
        expected = native["choices"][0]["message"]["content"]
        events = []
        identity = create(
            client,
            events,
            conversation="none",
            input=[message("a", "system"), raw_call, raw_result, reference("context")],
            max_output_tokens=2,
        )
        response = collect(client, events, {identity})[identity]
        check(events, response, expected)
        client.respond([{"role": "user", "content": "b"}])
        close(client)
    finally:
        client.close()


def independent_admission(address, trace):
    client = Client(address, trace)
    try:
        client.receive()
        invalid = [
            {"conversation": "missing"},
            {"input": None},
            {"input": [reference("missing")]},
            {"input": [message("a")] * 129},
            {"input": [{"type": "item_reference", "id": "", "extra": 1}]},
            {
                "input": [
                    {"type": "function_call_output", "call_id": "absent", "output": "a"}
                ]
            },
            {
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "same",
                        "name": "lookup",
                        "arguments": "{}",
                    }
                ]
                * 2
            },
            *(
                {"input": [{"type": "message", "role": "user", "content": [content]}]}
                for content in (
                    {"type": "input_audio", "audio": "AAA="},
                    {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                )
            ),
        ]
        for index, overrides in enumerate(invalid):
            identity = f"invalid-input-{index}"
            client.send(
                "response.create",
                response={"conversation": "none", **overrides},
                event_id=identity,
            )
            error = client.receive(allow_error=True)
            assert (
                error["type"] == "error" and error["error"]["event_id"] == identity
            ), error
        client.respond(
            [{"role": "system", "content": "a"}],
            overrides={"conversation": "none", "input": [], "instructions": "a"},
        )
        client.message("still-empty", "b")
        client.respond([{"role": "user", "content": "b"}])
        close(client)
    finally:
        client.close()


def independent_cancel_scope(address, trace):
    client, admin = Client(address, trace), HttpClient(address)
    drained = False
    recognition_drained = False
    try:
        client.receive()
        client.message("default-input", "a")
        admin.request("/v1/models/drain", {"model": "fixture-text"})
        drained = True
        events = []
        default = create(client, events)
        independent = create(client, events, conversation="none", input=[message("b")])
        client.send("response.cancel")
        responses = collect(client, events, {default})
        assert responses[default]["status"] == "cancelled", responses
        for event_id, kind, fields, code in (
            ("no-default", "response.cancel", {}, "response_cancel_not_active"),
            (
                "cancelled-response",
                "response.cancel",
                {"response_id": default},
                "response_cancel_not_active",
            ),
            (
                "unknown-response",
                "response.cancel",
                {"response_id": "missing"},
                "response_cancel_not_active",
            ),
            (
                "unknown-item",
                "conversation.item.delete",
                {"item_id": "missing"},
                "NOT_FOUND",
            ),
        ):
            client.send(kind, event_id=event_id, **fields)
            error = client.receive(allow_error=True)
            assert error["type"] == "error", error
            assert error["error"]["type"] == "invalid_request_error", error
            assert error["error"]["event_id"] == event_id, error
            assert error["error"]["code"] == code, error
            assert error["error"]["message"], error
        client.send("response.cancel", response_id=7)
        error = client.receive(allow_error=True)
        assert error["type"] == "error", error
        assert error["error"]["type"] == "invalid_request_error", error
        assert error["error"]["code"] == "INVALID_ARGUMENT", error
        admin.request("/v1/models/undrain", {"model": "fixture-text"})
        drained = False
        responses = collect(client, events, {independent})
        check(
            events,
            responses[independent],
            complete([{"role": "user", "content": "b"}], 2)[0],
        )
        client.send(
            "response.cancel", response_id=independent, event_id="completed-response"
        )
        error = client.receive(allow_error=True)
        assert error["type"] == "error", error
        assert error["error"]["event_id"] == "completed-response", error
        assert error["error"]["code"] == "response_cancel_not_active", error
        client.respond([{"role": "user", "content": "a"}])
        # Socket close drops the final snapshot reader while its original
        # canonical item has already been deleted and STT admission is held.
        admin.request("/v1/models/drain", {"model": "recognition"})
        recognition_drained = True
        client.send(
            "input_audio_buffer.append", audio=base64.b64encode(bytes(480)).decode()
        )
        client.send("input_audio_buffer.commit")
        item = client.until("conversation.item.added")[-1]["item"]["id"]
        create(client, [], conversation="none", input=[reference(item)])
        client.send("conversation.item.delete", item_id=item)
        client.until("conversation.item.deleted")
        close(client)
        admin.request("/v1/models/undrain", {"model": "recognition"})
        recognition_drained = False
    finally:
        if drained:
            admin.request("/v1/models/undrain", {"model": "fixture-text"})
        if recognition_drained:
            admin.request("/v1/models/undrain", {"model": "recognition"})
        client.close()


def independent_automatic_intent(address, trace):
    client, admin = Client(address, trace), HttpClient(address)
    drained = False
    try:
        client.receive()
        client.send(
            "session.update",
            session={
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "prefix_padding_ms": 1,
                            "silence_duration_ms": 1,
                            "create_response": True,
                            "interrupt_response": True,
                        }
                    }
                }
            },
        )
        client.until("session.updated")
        # Activity and STT are separate native registrations, as in a Silero/
        # Moonshine assistant. The VAD stop is observable while STT is held.
        admin.request("/v1/models/drain", {"model": "recognition"})
        drained = True
        samples = [0] * 96 + [64] * 192 + [0] * 480
        client.send(
            "input_audio_buffer.append",
            audio=base64.b64encode(struct.pack(f"<{len(samples)}h", *samples)).decode(),
        )
        events = client.until("input_audio_buffer.committed")
        auxiliary = create(
            client,
            events,
            conversation="none",
            input=[],
            instructions="b",
            metadata={"purpose": "status"},
        )
        responses = collect(client, events, {auxiliary})
        check(
            events,
            responses[auxiliary],
            complete([{"role": "system", "content": "b"}], 2)[0],
        )
        admin.request("/v1/models/undrain", {"model": "recognition"})
        drained = False
        automatic = client.until("response.done")
        response = automatic[-1]["response"]
        assert response["id"] != auxiliary and response["metadata"] is None, response
        check(
            automatic,
            response,
            complete([{"role": "user", "content": "A"}], 2)[0],
            independent=False,
        )
        close(client)
    finally:
        if drained:
            admin.request("/v1/models/undrain", {"model": "recognition"})
        client.close()
