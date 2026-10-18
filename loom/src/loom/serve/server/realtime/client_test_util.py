# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""RFC 6455 client and exact fixture-output checks for realtime tests."""

import base64
import json
import socket
import struct
from urllib.parse import quote

from loom.src.loom.serve.server.catalog_test_util import check_pcm, complete


class Client:
    """Independent RFC 6455 client with a durable ordered event transcript."""

    def __init__(self, address, trace, model="assistant"):
        host, port = address.rsplit(":", 1)
        self.socket = socket.create_connection((host, int(port)))
        self.input = self.socket.makefile("rb")
        self.trace = trace
        self.events = set()
        self.socket.sendall(
            f"GET /v1/realtime?model={quote(model, safe='')} HTTP/1.1\r\n".encode()
            + b"Host: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            b"Sec-WebSocket-Version: 13\r\n"
            b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"
        )
        status = self.input.readline()
        headers = {}
        while (line := self.input.readline()) != b"\r\n":
            assert line, "EOF during WebSocket handshake"
            name, value = line.decode().strip().split(":", 1)
            headers[name.lower()] = value.strip()
        if status != b"HTTP/1.1 101 Switching Protocols\r\n":
            body = (
                self.input.read(int(headers["content-length"]))
                if "content-length" in headers
                else self.input.read()
            )
            self.close()
            raise AssertionError((status, headers, body))
        assert headers["sec-websocket-accept"] == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="

    def send_frame(self, opcode, data):
        mask = b"\x12\x34\x56\x78"
        size = len(data)
        if size < 126:
            header = bytes([0x80 | opcode, 0x80 | size])
        elif size <= 65535:
            header = bytes([0x80 | opcode, 0xFE]) + struct.pack(">H", size)
        else:
            header = bytes([0x80 | opcode, 0xFF]) + struct.pack(">Q", size)
        self.socket.sendall(
            header + mask + bytes(byte ^ mask[i % 4] for i, byte in enumerate(data))
        )

    def send(self, kind, **fields):
        self.send_frame(
            1, json.dumps({"type": kind, **fields}, ensure_ascii=False).encode()
        )

    def read_frame(self):
        header = self.input.read(2)
        assert len(header) == 2, "EOF before a WebSocket frame"
        first, size = header
        assert first & 0x80 and not first & 0x70, header
        assert not size & 0x80, "server frames must not be masked"
        if size == 126:
            size = struct.unpack(">H", self.input.read(2))[0]
        elif size == 127:
            size = struct.unpack(">Q", self.input.read(8))[0]
        payload = self.input.read(size)
        assert len(payload) == size, "EOF inside a WebSocket frame"
        return first & 15, payload

    def receive(self, *, allow_error=False):
        opcode, payload = self.read_frame()
        assert opcode == 1, (opcode, payload)
        event = json.loads(payload)
        self.trace.write(json.dumps(event) + "\n")
        self.trace.flush()
        assert event["event_id"] not in self.events, event
        self.events.add(event["event_id"])
        assert allow_error or event["type"] != "error", event
        return event

    def until(self, kind):
        events = []
        while True:
            event = self.receive()
            events.append(event)
            if event["type"] == kind:
                return events

    def respond(self, messages, maximum=2, voice="positive", *, overrides=None):
        settings = {"max_output_tokens": maximum, **(overrides or {})}
        self.send("response.create", response=settings)
        events = self.until("response.done")
        check_response_metadata(events, settings.get("metadata"))
        audio_output = settings.get("output_modalities", ["audio"]) == ["audio"]
        delta_kind = (
            "response.output_audio_transcript.delta"
            if audio_output
            else "response.output_text.delta"
        )
        text = "".join(e["delta"] for e in events if e["type"] == delta_kind)
        expected, finish = complete(messages, maximum)
        assert text == expected, (text, expected, events)
        audio = b"".join(
            base64.b64decode(e["delta"], validate=True)
            for e in events
            if e["type"] == "response.output_audio.delta"
        )
        if audio_output:
            check_pcm(audio, text, 42, voice)
        else:
            assert not any("output_audio" in e["type"] for e in events), events
        response = events[-1]["response"]
        status = "completed" if finish == "stop" else "incomplete"
        assert response["status"] == status, response
        if text:
            item = response["output"][0]
            assert item["status"] == status, item
            assert (
                item["content"][0]["transcript" if audio_output else "text"] == text
            ), response
        else:
            assert response["output"] == [], response
        assert response["status_details"] == (
            None
            if finish == "stop"
            else {"type": "incomplete", "reason": "max_output_tokens"}
        ), response
        return response

    def message(self, identity, content):
        self.send(
            "conversation.item.create",
            item={
                "id": identity,
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": content}],
            },
        )
        item = self.until("conversation.item.done")[-1]["item"]
        assert item["id"] == identity and item["content"][0]["text"] == content, item

    def close(self):
        self.input.close()
        self.socket.close()


def check_response_metadata(events, metadata):
    responses = [
        e["response"]
        for e in events
        if e["type"] in ("response.created", "response.done")
    ]
    assert len(responses) == 2, events
    assert responses[0]["id"] == responses[1]["id"], responses
    assert all(r["metadata"] == metadata for r in responses), responses
