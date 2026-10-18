# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native PCM conversation through the production CLI and WebSocket endpoint."""

import base64
import json
import os
import socket
import struct
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

from loom_serve_testing.http_harness import running_server

from loom.src.loom.serve.server.catalog_test_util import (
    Client as HttpClient,
)
from loom.src.loom.serve.server.catalog_test_util import (
    check_pcm,
    complete,
    prepare_catalog,
    write_startup,
)


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
        self.send_frame(1, json.dumps({"type": kind, **fields}).encode())

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

    def respond(self, messages, maximum=2, voice="positive"):
        self.send("response.create", response={"max_output_tokens": maximum})
        events = self.until("response.done")
        text = "".join(
            e["delta"]
            for e in events
            if e["type"] == "response.output_audio_transcript.delta"
        )
        expected, finish = complete(messages, maximum)
        assert text == expected, (text, expected, events)
        audio = b"".join(
            base64.b64decode(e["delta"], validate=True)
            for e in events
            if e["type"] == "response.output_audio.delta"
        )
        check_pcm(audio, text, 42, voice)
        response = events[-1]["response"]
        status = "completed" if finish == "stop" else "incomplete"
        assert response["status"] == status, response
        item = response["output"][0]
        assert item["status"] == status, item
        assert item["content"][0]["transcript"] == text, response
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


def conversation(address, trace):
    client = Client(address, trace)
    try:
        created = client.receive()
        assert created["type"] == "session.created", created
        settings = created["session"]
        assert settings["audio"]["input"]["turn_detection"] is None, settings
        assert settings["audio"]["input"]["format"] == {
            "type": "audio/pcm",
            "rate": 24000,
        }, settings
        for malformed in (
            '{"type":"session.update","type":"response.create","session":{}}',
            '{"type":"session.update","session":{},"se\\u0073sion":{}}',
            '{"type":"session.update","session":{"instructions":"x","max_output_tokens":0}}',
            '{"type":"input_audio_buffer.append","audio":null}',
            '{"type":"response.create","response":{"output_modalities":["audio","text"]}}',
            '{"type":"input_audio_buffer.clear"} {}',
        ):
            client.send_frame(1, malformed.encode())
            assert client.receive(allow_error=True)["type"] == "error", malformed
        client.send("session.update", session={})
        assert client.until("session.updated")[-1]["session"] == settings
        # The 24->16 kHz converter must flush its final lookahead on commit,
        # even when no more microphone frames ever arrive on this connection.
        for count in (1, 7, 93, 139):
            client.send(
                "input_audio_buffer.append",
                audio=base64.b64encode(bytes(2 * count)).decode(),
            )
        client.send("input_audio_buffer.commit")
        events = client.until("conversation.item.input_audio_transcription.completed")
        transcript = events[-1]
        assert transcript["transcript"] == "A", transcript
        assert transcript["usage"] == {"type": "duration", "seconds": 0.01}, transcript
        assert sum(e["type"] == "input_audio_buffer.committed" for e in events) == 1
        assert not any(e["type"] == "response.created" for e in events), events
        messages = [{"role": "user", "content": "A"}]
        response = client.respond(messages)
        assert response["usage"]["input_tokens"] == 1, response
        messages.append(
            {
                "role": "assistant",
                "content": response["output"][0]["content"][0]["transcript"],
            }
        )
        # Voice policy cannot invalidate compatible text history. A later
        # truncation removes only the unaligned transcript of the heard item.
        client.send(
            "session.update", session={"audio": {"output": {"voice": "negative"}}}
        )
        updated = client.until("session.updated")[-1]["session"]
        assert updated["audio"]["output"]["voice"] == "negative", updated
        client.message("second", "aB")
        messages.append({"role": "user", "content": "aB"})
        second = client.respond(messages, maximum=16, voice="negative")
        assert second["usage"]["input_token_details"]["cached_tokens"] > 0, second
        second_item = second["output"][0]
        assert len(second_item["content"][0]["transcript"]) >= 3, second_item
        client.send(
            "conversation.item.truncate",
            item_id=second_item["id"],
            content_index=0,
            audio_end_ms=1,
        )
        truncated = client.until("conversation.item.truncated")[-1]
        assert (
            truncated["item_id"] == second_item["id"] and truncated["audio_end_ms"] == 1
        ), truncated
        client.message("third", "X")
        messages.append({"role": "user", "content": "X"})
        third = client.respond(messages, voice="negative")
        assert third["usage"]["input_token_details"]["cached_tokens"] > 0, third
        admin = HttpClient(address)
        admin.request("/v1/models/drain", {"model": "fixture-text"})
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
        # Close must join checkpoint release even while inference is drained.
        assert client.input.read(1) == b""
        admin.request("/v1/models/undrain", {"model": "fixture-text"})
    finally:
        client.close()


def cancellation(address, trace):
    admin = HttpClient(address)
    admin.request("/v1/models/drain", {"model": "fixture-text"})
    client = Client(address, trace)
    try:
        assert client.receive()["type"] == "session.created"
        client.message("pending", "a")
        client.send("response.create")
        events = client.until("response.content_part.added")
        identity = next(
            e["response"]["id"] for e in events if e["type"] == "response.created"
        )
        # Drained native admission is an explicit readiness boundary: this
        # cancellation cannot race with a fixture that already completed.
        client.send("response.cancel", response_id=identity)
        events = client.until("response.done")
        response = events[-1]["response"]
        assert response["id"] == identity and response["status"] == "cancelled", (
            response
        )
        assert not any(e["type"].endswith(".delta") for e in events), events
        client.send("conversation.item.delete", item_id=response["output"][0]["id"])
        client.until("conversation.item.deleted")
        admin.request("/v1/models/undrain", {"model": "fixture-text"})
        client.respond([{"role": "user", "content": "a"}])
        client.send("input_audio_buffer.append", audio="AAAAAA==")
        client.send("input_audio_buffer.clear")
        client.until("input_audio_buffer.cleared")
        client.send("input_audio_buffer.commit", event_id="empty-commit")
        error = client.receive(allow_error=True)
        assert (
            error["type"] == "error" and error["error"]["event_id"] == "empty-commit"
        ), error
        client.send(
            "input_audio_buffer.append",
            audio=base64.b64encode(bytes(480)).decode(),
        )
        client.send("input_audio_buffer.commit")
        transcript = client.until(
            "conversation.item.input_audio_transcription.completed"
        )[-1]
        assert transcript["transcript"] == "A", transcript
        assert transcript["usage"] == {"type": "duration", "seconds": 0.01}, transcript
        client.until("conversation.item.done")
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        client.close()


def pipelined_microphone_items(address, trace):
    client = Client(address, trace)
    admin = HttpClient(address)
    drained = False
    try:
        assert client.receive()["type"] == "session.created"
        admin.request("/v1/models/drain", {"model": "recognition"})
        drained = True
        identities = []
        for frames in (240, 480):
            client.send(
                "input_audio_buffer.append",
                audio=base64.b64encode(bytes(frames * 2)).decode(),
            )
            client.send("input_audio_buffer.commit")
            committed = client.until("input_audio_buffer.committed")[-1]
            identities.append(committed["item_id"])
        admin.request("/v1/models/undrain", {"model": "recognition"})
        drained = False
        transcripts = {}
        done = set()
        while len(transcripts) != 2 or len(done) != 2:
            event = client.receive()
            if event["type"] == "conversation.item.input_audio_transcription.completed":
                transcripts[event["item_id"]] = event
            elif event["type"] == "conversation.item.done":
                done.add(event["item"]["id"])
        for identity, seconds in zip(identities, (0.01, 0.02), strict=True):
            assert transcripts[identity]["transcript"] == "A", transcripts
            assert transcripts[identity]["usage"] == {
                "type": "duration",
                "seconds": seconds,
            }, transcripts
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        if drained:
            admin.request("/v1/models/undrain", {"model": "recognition"})
        client.close()


def automatic_microphone(address, trace, *, partitioned, create_response):
    client = Client(address, trace)
    try:
        assert client.receive()["type"] == "session.created"
        for invalid in (
            {"type": "semantic_vad"},
            {"type": "server_vad", "threshold": 1.1},
            {"type": "server_vad", "prefix_padding_ms": 2000},
            {"type": "server_vad", "silence_duration_ms": -1},
            {"type": "server_vad", "idle_timeout_ms": 1000},
        ):
            client.send(
                "session.update",
                session={"audio": {"input": {"turn_detection": invalid}}},
            )
            assert client.receive(allow_error=True)["type"] == "error", invalid
        policy = {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 1,
            "silence_duration_ms": 1,
            "create_response": create_response,
            "interrupt_response": True,
        }
        client.send(
            "session.update", session={"audio": {"input": {"turn_detection": policy}}}
        )
        updated = client.until("session.updated")[-1]["session"]
        assert updated["audio"]["input"]["turn_detection"] == policy, updated
        samples = [0] * 96 + [64] * 192 + [0] * 480
        if not create_response:
            samples *= 2
        raw = struct.pack(f"<{len(samples)}h", *samples)
        position = 0
        parts = [1, 7, 93] if partitioned else [len(samples)]
        while position < len(raw):
            count = parts.pop(0) if parts else 137
            payload = raw[position : position + count * 2]
            client.send(
                "input_audio_buffer.append", audio=base64.b64encode(payload).decode()
            )
            position += len(payload)
        count = 1 if create_response else 2
        starts, stops, transcripts, done, events = {}, {}, {}, set(), []
        response_done = False
        while len(done) != count or (create_response and not response_done):
            event = client.receive()
            events.append(event)
            kind = event["type"]
            response_done |= kind == "response.done"
            if kind == "input_audio_buffer.speech_started":
                starts[event["item_id"]] = event["audio_start_ms"]
            elif kind == "input_audio_buffer.speech_stopped":
                stops[event["item_id"]] = event["audio_end_ms"]
            elif kind == "conversation.item.input_audio_transcription.completed":
                transcripts[event["item_id"]] = event
            elif kind == "conversation.item.done" and event["item"]["role"] == "user":
                done.add(event["item"]["id"])
        assert len(starts) == len(stops) == len(transcripts) == count, events
        intervals = []
        for identity, begin in starts.items():
            end = stops[identity]
            assert 0 <= begin < end, events
            assert transcripts[identity]["transcript"] == "A", events
            intervals.append((begin, end, transcripts[identity]["usage"]["seconds"]))
        assert all(
            left[1] <= right[0] for left, right in zip(intervals, intervals[1:])
        ), intervals
        messages = [{"role": "user", "content": "A"}] * count
        if create_response:
            text = "".join(
                e["delta"]
                for e in events
                if e["type"] == "response.output_audio_transcript.delta"
            )
            assert text == complete(messages, 2)[0], events
            audio = b"".join(
                base64.b64decode(e["delta"])
                for e in events
                if e["type"] == "response.output_audio.delta"
            )
            check_pcm(audio, text, 42)
        else:
            assert not any(e["type"] == "response.created" for e in events), events
            client.respond(messages)
        # Changing classified input policy requires an explicit ordered cut.
        client.send(
            "session.update", session={"audio": {"input": {"turn_detection": None}}}
        )
        assert client.receive(allow_error=True)["type"] == "error"
        client.send("input_audio_buffer.clear")
        client.until("input_audio_buffer.cleared")
        client.send(
            "session.update", session={"audio": {"input": {"turn_detection": None}}}
        )
        updated = client.until("session.updated")[-1]["session"]
        assert updated["audio"]["input"]["turn_detection"] is None, updated
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
        return intervals
    finally:
        client.close()


def automatic_explicit_boundaries(address, trace):
    client = Client(address, trace)
    try:
        assert client.receive()["type"] == "session.created"
        client.send(
            "session.update",
            session={
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            # Retain the full quiet interval for explicit commit
                            # regardless of concurrent detector progress.
                            "prefix_padding_ms": 20,
                            "silence_duration_ms": 100,
                            "create_response": True,
                        }
                    }
                }
            },
        )
        client.until("session.updated")
        raw = struct.pack("<288h", *([64] * 192 + [0] * 96))
        client.send("input_audio_buffer.append", audio=base64.b64encode(raw).decode())
        started = client.until("input_audio_buffer.speech_started")[-1]
        assert started["audio_start_ms"] == 0, started
        client.send("input_audio_buffer.commit")
        client.send("input_audio_buffer.append", audio=base64.b64encode(raw).decode())
        client.send("input_audio_buffer.clear")
        client.send(
            "input_audio_buffer.append", audio=base64.b64encode(bytes(960)).decode()
        )
        client.send("input_audio_buffer.commit")
        transcripts, events, done = [], [], set()
        while len(done) != 2:
            event = client.receive()
            events.append(event)
            if event["type"] == "conversation.item.input_audio_transcription.completed":
                transcripts.append(event)
            elif event["type"] == "conversation.item.done":
                done.add(event["item"]["id"])
        assert sorted(e["usage"]["seconds"] for e in transcripts) == [0.012, 0.02], (
            transcripts
        )
        assert all(e["transcript"] == "A" for e in transcripts), transcripts
        assert not any(e["type"] == "response.created" for e in events), events
        client.send(
            "session.update", session={"audio": {"input": {"turn_detection": None}}}
        )
        client.until("session.updated")
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        client.close()


def automatic_response_policy(address, trace, *, interrupt):
    client = Client(address, trace)
    admin = HttpClient(address)
    drained = False
    try:
        assert client.receive()["type"] == "session.created"
        client.send(
            "session.update",
            session={
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "prefix_padding_ms": 1,
                            "silence_duration_ms": 1,
                            "create_response": not interrupt,
                            "interrupt_response": interrupt,
                        }
                    }
                }
            },
        )
        client.until("session.updated")
        client.message("before", "a")
        admin.request("/v1/models/drain", {"model": "speech"})
        drained = True
        client.send("response.create")
        response = client.until("response.created")[-1]["response"]
        raw = struct.pack("<768h", *([0] * 96 + [64] * 192 + [0] * 480))
        client.send("input_audio_buffer.append", audio=base64.b64encode(raw).decode())
        events = client.until("conversation.item.input_audio_transcription.completed")
        assert events[-1]["transcript"] == "A", events
        assert any(e["type"] == "input_audio_buffer.speech_started" for e in events), (
            events
        )
        if interrupt:
            while not any(e["type"] == "response.done" for e in events):
                events.append(client.receive())
            cancelled = next(
                e["response"] for e in events if e["type"] == "response.done"
            )
            assert (
                cancelled["id"] == response["id"] and cancelled["status"] == "cancelled"
            ), events
            item = cancelled["output"][0]
            assert item["content"][0]["transcript"] is None, item
            client.send("response.create")
            client.until("response.created")
            # The old unaligned text is already absent. A late playback report
            # cannot cancel or invalidate the correctly composed successor.
            client.send(
                "conversation.item.truncate",
                item_id=item["id"],
                content_index=0,
                audio_end_ms=0,
            )
            client.until("conversation.item.truncated")
        else:
            assert not any(
                e["type"] in ("response.done", "response.created") for e in events
            ), events
        admin.request("/v1/models/undrain", {"model": "speech"})
        drained = False
        messages = [{"role": "user", "content": "a"}]
        if not interrupt:
            first = client.until("response.done")
            assert first[-1]["response"]["id"] == response["id"], first
            assert first[-1]["response"]["status"] == "incomplete", first
            text = first[-1]["response"]["output"][0]["content"][0]["transcript"]
            assert text == complete(messages, 2)[0], first
            messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": "A"})
        second = client.until("response.done")
        response = second[-1]["response"]
        assert response["status"] in ("completed", "incomplete"), response
        text = response["output"][0]["content"][0]["transcript"]
        assert text == complete(messages, 2)[0], response
        audio = b"".join(
            base64.b64decode(e["delta"])
            for e in second
            if e["type"] == "response.output_audio.delta"
        )
        check_pcm(audio, text, 42)
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        if drained:
            admin.request("/v1/models/undrain", {"model": "speech"})
        client.close()


def rejected_activity_geometry(address, trace):
    client = Client(address, trace, model="bounded")
    try:
        assert client.receive()["session"]["model"] == "bounded"
        client.send(
            "session.update",
            session={
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "prefix_padding_ms": 0,
                        }
                    }
                }
            },
        )
        client.until("session.updated")
        client.send("input_audio_buffer.append", audio="AAAAAA==")
        failure = client.receive(allow_error=True)
        assert failure["type"] == "error", failure
        assert failure["error"]["type"] == "server_error", failure
        assert "input_capacity" in failure["error"]["message"], failure
        assert client.read_frame() == (8, struct.pack(">H", 1011))
        assert client.input.read(1) == b""
        status = HttpClient(address).request("/v1/status")
        assert len(status["data"]) == 3, status
    finally:
        client.close()


def rejected_capture(address, trace):
    client = Client(address, trace)
    try:
        assert client.receive()["type"] == "session.created"
        client.message("capture", "a")
        client.send("response.create")
        while True:
            event = client.receive(allow_error=True)
            if event["type"] == "error":
                assert event["error"]["type"] == "server_error", event
                break
        # A profile without retained text cannot capture its completed prefix.
        # That rejected operation owns no pin and must not abort the server.
        assert client.read_frame() == (8, struct.pack(">H", 1011))
        assert client.input.read(1) == b""
        admin = HttpClient(address)
        status = admin.request("/v1/status")
        assert len(status["data"]) == 3, status
        result = admin.request(
            "/v1/chat/completions",
            admin.text_payload([{"role": "user", "content": "a"}], 2),
        )
        assert result["text"] == "bb", result
    finally:
        client.close()


def main():
    executable = str(Path(sys.argv[1]).resolve())
    fixtures = [Path(value).resolve() for value in sys.argv[2:5]]
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    artifacts = Path(
        os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]
        if "TEST_UNDECLARED_OUTPUTS_DIR" in os.environ
        else tempfile.mkdtemp()
    )
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        catalog, flags = prepare_catalog(fixtures, directory)
        text = next(m for m in catalog["models"] if m["definition"]["kind"] == "text")
        text["definition"].update(session_cache_capacity=8, checkpoint_capacity=8)
        write_startup(flags, catalog)
        profile = directory / "assistant.json"
        profile.write_text(
            json.dumps(
                {
                    "name": "assistant",
                    "text": "fixture-text",
                    "recognition": "recognition",
                    "activity": "recognition",
                    "speech": "speech",
                    "voice": "positive",
                    "max_output_tokens": 2,
                    "seed": 42,
                }
            )
        )
        bounded = directory / "bounded.json"
        bounded.write_text(
            json.dumps(
                {
                    **json.loads(profile.read_text()),
                    "name": "bounded",
                    "input_capacity": 4096,
                }
            )
        )
        command = [
            executable,
            f"--flagfile={flags}",
            *sys.argv[6:],
            f"--workers={workers}",
            "--listen=127.0.0.1:0",
            "--heartbeat_ms=0",
            f"--realtime_assistant={profile}",
            f"--realtime_assistant={bounded}",
        ]
        with running_server(command, artifacts / "server.log") as (process, address):
            assert address is not None, process.poll()
            with (artifacts / "wire.jsonl").open("w") as trace:
                conversation(address, trace)
                cancellation(address, trace)
                pipelined_microphone_items(address, trace)
                packed = automatic_microphone(
                    address, trace, partitioned=False, create_response=False
                )
                partitioned = automatic_microphone(
                    address, trace, partitioned=True, create_response=False
                )
                assert packed == partitioned, (packed, partitioned)
                automatic_microphone(
                    address, trace, partitioned=True, create_response=True
                )
                automatic_explicit_boundaries(address, trace)
                automatic_response_policy(address, trace, interrupt=True)
                automatic_response_policy(address, trace, interrupt=False)
                rejected_activity_geometry(address, trace)
        text["definition"].update(session_cache_capacity=0, checkpoint_capacity=0)
        write_startup(flags, catalog)
        with running_server(command, artifacts / "rejection-server.log") as (
            process,
            address,
        ):
            assert address is not None, process.poll()
            with (artifacts / "rejection-wire.jsonl").open("w") as trace:
                rejected_capture(address, trace)


if __name__ == "__main__":
    main()
