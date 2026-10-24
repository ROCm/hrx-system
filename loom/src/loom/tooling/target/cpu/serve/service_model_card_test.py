# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks portable service packages against the source fixtures' dense oracles."""

import base64
import http.client
import json
import struct
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from catalog_test_util import Client, check_pcm, check_png
from catalog_test_util import check_decision as check_decision_result
from loom_serve_testing.http_harness import running_server


def check_image(client, model):
    def request(index):
        prompt = "ab" + "a" * (index % 2)
        result = client.request(
            "/v1/images/generations",
            {
                "model": model,
                "prompt": prompt,
                "seed": index,
                "response_format": "b64_json",
            },
        )
        check_png(result["data"][0]["b64_json"], prompt, index)

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(request, range(8)))
    assert client.request("/tokenize", {"model": model, "content": "abxa"}) == {
        "tokens": [0, 1, 2, 0]
    }
    # Adapter selection belongs to each request, independently of the card.
    result = client.request(
        "/v1/images/generations",
        {
            "model": model,
            "prompt": "aba",
            "seed": 1,
            "loras": [{"id": "red", "scale": -2}],
        },
    )
    check_png(result["data"][0]["b64_json"], "aba", 0)


def check_embedding(client, model):
    def request(index):
        inputs = ["a" * (index + 1), "bax", "xx"]
        result = client.request(
            "/v1/embeddings",
            {"model": model, "input": inputs, "dimensions": 2},
        )
        assert [row["index"] for row in result["data"]] == [0, 1, 2], result
        assert [row["embedding"] for row in result["data"]] == [
            [len(value) + sum("abx".index(token) for token in value), len(value)]
            for value in inputs
        ], result
        assert result["usage"]["prompt_tokens"] == sum(map(len, inputs)), result

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(request, range(8)))
    assert client.request("/tokenize", {"model": model, "content": "abxa"}) == {
        "tokens": [0, 1, 2, 0]
    }


def check_decision(client, model):
    for _ in range(4):
        result = client.request(
            "/v1/systemone",
            {
                "model": model,
                "state": "a",
                "questions": {
                    "truth": {
                        "type": "noul",
                        "instructions": "???",
                        "criteria": {"false": "", "true": "x"},
                    }
                },
            },
        )
        check_decision_result(result)


def check_speech(client, model, *, tokenizer=True):
    def request(index):
        text = "ab" + "a" * index
        voice = "positive" if index % 2 else "negative"
        result = client.request(
            "/v1/audio/speech",
            {
                "model": model,
                "input": text,
                "voice": voice,
                "seed": index,
                "response_format": "pcm",
            },
        )
        check_pcm(result, text, index, voice)

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(request, range(8)))
    if tokenizer:
        assert client.request("/tokenize", {"model": model, "content": "abba"}) == {
            "tokens": [0, 1, 1, 0]
        }


def check_speech_voice(client, model):
    def upload(voice, value, expected=200):
        boundary = "voice-boundary"
        body = (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="id"\r\n\r\n{voice}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="sample"\r\n'
                "Content-Type: audio/pcm;rate=44100;channels=2;format=f32le\r\n\r\n"
            ).encode()
            + struct.pack("<8f", *([value] * 8))
            + f"\r\n--{boundary}--\r\n".encode()
        )
        connection = http.client.HTTPConnection(client.address)
        try:
            connection.request(
                "POST",
                "/v1/audio/voices",
                body,
                {"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
            response = connection.getresponse()
            result = response.read()
            assert response.status == expected, (response.status, result)
            if expected == 200:
                assert json.loads(result) == {"id": voice}, result
        finally:
            connection.close()

    upload("positive", 0.125)
    upload("negative", -0.125)
    upload("positive", 0.25, 409)
    upload("third", 0.25, 503)
    check_speech(client, model, tokenizer=False)
    # Removing voices remains available while inference admission is drained.
    client.request("/v1/models/drain", {"model": model})
    for voice in ("positive", "negative"):
        assert client.request(
            "/v1/audio/voices/remove", {"model": model, "id": voice}
        ) == {"id": voice}
    client.request(
        "/v1/audio/voices/remove", {"model": model, "id": "missing"}, status=404
    )
    client.request("/v1/models/undrain", {"model": model})
    upload("negative", -0.125)
    upload("positive", 0.125)
    check_speech(client, model, tokenizer=False)


def check_sequence(client, model):
    history = []
    handle = None
    for tokens in ([0, 1, 2, 0, 1, 1, 0, 0, 2, 1, 0], [1], [2, 0]):
        position = len(history)
        body = {"model": model, "position": position, "token_ids": tokens}
        if handle:
            body["session"] = handle
        connection = http.client.HTTPConnection(client.address)
        try:
            connection.request(
                "POST",
                "/v1/sequences",
                json.dumps(body),
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            payload = response.read().decode()
            assert response.status == 200, payload
            assert response.getheader("Content-Type") == "text/event-stream"
        finally:
            connection.close()
        events = []
        for frame in payload.strip().split("\n\n"):
            event, data = frame.splitlines()
            assert event.startswith("event: ") and data.startswith("data: "), frame
            events.append((event[7:], json.loads(data[6:])))
        history.extend(tokens)
        assert events[0][0] == "attached", events
        attached = events[0][1]
        handle = handle or attached["session"]
        assert len(handle) == 32, attached
        assert attached == {"session": handle, "position": position}, attached
        for kind, value in events[1:-1]:
            assert kind == "chunk", events
            assert value["session"] == handle, value
            assert value["position"] == position, value
            end = value["next_position"]
            assert position < end <= len(history), value
            assert value["token_ids"] == history[position:end], value
            # Dense recurrence, independent of page storage and chunk geometry.
            hidden = 0
            for token in history[:end]:
                hidden = (5 * hidden + token + 2) % 97
            assert value["encoding"] == "base64", value
            assert base64.b64decode(value["data"], validate=True) == struct.pack(
                "<HH", hidden >> 16, hidden & 0xFFFF
            ), value
            position = end
        assert position == len(history), events
        assert events[-1] == (
            "done",
            {"session": handle, "position": len(history)},
        ), events


def check_service(server, kind, deployment, model):
    server = str(Path(server).absolute())
    deployment = str(Path(deployment).absolute())
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        log = directory / "server.log"
        with running_server(
            [
                server,
                "--device=task",
                "--workers=4",
                "--pool_backing=fixed",
                "--listen=127.0.0.1:0",
                "--model=" + deployment,
            ],
            log,
            cwd=directory,
        ) as (process, address):
            assert address is not None, log.read_text()
            client = Client(address)
            try:
                {
                    "image": check_image,
                    "embedding": check_embedding,
                    "decision": check_decision,
                    "speech": check_speech,
                    "speech_voice": check_speech_voice,
                    "sequence": check_sequence,
                }[kind](client, model)
            except (AssertionError, RuntimeError) as error:
                status = client.request("/v1/status")
                raise AssertionError(
                    f"{error}\nmodel status: {status}\n{log.read_text()}"
                ) from error
        assert process.returncode == 0, log.read_text()


def main(server, kind, deployment):
    deployment = Path(deployment).absolute()
    for path in (deployment, deployment.parent / "model.json"):
        document = json.loads(path.read_text())
        model = (
            document["model"]["id"]
            if "schema_version" in document
            else document["name"]
        )
        check_service(server, kind, str(path), model)


if __name__ == "__main__":
    main(*sys.argv[1:])
