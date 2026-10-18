# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Multipart transcription through the real HTTP server and source-JIT model."""

import argparse
import http.client
import json
import struct
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from loom_serve_testing.http_harness import control, running_server, wait_status

parser = argparse.ArgumentParser()
parser.add_argument("server", type=Path)
parser.add_argument("model", type=Path)
parser.add_argument("audio_fixture", type=Path)
for codec in ("wav", "mp3", "flac"):
    parser.add_argument("--" + codec, action="store_true")
options, test_arguments = parser.parse_known_args()
SERVER = options.server.resolve()
PACKAGE = options.model.resolve().parent
AUDIO_FIXTURES = options.audio_fixture.resolve().parent
PCM_TYPE = "audio/pcm;rate=16000;channels=1"


def pcm(frames):
    return struct.pack("<h", 8192) * frames


def wave(frames, rate=16000, channels=1):
    samples = pcm(frames * channels)
    return (
        struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + len(samples),
            b"WAVE",
            b"fmt ",
            16,
            1,
            channels,
            rate,
            rate * channels * 2,
            channels * 2,
            16,
            b"data",
            len(samples),
        )
        + samples
    )


def multipart(audio, fields, *, file_first=False, audio_type=PCM_TYPE):
    boundary = "loom-transcription-test"
    parts = [
        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        + value.encode()
        for name, value in fields
    ]
    # A filename is opaque. Containers carry geometry; raw PCM declares it.
    file = (
        b'Content-Disposition: form-data; name="file"; filename="../../voice.bin"'
        + f"\r\nContent-Type: {audio_type}\r\n\r\n".encode()
        + audio
    )
    parts.insert(0 if file_first else len(parts), file)
    marker = f"--{boundary}".encode()
    body = b"".join(marker + b"\r\n" + part + b"\r\n" for part in parts)
    return body + marker + b"--\r\n", f"multipart/form-data; boundary={boundary}"


class TranscriptionClientTest(unittest.TestCase):
    def request(
        self,
        address,
        audio,
        fields=(),
        *,
        chunked=False,
        file_first=False,
        audio_type=PCM_TYPE,
        expected=200,
        text=False,
    ):
        body, content_type = multipart(
            audio,
            [("model", "recognition"), *fields],
            file_first=file_first,
            audio_type=audio_type,
        )
        connection = http.client.HTTPConnection(address.hostname, address.port)
        try:
            # Deliberately split headers, boundaries, and audio frames across
            # transfer chunks, as an asynchronous multipart client may do.
            payload = (
                (body[index : index + 17] for index in range(0, len(body), 17))
                if chunked
                else body
            )
            connection.request(
                "POST",
                "/v1/audio/transcriptions",
                payload,
                {"Content-Type": content_type},
                encode_chunked=chunked,
            )
            response = connection.getresponse()
            result = response.read()
            self.assertEqual(response.status, expected, result)
            self.assertEqual(
                response.getheader("Content-Type"),
                "text/plain; charset=utf-8" if text else "application/json",
            )
            return result.decode() if text else json.loads(result)
        finally:
            connection.close()

    def test_uploads_share_bounded_recognition(self):
        for workers in (1, 4):
            with (
                self.subTest(workers=workers),
                tempfile.TemporaryDirectory() as temporary,
            ):
                directory = Path(temporary)
                declaration = {
                    "name": "recognition",
                    "admission": "drained",
                    "base_directory": str(PACKAGE),
                    "definition": {
                        "kind": "recognition",
                        "prepare": "prepare.loom",
                        "control": "control.loom",
                        "sources": ["generate.loom"],
                        "assets": str(PACKAGE),
                        "pending_requests": 8,
                    },
                }
                (directory / "model.json").write_text(json.dumps(declaration))
                log = directory / "server.log"
                with running_server(
                    [
                        str(SERVER),
                        "--model=model.json",
                        "--device=task",
                        "--pool_backing=fixed",
                        f"--workers={workers}",
                        "--listen=127.0.0.1:0",
                    ],
                    log,
                    cwd=directory,
                ) as (process, bound):
                    self.assertIsNotNone(bound, log.read_text())
                    address = urlsplit(f"http://{bound}")
                    selector = json.dumps({"model": "recognition"})
                    primary = json.dumps(
                        {"model": "recognition", "component": "primary"}
                    )
                    state = control(
                        address, "POST", "/v1/models/pin", 200, body=selector
                    )["loom"]
                    self.assertEqual(state["registration"], "registered")
                    self.assertEqual(set(state["parameters"]), {"primary"})
                    self.assertTrue(state["parameters"]["primary"]["pinned"])
                    self.assertNotIn("active", state["parameters"]["primary"])
                    self.assertNotIn("compilation", state)
                    for operation in ("pin", "unpin", "activate", "deactivate"):
                        control(
                            address,
                            "POST",
                            f"/v1/models/{operation}",
                            404,
                            body=json.dumps(
                                {"model": "recognition", "component": "image"}
                            ),
                        )

                    # Explicit warmup prepares this primary component without
                    # opening admission or requiring an inference request.
                    control(address, "POST", "/v1/models/activate", 202, body=primary)
                    while True:
                        state = control(address, "GET", "/v1/status", 200)["data"][0][
                            "loom"
                        ]
                        self.assertNotIn("error", state)
                        if state["parameter_operation"] == "idle":
                            break
                    self.assertEqual(state["registration"], "ready")
                    self.assertEqual(state["status"]["admission"], "drained")
                    self.assertTrue(state["parameters"]["primary"]["pinned"])
                    self.assertTrue(state["parameters"]["primary"]["active"])
                    state = control(
                        address, "POST", "/v1/models/unpin", 200, body=primary
                    )["loom"]
                    self.assertFalse(state["parameters"]["primary"]["pinned"])
                    # Fixed backing cannot be released through parameter trim.
                    control(address, "POST", "/v1/models/deactivate", 409, body=primary)
                    control(address, "POST", "/v1/models/undrain", 200, body=selector)
                    wait_status(address, model="recognition", phase="idle")
                    model = control(address, "GET", "/v1/models", 200)["data"][0]
                    self.assertEqual(model["id"], "recognition")
                    self.assertEqual(model["input_modalities"], ["audio"])
                    self.assertEqual(model["output_modalities"], ["text", "activity"])

                    # Different durations and output forms cross conversion,
                    # input-credit, and three-frame source work boundaries.
                    lengths = (1, 2, 3, 7, 19, 511, 1607, 2048)
                    with ThreadPoolExecutor(max_workers=8) as clients:
                        requests = [
                            clients.submit(
                                self.request,
                                address,
                                pcm(length),
                                [("response_format", "text" if index % 2 else "json")],
                                chunked=index % 3 == 0,
                                file_first=index % 2 == 0,
                                text=index % 2 == 1,
                            )
                            for index, length in enumerate(lengths)
                        ]
                        for index, (length, request) in enumerate(
                            zip(lengths, requests)
                        ):
                            expected = chr(ord("A") + (length // 4) % 26)
                            self.assertEqual(
                                request.result(),
                                expected if index % 2 else {"text": expected},
                            )

                    # A mixed cohort exercises independent codec and rate
                    # conversion histories. FLAC metadata is larger than the
                    # model's raw PCM capacity, but its decoded frames fit.
                    uploads = [
                        (pcm(19), PCM_TYPE, True, "E"),
                        (pcm(27), "audio/pcm;rate=24000;channels=1", True, "E"),
                        (pcm(9), "audio/pcm;rate=8000;channels=1", True, "E"),
                        (wave(1607), "audio/wav", options.wav, "L"),
                        (wave(19, channels=2), "audio/wav", options.wav, "E"),
                        (wave(1607), "application/octet-stream", options.wav, "L"),
                        (
                            (AUDIO_FIXTURES / "sample.flac").read_bytes(),
                            "audio/flac",
                            options.flac,
                            "D",
                        ),
                        (
                            (AUDIO_FIXTURES / "sample.mp3").read_bytes(),
                            "audio/mpeg",
                            options.mp3,
                            "D",
                        ),
                    ]
                    with ThreadPoolExecutor(max_workers=8) as clients:
                        requests = [
                            clients.submit(
                                self.request,
                                address,
                                audio,
                                audio_type=audio_type,
                                chunked=True,
                                expected=200 if enabled else 400,
                            )
                            for audio, audio_type, enabled, _ in uploads
                        ]
                        for (_, _, enabled, expected), request in zip(
                            uploads, requests
                        ):
                            result = request.result()
                            if enabled:
                                self.assertEqual(result, {"text": expected})
                            else:
                                self.assertIn("error", result)
                    self.request(
                        address, wave(2049), audio_type="audio/wav", expected=400
                    )

                    for fields in (
                        [("stream", "true")],
                        [("temperature", "nan")],
                        [("response_format", "verbose_json")],
                        [("prompt", ""), ("prompt", "again")],
                        [("unknown", "value")],
                    ):
                        result = self.request(address, pcm(7), fields, expected=400)
                        self.assertIn("error", result)
                    for audio in (b"\x00", pcm(2049)):
                        self.assertIn(
                            "error", self.request(address, audio, expected=400)
                        )
                    self.request(
                        address,
                        pcm(1025),
                        audio_type="audio/pcm;rate=8000;channels=1",
                        expected=400,
                    )
                    # Rejections release both host and model capacity.
                    self.assertEqual(
                        self.request(address, pcm(19), [("stream", "false")]),
                        {"text": "E"},
                    )
                    for audio_type in (
                        "audio/pcm",
                        "audio/pcm;rate=0;channels=1",
                        "audio/pcm;rate=16000;channels=2",
                        "audio/pcm;rate=16000;channels=1;format=f32le",
                    ):
                        self.request(
                            address, pcm(19), audio_type=audio_type, expected=400
                        )
                    self.request(
                        address,
                        b"not an audio file",
                        audio_type="application/octet-stream",
                        expected=400,
                    )
                    self.request(address, pcm(19), audio_type="audio/wav", expected=400)
                    # Signed conversion includes both extrema and cancellation
                    # to exact zero; every prefix visible to source is nonnegative.
                    signed = struct.pack("<6h", 32767, 1, -32768, 16384, -16384, 8192)
                    self.assertEqual(self.request(address, signed * 64), {"text": "Q"})
                self.assertEqual(process.returncode, 0, log.read_text())


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *test_arguments])
