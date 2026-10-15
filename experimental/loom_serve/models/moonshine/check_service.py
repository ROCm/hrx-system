#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare concurrent HTTP transcriptions with exact reference text.

The manifest is a JSON array of {audio, text, content_type} records. Audio paths
are relative to the manifest; content_type may be omitted for WAV, MP3, or FLAC.
Raw PCM records declare audio/pcm;rate=16000;channels=1 explicitly. References
describe the exact supplied audio, including any preprocessing or quantization.
"""

import argparse
import http.client
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

CONTENT_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac"}


def transcribe(base_url, model, audio, content_type, text):
    address = urlsplit(base_url)
    if address.scheme not in ("http", "https") or not address.hostname:
        raise ValueError("--base-url must be an HTTP or HTTPS API URL")
    boundary = "moonshine-" + uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
            f"{model}\r\n--{boundary}\r\nContent-Disposition: form-data; "
            f'name="file"; filename="recording{audio.suffix}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
        + audio.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    connection_type = (
        http.client.HTTPSConnection
        if address.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(address.hostname, address.port)
    try:
        connection.request(
            "POST",
            address.path.rstrip("/") + "/audio/transcriptions",
            (body[index : index + 1031] for index in range(0, len(body), 1031)),
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
            encode_chunked=True,
        )
        response = connection.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(f"{audio}: HTTP {response.status}: {payload!r}")
        actual = json.loads(payload)
        if actual != {"text": text}:
            raise AssertionError(f"{audio}: expected {text!r}, received {actual!r}")
        return {"audio": str(audio), **actual}
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="moonshine-streaming-tiny")
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    records = json.loads(arguments.manifest.read_text())
    if not isinstance(records, list) or not records:
        parser.error("the manifest must contain a nonempty array")
    inputs = []
    for record in records:
        audio = (arguments.manifest.parent / record["audio"]).resolve(strict=True)
        content_type = record.get("content_type", CONTENT_TYPES.get(audio.suffix))
        if not content_type:
            parser.error(f"{audio}: specify content_type for this input format")
        inputs.append((audio, content_type, record["text"]))
    with ThreadPoolExecutor(max_workers=arguments.workers) as clients:
        results = clients.map(
            lambda row: transcribe(arguments.base_url, arguments.model, *row), inputs
        )
        for result in results:
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
