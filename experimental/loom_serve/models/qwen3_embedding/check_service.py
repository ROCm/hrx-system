#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check a ready embedding deployment through its public HTTP contracts."""

import argparse
import base64
import json
import math
import struct
import urllib.error
import urllib.request


def check(endpoint: str, model: str) -> None:
    def request(path, body, status=200):
        message = urllib.request.Request(
            endpoint.rstrip("/") + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(message)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            payload = response.read()
            if response.status != status:
                raise AssertionError((path, response.status, status, payload))
            return json.loads(payload)

    def embed(inputs, dimensions=1024, encoding="base64"):
        value = request(
            "/v1/embeddings",
            {
                "model": model,
                "input": inputs,
                "dimensions": dimensions,
                "encoding_format": encoding,
            },
        )
        vectors = []
        for index, row in enumerate(value["data"]):
            assert row["index"] == index
            if encoding == "base64":
                raw = base64.b64decode(row["embedding"], validate=True)
                vector = struct.unpack(f"<{dimensions}f", raw)
            else:
                vector = row["embedding"]
            assert len(vector) == dimensions and all(map(math.isfinite, vector))
            assert abs(math.sqrt(sum(x * x for x in vector)) - 1.0) < 2e-6
            vectors.append(tuple(vector))
        return vectors, value["usage"]

    texts = [
        "A small embedding model for document retrieval.",
        "The library is open on Saturday.",
        "Bonjour, le monde.",
    ]
    tokens = [
        request("/tokenize", {"model": model, "content": text, "add_special": True})[
            "tokens"
        ]
        for text in texts
    ]
    batched, usage = embed(texts)
    assert len(batched) == len(texts)
    count = sum(map(len, tokens))
    assert usage["prompt_tokens"] == usage["total_tokens"] == count
    for text, ids, expected in zip(texts, tokens, batched):
        single, _ = embed(text)
        explicit, _ = embed(ids)
        assert single == explicit == [expected]
    permuted, _ = embed(list(reversed(texts)))
    assert permuted == list(reversed(batched))
    encoded, _ = embed(texts[0], encoding="float")
    for actual, expected in zip(encoded[0], batched[0]):
        assert abs(actual - expected) <= 1e-8
    shortened, _ = embed(texts[0], dimensions=64)
    prefix = batched[0][:64]
    norm = math.sqrt(sum(value * value for value in prefix))
    assert (
        max(
            abs(actual - expected / norm)
            for actual, expected in zip(shortened[0], prefix)
        )
        < 2e-6
    )
    # A complete cohort exercises all sixteen independently masked rows.
    sixteen, usage = embed([[1]] * 16)
    assert len(sixteen) == 16 and all(row == sixteen[0] for row in sixteen)
    assert usage["prompt_tokens"] == usage["total_tokens"] == 16
    for body in (
        {"input": [1] * 129},
        {"input": [[1]] * 17},
        {"input": [151669]},
        {"input": [1], "dimensions": 1025},
    ):
        request("/v1/embeddings", {"model": model, **body}, status=400)
    print(
        "Embedding contracts pass: tokenization, input forms, cohort isolation, "
        "row order, dimensions, encodings, usage, and rejection boundaries."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen3-embedding-0.6b")
    arguments = parser.parse_args()
    check(arguments.url, arguments.model)


if __name__ == "__main__":
    main()
