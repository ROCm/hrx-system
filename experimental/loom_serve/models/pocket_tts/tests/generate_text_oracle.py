#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Records normalization output from the pinned Pocket TTS 3.3.0 reference.

Run in the reference environment with CPython 3.12 and pocket-tts==3.3.0.
Only text preparation is imported; no checkpoint is loaded.
"""

import importlib.metadata
import struct
import unicodedata
from pathlib import Path

from pocket_tts.models.text_chunking import prepare_text_prompt


def main() -> None:
    if (
        importlib.metadata.version("pocket-tts") != "3.3.0"
        or unicodedata.unidata_version != "15.0.0"
    ):
        raise SystemExit("The oracle requires Pocket TTS 3.3.0 and Unicode 15.0.0")
    examples = [
        "",
        " \t\r\n",
        "hello",
        "HELLO",
        "one two three four",
        "one two three four five",
        "a\x00b",
        "a\x1cb\x85c\u2009d\u3000e",
        "你好世界",
        "𝔞 message",
        "ǳ message",
        "hello\n\rworld",
        "hello\r\nworld",
        "hello\t\tworld",
        '"',
        " ) ] ” ’ » ",
        " ; - , : ",
        " , ; : ) ] ",
        "line\n\n\n\n\nends",
    ]
    examples.extend("a" + " " * count + "b" for count in range(1, 10))
    for point in range(0x110000):
        character = chr(point)
        if character.isspace():
            examples.append(character + "hi" + character + "there" + character)
        if len(character.upper()) > 1:
            examples.append(character + "eta")
    for ending in ".!?… ,;:-–—":
        for closer in ("", '"', "'", "”", "’", ")", "]", "»", " ” ]"):
            examples.append("hi" + ending + closer)
            examples.append("hi" + ending + " " + closer)
    inputs = bytearray()
    outputs = bytearray()
    rows = []
    for example in dict.fromkeys(examples):
        if example.strip():
            normalized, tail = prepare_text_prompt(example, False, False)
            tail += 2
        else:
            # Pure normalization exposes emptiness to request admission.
            normalized, tail = "", 5
        input_bytes = example.encode()
        output_bytes = normalized.encode()
        rows.append(
            (len(inputs), len(input_bytes), len(outputs), len(output_bytes), tail)
        )
        inputs.extend(input_bytes)
        outputs.extend(output_bytes)
    records = b"".join(struct.pack("<5I", *row) for row in rows)
    declarations = [f"global.constant @normalization_count : index = {len(rows)}"]
    for name, data in (("records", records), ("inputs", inputs), ("outputs", outputs)):
        if len(data) * 2 + 100 >= 65535:
            raise ValueError("Oracle literal exceeds the text format's source columns")
        declarations.append(
            f'global.rodata.def @normalization_{name} = align(4) bytes("{data.hex()}")'
        )
    path = Path(__file__).with_name("text_normalization.loom")
    prefix, body = path.read_text().split("// BEGIN GENERATED POCKET ORACLE\n", 1)
    _, suffix = body.split("// END GENERATED POCKET ORACLE", 1)
    path.write_text(
        prefix
        + "// BEGIN GENERATED POCKET ORACLE\n"
        + "\n\n".join(declarations)
        + "\n\n// END GENERATED POCKET ORACLE"
        + suffix
    )


if __name__ == "__main__":
    main()
