# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Writes one scalar checkpoint shard for the weighted text fixture."""

import argparse
import json
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True)
    parser.add_argument("--value", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = {args.key: {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}}
    header = json.dumps(metadata, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    args.output.write_bytes(
        struct.pack("<Q", len(header)) + header + struct.pack("<i", args.value)
    )


if __name__ == "__main__":
    main()
