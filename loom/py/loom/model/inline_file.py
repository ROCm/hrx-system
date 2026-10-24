# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Materializes an inline model-card file as a declared build artifact."""

import argparse
import base64
from pathlib import Path

from loom.model import ManifestError, ModelCard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--preset")
    parser.add_argument("--service")
    parser.add_argument("--variant")
    parser.add_argument("--programs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = (
        ModelCard.parse(args.card.read_text(encoding="utf-8"))
        .select(
            args.preset,
            service=args.service,
            variant=args.variant,
            programs=args.programs,
        )
        .files
    )
    origin = files[args.file].origin if args.file in files else None
    if not isinstance(origin, dict) or origin.get("kind") != "inline":
        raise ManifestError(f"{args.file!r} is not a declared inline file")
    args.output.write_bytes(base64.b64decode(origin["base64"], validate=True))


if __name__ == "__main__":
    main()
