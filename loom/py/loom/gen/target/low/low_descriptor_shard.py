# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: checked-in target-low descriptor set -> C descriptor tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[4]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.target.low.low_descriptors import write_descriptor_set_to_paths  # noqa: E402
from loom.target.descriptor_sets import resolve_descriptor_set  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate target-low descriptor C tables from checked-in descriptor data.")
    parser.add_argument(
        "--descriptor-set",
        required=True,
        metavar="KEY",
        help="Descriptor set key or explicit shard alias to generate.",
    )
    parser.add_argument(
        "--header",
        required=True,
        type=Path,
        help="Generated descriptor header path.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Generated descriptor source path.",
    )
    args = parser.parse_args(argv)

    try:
        descriptor_set = resolve_descriptor_set(args.descriptor_set)
    except ValueError as exc:
        parser.error(str(exc))
    write_descriptor_set_to_paths(
        descriptor_set,
        header_path=args.header,
        source_path=args.source,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
