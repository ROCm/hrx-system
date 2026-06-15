# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: Python target contract fragment -> C lower-rule ABI table."""

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

from loom.gen.target.contracts.lower_rules import write_lower_rule_set_to_paths  # noqa: E402
from loom.target.contract_fragments import resolve_contract_fragment  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate target-low lower-rule C tables from Python data.")
    parser.add_argument(
        "--contract-fragment",
        required=True,
        metavar="KEY",
        help="Contract fragment key or alias to generate.",
    )
    parser.add_argument(
        "--header",
        required=True,
        type=Path,
        help="Generated lower-rule header path.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Generated lower-rule source path.",
    )
    args = parser.parse_args(argv)

    try:
        registration = resolve_contract_fragment(args.contract_fragment)
    except ValueError as exc:
        parser.error(str(exc))
    write_lower_rule_set_to_paths(
        registration.load(),
        dialect_ops=registration.load_dialect_ops(),
        header_path=args.header,
        source_path=args.source,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
