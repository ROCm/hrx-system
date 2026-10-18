# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Produces a Windows export definition from the configured library objects."""

import argparse
import subprocess
from pathlib import Path


def export_declarations(text, prefixes, *, dumpbin=False):
    """Selects defined C symbols from the compiler's symbol-table output."""
    exports = {}
    for line in text.splitlines():
        if dumpbin:
            fields, separator, name = line.partition(" | ")
            if not separator or "External" not in fields or "UNDEF" in fields:
                continue
            name = name.split()[0]
            kind = "T" if "()" in fields else "D"
        else:
            fields = line.split()
            if len(fields) < 2:
                continue
            name, kind = fields[:2]
        if not name.startswith(tuple(prefixes)):
            continue
        if kind not in "TDBRSCVW":
            raise ValueError(f"unsupported exported symbol: {line}")
        declaration = name + ("" if kind in "TW" else " DATA")
        previous = exports.setdefault(name, declaration)
        if previous != declaration:
            raise ValueError(f"conflicting exported symbol kinds: {name}")
    if not exports:
        raise ValueError("the configured library defines no matching exports")
    return [exports[name] for name in sorted(exports)]


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    tool = parser.add_mutually_exclusive_group(required=True)
    tool.add_argument("--nm")
    tool.add_argument("--dumpbin")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", action="append", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    # Both symbol tools accept response files, avoiding Windows command limits.
    response = args.output.with_suffix(".nm.rsp")
    response.write_text(
        "\n".join(
            '"' + path.replace("\\", "/").replace('"', '\\"') + '"'
            for path in args.inputs
        ),
        encoding="utf-8",
    )
    try:
        command = (
            [args.dumpbin, "/nologo", "/symbols"]
            if args.dumpbin
            else [args.nm, "--defined-only", "--extern-only", "--format=posix"]
        )
        symbols = subprocess.check_output(
            command + ["@" + str(response)],
            text=True,
        )
    finally:
        response.unlink()

    exports = export_declarations(symbols, args.prefix, dumpbin=bool(args.dumpbin))
    args.output.write_text(
        "EXPORTS\n" + "\n".join("  " + declaration for declaration in exports) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
