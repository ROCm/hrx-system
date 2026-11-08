# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executes source observations after public C++ export and reimport."""

import argparse
import re
import subprocess
from pathlib import Path


def invoke(tool: str, arguments: list[str], source: bytes) -> bytes:
    result = subprocess.run(
        [str(Path(tool).resolve()), *arguments],
        input=source,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"{tool} exited with {result.returncode}:\n"
            + result.stdout.decode("utf-8", errors="replace")
            + result.stderr.decode("utf-8", errors="replace")
        )
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("importer", "exporter", "checker"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--source", required=True, action="append")
    args = parser.parse_args()
    count = 0
    for source_path in args.source:
        text = Path(source_path).read_text(encoding="utf-8")
        cases = re.split(r"^// ====\s*$", text, flags=re.MULTILINE)
        for ordinal, case in enumerate(cases, 1):
            source, expected = case.split("// ----\n")
            try:
                bytecode = invoke(
                    args.importer,
                    ["--root=run", "--cleanup=false", "--to=bc", "-"],
                    source.encode("utf-8"),
                )
                exported = invoke(args.exporter, ["-"], bytecode)
                fixture = (
                    b"// RUN: emit vm-object-lifetime\n// INPUT: cxx root=run\n"
                    + exported
                    + b"\n// ----\n"
                    + expected.encode("utf-8")
                )
                invoke(args.checker, ["--input-format=cxx", "-"], fixture)
            except RuntimeError as error:
                raise RuntimeError(f"{source_path}, case {ordinal}: {error}") from error
            count += 1
    print(f"{count} exported programs preserve their observations")


if __name__ == "__main__":
    main()
