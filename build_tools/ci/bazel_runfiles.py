# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reads the names Bazel writes in a runfiles manifest."""


def parse_manifest_line(line: str) -> tuple[str, str | None]:
    """Returns the logical name and target. An empty target creates an empty file."""
    line = line.rstrip("\r\n")
    escaped = line.startswith(" ")
    name, separator, target = line[escaped:].partition(" ")
    if not separator:
        return name, None
    if escaped:
        name = name.replace(r"\s", " ").replace(r"\n", "\n").replace(r"\b", "\\")
        target = target.replace(r"\n", "\n").replace(r"\b", "\\")
    return name, target
