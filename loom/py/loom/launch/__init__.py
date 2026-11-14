# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Launcher for published Loom model profiles.

A profile is one published Loom package: a portable model card, its parameter
and program payloads, the chat policy its generator emitted, and the
measurements its publisher recorded. This package discovers profiles on a model
hub, installs them through the ordinary acquisition path, composes a deployment
document for the local machine, supervises the native server, and instruments
the OpenAI-compatible endpoint it serves.
"""

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Runs the `loom-launch` command line."""
    from loom.launch.cli import main as cli_main

    return cli_main(argv)
