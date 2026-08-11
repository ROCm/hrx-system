# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for Loom documentation source assembly hooks."""

from __future__ import annotations

import unittest

import prepare


class _FakeServer:
    def __init__(self) -> None:
        self.unwatched_paths: list[str] = []

    def unwatch(self, path: str) -> None:
        self.unwatched_paths.append(path)


class PrepareTest(unittest.TestCase):
    def test_live_reload_does_not_watch_generated_source(self) -> None:
        server = _FakeServer()

        result = prepare.on_serve(
            server,
            config={"docs_dir": "/generated/loom-docs/mkdocs-source"},
            builder=lambda: None,
        )

        self.assertIs(result, server)
        self.assertEqual(
            server.unwatched_paths,
            ["/generated/loom-docs/mkdocs-source"],
        )


if __name__ == "__main__":
    unittest.main()
