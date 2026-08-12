# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for Loom documentation source assembly hooks."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import prepare


class _FakeServer:
    def __init__(self) -> None:
        self.unwatched_paths: list[str] = []

    def unwatch(self, path: str) -> None:
        self.unwatched_paths.append(path)


class PrepareTest(unittest.TestCase):
    def test_config_adds_generated_source_to_snippet_search_path(self) -> None:
        config = {
            "mdx_configs": {
                "pymdownx.snippets": {
                    "base_path": ["loom/docs"],
                }
            }
        }

        with TemporaryDirectory() as temporary_directory:
            with mock.patch.dict(
                prepare.os.environ,
                {"LOOM_DOCS_WORK_DIR": temporary_directory},
            ):
                staged_source_root = prepare._staged_source_root()
                result = prepare.on_config(config)
                self.assertFalse(staged_source_root.exists())

        self.assertIs(result, config)
        self.assertEqual(
            config["mdx_configs"]["pymdownx.snippets"]["base_path"],
            ["loom/docs", str(staged_source_root)],
        )

    def test_example_generators_are_discovered_by_package(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            example_root = temporary_root / "source"
            work_root = temporary_root / "work"
            first_generator = example_root / "alpha" / prepare.DOC_GENERATOR_NAME
            second_generator = (
                example_root / "nested" / "beta" / prepare.DOC_GENERATOR_NAME
            )
            first_generator.parent.mkdir(parents=True)
            second_generator.parent.mkdir(parents=True)
            first_generator.touch()
            second_generator.touch()
            stale_output = work_root / "examples" / "stale.loom"
            stale_output.parent.mkdir(parents=True)
            stale_output.touch()

            with (
                mock.patch.object(prepare, "EXAMPLE_SOURCE_ROOT", example_root),
                mock.patch.object(prepare.subprocess, "run") as run,
            ):
                output_root = prepare._generate_example_outputs(work_root)

            self.assertEqual(output_root, work_root / "examples")
            self.assertFalse(stale_output.exists())
            self.assertEqual(
                [call.args[0] for call in run.call_args_list],
                [
                    [str(first_generator), str(output_root / "alpha")],
                    [
                        str(second_generator),
                        str(output_root / "nested" / "beta"),
                    ],
                ],
            )
            for call in run.call_args_list:
                self.assertEqual(call.kwargs["cwd"], prepare.REPO_ROOT)
                self.assertTrue(call.kwargs["check"])

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
