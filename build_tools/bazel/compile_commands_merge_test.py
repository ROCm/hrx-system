# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for compile_commands_merge.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_tools.bazel import compile_commands_merge


class CompileCommandsMergeTest(unittest.TestCase):
    def test_merges_and_sorts_fragments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_fragment = root / "first.json"
            second_fragment = root / "second.json"
            output = root / "compile_commands.json"
            first_fragment.write_text(
                json.dumps(
                    [
                        {
                            "arguments": ["clang", "-c", "z.c"],
                            "directory": ".",
                            "file": "z.c",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            second_fragment.write_text(
                json.dumps(
                    [
                        {
                            "arguments": ["clang", "-c", "a.c"],
                            "directory": ".",
                            "file": "a.c",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                0,
                compile_commands_merge.main(
                    [
                        "compile_commands_merge.py",
                        str(output),
                        str(first_fragment),
                        str(second_fragment),
                    ]
                ),
            )

            merged_entries = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                ["a.c", "z.c"], [entry["file"] for entry in merged_entries]
            )

    def test_rewrites_command_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fragment = root / "fragment.json"
            output = root / "out" / "compile_commands.json"
            command_directory = root / "execroot"
            fragment.write_text(
                json.dumps(
                    [
                        {
                            "arguments": ["clang", "-c", "a.c"],
                            "directory": ".",
                            "file": "a.c",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            compile_commands_merge.write_merged_compile_commands(
                output_path=output,
                fragment_paths=[fragment],
                command_directory=command_directory,
            )

            merged_entries = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(str(command_directory), merged_entries[0]["directory"])

    def test_finds_compile_command_fragments_from_bep(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_fragment = root / "first fragment.compile_commands.json"
            second_fragment = root / "second.compile_commands.json"
            ignored_file = root / "ignored.txt"
            build_events = root / "build_events.json"
            for path in (first_fragment, second_fragment, ignored_file):
                path.write_text("[]\n", encoding="utf-8")
            build_events.write_text(
                "\n".join(
                    json.dumps(event)
                    for event in [
                        {
                            "id": {"namedSet": {"id": "child"}},
                            "namedSetOfFiles": {
                                "files": [
                                    {
                                        "name": ignored_file.name,
                                        "uri": ignored_file.as_uri(),
                                    },
                                    {
                                        "name": first_fragment.name,
                                        "uri": first_fragment.as_uri(),
                                    },
                                ],
                            },
                        },
                        {
                            "id": {"namedSet": {"id": "root"}},
                            "namedSetOfFiles": {
                                "files": [
                                    {
                                        "name": first_fragment.name,
                                        "uri": first_fragment.as_uri(),
                                    },
                                    {
                                        "name": second_fragment.name,
                                        "uri": second_fragment.as_uri(),
                                    },
                                ],
                                "fileSets": [{"id": "child"}],
                            },
                        },
                        {
                            "completed": {
                                "outputGroup": [
                                    {
                                        "name": "iree_compile_commands_fragments",
                                        "fileSets": [{"id": "root"}],
                                    }
                                ],
                            },
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                [first_fragment, second_fragment],
                compile_commands_merge.fragment_paths_from_bep(build_events),
            )


if __name__ == "__main__":
    unittest.main()
