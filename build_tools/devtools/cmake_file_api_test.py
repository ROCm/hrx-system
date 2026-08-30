# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.devtools import cmake_file_api


class CMakeFileApiTest(unittest.TestCase):
    def test_resolve_executable_uses_codemodel_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir)
            reply_dir = build_dir / ".cmake/api/v1/reply"
            reply_dir.mkdir(parents=True)
            self.write_json(
                cmake_file_api.target_aliases_path(build_dir),
                {
                    "iree::tools::iree-dump-cpuinfo": "runtime_src_tools_iree-dump-cpuinfo"
                },
            )
            self.write_json(
                reply_dir / "index-1.json",
                {"objects": [{"kind": "codemodel", "jsonFile": "codemodel-v2.json"}]},
            )
            self.write_json(
                reply_dir / "codemodel-v2.json",
                {
                    "configurations": [
                        {
                            "targets": [
                                {
                                    "name": "iree-dump-cpuinfo",
                                    "jsonFile": "target-iree-dump-cpuinfo.json",
                                },
                                {
                                    "name": "runtime_src_tools_iree-dump-cpuinfo",
                                    "jsonFile": "target-iree-dump-cpuinfo.json",
                                },
                                {
                                    "name": "iree-base",
                                    "jsonFile": "target-iree-base.json",
                                },
                            ]
                        }
                    ]
                },
            )
            self.write_json(
                reply_dir / "target-iree-dump-cpuinfo.json",
                {
                    "name": "iree-dump-cpuinfo",
                    "type": "EXECUTABLE",
                    "artifacts": [{"path": "tools/iree-dump-cpuinfo"}],
                },
            )
            self.write_json(
                reply_dir / "target-iree-base.json",
                {
                    "name": "iree-base",
                    "type": "STATIC_LIBRARY",
                    "artifacts": [{"path": "lib/libiree-base.a"}],
                },
            )

            target = cmake_file_api.resolve_executable(build_dir, "iree-dump-cpuinfo")

            self.assertEqual(target.name, "iree-dump-cpuinfo")
            self.assertEqual(target.path, build_dir / "tools/iree-dump-cpuinfo")

            alias_target = cmake_file_api.resolve_executable(
                build_dir,
                "iree::tools::iree-dump-cpuinfo",
            )

            self.assertEqual(alias_target.path, build_dir / "tools/iree-dump-cpuinfo")

    def test_resolve_executable_uses_unique_artifact_name(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir)
            reply_dir = build_dir / ".cmake/api/v1/reply"
            reply_dir.mkdir(parents=True)
            self.write_json(
                reply_dir / "index-1.json",
                {"objects": [{"kind": "codemodel", "jsonFile": "codemodel-v2.json"}]},
            )
            self.write_json(
                reply_dir / "codemodel-v2.json",
                {
                    "configurations": [
                        {
                            "targets": [
                                {
                                    "name": "loom_tools_loom-compile_loom-compile",
                                    "jsonFile": "target-loom-compile.json",
                                }
                            ]
                        }
                    ]
                },
            )
            self.write_json(
                reply_dir / "target-loom-compile.json",
                {
                    "name": "loom_tools_loom-compile_loom-compile",
                    "type": "EXECUTABLE",
                    "artifacts": [
                        {"path": "loom/src/loom/tools/loom-compile/loom-compile.exe"}
                    ],
                },
            )

            with mock.patch.object(
                cmake_file_api,
                "executable_targets",
                wraps=cmake_file_api.executable_targets,
            ) as executable_targets:
                resolved_names = cmake_file_api.resolve_target_names(
                    build_dir, ["loom-compile", "loom-compile"]
                )

            self.assertEqual(
                resolved_names,
                [
                    "loom_tools_loom-compile_loom-compile",
                    "loom_tools_loom-compile_loom-compile",
                ],
            )
            executable_targets.assert_called_once_with(build_dir)

            target = cmake_file_api.resolve_executable(build_dir, "loom-compile")

            self.assertEqual(target.name, "loom_tools_loom-compile_loom-compile")
            self.assertEqual(
                target.path,
                build_dir / "loom/src/loom/tools/loom-compile/loom-compile.exe",
            )

    def test_resolve_target_name_rejects_ambiguous_artifact_name(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir)
            reply_dir = build_dir / ".cmake/api/v1/reply"
            reply_dir.mkdir(parents=True)
            self.write_json(
                reply_dir / "index-1.json",
                {"objects": [{"kind": "codemodel", "jsonFile": "codemodel-v2.json"}]},
            )
            self.write_json(
                reply_dir / "codemodel-v2.json",
                {
                    "configurations": [
                        {
                            "targets": [
                                {"name": "first_tool", "jsonFile": "first.json"},
                                {"name": "second_tool", "jsonFile": "second.json"},
                            ]
                        }
                    ]
                },
            )
            for target_name, json_file, artifact_path in (
                ("first_tool", "first.json", "first/tool.exe"),
                ("second_tool", "second.json", "second/tool.exe"),
            ):
                self.write_json(
                    reply_dir / json_file,
                    {
                        "name": target_name,
                        "type": "EXECUTABLE",
                        "artifacts": [{"path": artifact_path}],
                    },
                )

            with self.assertRaisesRegex(
                cmake_file_api.FileApiError,
                "matching targets: first_tool, second_tool",
            ):
                cmake_file_api.resolve_target_name(build_dir, "tool")

    def test_resolve_target_name_prefers_exact_non_executable_target(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir)
            reply_dir = build_dir / ".cmake/api/v1/reply"
            reply_dir.mkdir(parents=True)
            self.write_json(
                reply_dir / "index-1.json",
                {"objects": [{"kind": "codemodel", "jsonFile": "codemodel-v2.json"}]},
            )
            self.write_json(
                reply_dir / "codemodel-v2.json",
                {
                    "configurations": [
                        {
                            "targets": [
                                {"name": "tool", "jsonFile": "library.json"},
                                {
                                    "name": "package_tool",
                                    "jsonFile": "executable.json",
                                },
                            ]
                        }
                    ]
                },
            )
            self.write_json(
                reply_dir / "executable.json",
                {
                    "name": "package_tool",
                    "type": "EXECUTABLE",
                    "artifacts": [{"path": "package/tool.exe"}],
                },
            )

            self.assertEqual(
                cmake_file_api.resolve_target_name(build_dir, "tool"),
                "tool",
            )

    def test_resolve_executable_reports_missing_reply(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaises(cmake_file_api.FileApiError):
                cmake_file_api.resolve_executable(
                    Path(temporary_dir),
                    "iree-dump-cpuinfo",
                )

    def write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
