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

from build_tools.devtools import cmake_file_api


class CMakeFileApiTest(unittest.TestCase):
    def test_resolve_executable_uses_codemodel_output_name(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir)
            self.write_json(
                cmake_file_api.target_aliases_path(build_dir),
                {"iree::tools::iree-run-module": "runtime_src_tools_iree-run-module"},
            )
            self.write_codemodel(
                build_dir,
                [
                    {
                        "name": "runtime_src_tools_iree-run-module",
                        "nameOnDisk": "iree-run-module",
                        "type": "EXECUTABLE",
                        "artifacts": [{"path": "tools/iree-run-module"}],
                    },
                    {
                        "name": "iree-base",
                        "type": "STATIC_LIBRARY",
                        "artifacts": [{"path": "lib/libiree-base.a"}],
                    },
                ],
            )

            target = cmake_file_api.resolve_executable(build_dir, "iree-run-module")

            self.assertEqual(target.name, "runtime_src_tools_iree-run-module")
            self.assertEqual(target.output_name, "iree-run-module")
            self.assertEqual(target.path, build_dir / "tools/iree-run-module")

            alias_target = cmake_file_api.resolve_executable(
                build_dir,
                "iree::tools::iree-run-module",
            )

            self.assertEqual(alias_target.path, build_dir / "tools/iree-run-module")

    def test_exact_target_precedes_executable_output_name(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir)
            self.write_codemodel(
                build_dir,
                [
                    {
                        "name": "iree-serve-device",
                        "type": "STATIC_LIBRARY",
                        "artifacts": [{"path": "lib/libiree-serve-device.a"}],
                    },
                    {
                        "name": "runtime_src_tools_iree-serve-device",
                        "nameOnDisk": "iree-serve-device",
                        "type": "EXECUTABLE",
                        "artifacts": [{"path": "tools/iree-serve-device"}],
                    },
                ],
            )

            self.assertEqual(
                cmake_file_api.resolve_target_name(build_dir, "iree-serve-device"),
                "iree-serve-device",
            )

    def test_ambiguous_executable_output_name_reports_concrete_targets(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir)
            self.write_codemodel(
                build_dir,
                [
                    {
                        "name": "first_tool",
                        "nameOnDisk": "shared-tool",
                        "type": "EXECUTABLE",
                        "artifacts": [{"path": "first/shared-tool"}],
                    },
                    {
                        "name": "second_tool",
                        "nameOnDisk": "shared-tool",
                        "type": "EXECUTABLE",
                        "artifacts": [{"path": "second/shared-tool"}],
                    },
                ],
            )

            with self.assertRaisesRegex(
                cmake_file_api.FileApiError,
                "ambiguous.*first_tool.*second_tool",
            ):
                cmake_file_api.resolve_target_name(build_dir, "shared-tool")

    def test_windows_executable_output_name_omits_suffix(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir)
            self.write_codemodel(
                build_dir,
                [
                    {
                        "name": "runtime_src_tools_iree-serve-device",
                        "nameOnDisk": "iree-serve-device.exe",
                        "type": "EXECUTABLE",
                        "artifacts": [{"path": "tools/iree-serve-device.exe"}],
                    }
                ],
            )

            self.assertEqual(
                cmake_file_api.resolve_target_name(build_dir, "iree-serve-device"),
                "runtime_src_tools_iree-serve-device",
            )
            self.assertEqual(
                cmake_file_api.resolve_target_name(build_dir, "iree-serve-device.exe"),
                "runtime_src_tools_iree-serve-device",
            )

    def test_unknown_target_is_preserved_without_file_api_reply(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            self.assertEqual(
                cmake_file_api.resolve_target_name(
                    Path(temporary_dir),
                    "custom-target",
                ),
                "custom-target",
            )

    def test_invalid_file_api_reply_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir)
            index_path = build_dir / ".cmake/api/v1/reply/index-1.json"
            index_path.parent.mkdir(parents=True)
            index_path.write_text("not JSON", encoding="utf-8")

            with self.assertRaisesRegex(
                cmake_file_api.FileApiError,
                "reply file is invalid",
            ):
                cmake_file_api.resolve_target_name(build_dir, "iree-run-module")

    def test_resolve_executable_reports_missing_reply(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaises(cmake_file_api.FileApiError):
                cmake_file_api.resolve_executable(
                    Path(temporary_dir),
                    "iree-run-module",
                )

    def write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def write_codemodel(
        self,
        build_dir: Path,
        targets: list[dict[str, object]],
    ) -> None:
        reply_dir = build_dir / ".cmake/api/v1/reply"
        target_refs = []
        for index, target in enumerate(targets):
            json_file = f"target-{index}.json"
            target_refs.append({"name": target["name"], "jsonFile": json_file})
            self.write_json(reply_dir / json_file, target)
        self.write_json(
            reply_dir / "codemodel-v2.json",
            {"configurations": [{"targets": target_refs}]},
        )
        self.write_json(
            reply_dir / "index-1.json",
            {"objects": [{"kind": "codemodel", "jsonFile": "codemodel-v2.json"}]},
        )


if __name__ == "__main__":
    unittest.main()
