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
