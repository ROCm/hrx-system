# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from build_tools.ci import native_test_artifact


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


class NativeTestArtifactTest(unittest.TestCase):
    def test_stages_typed_payload_with_runfiles_and_shared_dwp(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workspace = root / "workspace"
            execution_root = root / "output_base" / "execroot" / "_main"
            external_root = root / "output_base" / "external"
            workspace.mkdir()
            execution_root.mkdir(parents=True)
            external_root.mkdir(parents=True)

            binary = execution_root / "bazel-out/bin/pkg/test"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"\x7fELFfixture")
            binary.chmod(0o755)
            data_file = workspace / "pkg/data file.txt"
            data_file.parent.mkdir()
            data_file.write_text("runtime data\n", encoding="utf-8")
            manifest = binary.with_name("test.runfiles_manifest")
            manifest.write_text(
                "_main/pkg/test bazel-out/bin/pkg/test\n"
                f" _main/pkg/data\\sfile.txt {data_file}\n",
                encoding="utf-8",
            )
            metadata = binary.with_name("test.native-artifact.json")
            metadata.write_text(
                json.dumps(
                    {
                        "arguments": ["--data=pkg/data file.txt"],
                        "environment": {},
                        "executable": "bazel-out/bin/pkg/test",
                        "inherited_environment": [],
                        "label": "@@//pkg:test",
                        "marked_arguments": [
                            "--data=__IREE_BAZEL_RUNFILE_PATH_BEGIN__"
                            "pkg/data file.txt"
                            "__IREE_BAZEL_RUNFILE_PATH_END__"
                        ],
                        "repository_mapping": None,
                        "rule_kind": "cc_test",
                        "runfiles_manifest": "bazel-out/bin/pkg/test.runfiles_manifest",
                        "size": "small",
                        "tags": ["fixture"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            debug_file = execution_root / "bazel-out/bin/pkg/test.pic.dwo"
            debug_file.write_bytes(b"split dwarf")

            build_events = root / "build-events.jsonl"
            events = [
                {"started": {"workspaceDirectory": str(workspace)}},
                {"workspaceInfo": {"localExecRoot": str(execution_root)}},
                {
                    "id": {"namedSet": {"id": "payload-child"}},
                    "namedSetOfFiles": {
                        "files": [{"name": data_file.name, "uri": data_file.as_uri()}]
                    },
                },
                {
                    "id": {"namedSet": {"id": "payload"}},
                    "namedSetOfFiles": {
                        "files": [
                            {"name": metadata.name, "uri": metadata.as_uri()},
                            {"name": binary.name, "uri": binary.as_uri()},
                            {"name": manifest.name, "uri": manifest.as_uri()},
                        ],
                        "fileSets": [{"id": "payload-child"}],
                    },
                },
                {
                    "id": {"namedSet": {"id": "debug"}},
                    "namedSetOfFiles": {
                        "files": [{"name": debug_file.name, "uri": debug_file.as_uri()}]
                    },
                },
                {
                    "id": {
                        "targetCompleted": {
                            "label": "//pkg:test",
                            "configuration": {"id": "fixture-config"},
                        }
                    },
                    "completed": {
                        "success": True,
                        "outputGroup": [
                            {
                                "name": native_test_artifact.PAYLOAD_OUTPUT_GROUP,
                                "fileSets": [{"id": "payload"}],
                            },
                            {
                                "name": native_test_artifact.DEBUG_OUTPUT_GROUP,
                                "fileSets": [{"id": "debug"}],
                            },
                        ],
                    },
                },
                {"finished": {"overallSuccess": True}},
            ]
            build_events.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            fake_objcopy = root / "llvm-objcopy"
            _write_executable(
                fake_objcopy,
                "#!/usr/bin/env python3\n"
                "import shutil, sys\n"
                "shutil.copyfile(sys.argv[-2], sys.argv[-1])\n",
            )
            fake_dwp = root / "llvm-dwp"
            _write_executable(
                fake_dwp,
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "pathlib.Path(sys.argv[sys.argv.index('-o') + 1]).write_bytes(b'DWP')\n",
            )

            staging_root = root / "staging"
            result = native_test_artifact.stage_native_artifact(
                build_events_paths=[build_events],
                staging_root=staging_root,
                package_path=Path("artifacts/ci/fixture"),
                revision="a" * 40,
                llvm_dwp=fake_dwp,
                llvm_objcopy=fake_objcopy,
                platform_constraints=[
                    "@platforms//os:linux",
                    "@platforms//cpu:x86_64",
                ],
            )

            relocated_root = root / "relocated"
            staging_root.rename(relocated_root)
            package = relocated_root / "artifacts/ci/fixture"
            copied_binary = package / "files/execroot/bazel-out/bin/pkg/test"
            adjacent_dwp = copied_binary.with_name("test.dwp")
            shared_dwp = package / ".debug/all.dwp"
            self.assertEqual(
                os.stat(shared_dwp).st_ino,
                os.stat(adjacent_dwp).st_ino,
            )
            runfile = package / "runfiles/native_pkg_test/_main/pkg/test"
            self.assertTrue(runfile.is_symlink())
            self.assertEqual(copied_binary, runfile.resolve())
            self.assertEqual(
                "runtime data\n",
                package.joinpath(
                    "runfiles/native_pkg_test/_main/pkg/data file.txt"
                ).read_text(encoding="utf-8"),
            )
            generated_build = package.joinpath("BUILD.bazel").read_text(
                encoding="utf-8"
            )
            self.assertIn("native_artifact_test(", generated_build)
            self.assertIn("files/execroot/bazel-out/bin/pkg/test.dwp", generated_build)
            self.assertIn("__IREE_BAZEL_RUNFILE_PATH_BEGIN__", generated_build)
            self.assertIn('"@platforms//os:linux"', generated_build)
            self.assertEqual(1, result["target_count"])
            self.assertEqual(1, result["debug_alias_count"])

    def test_rejects_unsuccessful_bazel_build(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            build_events = Path(temporary_directory) / "build-events.jsonl"
            build_events.write_text(
                json.dumps({"finished": {"overallSuccess": False}}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "did not finish successfully"):
                native_test_artifact.load_build_export(build_events)

    def test_verifies_artifact_and_checkout_revision(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            package = Path(temporary_directory) / "package"
            package.mkdir()
            revision = "a" * 40
            package.joinpath("manifest.json").write_text(
                json.dumps(
                    {
                        "revision": revision,
                        "target_count": 12,
                        "test_count": 10,
                        "tool_count": 2,
                    }
                ),
                encoding="utf-8",
            )

            result = native_test_artifact.verify_artifact(
                package_directory=package,
                expected_revision=revision,
                checkout_revision=revision,
            )

            self.assertEqual(12, result["target_count"])
            self.assertEqual(10, result["test_count"])
            self.assertEqual(2, result["tool_count"])

            with self.assertRaisesRegex(ValueError, "checkout revision"):
                native_test_artifact.verify_artifact(
                    package_directory=package,
                    expected_revision=revision,
                    checkout_revision="b" * 40,
                )
            with self.assertRaisesRegex(ValueError, "artifact revision"):
                native_test_artifact.verify_artifact(
                    package_directory=package,
                    expected_revision="b" * 40,
                    checkout_revision="b" * 40,
                )


if __name__ == "__main__":
    unittest.main()
