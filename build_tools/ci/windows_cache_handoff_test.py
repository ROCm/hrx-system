# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib
import http.server
import json
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path

from build_tools.ci import windows_cache_handoff as handoff


class _CasHandler(http.server.BaseHTTPRequestHandler):
    blobs: dict[str, bytes] = {}

    def do_PUT(self):
        digest = self.path.rsplit("/", 1)[-1]
        content = self.rfile.read(int(self.headers["Content-Length"]))
        if hashlib.sha256(content).hexdigest() != digest:
            self.send_error(400)
            return
        self.blobs[digest] = content
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        digest = self.path.rsplit("/", 1)[-1]
        content = self.blobs.get(digest)
        if content is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, _format, *_args):
        pass


class WindowsCacheHandoffTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _CasHandler)
        cls.server_thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.server_thread.start()
        cls.cas_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join()

    def setUp(self):
        _CasHandler.blobs.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_fixture(self):
        execution_root = self.root / "execroot"
        executable = execution_root / "bazel-out/bin/test.exe"
        executable.parent.mkdir(parents=True)
        if os.name == "nt":
            executable.write_bytes(b"@echo off\r\nexit /b 0\r\n")
        else:
            executable.write_bytes(b"#!/bin/sh\nexit 0\n")
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        repository_mapping = execution_root / "bazel-out/bin/test.exe.repo_mapping"
        repository_mapping.write_bytes(b",_main,_main\n")
        Path(str(executable) + ".runfiles_manifest").write_text(
            f"_main/bin/test.exe {executable}\n_repo_mapping {repository_mapping}\n",
            encoding="utf-8",
        )
        target_metadata = self.root / "target.json"
        target_metadata.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "target": {
                                "rule": {
                                    "name": "//example:test",
                                    "attribute": [
                                        {
                                            "name": "args",
                                            "stringListValue": ["--plain"],
                                        },
                                        {
                                            "name": "exec_properties",
                                            "stringDictValue": [
                                                {"key": "pool", "value": "windows"}
                                            ],
                                        },
                                        {
                                            "name": "flaky",
                                            "booleanValue": False,
                                        },
                                        {"name": "local", "booleanValue": False},
                                        {
                                            "name": "shard_count",
                                            "intValue": -1,
                                        },
                                        {"name": "size", "stringValue": "small"},
                                        {
                                            "name": "tags",
                                            "stringListValue": ["requires-windows"],
                                        },
                                        {
                                            "name": "timeout",
                                            "stringValue": "short",
                                        },
                                    ],
                                }
                            }
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        provider_metadata = self.root / "providers.json"
        provider_metadata.write_text(
            json.dumps(
                {
                    "argument_provider_count": 0,
                    "arguments": [],
                    "environment": {"FIXTURE_ENV": "present"},
                    "executable": executable.relative_to(execution_root).as_posix(),
                    "inherited_environment": ["PATH"],
                    "label": "@@//example:test",
                    "marked_arguments": [],
                    "marked_environment": {},
                }
            ),
            encoding="utf-8",
        )
        return execution_root, target_metadata, provider_metadata

    def test_emit_uploads_exact_runfiles_and_preserves_test_metadata(self):
        execution_root, target_metadata, provider_metadata = self._write_fixture()
        output_dir = self.root / "package"
        manifest = handoff.emit_handoff_package(
            target_metadata_path=target_metadata,
            provider_metadata_path=provider_metadata,
            execution_root=execution_root,
            cas_url=self.cas_url,
            output_dir=output_dir,
        )
        self.assertEqual(manifest["source_label"], "//example:test")
        self.assertEqual(manifest["executable"], "_main/bin/test.exe")
        self.assertEqual(manifest["test"]["arguments"], ["--plain"])
        self.assertEqual(manifest["test"]["environment"], {"FIXTURE_ENV": "present"})
        self.assertEqual(manifest["test"]["inherited_environment"], ["PATH"])
        self.assertEqual(len(_CasHandler.blobs), 2)
        build_text = (output_dir / "BUILD.bazel").read_text(encoding="utf-8")
        self.assertIn("windows_cache_handoff_test(", build_text)
        self.assertIn("exec_properties = {'pool': 'windows'}", build_text)
        self.assertIn("tags = ['requires-windows']", build_text)

    def test_run_materializes_verified_bytes_before_execution(self):
        execution_root, target_metadata, provider_metadata = self._write_fixture()
        output_dir = self.root / "package"
        handoff.emit_handoff_package(
            target_metadata_path=target_metadata,
            provider_metadata_path=provider_metadata,
            execution_root=execution_root,
            cas_url=self.cas_url,
            output_dir=output_dir,
        )
        self.assertEqual(handoff.run_handoff_test(output_dir / "test.json"), 0)

    def test_run_rejects_corrupt_cas_bytes(self):
        execution_root, target_metadata, provider_metadata = self._write_fixture()
        output_dir = self.root / "package"
        manifest = handoff.emit_handoff_package(
            target_metadata_path=target_metadata,
            provider_metadata_path=provider_metadata,
            execution_root=execution_root,
            cas_url=self.cas_url,
            output_dir=output_dir,
        )
        digest = manifest["runfiles"][0]["sha256"]
        _CasHandler.blobs[digest] = b"corrupt"
        with self.assertRaisesRegex(handoff.HandoffError, "verified as"):
            handoff.run_handoff_test(output_dir / "test.json")

    def test_parse_runfiles_manifest_decodes_escaped_values(self):
        physical = self.root / "file with space"
        physical.write_bytes(b"data")
        manifest = self.root / "MANIFEST"
        encoded_physical = str(physical).replace(" ", r"\s")
        manifest.write_text(
            f" _main/file\\swith\\sspace {encoded_physical}\n_main/empty \n",
            encoding="utf-8",
        )
        self.assertEqual(
            handoff.parse_runfiles_manifest(manifest),
            [
                handoff.Runfile("_main/file with space", physical),
                handoff.Runfile("_main/empty", None),
            ],
        )

    def test_parse_runfiles_manifest_rejects_windows_case_collision(self):
        manifest = self.root / "MANIFEST"
        manifest.write_text(
            f"_main/Data {self.root / 'one'}\n_main/data {self.root / 'two'}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(handoff.HandoffError, "case-colliding"):
            handoff.parse_runfiles_manifest(manifest)

    def test_audit_rejects_native_compilation(self):
        execution_log = self.root / "execution.json"
        execution_log.write_text(
            json.dumps({"mnemonic": "Symlink", "targetLabel": "//:runner"})
            + json.dumps(
                {
                    "mnemonic": "CppCompile",
                    "targetLabel": "//example:test",
                    "runner": "local",
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(handoff.HandoffError, "CppCompile"):
            handoff.audit_consumer_execution(execution_log, self.root / "receipt.json")

    def test_audit_writes_action_summary(self):
        execution_log = self.root / "execution.json"
        execution_log.write_text(
            json.dumps({"mnemonic": "Symlink", "targetLabel": "//:runner"}),
            encoding="utf-8",
        )
        receipt = self.root / "receipt.json"
        handoff.audit_consumer_execution(execution_log, receipt)
        self.assertEqual(
            json.loads(receipt.read_text(encoding="utf-8")),
            {
                "entry_count": 1,
                "mnemonics": ["Symlink"],
                "native_cpp_action_count": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
