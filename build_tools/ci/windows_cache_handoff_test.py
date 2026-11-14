# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import copy
import hashlib
import http.server
import json
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from build_tools.ci import windows_cache_handoff as handoff


class _CasHandler(http.server.BaseHTTPRequestHandler):
    blobs: dict[str, bytes] = {}
    puts: list[str] = []
    heads: list[str] = []

    def do_PUT(self):
        digest = self.path.rsplit("/", 1)[-1]
        content = self.rfile.read(int(self.headers["Content-Length"]))
        if hashlib.sha256(content).hexdigest() != digest:
            self.send_error(400)
            return
        self.blobs[digest] = content
        self.puts.append(digest)
        self.send_response(200)
        self.end_headers()

    def do_HEAD(self):
        digest = self.path.rsplit("/", 1)[-1]
        self.heads.append(digest)
        content = self.blobs.get(digest)
        if content is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(content)))
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
        _CasHandler.puts.clear()
        _CasHandler.heads.clear()
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
                    "compatible": True,
                    "environment": {"FIXTURE_ENV": "present"},
                    "executable": executable.relative_to(execution_root).as_posix(),
                    "executable_logical_path": "_main/bin/test.exe",
                    "executable_provider_count": 0,
                    "inherited_environment": ["PATH"],
                    "label": "@@//example:test",
                    "marked_arguments": [],
                    "marked_environment": {},
                    "runfiles": [
                        {
                            "logical_path": "_main/bin/test.exe",
                            "physical_path": executable.relative_to(
                                execution_root
                            ).as_posix(),
                        },
                        {
                            "logical_path": "_repo_mapping",
                            "physical_path": repository_mapping.relative_to(
                                execution_root
                            ).as_posix(),
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return execution_root, target_metadata, provider_metadata

    def test_emit_uploads_exact_runfiles_and_preserves_test_metadata(self):
        execution_root, target_metadata, provider_metadata = self._write_fixture()
        output_dir = self.root / "package"
        summary = handoff.emit_handoff_packages(
            target_metadata_path=target_metadata,
            provider_metadata_path=provider_metadata,
            execution_root=execution_root,
            cas_url=self.cas_url,
            output_dir=output_dir,
        )
        manifest = json.loads(
            (output_dir / summary["tests"][0]["manifest"]).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["source_label"], "//example:test")
        self.assertEqual(manifest["executable"], "_main/bin/test.exe")
        self.assertEqual(manifest["test"]["arguments"], ["--plain"])
        self.assertEqual(manifest["test"]["environment"], {"FIXTURE_ENV": "present"})
        self.assertEqual(manifest["test"]["inherited_environment"], ["PATH"])
        self.assertEqual(len(_CasHandler.blobs), 2)
        self.assertCountEqual(
            _CasHandler.heads,
            [digest for digest in _CasHandler.blobs for _ in range(2)],
        )
        self.assertCountEqual(_CasHandler.puts, _CasHandler.blobs)
        self.assertEqual(summary["compatible_test_count"], 1)
        self.assertEqual(summary["unique_blob_count"], 2)
        self.assertEqual(summary["cas_url"], self.cas_url)
        self.assertEqual(
            {(entry["sha256"], entry["size"]) for entry in summary["blobs"]},
            {(digest, len(content)) for digest, content in _CasHandler.blobs.items()},
        )
        self.assertEqual(
            (output_dir / "MODULE.bazel").read_text(encoding="utf-8"),
            'module(name = "hrx_windows_cache_handoff")\n',
        )
        self.assertTrue((output_dir / "handoff/build_defs.bzl").is_file())
        self.assertTrue((output_dir / "handoff/runner.py").is_file())
        build_text = (output_dir / "example/BUILD.bazel").read_text(encoding="utf-8")
        self.assertIn('load("//handoff:build_defs.bzl"', build_text)
        self.assertIn("windows_cache_handoff_test(", build_text)
        self.assertIn("name = 'test'", build_text)
        self.assertIn("exec_properties = {'pool': 'windows'}", build_text)
        self.assertIn("tags = ['requires-windows']", build_text)

    def test_emit_reuses_blobs_already_owned_by_the_cas(self):
        execution_root, target_metadata, provider_metadata = self._write_fixture()
        first_output_dir = self.root / "first-package"
        handoff.emit_handoff_packages(
            target_metadata_path=target_metadata,
            provider_metadata_path=provider_metadata,
            execution_root=execution_root,
            cas_url=self.cas_url,
            output_dir=first_output_dir,
        )
        first_puts = list(_CasHandler.puts)

        second_output_dir = self.root / "second-package"
        handoff.emit_handoff_packages(
            target_metadata_path=target_metadata,
            provider_metadata_path=provider_metadata,
            execution_root=execution_root,
            cas_url=self.cas_url,
            output_dir=second_output_dir,
        )

        self.assertEqual(_CasHandler.puts, first_puts)
        self.assertEqual(len(_CasHandler.heads), 4 * len(_CasHandler.blobs))

    def test_residency_gate_rejects_evicted_blob(self):
        source = self.root / "blob"
        source.write_bytes(b"handoff product")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()

        with self.assertRaisesRegex(
            handoff.HandoffError,
            rf"evicted 1 handoff blob.*{digest}",
        ):
            handoff._verify_blobs_resident(
                handoff.CasClient(self.cas_url),
                {digest: handoff.BlobSource(path=source, size=source.stat().st_size)},
                jobs=1,
            )

    def test_emit_resolves_rootpath_arguments_against_exact_runfiles(self):
        execution_root, target_metadata, provider_metadata = self._write_fixture()
        library = execution_root / "bazel-out/bin/libamdf/amdf.dll"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"amdf")
        provider = json.loads(provider_metadata.read_text(encoding="utf-8"))
        rootpath = "libamdf/amdf.dll"
        marked_rootpath = (
            handoff.RUNFILES_PATH_BEGIN + rootpath + handoff.RUNFILES_PATH_END
        )
        provider.update(
            {
                "argument_provider_count": 1,
                "arguments": ["--amdf_library=" + rootpath],
                "environment": {"AMDF_LIBRARY": rootpath},
                "marked_arguments": ["--amdf_library=" + marked_rootpath],
                "marked_environment": {"AMDF_LIBRARY": marked_rootpath},
            }
        )
        provider["runfiles"].append(
            {
                "logical_path": "_main/" + rootpath,
                "physical_path": library.relative_to(execution_root).as_posix(),
            }
        )
        provider_metadata.write_text(json.dumps(provider), encoding="utf-8")

        output_dir = self.root / "package"
        summary = handoff.emit_handoff_packages(
            target_metadata_path=target_metadata,
            provider_metadata_path=provider_metadata,
            execution_root=execution_root,
            cas_url=self.cas_url,
            output_dir=output_dir,
        )
        manifest_path = output_dir / summary["tests"][0]["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        handoff_path = (
            handoff.HANDOFF_PATH_BEGIN + "_main/" + rootpath + handoff.HANDOFF_PATH_END
        )
        self.assertEqual(
            manifest["test"]["arguments"], ["--amdf_library=" + handoff_path]
        )
        self.assertEqual(
            manifest["test"]["environment"], {"AMDF_LIBRARY": handoff_path}
        )

        observed = {}

        def observe_run(command, *, cwd, env, **_kwargs):
            argument_path = command[1].split("=", 1)[1]
            observed["argument_path"] = argument_path
            observed["environment_path"] = env["AMDF_LIBRARY"]
            observed["argument"] = (cwd / argument_path).read_bytes()
            observed["environment"] = (cwd / env["AMDF_LIBRARY"]).read_bytes()
            return mock.Mock(returncode=0)

        with mock.patch.object(handoff.subprocess, "run", side_effect=observe_run):
            self.assertEqual(handoff.run_handoff_test(manifest_path), 0)
        self.assertEqual(
            observed,
            {
                "argument_path": rootpath,
                "environment_path": rootpath,
                "argument": b"amdf",
                "environment": b"amdf",
            },
        )

    def test_run_materializes_verified_bytes_before_execution(self):
        execution_root, target_metadata, provider_metadata = self._write_fixture()
        output_dir = self.root / "package"
        summary = handoff.emit_handoff_packages(
            target_metadata_path=target_metadata,
            provider_metadata_path=provider_metadata,
            execution_root=execution_root,
            cas_url=self.cas_url,
            output_dir=output_dir,
        )
        manifest_path = output_dir / summary["tests"][0]["manifest"]
        self.assertEqual(handoff.run_handoff_test(manifest_path), 0)

    def test_resolve_external_rootpath_from_main_workspace(self):
        value = (
            "--tool="
            + handoff.HANDOFF_PATH_BEGIN
            + "rules_python/python.exe"
            + handoff.HANDOFF_PATH_END
        )

        self.assertEqual(
            handoff._resolve_handoff_value(value, "_main"),
            "--tool=../rules_python/python.exe",
        )

    def test_prefetch_materializes_one_shared_hard_link_store(self):
        execution_root, target_metadata, provider_metadata = self._write_fixture()
        output_dir = self.root / "package"
        summary = handoff.emit_handoff_packages(
            target_metadata_path=target_metadata,
            provider_metadata_path=provider_metadata,
            execution_root=execution_root,
            cas_url=self.cas_url,
            output_dir=output_dir,
        )
        content_root = self.root / "content"
        handoff.prefetch_handoff_blobs(
            output_dir / "handoff-summary.json", content_root, jobs=2
        )
        self.assertEqual(
            {path.name for path in content_root.iterdir()},
            {entry["sha256"] for entry in summary["blobs"]},
        )

        manifest_path = output_dir / summary["tests"][0]["manifest"]
        with (
            mock.patch.dict(
                os.environ,
                {handoff.CAS_DIRECTORY_ENVIRONMENT_VARIABLE: str(content_root)},
            ),
            mock.patch.object(handoff.os, "link", wraps=handoff.os.link) as hard_link,
        ):
            self.assertEqual(handoff.run_handoff_test(manifest_path), 0)
        self.assertEqual(hard_link.call_count, len(summary["blobs"]))

    def test_prefetch_rejects_corrupt_existing_content(self):
        execution_root, target_metadata, provider_metadata = self._write_fixture()
        output_dir = self.root / "package"
        summary = handoff.emit_handoff_packages(
            target_metadata_path=target_metadata,
            provider_metadata_path=provider_metadata,
            execution_root=execution_root,
            cas_url=self.cas_url,
            output_dir=output_dir,
        )
        content_root = self.root / "content"
        content_root.mkdir()
        first_blob = summary["blobs"][0]
        (content_root / first_blob["sha256"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(handoff.HandoffError, "verified as"):
            handoff.prefetch_handoff_blobs(
                output_dir / "handoff-summary.json", content_root
            )

    def test_subject_manifest_matches_native_windows_launcher_format(self):
        manifest = self.root / "MANIFEST"
        runfiles_root = self.root / "runfiles tree"
        entries = [{"path": "_main/bin/test.exe"}]

        handoff._write_runfiles_manifest(manifest, entries, runfiles_root)

        line = manifest.read_text(encoding="utf-8").rstrip("\n")
        logical_path, separator, physical_path = line.partition(" ")
        self.assertEqual(separator, " ")
        self.assertEqual(logical_path, "_main/bin/test.exe")
        self.assertEqual(
            physical_path,
            str(runfiles_root / "_main/bin/test.exe"),
        )
        self.assertNotIn(r"\s", line)

    def test_subject_manifest_rejects_unrepresentable_logical_path(self):
        with self.assertRaisesRegex(
            handoff.HandoffError, "cannot represent runfile path"
        ):
            handoff._write_runfiles_manifest(
                self.root / "MANIFEST",
                [{"path": "_main/path with space"}],
                self.root / "runfiles",
            )

    def test_emit_batches_packages_and_uploads_shared_blobs_once(self):
        execution_root, target_metadata, provider_metadata = self._write_fixture()
        second_executable = execution_root / "bazel-out/bin/second.exe"
        if os.name == "nt":
            second_executable.write_bytes(b"@echo off\r\nrem second\r\nexit /b 0\r\n")
        else:
            second_executable.write_bytes(b"#!/bin/sh\n# second\nexit 0\n")
        second_executable.chmod(second_executable.stat().st_mode | stat.S_IXUSR)
        repository_mapping = execution_root / "bazel-out/bin/test.exe.repo_mapping"
        Path(str(second_executable) + ".runfiles_manifest").write_text(
            f"_main/bin/second.exe {second_executable}\n"
            f"_repo_mapping {repository_mapping}\n",
            encoding="utf-8",
        )

        target_payload = json.loads(target_metadata.read_text(encoding="utf-8"))
        first_result = target_payload["results"][0]
        second_result = copy.deepcopy(first_result)
        second_result["target"]["rule"]["name"] = "//other:second"
        incompatible_result = copy.deepcopy(first_result)
        incompatible_result["target"]["rule"]["name"] = "//other:incompatible"
        non_native_result = copy.deepcopy(first_result)
        non_native_result["target"]["rule"]["name"] = "//other:script"
        target_payload["results"].extend(
            [second_result, incompatible_result, non_native_result]
        )
        target_metadata.write_text(json.dumps(target_payload), encoding="utf-8")

        first_provider = json.loads(provider_metadata.read_text(encoding="utf-8"))
        second_provider = copy.deepcopy(first_provider)
        second_provider.update(
            {
                "executable": second_executable.relative_to(execution_root).as_posix(),
                "executable_logical_path": "_main/bin/second.exe",
                "label": "@@//other:second",
                "runfiles": [
                    {
                        "logical_path": "_main/bin/second.exe",
                        "physical_path": second_executable.relative_to(
                            execution_root
                        ).as_posix(),
                    },
                    {
                        "logical_path": "_main/bin/second_wrapper.exe",
                        "physical_path": second_executable.relative_to(
                            execution_root
                        ).as_posix(),
                    },
                    {
                        "logical_path": "_repo_mapping",
                        "physical_path": repository_mapping.relative_to(
                            execution_root
                        ).as_posix(),
                    },
                ],
            }
        )
        incompatible_provider = copy.deepcopy(first_provider)
        incompatible_provider.update(
            {
                "compatible": False,
                "executable": None,
                "label": "@@//other:incompatible",
            }
        )
        non_native_provider = copy.deepcopy(first_provider)
        non_native_provider.update(
            {
                "executable": "bazel-out/bin/script.py",
                "executable_logical_path": "_main/bin/script.py",
                "label": "@@//other:script",
                "runfiles": [],
            }
        )
        provider_metadata.write_text(
            "\n".join(
                json.dumps(value)
                for value in (
                    first_provider,
                    second_provider,
                    incompatible_provider,
                    non_native_provider,
                )
            )
            + "\n",
            encoding="utf-8",
        )

        output_dir = self.root / "package"
        summary = handoff.emit_handoff_packages(
            target_metadata_path=target_metadata,
            provider_metadata_path=provider_metadata,
            execution_root=execution_root,
            cas_url=self.cas_url,
            output_dir=output_dir,
        )

        self.assertEqual(summary["selected_test_count"], 4)
        self.assertEqual(summary["compatible_test_count"], 2)
        self.assertEqual(summary["incompatible_test_count"], 1)
        self.assertEqual(summary["incompatible_labels"], ["//other:incompatible"])
        self.assertEqual(summary["non_native_test_count"], 1)
        self.assertEqual(summary["non_native_labels"], ["//other:script"])
        self.assertEqual(summary["runfile_reference_count"], 5)
        self.assertEqual(summary["unique_blob_count"], 3)
        self.assertEqual(len(_CasHandler.blobs), 3)
        self.assertTrue((output_dir / "example/BUILD.bazel").is_file())
        self.assertTrue((output_dir / "other/BUILD.bazel").is_file())
        other_build = (output_dir / "other/BUILD.bazel").read_text(encoding="utf-8")
        self.assertIn("name = 'second'", other_build)
        self.assertNotIn("incompatible", other_build)
        self.assertNotIn("script", other_build)

    def test_run_forwards_bazel_test_arguments(self):
        execution_root, target_metadata, provider_metadata = self._write_fixture()
        output_dir = self.root / "package"
        summary = handoff.emit_handoff_packages(
            target_metadata_path=target_metadata,
            provider_metadata_path=provider_metadata,
            execution_root=execution_root,
            cas_url=self.cas_url,
            output_dir=output_dir,
        )
        manifest_path = output_dir / summary["tests"][0]["manifest"]
        completed = mock.Mock(returncode=0)
        with (
            mock.patch.dict(
                os.environ,
                {handoff.PYTHON_ENVIRONMENT_VARIABLE: "C:/host/python.exe"},
            ),
            mock.patch.object(handoff.subprocess, "run", return_value=completed) as run,
        ):
            self.assertEqual(
                handoff.run_handoff_test(
                    manifest_path, ["--gtest_filter=Fixture.Selected"]
                ),
                0,
            )
        self.assertEqual(run.call_args.args[0][-1], "--gtest_filter=Fixture.Selected")
        self.assertNotIn(
            handoff.PYTHON_ENVIRONMENT_VARIABLE, run.call_args.kwargs["env"]
        )

    def test_run_rejects_corrupt_cas_bytes(self):
        execution_root, target_metadata, provider_metadata = self._write_fixture()
        output_dir = self.root / "package"
        summary = handoff.emit_handoff_packages(
            target_metadata_path=target_metadata,
            provider_metadata_path=provider_metadata,
            execution_root=execution_root,
            cas_url=self.cas_url,
            output_dir=output_dir,
        )
        manifest_path = output_dir / summary["tests"][0]["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = manifest["runfiles"][0]["sha256"]
        _CasHandler.blobs[digest] = b"corrupt"
        with self.assertRaisesRegex(handoff.HandoffError, "verified as"):
            handoff.run_handoff_test(manifest_path)

    def test_find_manifest_resolves_windows_runfiles_manifest(self):
        physical_manifest = self.root / "compiler_runtime_test.json"
        physical_manifest.write_text("{}", encoding="utf-8")
        runfiles_manifest = self.root / "runner.runfiles_manifest"
        runfiles_manifest.write_text(
            f"_main/generated/compiler_runtime_test.json {physical_manifest}\n",
            encoding="utf-8",
        )
        with mock.patch.dict(
            os.environ,
            {
                handoff.MANIFEST_ENVIRONMENT_VARIABLE: (
                    "generated/compiler_runtime_test.json"
                ),
                "RUNFILES_MANIFEST_FILE": str(runfiles_manifest),
                "TEST_SRCDIR": str(self.root / "unmaterialized-runfiles"),
                "TEST_WORKSPACE": "_main",
            },
        ):
            self.assertEqual(
                handoff._find_manifest_from_environment(), physical_manifest
            )

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
