# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from xml.sax.saxutils import escape

from build_tools.ci import windows_diagnostics as diagnostics


class FakeRegistry:
    KEY_READ = 1
    KEY_WRITE = 2
    KEY_WOW64_64KEY = 4
    HKEY_LOCAL_MACHINE = object()
    REG_EXPAND_SZ = 10
    REG_DWORD = 11

    def __init__(self, keys=None):
        self.keys = {} if keys is None else keys

    def OpenKey(self, _root, path, _reserved=0, _access=0):
        if path not in self.keys:
            raise FileNotFoundError(path)
        return path

    def CreateKeyEx(self, _root, path, _reserved=0, _access=0):
        parent = path.rpartition("\\")[0]
        self.keys.setdefault(parent, {})
        self.keys.setdefault(path, {})
        return path

    def QueryValueEx(self, key, name):
        try:
            return self.keys[key][name]
        except KeyError:
            raise FileNotFoundError(name) from None

    def SetValueEx(self, key, name, _reserved, value_type, value):
        self.keys[key][name] = (value, value_type)

    def DeleteValue(self, key, name):
        try:
            del self.keys[key][name]
        except KeyError:
            raise FileNotFoundError(name) from None

    def QueryInfoKey(self, key):
        prefix = key + "\\"
        child_count = sum(
            path.startswith(prefix) and "\\" not in path[len(prefix) :]
            for path in self.keys
        )
        return child_count, len(self.keys[key]), 0

    def CloseKey(self, _key):
        pass

    def DeleteKeyEx(self, _root, path, _access=0, _reserved=0):
        if any(key.startswith(path + "\\") for key in self.keys):
            raise OSError(f"registry key has children: {path}")
        del self.keys[path]


class WindowsDiagnosticsTest(unittest.TestCase):
    def test_local_dump_registry_is_bounded_and_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = FakeRegistry()
            dump_directory = Path(directory) / "dumps"

            with mock.patch.object(diagnostics, "winreg", registry):
                with diagnostics.local_dump_collection(dump_directory) as state:
                    self.assertTrue(state["enabled"])
                    self.assertTrue(dump_directory.is_dir())
                    for executable in diagnostics.LOCAL_DUMP_EXECUTABLES:
                        values = registry.keys[
                            diagnostics.LOCAL_DUMPS_REGISTRY_PATH + "\\" + executable
                        ]
                        self.assertEqual(
                            values,
                            {
                                "DumpFolder": (
                                    str(dump_directory),
                                    registry.REG_EXPAND_SZ,
                                ),
                                "DumpCount": (
                                    diagnostics.LOCAL_DUMP_COUNT,
                                    registry.REG_DWORD,
                                ),
                                "DumpType": (1, registry.REG_DWORD),
                            },
                        )

            self.assertTrue(state["restored"])
            self.assertEqual(registry.keys, {})

    def test_local_dump_registry_restores_existing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = "compile_artifact.exe"
            application_key = diagnostics.LOCAL_DUMPS_REGISTRY_PATH + "\\" + executable
            prior_values = {
                "DumpFolder": (r"C:\existing", 20),
                "DumpCount": (7, 21),
                "DumpType": (2, 21),
            }
            registry = FakeRegistry(
                {
                    diagnostics.LOCAL_DUMPS_REGISTRY_PATH: {},
                    application_key: prior_values.copy(),
                }
            )

            with (
                mock.patch.object(diagnostics, "winreg", registry),
                mock.patch.object(diagnostics, "LOCAL_DUMP_EXECUTABLES", (executable,)),
            ):
                with diagnostics.local_dump_collection(Path(directory) / "dumps"):
                    pass

            self.assertEqual(registry.keys[application_key], prior_values)

    def test_local_dump_setup_failure_fails_a_successful_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = FakeRegistry()
            registry.OpenKey = mock.Mock(
                side_effect=PermissionError("registry write denied")
            )

            with (
                mock.patch.object(diagnostics, "winreg", registry),
                contextlib.redirect_stderr(io.StringIO()) as stderr,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = diagnostics.run(
                    [sys.executable, "-c", "pass"],
                    cwd=root,
                    env=None,
                    artifact_dir=root / "artifacts",
                    label="successful command",
                )

            self.assertEqual(result, 1)
            self.assertIn("Local dump setup failed", stderr.getvalue())
            phase = next((root / "artifacts").iterdir())
            command = json.loads((phase / "command.json").read_text())
            self.assertIn(
                "registry write denied", command["local_dumps"]["setup_error"]
            )

    def test_local_dumps_are_inventoried_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            dump = artifacts / "crash-dumps" / "tool.exe.123.dmp"
            dump.parent.mkdir()
            dump.write_bytes(b"minidump")
            capture = diagnostics.Capture(artifacts, {})

            self.assertEqual(diagnostics.capture_local_dumps(capture), [str(dump)])
            record = capture.files[str(dump)]
            self.assertEqual(
                record["artifact"], str(Path("crash-dumps") / "tool.exe.123.dmp")
            )
            self.assertEqual(record["sha256"], hashlib.sha256(b"minidump").hexdigest())
            self.assertNotIn("copy", record)

    def test_process_output_and_status_survive_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for code in (0, 7):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = diagnostics.run(
                        [
                            sys.executable,
                            "-c",
                            "import sys; "
                            "sys.stdout.buffer.write(b'out\\xff\\n'); "
                            "sys.stdout.flush(); sys.stderr.buffer.write(b'err\\n'); "
                            f"sys.exit({code})",
                        ],
                        cwd=root,
                        env=None,
                        artifact_dir=root / "artifacts",
                        label="Build repeated phase",
                    )
                self.assertEqual(result, code)
            phases = list((root / "artifacts").iterdir())
            self.assertEqual(len(phases), 2)
            for phase in phases:
                self.assertEqual((phase / "output.log").read_bytes(), b"out\xff\nerr\n")
                command = json.loads((phase / "command.json").read_text())
                self.assertEqual(
                    (phase / "manifest.json").exists(), command["returncode"] != 0
                )

    def test_ninja_inputs_and_response_files_are_copied_before_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build with spaces"
            build.mkdir()
            artifacts = root / "artifacts"
            artifacts.mkdir()
            tool_dir = root / "tools"
            tool_dir.mkdir()
            (tool_dir / "llvm-nm.exe").touch()
            (build / "CMakeCache.txt").write_text(
                "CMAKE_MAKE_PROGRAM:FILEPATH=ninja\n"
                f"CMAKE_C_COMPILER:FILEPATH={tool_dir}/clang-cl.exe\n"
            )
            objects = {
                "backend.obj": b"backend bytes",
                "common.lib": b"archive bytes",
                "common source.obj": b"archive member bytes",
            }
            for name, contents in objects.items():
                (build / name).write_bytes(contents)
            (build / "link.rsp").write_bytes(
                '"common source.obj" common.lib'.encode("utf-16")
            )
            # The graph fits alone but competes with the actual failed-link
            # inputs for the capture budget, as in large generated CMake builds.
            (build / "build.ninja").write_bytes(b"graph" * 20)
            capture_budget = sum(len(contents) for contents in objects.values())
            capture_budget += (build / "CMakeCache.txt").stat().st_size
            capture_budget += (build / "link.rsp").stat().st_size
            (artifacts / "output.log").write_text(
                "FAILED: [code=4294967295] test with spaces.exe test.pdb\n"
            )
            (root / "sdk.lib").write_bytes(b"installed library")

            def native_tool(argv, **kwargs):
                if argv[0] == "wevtutil.exe":
                    self.assertEqual(argv[2], "Application")
                    kwargs["stdout"].write(b"<Events/>")
                elif argv[1:4] == ["-t", "inputs", "--no-shell-escape"]:
                    self.assertEqual(argv[4], "test with spaces.exe")
                    kwargs["stdout"].write(
                        ("\n".join(objects) + "\n../sdk.lib\n").encode()
                    )
                elif argv[1:3] == ["-t", "commands"]:
                    kwargs["stdout"].write(b"lld-link @link.rsp")
                else:
                    self.assertEqual(argv[1], "--print-armap")
                    snapshot = Path(argv[2])
                    self.assertEqual(snapshot.parent, artifacts)
                    self.assertIn(snapshot.read_bytes(), objects.values())
                    kwargs["stdout"].write(b"T defined_symbol\n")
                return subprocess.CompletedProcess(argv, 0)

            # The native graph query and symbol reader are dependencies. The
            # collector, filesystem snapshots, and manifest writer all run.
            with (
                mock.patch.object(subprocess, "run", side_effect=native_tool),
                mock.patch.object(diagnostics, "MAX_CAPTURE_BYTES", capture_budget),
            ):
                diagnostics.capture_failure(
                    artifacts, root, {}, build, started_at=dt.datetime.now(dt.UTC)
                )
            manifest = json.loads((artifacts / "manifest.json").read_text())
            records = {
                Path(record["path"]).name: record for record in manifest["files"]
            }
            for name, contents in objects.items():
                (build / name).write_bytes(b"rebuild replaced the original")
                self.assertIn("copy", records[name], records[name])
                self.assertEqual(
                    (artifacts / records[name]["copy"]).read_bytes(), contents
                )
                self.assertEqual(
                    records[name]["sha256"], hashlib.sha256(contents).hexdigest()
                )
            self.assertEqual(
                (artifacts / records["link.rsp"]["copy"]).read_bytes(),
                (build / "link.rsp").read_bytes(),
            )
            self.assertNotIn("sdk.lib", records)
            self.assertEqual(
                records["build.ninja"]["error"], "capture byte limit exceeded"
            )
            self.assertEqual(
                sum("--print-armap" in tool["argv"] for tool in manifest["tools"]), 3
            )

    def test_header_access_and_bazel_parameters_are_observed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk = root / "Windows SDK" / "ucrt"
            sdk.mkdir(parents=True)
            (sdk / "wchar.h").write_text("#include <missing.h>\n")
            params = root / "bazel-out/bin/cache.obj.params"
            params.parent.mkdir(parents=True)
            params.write_bytes(b"/I SDK\r\n/c cache.cc\r\n")
            action_error = root / "actions" / "stderr-15695"
            action_error.parent.mkdir()
            action_error.write_bytes(b"compiler diagnostic\r\n")
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "output.log").write_text(
                f"{sdk}/wchar.h(17): fatal error C1083: Cannot open include file: 'missing.h': No such file or directory\n"
                "cl.exe @bazel-out/bin/cache.obj.params\n"
                f"ERROR: Couldn't delete action output directory: {action_error} (Permission denied)\n"
            )
            diagnostics.capture_failure(
                artifacts,
                root,
                {"Include": str(sdk), "A_SECRET": "not captured", "PATH": ""},
                None,
                started_at=dt.datetime.now(dt.UTC),
            )
            manifest_text = (artifacts / "manifest.json").read_text()
            manifest = json.loads(manifest_text)
            records = {
                Path(record["path"]).name: record for record in manifest["files"]
            }
            self.assertIn("sha256", records["wchar.h"])
            self.assertNotIn("copy", records["wchar.h"])
            self.assertIn("error", records["missing.h"])
            self.assertEqual(records["missing.h"]["parent_entries"], ["wchar.h"])
            self.assertEqual(
                (artifacts / records[params.name]["copy"]).read_bytes(),
                params.read_bytes(),
            )
            action_error.write_bytes(b"a later action replaced the diagnostic")
            self.assertEqual(
                (artifacts / records[action_error.name]["copy"]).read_bytes(),
                b"compiler diagnostic\r\n",
            )
            self.assertNotIn("A_SECRET", manifest_text)
            self.assertNotIn("not captured", manifest_text)

    def test_generator_output_ebadf_preserves_only_bazel_output_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "bazel-out"
            generated = output_root / "x64_windows-opt-exec/bin/pkg/generated"
            generated.mkdir(parents=True)
            (generated / "sibling.h").write_text("retained output\n")
            failed_output = generated / "catalog.c"
            outside = root / "host-secret"
            outside.write_text("must not be inspected\n")
            artifacts = root / "artifacts"
            artifacts.mkdir()
            relative_output = str(failed_output.relative_to(root))
            (artifacts / "output.log").write_text(
                "path.write_text(contents, encoding='utf-8')\n"
                "return io.open(self, mode, buffering, encoding, errors, newline)\n"
                f"OSError: [Errno 9] Bad file descriptor: {relative_output!r}\n"
                f"OSError: [Errno 9] Bad file descriptor: {str(outside)!r}\n"
            )

            diagnostics.capture_failure(
                artifacts,
                root,
                {"PATH": ""},
                None,
                started_at=dt.datetime.now(dt.UTC),
            )

            manifest = json.loads((artifacts / "manifest.json").read_text())
            files = {record["path"]: record for record in manifest["files"]}
            self.assertIn(str(failed_output), files)
            self.assertEqual(files[str(failed_output)]["errno"], 2)
            self.assertNotIn(str(outside), files)
            directories = {record["path"]: record for record in manifest["directories"]}
            self.assertEqual(directories[str(generated)]["entries"], ["sibling.h"])
            self.assertEqual(
                directories[str(generated)]["resolved_path"], str(generated)
            )
            self.assertIn(str(output_root), directories)
            self.assertNotIn(str(root), directories)
            if os.name == "nt":
                tools = {entry["output"]: entry for entry in manifest["tools"]}
                for name in (
                    "windows-file-access",
                    "windows-volume-info",
                    "windows-filter-instances",
                ):
                    output_name, tool = next(
                        (output_name, tool)
                        for output_name, tool in tools.items()
                        if name in output_name
                    )
                    self.assertEqual(
                        tool.get("returncode"),
                        0,
                        (tool, (artifacts / output_name).read_text(errors="replace")),
                    )
                access_output = next(
                    artifacts / name for name in tools if "windows-file-access" in name
                )
                self.assertTrue(
                    (artifacts / tools[access_output.name]["stderr_output"]).is_file()
                )
                access_text = access_output.read_text()
                observations = []
                for line_number, line in enumerate(access_text.splitlines(), start=1):
                    try:
                        observations.append(json.loads(line))
                    except json.JSONDecodeError:
                        self.fail(
                            f"{access_output.name} line {line_number} is not JSON: "
                            f"{line!r}\ncomplete output:\n{access_text}"
                        )
                file_access = [entry["file_access"] for entry in observations]
                self.assertEqual(file_access[0]["read"]["winerror"], 2)
                self.assertTrue(
                    all(
                        entry["read_directory"]["winerror"] == 0
                        for entry in file_access[1:]
                    )
                )

    def test_failed_process_image_survives_rebuild_and_resource_query_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "bazel-out" / "opt exec" / "bin" / "loom-link.exe"
            executable.parent.mkdir(parents=True)
            original = b"original executable image\x00\xff"
            executable.write_bytes(original)
            installed = root / "bazel-outside" / "sdk-tool.exe"
            installed.parent.mkdir()
            installed.write_bytes(b"installed toolchain image")
            missing = executable.with_name("missing.exe")
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "output.log").write_text(
                "\n".join(
                    f'CreateProcessW("{path}" --mode=merge): '
                    "Not enough memory resources are available (error: 8)"
                    for path in (executable, executable, installed, missing)
                )
            )

            # Native resource-query tools are dependencies; an unavailable query
            # must not lose the image that the real collector reads and hashes.
            with mock.patch.object(
                subprocess, "run", side_effect=FileNotFoundError("native query missing")
            ):
                diagnostics.capture_failure(
                    artifacts,
                    root,
                    {"PATH": ""},
                    None,
                    started_at=dt.datetime.now(dt.UTC),
                )
            executable.write_bytes(b"rebuilt image")
            manifest = json.loads((artifacts / "manifest.json").read_text())
            self.assertNotIn("capture_error", manifest)
            records = {record["path"]: record for record in manifest["files"]}
            self.assertEqual(set(records), {str(executable), str(missing)})
            record = records[str(executable)]
            self.assertEqual((artifacts / record["copy"]).read_bytes(), original)
            self.assertEqual(record["sha256"], hashlib.sha256(original).hexdigest())
            self.assertIn("error", records[str(missing)])
            self.assertTrue(all("error" in tool for tool in manifest["tools"]))

    def test_policy_capture_failure_preserves_command_and_other_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            denied = "Failed to open log. Access is denied.\r\n".encode("utf-16")
            policies = b'{"Policies": [{"PolicyID": "blocked-policy"}]}\r\n'

            def native_tool(argv, **kwargs):
                if argv[0] == "wevtutil.exe":
                    kwargs["stdout"].write(denied)
                    return subprocess.CompletedProcess(argv, 5)
                kwargs["stdout"].write(policies)
                return subprocess.CompletedProcess(argv, 0)

            # Only native diagnostic tools are substituted. The failed child,
            # log capture, command timestamps, and artifact writing are real.
            with (
                mock.patch.object(subprocess, "run", side_effect=native_tool),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = diagnostics.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys; print('CreateProcessW failed (error: 4551)'); "
                        "sys.exit(3)",
                    ],
                    cwd=root,
                    env=None,
                    artifact_dir=root / "artifacts",
                    label="Test",
                )
            self.assertEqual(result, 3)
            phase = next((root / "artifacts").iterdir())
            command = json.loads((phase / "command.json").read_text())
            manifest = json.loads((phase / "manifest.json").read_text())
            events, policy_list = [
                tool
                for tool in manifest["tools"]
                if "application-crash-events" not in tool["output"]
            ]
            self.assertEqual(events["returncode"], 5)
            self.assertEqual((phase / events["output"]).read_bytes(), denied)
            self.assertEqual(policy_list["returncode"], 0)
            self.assertEqual((phase / policy_list["output"]).read_bytes(), policies)
            started_at = dt.datetime.fromisoformat(command["started_at"])
            timestamp = started_at.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            )
            self.assertIn(
                f"TimeCreated[@SystemTime >= '{timestamp}']", events["argv"][3]
            )

    def test_silent_failure_retains_crash_images_before_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "bazel-out/bin/tool & \N{GREEK SMALL LETTER LAMDA}.exe"
            module = root / "bazel-out/bin/runtime/library.dll"
            installed = root / "bazel-outside/system.dll"
            program_data = root / "ProgramData"
            wer_dump = program_data / "Microsoft/Windows/WER/Temp/WER.tmp.dmp"
            unrelated_dump = wer_dump.with_name("unrelated.tmp.dmp")
            original = {
                executable: b"executable image",
                module: b"runtime image",
                executable.with_suffix(".pdb"): b"executable symbols",
                module.with_suffix(".pdb"): b"runtime symbols",
                wer_dump: b"minidump",
            }
            for path, contents in {
                **original,
                installed: b"installed image",
                unrelated_dump: b"unrelated minidump",
            }.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
            event = (
                '<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">'
                '<System><Provider Name="Application Error"/><EventID>1000</EventID>'
                '</System><EventData><Data Name="ExceptionCode">c0000005</Data>'
                '<Data Name="FaultingOffset">0000000000001234</Data>'
                f'<Data Name="AppPath">{escape(str(executable))}</Data>'
                f'<Data Name="ModulePath">{escape(str(module))}</Data>'
                "</EventData></Event>"
            )
            wer_fields = [""] * 15
            wer_fields[5] = "compile_artifact.exe"
            wer_event = (
                '<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">'
                '<System><Provider Name="Windows Error Reporting"/>'
                "<EventID>1001</EventID><Version>0</Version></System><EventData>"
                + "".join(f"<Data>{value}</Data>" for value in wer_fields)
                + f"<Data>\n{escape(str(wer_dump))}\n</Data>"
                + "</EventData></Event>"
            )
            # Repeated failures share images; installed modules must remain
            # identified in the raw events without being copied into artifacts.
            events = (
                "<Events>"
                + event
                + event.replace(escape(str(module)), escape(str(installed)))
                + wer_event
                + wer_event.replace("compile_artifact.exe", "python.exe").replace(
                    escape(str(wer_dump)), escape(str(unrelated_dump))
                )
                + "</Events>"
            ).encode("utf-16")

            def native_tool(argv, **kwargs):
                self.assertEqual(argv[:3], ["wevtutil.exe", "qe", "Application"])
                self.assertIn("Provider[@Name='Application Error']", argv[3])
                self.assertIn("Provider[@Name='Windows Error Reporting']", argv[3])
                self.assertIn("/c:64", argv)
                kwargs["stdout"].write(events)
                return subprocess.CompletedProcess(argv, 0)

            # The child exits silently; only the native event reader is replaced.
            # Run the real phase collector, XML parser, snapshots, and manifest.
            with (
                mock.patch.object(subprocess, "run", side_effect=native_tool),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                environment = os.environ.copy()
                environment["PROGRAMDATA"] = str(program_data)
                result = diagnostics.run(
                    [sys.executable, "-c", "raise SystemExit(3)"],
                    cwd=root,
                    env=environment,
                    artifact_dir=root / "artifacts",
                    label="silent test failure",
                )
            self.assertEqual(result, 3)
            phase = next((root / "artifacts").iterdir())
            self.assertEqual((phase / "output.log").read_bytes(), b"")
            manifest = json.loads((phase / "manifest.json").read_text())
            self.assertNotIn("capture_error", manifest)
            query = manifest["tools"][0]
            self.assertEqual(query["returncode"], 0)
            self.assertEqual((phase / query["output"]).read_bytes(), events)
            started_at = dt.datetime.fromisoformat(
                json.loads((phase / "command.json").read_text())["started_at"]
            )
            timestamp = started_at.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            )
            self.assertIn(
                f"TimeCreated[@SystemTime >= '{timestamp}']", query["argv"][3]
            )
            records = {Path(record["path"]): record for record in manifest["files"]}
            self.assertEqual(set(records), set(original))
            for path, contents in original.items():
                path.write_bytes(b"a later phase replaced these bytes")
                record = records[path]
                self.assertEqual((phase / record["copy"]).read_bytes(), contents)
                self.assertEqual(record["sha256"], hashlib.sha256(contents).hexdigest())

    def test_classic_application_error_retains_build_owned_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "bazel-out/bin/compiler.exe"
            module = root / "bazel-out/bin/compiler.dll"
            installed = root / "system.dll"
            other_provider = root / "bazel-out/bin/other.exe"
            original = {
                executable: b"compiler image",
                module: b"compiler runtime",
                executable.with_suffix(".pdb"): b"compiler symbols",
            }
            for path, contents in {
                **original,
                installed: b"system image",
                other_provider: b"not an application error image",
            }.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
            # Version-zero Application Error events on Windows CI contain
            # unnamed insertion strings, including the two paths at 10 and 11.
            fields = [
                "compiler.exe",
                "0.0.0.0",
                "5bf93434",
                "compiler.dll",
                "0.0.0.0",
                "5bf93434",
                "c0000005",
                "00000000002a7001",
                "5b1c",
                "01dd6dafec3f143a",
                str(executable),
                str(module),
                "report-id",
                "",
                "",
            ]
            event = (
                '<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">'
                '<System><Provider Name="Application Error"/>'
                '<EventID Qualifiers="0">1000</EventID><Version>0</Version>'
                "</System><EventData>"
                + "".join(f"<Data>{escape(value)}</Data>" for value in fields)
                + "</EventData></Event>"
            )
            events = (
                "<Events>"
                + event
                + event.replace(escape(str(module)), escape(str(installed)))
                + event.replace("Application Error", "Windows Error Reporting")
                .replace(">1000<", ">1001<")
                .replace(escape(str(executable)), escape(str(other_provider)))
                + "</Events>"
            ).encode("utf-16")

            def native_tool(argv, **kwargs):
                kwargs["stdout"].write(events)
                return subprocess.CompletedProcess(argv, 0)

            artifacts = root / "artifacts"
            artifacts.mkdir()
            capture = diagnostics.Capture(artifacts, {})
            with mock.patch.object(subprocess, "run", side_effect=native_tool):
                diagnostics.capture_application_crashes(
                    capture, root, None, dt.datetime.now(dt.UTC)
                )
            self.assertEqual(set(capture.files), {str(path) for path in original})
            for path, contents in original.items():
                self.assertEqual(
                    (artifacts / capture.files[str(path)]["copy"]).read_bytes(),
                    contents,
                )

    def test_crash_images_precede_symbols_in_the_capture_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "cmake build"
            build.mkdir()
            executable = build / "tool.exe"
            module = build / "runtime.dll"
            executable.write_bytes(b"image")
            module.write_bytes(b"library")
            executable.with_suffix(".pdb").write_bytes(b"large symbols")
            artifacts = root / "artifacts"
            artifacts.mkdir()
            events = (
                "<Events><Event><EventData>"
                f'<Data Name="AppPath">{escape(str(executable))}</Data>'
                f'<Data Name="ModulePath">{escape(str(module))}</Data>'
                "</EventData></Event></Events>"
            ).encode("utf-16")

            def native_tool(argv, **kwargs):
                kwargs["stdout"].write(events)
                return subprocess.CompletedProcess(argv, 0)

            capture = diagnostics.Capture(artifacts, {})
            capture.remaining_bytes = len(b"imagelibrary")
            with mock.patch.object(subprocess, "run", side_effect=native_tool):
                diagnostics.capture_application_crashes(
                    capture, root, build, dt.datetime.now(dt.UTC)
                )
            for path in (executable, module):
                record = capture.files[str(path)]
                self.assertEqual(
                    (artifacts / record["copy"]).read_bytes(), path.read_bytes()
                )
            self.assertEqual(
                capture.files[str(executable.with_suffix(".pdb"))]["error"],
                "capture byte limit exceeded",
            )

    def test_failed_bazel_tests_retain_runfiles_before_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            workspace = root / "workspace with spaces"
            output = workspace / "bazel-out/x64_windows-fastbuild"
            binary_root = output / "bin"
            compiler = binary_root / "tools/compiler.exe"
            runtime = binary_root / "tools/runtime.dll"
            symbols = compiler.with_suffix(".pdb")
            source = workspace / "inputs/literal/name with spaces.loom"
            installed = root / "sdk/runtime.dll"
            originals = {
                compiler: b"failed compiler image",
                runtime: b"runtime image",
                source: b"authored source",
            }
            manifests = []
            logs = []
            for index, suffix in enumerate(
                (".runfiles_manifest", ".exe.runfiles_manifest")
            ):
                target = binary_root / f"tests/failure{index}"
                wrapper = target.with_name(target.name + ("" if index == 0 else ".exe"))
                fixture = binary_root / f"tests/input{index}.loombc"
                manifest = target.with_name(target.name + suffix)
                log = output / f"testlogs/tests/failure{index}/test.log"
                logs.append(log)
                manifests.append(manifest)
                originals.update(
                    {
                        wrapper: b"test launcher",
                        fixture: f"linked input {index}".encode(),
                        log: b"Assertion failed: binding.type != INVALID\n",
                    }
                )
                # Place symbols first to ensure manifest order cannot consume
                # the budget before other tests' executable inputs are copied.
                escaped_source = str(source).replace("\\", r"\b")
                originals[manifest] = (
                    f"_main/compiler.pdb {symbols}\n"
                    f"_main/compiler.exe {compiler}\n"
                    f"_main/runtime.dll {runtime}\n"
                    f"_main/input.loombc {fixture}\n"
                    f" _main/source\\sname\\n {escaped_source}\n"
                    "_main/empty.py \n"
                    f"sdk/runtime.dll {installed}\n"
                ).encode()
            for path, contents in {
                **originals,
                symbols: b"optional symbols" * 100,
                installed: b"installed SDK image",
            }.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
            summary = "\n".join(
                f"//tests:failure{index} FAILED in 0.1s\n  {log}"
                for index, log in enumerate(logs)
            )
            summary += "\n//tests:success PASSED in 0.1s\n"

            def native_tool(argv, **kwargs):
                self.assertEqual(argv[:3], ["wevtutil.exe", "qe", "Application"])
                kwargs["stdout"].write(b"<Events/>")
                return subprocess.CompletedProcess(argv, 0)

            # Exercise the actual failed-command boundary. Only the external
            # event reader is replaced; capture follows real manifests/files.
            with (
                mock.patch.object(subprocess, "run", side_effect=native_tool),
                mock.patch.object(
                    diagnostics,
                    "MAX_CAPTURE_BYTES",
                    sum(len(contents) for contents in originals.values()),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = diagnostics.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys; print(sys.argv[1]); raise SystemExit(3)",
                        summary,
                    ],
                    cwd=workspace,
                    env=None,
                    artifact_dir=root / "artifacts",
                    label="failed Bazel tests",
                )
            self.assertEqual(result, 3)
            phase = next((root / "artifacts").iterdir())
            metadata = json.loads((phase / "manifest.json").read_text())
            self.assertNotIn("capture_error", metadata)
            records = {Path(record["path"]): record for record in metadata["files"]}
            self.assertEqual(set(records), {*originals, symbols})
            for path, contents in originals.items():
                path.write_bytes(b"later build replaced the original")
                record = records[path]
                self.assertEqual((phase / record["copy"]).read_bytes(), contents)
                self.assertEqual(record["sha256"], hashlib.sha256(contents).hexdigest())
            self.assertEqual(records[symbols]["error"], "capture byte limit exceeded")
            self.assertEqual(len(metadata["bazel_tests"]), 2)
            for index, test in enumerate(metadata["bazel_tests"]):
                self.assertEqual(test["manifest"], str(manifests[index]))
                entries = {entry["name"]: entry for entry in test["runfiles"]}
                self.assertEqual(entries["_main/source name\n"]["path"], str(source))
                self.assertTrue(entries["_main/empty.py"]["empty"])
                self.assertEqual(
                    entries["sdk/runtime.dll"]["excluded"],
                    "runfile is outside the build and source trees",
                )

    def test_bazel_manifest_and_input_failures_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            artifacts = root / "artifacts"
            artifacts.mkdir()
            output = root / "bazel-out/config"
            binary_root = output / "bin/tests"
            binary_root.mkdir(parents=True)
            missing_input = binary_root / "missing.loombc"
            manifest = binary_root / "failure.runfiles_manifest"
            manifest.write_text(
                "malformed-entry\r\n"
                "_main/relative relative/input.loombc\r\n"
                f"_main/missing {missing_input}\r\n",
                encoding="utf-8",
                newline="",
            )
            logs = [
                root / "outside/test.log",
                root / "bazel-out/wrong-layout/test.log",
                output / "testlogs/tests/no_manifest/test.log",
                output / "testlogs/tests/failure/test.log",
            ]
            for log in logs:
                log.parent.mkdir(parents=True, exist_ok=True)
                log.touch()
            capture = diagnostics.Capture(artifacts, {})
            tests = diagnostics.capture_bazel_tests(
                capture, root, "\n".join(f"  {log}" for log in logs)
            )
            self.assertEqual(len(tests), 4)
            self.assertEqual(
                tests[0]["error"], "test log is outside the Bazel output tree"
            )
            self.assertEqual(
                tests[1]["error"],
                "test log does not identify a configured Bazel test",
            )
            self.assertEqual(
                tests[2]["error"], "failed test runfiles manifest is missing"
            )
            self.assertEqual(
                tests[2]["manifest_candidates"],
                [
                    str(binary_root / f"no_manifest{suffix}")
                    for suffix in (".runfiles_manifest", ".exe.runfiles_manifest")
                ],
            )
            entries = tests[3]["runfiles"]
            self.assertEqual(entries[0]["error"], "invalid runfiles manifest entry")
            self.assertEqual(entries[0]["line"], 1)
            self.assertEqual(
                entries[1]["error"], "runfile target is not an absolute path"
            )
            self.assertIn("error", capture.files[str(missing_input)])
            self.assertNotIn(str(logs[0]), capture.files)

    def test_bazel_manifest_capture_limit_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "bazel-out/config"
            log = output / "testlogs/tests/failure/test.log"
            log.parent.mkdir(parents=True)
            log.touch()
            manifest = output / "bin/tests/failure.runfiles_manifest"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("_main/empty.py \n")
            capture = diagnostics.Capture(root, {})
            capture.remaining_bytes = 0
            tests = diagnostics.capture_bazel_tests(capture, root, f"  {log}\n")
            self.assertEqual(
                tests[0]["error"],
                "failed test runfiles manifest could not be captured",
            )
            self.assertEqual(
                capture.files[str(manifest)]["error"], "capture byte limit exceeded"
            )

    def test_bad_crash_events_preserve_other_failure_evidence(self):
        for output, returncode in ((b"Access denied", 5), (b"invalid XML", 0)):
            with (
                self.subTest(returncode=returncode),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                executable = root / "bazel-out/bin/tool.exe"
                executable.parent.mkdir(parents=True)
                executable.write_bytes(b"failed launch image")
                artifacts = root / "artifacts"
                artifacts.mkdir()
                (artifacts / "output.log").write_text(f'CreateProcessW("{executable}")')

                def native_tool(argv, **kwargs):
                    kwargs["stdout"].write(output)
                    return subprocess.CompletedProcess(argv, returncode)

                with mock.patch.object(subprocess, "run", side_effect=native_tool):
                    diagnostics.capture_failure(
                        artifacts,
                        root,
                        {"PATH": ""},
                        None,
                        started_at=dt.datetime.now(dt.UTC),
                    )
                manifest = json.loads((artifacts / "manifest.json").read_text())
                self.assertNotIn("capture_error", manifest)
                query = manifest["tools"][0]
                self.assertEqual(query["returncode"], returncode)
                self.assertEqual((artifacts / query["output"]).read_bytes(), output)
                if returncode == 0:
                    self.assertIn("capture_error", query)
                self.assertEqual(
                    (artifacts / manifest["files"][0]["copy"]).read_bytes(),
                    b"failed launch image",
                )

    def test_unreadable_files_and_limits_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = diagnostics.Capture(root, {})
            unreadable = root / "directory.obj"
            unreadable.mkdir()
            self.assertIn("error", capture.file(unreadable, copy=True))
            self.assertIn("error", capture.file(root / "missing.obj", copy=True))
            oversized = root / "large.obj"
            oversized.write_bytes(b"1234")
            capture.remaining_bytes = 3
            self.assertEqual(
                capture.file(oversized, copy=True)["error"],
                "capture byte limit exceeded",
            )

    def test_resource_capture_preserves_failure_when_event_access_is_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def native_tool(argv, **kwargs):
                if argv[0] == "wevtutil.exe":
                    kwargs["stdout"].write(b"Access is denied.\n")
                    return subprocess.CompletedProcess(argv, 5)
                kwargs["stdout"].write(b'{"memory": {}, "processes": []}\n')
                return subprocess.CompletedProcess(argv, 0)

            for error_code in (8, 14, 1450, 1455):
                with (
                    self.subTest(error_code=error_code),
                    mock.patch.object(subprocess, "run", side_effect=native_tool),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    artifacts = root / str(error_code)
                    result = diagnostics.run(
                        [
                            sys.executable,
                            "-c",
                            "import sys; "
                            f"print('CreateProcessW failed (error: {error_code})'); "
                            "sys.exit(34)",
                        ],
                        cwd=root,
                        env=None,
                        artifact_dir=artifacts,
                        label="resource failure",
                    )
                    self.assertEqual(result, 34)
                    phase = next(artifacts.iterdir())
                    manifest = json.loads((phase / "manifest.json").read_text())
                    events, snapshot = [
                        tool
                        for tool in manifest["tools"]
                        if "application-crash-events" not in tool["output"]
                    ]
                    self.assertEqual(events["returncode"], 5)
                    self.assertEqual(snapshot["returncode"], 0)
                    self.assertIn(
                        "Access is denied", (phase / events["output"]).read_text()
                    )
                    self.assertEqual(
                        json.loads((phase / snapshot["output"]).read_text())[
                            "processes"
                        ],
                        [],
                    )
                    command = json.loads((phase / "command.json").read_text())
                    timestamp = (
                        dt.datetime.fromisoformat(command["started_at"])
                        .isoformat(timespec="milliseconds")
                        .replace("+00:00", "Z")
                    )
                    self.assertIn(
                        f"TimeCreated[@SystemTime >= '{timestamp}']", events["argv"][3]
                    )

    @unittest.skipUnless(os.name == "nt", "requires native Windows event queries")
    def test_native_crash_query_preserves_silent_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                result = diagnostics.run(
                    [sys.executable, "-c", "raise SystemExit(3)"],
                    cwd=root,
                    env=None,
                    artifact_dir=root / "artifacts",
                    label="silent failure",
                )
            self.assertEqual(result, 3)
            phase = next((root / "artifacts").iterdir())
            manifest = json.loads((phase / "manifest.json").read_text())
            self.assertNotIn("capture_error", manifest)
            query = manifest["tools"][0]
            self.assertEqual(query["returncode"], 0, query)
            self.assertNotIn("capture_error", query)

    @unittest.skipUnless(os.name == "nt", "requires native Windows resource queries")
    def test_native_resource_snapshot_preserves_process_launch_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                result = diagnostics.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys; "
                        "print('CreateProcessW failed (error: 8)'); sys.exit(34)",
                    ],
                    cwd=root,
                    env=None,
                    artifact_dir=root / "artifacts",
                    label="resource failure",
                )
            self.assertEqual(result, 34)
            phase = next((root / "artifacts").iterdir())
            manifest = json.loads((phase / "manifest.json").read_text())
            events, snapshot = [
                tool
                for tool in manifest["tools"]
                if "application-crash-events" not in tool["output"]
            ]
            self.assertEqual(events["returncode"], 0, events)
            self.assertEqual(snapshot["returncode"], 0, snapshot)
            state = json.loads((phase / snapshot["output"]).read_text())
            self.assertGreater(int(state["memory"]["TotalVisibleMemorySize"]), 0)
            self.assertGreater(len(state["processes"]), 0)
            self.assertLessEqual(len(state["processes"]), 32)
            self.assertTrue(all(process["Id"] >= 0 for process in state["processes"]))

    def test_native_diagnostic_failure_keeps_output_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = diagnostics.Capture(root, os.environ)
            output = capture.tool(
                [sys.executable, "-c", "import sys; print('tool error'); sys.exit(23)"],
                root,
                "tool",
            )
            self.assertIsNone(output)
            self.assertEqual(capture.tools[0]["returncode"], 23)
            self.assertIn("tool error", (root / capture.tools[0]["output"]).read_text())

    def test_protocol_tool_separates_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = diagnostics.Capture(root, os.environ)
            output = capture.tool(
                [
                    sys.executable,
                    "-c",
                    "import sys; "
                    "print('{\"record\": 42}'); "
                    "print('loader warning', file=sys.stderr)",
                ],
                root,
                "protocol",
                separate_stderr=True,
            )

            record = capture.tools[0]
            self.assertEqual(output, root / record["output"])
            self.assertEqual(
                json.loads((root / record["output"]).read_text()), {"record": 42}
            )
            self.assertEqual(
                (root / record["stderr_output"]).read_text().strip(),
                "loader warning",
            )

    def test_metadata_write_failure_preserves_build_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_write_text = Path.write_text

            def write_text(path, *args, **kwargs):
                if path.name == "command.json":
                    raise PermissionError("worker cannot write command metadata")
                return real_write_text(path, *args, **kwargs)

            with (
                mock.patch.object(Path, "write_text", write_text),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()) as errors,
            ):
                result = diagnostics.run(
                    [sys.executable, "-c", "import sys; sys.exit(29)"],
                    cwd=root,
                    env=None,
                    artifact_dir=root / "artifacts",
                    label="failed",
                )
            self.assertEqual(result, 29)
            self.assertIn("worker cannot write", errors.getvalue())

    def test_log_write_failure_still_drains_the_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_open = Path.open

            def open_file(path, *args, **kwargs):
                stream = real_open(path, *args, **kwargs)
                if path.name == "output.log" and args and args[0] == "wb":
                    stream = mock.Mock(wraps=stream)
                    stream.write.side_effect = OSError("disk full")
                return stream

            with (
                mock.patch.object(Path, "open", open_file),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()) as errors,
            ):
                result = diagnostics.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys; sys.stdout.buffer.write(b'x' * 262144); sys.exit(37)",
                    ],
                    cwd=root,
                    env=None,
                    artifact_dir=root / "artifacts",
                    label="failed",
                )
            self.assertEqual(result, 37)
            command = json.loads(
                next((root / "artifacts").glob("*/command.json")).read_text()
            )
            self.assertEqual(command["log_error"], "disk full")
            self.assertIn("Log write failed", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
