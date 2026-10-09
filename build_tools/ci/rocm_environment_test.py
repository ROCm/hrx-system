# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from build_tools.ci import rocm_environment


class NativeGpuSelectionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.topology = self.root / "topology"
        self.drm = self.root / "drm"
        self.devices = self.root / "dri"
        for directory in (self.topology, self.drm, self.devices):
            directory.mkdir()
        # Native ROCr output identifies hardware, not its enumeration ordinal.
        self.output = (
            "Agent 1\n  Uuid:                    CPU-XX\n"
            "  Device Type:             CPU\n  BDFID:                   0\n"
            "Agent 2\n  Uuid:                    GPU-af47ea8fb2cb9b42\n"
            "  Device Type:             GPU\n  BDFID:                   17920\n"
        )
        self.add_node("7", 144, 17920, 0xAF47EA8FB2CB9B42)
        self.add_node("2", 168, 42496, 0x0123456789ABCDEF)
        node = self.drm / "renderD144"
        node.mkdir()
        (node / "dev").write_text("226:144\n")
        self.metadata = SimpleNamespace(
            st_mode=stat.S_IFCHR | 0o660, st_rdev=os.makedev(226, 144)
        )

    def add_node(self, name, minor, location, unique):
        node = self.topology / name
        node.mkdir()
        (node / "gpu_id").write_text(str(int(name) + 100))
        (node / "properties").write_text(
            f"location_id {location}\nunique_id {unique}\ndrm_render_minor {minor}\n"
        )

    def select(self):
        # Only the native character-device stat is substituted; discovery and
        # correlation read the external ROCr/sysfs records through real files.
        with mock.patch.object(Path, "stat", return_value=self.metadata):
            return rocm_environment.select_native_gpu(
                self.output, self.topology, self.drm, self.devices
            )

    def test_correlates_uuid_and_pci_to_render_identity(self):
        self.assertEqual(self.select(), "linux_device:226:144")

    def test_uuid_disambiguates_same_bdf_in_another_domain(self):
        self.add_node("9", 176, 17920, 0x1122334455667788)
        self.assertEqual(self.select(), "linux_device:226:144")

    def test_bdf_must_still_match_the_uuid(self):
        self.output = self.output.replace("17920", "42496")
        with self.assertRaisesRegex(RuntimeError, "matches 0 KFD nodes"):
            self.select()

    def test_single_bdf_can_identify_an_agent_without_a_uuid(self):
        self.output = self.output.replace("GPU-af47ea8fb2cb9b42", "GPU-XX")
        self.assertEqual(self.select(), "linux_device:226:144")

    def test_ambiguous_bdf_without_uuid_is_rejected(self):
        self.output = self.output.replace("GPU-af47ea8fb2cb9b42", "GPU-XX")
        self.add_node("9", 176, 17920, 0x1122334455667788)
        with self.assertRaisesRegex(RuntimeError, "matches 2 KFD nodes"):
            self.select()

    def test_malformed_uuid_does_not_discard_identity(self):
        self.output = self.output.replace("GPU-af47ea8fb2cb9b42", "GPU-invalid")
        with self.assertRaisesRegex(RuntimeError, "Unrecognized ROCr GPU UUID"):
            self.select()

    def test_missing_or_multiple_visible_gpus_require_an_allocation(self):
        for output in ("", self.output + self.output):
            with self.subTest(output=output):
                self.output = output
                with self.assertRaisesRegex(RuntimeError, "requires one visible"):
                    self.select()

    def test_render_node_device_number_must_match(self):
        self.metadata.st_rdev = os.makedev(226, 168)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.select()

    def test_regular_file_cannot_substitute_for_render_node(self):
        self.metadata.st_mode = stat.S_IFREG | 0o660
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.select()

    def test_drm_and_kfd_minor_must_match(self):
        (self.drm / "renderD144" / "dev").write_text("226:168\n")
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.select()


class RocmSupervisorTest(unittest.TestCase):
    def test_selects_gnu_supervisor_when_default_is_another_implementation(self):
        paths = {"timeout": "/usr/bin/timeout", "gnutimeout": "/usr/bin/gnutimeout"}
        with (
            mock.patch.object(rocm_environment.shutil, "which", side_effect=paths.get),
            mock.patch.object(
                rocm_environment.subprocess,
                "run",
                side_effect=[
                    subprocess.CompletedProcess(
                        [], 0, "timeout (uutils coreutils) 0.2.2"
                    ),
                    subprocess.CompletedProcess([], 0, "timeout (GNU coreutils) 9.5"),
                ],
            ),
        ):
            self.assertEqual(
                rocm_environment.require_gnu_timeout(), paths["gnutimeout"]
            )

    def test_missing_or_incompatible_supervisor_fails_explicitly(self):
        for path in (None, "/usr/bin/timeout"):
            with (
                self.subTest(path=path),
                mock.patch.object(rocm_environment.shutil, "which", return_value=path),
                mock.patch.object(
                    rocm_environment.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(
                        [], 0, "timeout (uutils coreutils) 0.2.2"
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "GNU coreutils timeout is required"
                ):
                    rocm_environment.require_gnu_timeout()


class RocmEnvironmentTest(unittest.TestCase):
    def probe_command(self, native_script: str, **options) -> list[str]:
        script = (
            "import signal, sys; "
            "from build_tools.ci.rocm_environment import "
            "run_probe, handle_termination, require_gnu_timeout; "
            "signal.signal(signal.SIGTERM, handle_termination); "
            f"sys.exit(run_probe([sys.executable, '-c', {native_script!r}], "
            f"timeout_tool=require_gnu_timeout(), **{options!r}))"
        )
        return [sys.executable, "-u", "-c", script]

    def test_success_and_native_errors_preserve_exit_status(self):
        for native_script, expected in (
            ("print('native probe succeeded')", 0),
            ("import sys; sys.exit(23)", 23),
            ("import os, signal; os.kill(os.getpid(), signal.SIGUSR1)", 138),
        ):
            with self.subTest(status=expected):
                result = subprocess.run(
                    self.probe_command(native_script),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    result.returncode, expected, result.stdout + result.stderr
                )
                self.assertNotIn("capturing process state", result.stdout)

    def test_timeout_captures_native_wait_state_before_termination(self):
        result = subprocess.run(
            self.probe_command(
                "import os, signal; print(f'NATIVE_PID={os.getpid()}', flush=True); "
                "signal.pause()",
                timeout_seconds=2,
                kill_after_seconds=1,
                diagnostic_delay_seconds=1,
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 124, result.stdout + result.stderr)
        native_id = result.stdout.split("NATIVE_PID=", 1)[1].splitlines()[0]
        self.assertIn(f"/proc/{native_id}/status:", result.stdout)
        self.assertIn(f"/proc/{native_id}/task/{native_id}/wchan:", result.stdout)
        self.assertIn("Recent kernel warnings and errors", result.stdout)

    def test_native_process_ignoring_term_requires_kill(self):
        result = subprocess.run(
            self.probe_command(
                "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('ignoring TERM', flush=True); signal.pause()",
                timeout_seconds=2,
                kill_after_seconds=1,
                diagnostic_delay_seconds=1,
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertIn("ignoring TERM", result.stdout)
        self.assertEqual(result.returncode, 137, result.stdout + result.stderr)

    def test_cancellation_reaches_native_child(self):
        process = subprocess.Popen(
            self.probe_command(
                "import os, signal; print(os.getpid(), flush=True); signal.pause()",
                kill_after_seconds=1,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # The child explicitly announces readiness before cancellation.
            native_id = int(process.stdout.readline())
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate()
            self.assertEqual(process.returncode, 143, stdout + stderr)
            try:
                status = Path(f"/proc/{native_id}/status").read_text()
            except FileNotFoundError:
                pass
            else:
                # A container's init may not have reaped the terminated child.
                self.assertIn("State:\tZ", status)
        finally:
            if process.poll() is None:
                process.terminate()
                process.communicate()

    def test_kernel_log_access_failure_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            dmesg = Path(temporary_dir) / "dmesg"
            dmesg.write_text(
                "#!/bin/sh\necho 'kernel log permission denied'\nexit 13\n"
            )
            dmesg.chmod(0o755)
            result = subprocess.run(
                [
                    sys.executable,
                    "build_tools/ci/rocm_environment.py",
                    "--diagnose",
                    str(os.getpid()),
                ],
                env={**os.environ, "PATH": temporary_dir},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("kernel log permission denied", result.stdout)
            self.assertIn("Kernel log unavailable (dmesg exit 13)", result.stdout)

    def test_stalled_diagnostics_are_bounded_separately(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            dmesg = Path(temporary_dir) / "dmesg"
            dmesg.write_text(
                "#!/bin/sh\ntrap '' TERM\necho 'stalled kernel reader'\nexec sleep 60\n"
            )
            dmesg.chmod(0o755)
            result = subprocess.run(
                self.probe_command(
                    "import signal; signal.pause()",
                    timeout_seconds=2,
                    kill_after_seconds=1,
                    diagnostic_delay_seconds=1,
                ),
                env={
                    **os.environ,
                    "PATH": temporary_dir + os.pathsep + os.environ["PATH"],
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 124, result.stdout + result.stderr)
            self.assertIn("Process diagnostics incomplete", result.stdout)


if __name__ == "__main__":
    unittest.main()
