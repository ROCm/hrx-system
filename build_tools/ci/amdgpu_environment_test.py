# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.ci import amdgpu_environment, gpu_assignment


class NativeGpuAdmissionTest(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.dri = root / "dri"
        self.drm = root / "drm"
        self.topology = root / "nodes"
        self.exports = root / "job.env"
        self.dri.mkdir()
        self.topology.mkdir()
        (self.topology.parent / "generation_id").write_text("1")
        self.enterContext(
            mock.patch.multiple(
                gpu_assignment,
                DRI=self.dri,
                DRM_CLASS=self.drm,
                KFD_NODES=self.topology,
            )
        )
        self.enterContext(
            mock.patch.dict(os.environ, {"GITHUB_ENV": str(self.exports)}, clear=True)
        )
        self.addCleanup(signal.signal, signal.SIGTERM, signal.getsignal(signal.SIGTERM))
        self.endpoints = {}

    def gpu(self, source, target, node_id, uuid=0):
        # Real character-device numbers exercise the kernel identity check.
        number = Path(source).stat().st_rdev
        major, minor = os.major(number), os.minor(number)
        path = self.dri / f"renderD{minor}"
        path.symlink_to(source)
        entry = self.drm / path.name
        (entry / "device").mkdir(parents=True)
        (entry / "device/vendor").write_text("0x1002")
        (entry / "dev").write_text(f"{major}:{minor}")
        node = self.topology / str(node_id)
        node.mkdir()
        (node / "properties").write_text(
            f"simd_count 64\nunique_id {uuid}\ndrm_render_minor {minor}\n"
        )
        self.endpoints[path] = {
            "uri": f"amdgpu://id/test-{minor}",
            "gpu": {"target": target},
            "native_identity": {"major": major, "minor": minor},
        }
        return path

    def run_job(self, *, target="gfx120X-all", activation=0, warnings=("", "")):
        self.queries = []
        self.activations = []

        def run_tool(command, *, stdout, **kwargs):
            self.assertEqual(command[:2], ["amdf-info", "select"])
            selector = next(
                arg.removeprefix("--device=")
                for arg in command
                if arg.startswith("--device=")
            )
            if "--activate" in command:
                self.activations.append(selector)
                self.assertFalse(self.exports.exists())
                return activation
            path = Path(selector)
            self.queries.append(path)
            endpoint = self.endpoints[path]
            json.dump({"endpoints": [endpoint] if endpoint else []}, stdout)
            return 0 if endpoint else 3

        with (
            mock.patch.object(
                amdgpu_environment, "require_gnu_timeout", return_value="timeout"
            ),
            mock.patch.object(
                amdgpu_environment, "read_kernel_warnings", side_effect=warnings
            ),
            mock.patch.object(amdgpu_environment, "run_probe", side_effect=run_tool),
        ):
            return amdgpu_environment.main(["--native", "amdf-info", target])

    def assert_admitted(self, path):
        self.assertEqual(self.activations, [self.endpoints[path]["uri"]])
        identity = gpu_assignment.native_identity(path.stat().st_rdev)
        self.assertEqual(
            dict(line.split("=", 1) for line in self.exports.read_text().splitlines()),
            {
                "HAL_AMD_GPU_ASSIGNMENT": identity,
                "AMDF_CTS_GPU_NATIVE_IDENTITY": identity,
            },
        )

    def test_selects_first_compatible_assigned_gpu(self):
        first = self.gpu("/dev/null", "gfx1100", 1)
        second = self.gpu("/dev/zero", "gfx1201", 2)
        os.environ["PARENT_GPU_DEVICES"] = f"{first},{second}"
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.queries, [first, second])
        self.assert_admitted(second)

    def test_child_and_ordinal_restrict_parent_before_any_native_query(self):
        first = self.gpu("/dev/null", "gfx1201", 10)
        second = self.gpu("/dev/zero", "gfx1201", 2)
        third = self.gpu("/dev/full", "gfx1201", 23)
        os.environ.update(
            PARENT_GPU_DEVICES=f"{first},{second},{third}",
            CHILD_GPU_DEVICES=f"{third},{second}",
            ROCR_VISIBLE_DEVICES="0",
        )
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.queries, [second])
        self.assert_admitted(second)

    def test_granted_render_node_without_a_libamdf_endpoint(self):
        older = self.gpu("/dev/null", "gfx1030", 1)
        supported = self.gpu("/dev/zero", "gfx1100", 2)
        self.endpoints[older] = None
        os.environ["PARENT_GPU_DEVICES"] = str(older)
        self.assertNotEqual(self.run_job(target="gfx110X-all"), 0)
        self.assertFalse(self.activations)
        self.assertFalse(self.exports.exists())

        os.environ["PARENT_GPU_DEVICES"] = f"{older},{supported}"
        self.assertEqual(self.run_job(target="gfx110X-all"), 0)
        self.assertEqual(self.queries, [older, supported])
        self.assert_admitted(supported)

    def test_legacy_rank_selects_the_partition_without_a_unique_uuid(self):
        cpu = self.topology / "0"
        cpu.mkdir()
        (cpu / "properties").write_text("simd_count 0\n")
        self.gpu("/dev/null", "gfx942", 2, uuid=0x1234)
        second = self.gpu("/dev/zero", "gfx942", 13, uuid=0x1234)
        os.environ["ROCR_VISIBLE_DEVICES"] = "1"
        self.assertEqual(self.run_job(target="gfx942"), 0)
        self.assertEqual(self.queries, [second])
        self.assert_admitted(second)

    def test_missing_mount_cannot_shift_an_ordinal_onto_another_job(self):
        missing = self.gpu("/dev/null", "gfx1200", 1)
        self.gpu("/dev/zero", "gfx1200", 2)
        missing.unlink()
        os.environ["ROCR_VISIBLE_DEVICES"] = "0"
        self.assertNotEqual(self.run_job(), 0)
        self.assertFalse(self.queries)
        self.assertFalse(self.exports.exists())

    def test_uuid_grant_and_native_assignment_can_narrow_to_one_card(self):
        first = self.gpu("/dev/null", "gfx1200", 1, uuid=0xAABBCC0000000001)
        second = self.gpu("/dev/zero", "gfx1200", 2, uuid=0xBBCCDD0000000002)
        os.environ.update(
            ROCR_VISIBLE_DEVICES="GPU-aab,GPU-BBc",
            HAL_AMD_GPU_ASSIGNMENT=gpu_assignment.native_identity(
                second.stat().st_rdev
            ),
        )
        self.assertEqual(self.run_job(), 0)
        self.assertNotIn(first, self.queries)
        self.assert_admitted(second)

    def test_shared_job_never_invents_or_widens_an_assignment(self):
        first = self.gpu("/dev/null", "gfx1200", 1, uuid=0xAABB000000000000)
        second = self.gpu("/dev/zero", "gfx1200", 2, uuid=0xAABB000000000001)
        identity = gpu_assignment.native_identity(second.stat().st_rdev)
        for settings in (
            {},
            {"HIP_VISIBLE_DEVICES": "0"},
            {"AMDF_CTS_GPU_NATIVE_IDENTITY": identity},
            {"PARENT_GPU_DEVICES": str(first), "CHILD_GPU_DEVICES": str(second)},
            {"CHILD_GPU_DEVICES": str(first), "ROCR_VISIBLE_DEVICES": "1"},
            {
                "PARENT_GPU_DEVICES": str(first),
                "AMDF_CTS_GPU_NATIVE_IDENTITY": identity,
            },
            {"PARENT_GPU_DEVICES": str(first), "HAL_AMD_GPU_ASSIGNMENT": identity},
            {"ROCR_VISIBLE_DEVICES": "GPU-aabb"},
            {"ROCR_VISIBLE_DEVICES": "0,2"},
            {"ROCR_VISIBLE_DEVICES": "0,0"},
            {"ROCR_VISIBLE_DEVICES": ""},
            {"PARENT_GPU_DEVICES": str(first), "CHILD_GPU_DEVICES": ""},
            {
                "PARENT_GPU_DEVICES": str(first),
                "AMDF_CTS_GPU_PEER_NATIVE_IDENTITY": identity,
            },
        ):
            with self.subTest(settings=settings), mock.patch.dict(os.environ, settings):
                self.assertNotEqual(self.run_job(), 0)
                self.assertFalse(self.queries)
                self.assertFalse(self.activations)
                self.assertFalse(self.exports.exists())

    def test_dedicated_vm_requires_one_gpu_before_target_filtering(self):
        os.environ["RUNNER_NAME"] = (
            "azure-linux-u2404-gfx1101-gpu-rocm-scale-runner-abcd"
        )
        first = self.gpu("/dev/null", "gfx1101", 1)
        self.assertEqual(self.run_job(target="gfx110X-all"), 0)
        self.assert_admitted(first)
        self.exports.unlink()
        self.gpu("/dev/zero", "gfx1201", 2)
        self.assertNotEqual(self.run_job(target="gfx110X-all"), 0)
        self.assertFalse(self.queries)
        self.assertFalse(self.exports.exists())

    def test_failed_activation_or_kernel_check_publishes_no_assignment(self):
        path = self.gpu("/dev/null", "gfx1200", 1)
        os.environ["PARENT_GPU_DEVICES"] = str(path)
        for activation, warnings in (
            (23, ("", "")),
            (0, ("", "amdgpu: Fence fallback timer expired on ring gfx_0.0.0")),
        ):
            with self.subTest(activation=activation, warnings=warnings):
                self.assertNotEqual(
                    self.run_job(activation=activation, warnings=warnings), 0
                )
                self.assertEqual(self.activations, [self.endpoints[path]["uri"]])
                self.assertFalse(self.exports.exists())


class AmdgpuKernelHealthTest(unittest.TestCase):
    def test_rejects_systemic_boot_interrupt_failure(self):
        output = "\n".join(
            [
                "amdgpu: Fence fallback timer expired on ring gfx_0.0.0",
                "amdgpu: Fence fallback timer expired on ring comp_1.0.0",
                "amdgpu: Fence fallback timer expired on ring sdma0",
                "amdgpu: Fence fallback timer expired on ring vcn_unified_0",
            ]
        )
        self.assertEqual(
            amdgpu_environment.find_amdgpu_interrupt_failures(output),
            output.splitlines(),
        )
        with self.assertRaisesRegex(
            RuntimeError, "refusing native GPU admission"
        ) as raised:
            amdgpu_environment.require_healthy_amdgpu_interrupts(output)
        self.assertIn(output.splitlines()[-1], str(raised.exception))

    def test_allows_one_historical_recovered_interrupt(self):
        output = (
            "amdgpu: Fence fallback timer expired on ring sdma0\n"
            "another kernel warning\n"
        )
        self.assertEqual(amdgpu_environment.find_amdgpu_interrupt_failures(output), [])

    def test_rejects_fallback_during_native_probe(self):
        baseline = "amdgpu: Fence fallback timer expired on ring sdma0\n"
        active = "amdgpu: Fence fallback timer expired on ring sdma1"
        self.assertEqual(
            amdgpu_environment.find_amdgpu_interrupt_failures(
                baseline, baseline + active
            ),
            [active],
        )

    def test_counts_repeated_fallback_lines(self):
        fallback = "amdgpu: Fence fallback timer expired on ring sdma0"
        self.assertEqual(
            amdgpu_environment.find_amdgpu_interrupt_failures(
                fallback, f"{fallback}\n{fallback}\n"
            ),
            [fallback],
        )


class GpuSupervisorTest(unittest.TestCase):
    def test_selects_gnu_supervisor_when_default_is_another_implementation(self):
        paths = {"timeout": "/usr/bin/timeout", "gnutimeout": "/usr/bin/gnutimeout"}
        with (
            mock.patch.object(
                amdgpu_environment.shutil, "which", side_effect=paths.get
            ),
            mock.patch.object(
                amdgpu_environment.subprocess,
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
                amdgpu_environment.require_gnu_timeout(), paths["gnutimeout"]
            )

    def test_missing_or_incompatible_supervisor_fails_explicitly(self):
        for path in (None, "/usr/bin/timeout"):
            with (
                self.subTest(path=path),
                mock.patch.object(
                    amdgpu_environment.shutil, "which", return_value=path
                ),
                mock.patch.object(
                    amdgpu_environment.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(
                        [], 0, "timeout (uutils coreutils) 0.2.2"
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "GNU coreutils timeout is required"
                ):
                    amdgpu_environment.require_gnu_timeout()


class GpuEnvironmentTest(unittest.TestCase):
    def probe_command(self, native_script: str, **options) -> list[str]:
        script = (
            "import signal, sys; "
            "from build_tools.ci.amdgpu_environment import "
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
        # Source checkouts in CI import build_tools from the working directory.
        # Bazel supplies PYTHONPATH, which would mask a broken child invocation.
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            self.probe_command(
                "import os, signal; print(f'NATIVE_PID={os.getpid()}', flush=True); "
                "signal.pause()",
                timeout_seconds=2,
                kill_after_seconds=1,
                diagnostic_delay_seconds=1,
            ),
            capture_output=True,
            env=environment,
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
                    "-m",
                    "build_tools.ci.amdgpu_environment",
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
