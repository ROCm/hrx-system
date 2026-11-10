# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercises the real exec boundary, arguments, and process resource policy."""

import pathlib
import resource
import signal
import subprocess
import sys
import unittest


def prepare_child(soft_limit=1024, hard_limit=None):
    _, inherited_hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(
        resource.RLIMIT_NOFILE,
        (soft_limit, inherited_hard_limit if hard_limit is None else hard_limit),
    )
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)


class LauncherTest(unittest.TestCase):
    def launch(self, fixed, arguments, **kwargs):
        return subprocess.Popen(
            [LAUNCHER, "x86-64", FIXTURE, *fixed, "--", *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        )

    def test_exec_preserves_arguments_and_process_ownership(self):
        fixed = ["--owned=a b", "--repeated=first", "--repeated=second"]
        arguments = ["", "a'b\"c", "a\nb", "$(false)", "$HOME", "--owned_suffix=ok"]
        process = self.launch(fixed, arguments, preexec_fn=prepare_child)
        output, errors = process.communicate()
        self.assertEqual(process.returncode, 0, errors)
        header, tokens = output.split(b"\n", 1)
        process_id, descriptor_limit, ignored_signal = map(int, header.split())
        self.assertEqual(process_id, process.pid)
        self.assertEqual(descriptor_limit, 8192)
        self.assertEqual(ignored_signal, 1)
        expected = b"".join(
            str(len(token.encode())).encode() + b":" + token.encode() + b"\n"
            for token in fixed + arguments
        )
        self.assertEqual(tokens, expected)

    def test_preserves_larger_descriptor_limit(self):
        process = self.launch([], [], preexec_fn=lambda: prepare_child(soft_limit=8193))
        output, errors = process.communicate()
        self.assertEqual(process.returncode, 0, errors)
        self.assertEqual(int(output.split()[1]), 8193)

    def test_reports_insufficient_hard_limit(self):
        process = self.launch([], [], preexec_fn=lambda: prepare_child(hard_limit=1024))
        output, errors = process.communicate()
        self.assertEqual(process.returncode, 2)
        self.assertEqual(output, b"")
        self.assertIn(b"RLIMIT_NOFILE >= 8192", errors)

    def test_rejects_overrides_before_exec(self):
        for argument in ["--owned", "--owned=changed"]:
            with self.subTest(argument=argument):
                process = self.launch(["--owned=fixed"], [argument])
                output, errors = process.communicate()
                self.assertEqual(process.returncode, 2)
                self.assertEqual(output, b"")
                self.assertIn(b"cannot override rocjitsu invocation option", errors)

    def test_reports_exec_failure(self):
        result = subprocess.run(
            [LAUNCHER, "x86-64", "/no/such/rocjitsu-consumer", "--"],
            capture_output=True,
        )
        self.assertEqual(result.returncode, 126)
        self.assertIn(b"exec /no/such/rocjitsu-consumer:", result.stderr)

    def test_rejects_unknown_host_cpu(self):
        result = subprocess.run(
            [LAUNCHER, "unknown", FIXTURE, "--"], capture_output=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"unsupported rocjitsu host CPU", result.stderr)

    def test_requires_argument_separator(self):
        result = subprocess.run(
            [LAUNCHER, "x86-64", FIXTURE, "--owned=fixed"], capture_output=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"requires -- before caller arguments", result.stderr)


if __name__ == "__main__":
    LAUNCHER = str(pathlib.Path(sys.argv.pop(1)).resolve())
    FIXTURE = str(pathlib.Path(sys.argv.pop(1)).resolve())
    unittest.main()
