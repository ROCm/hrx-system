# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch

from loom_serve_testing.process import creation_options, shutdown


def child():
    print("ready", flush=True)
    sys.stdin.readline()
    sys.exit(7)


class ProcessTest(unittest.TestCase):
    def start(self):
        process = subprocess.Popen(
            [sys.executable, __file__, "child"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **creation_options(),
        )
        self.addCleanup(self.clean, process)
        self.assertEqual(process.stdout.readline(), "ready\n")
        return process

    def clean(self, process):
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()

    def test_finished_child_retains_its_actual_exit_status(self):
        process = self.start()
        process.stdin.write("finish\n")
        process.stdin.flush()
        self.assertEqual(process.wait(), 7)
        self.assertEqual(shutdown(process), 7)
        self.assertEqual(shutdown(process), 7)

    def test_delivery_failure_is_reported_after_owned_child_is_joined(self):
        process = self.start()
        if os.name == "nt":
            failure = subprocess.CalledProcessError(
                1, "console helper", stderr="console delivery failed"
            )
            transport = patch("subprocess.run", side_effect=failure)
        else:
            send_signal = process.send_signal

            def failed_delivery(number):
                if number == signal.SIGTERM:
                    raise PermissionError("signal delivery failed")
                send_signal(number)

            failure = PermissionError()
            transport = patch.object(
                process, "send_signal", side_effect=failed_delivery
            )
        with transport, self.assertRaises(type(failure)):
            shutdown(process)
        self.assertIsNotNone(process.poll())
        self.assertNotEqual(process.returncode, 0)


if __name__ == "__main__":
    if sys.argv[1:] == ["child"]:
        child()
    else:
        unittest.main()
