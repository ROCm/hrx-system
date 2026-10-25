# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import select
import socket
import subprocess
import sys
import threading
import unittest
from contextlib import chdir, suppress
from pathlib import Path
from unittest.mock import patch

from loom.model.fetch_source import FetchError
from loom.model.fetch_transfer import run_transfer


def _context():
    return os.getpid(), os.getcwd(), os.environ.get("LOOM_TRANSFER_TEST_CONTEXT")


def _echo(value):
    return value


def _expected_error():
    raise FetchError("expected transfer failure")


def _unexpected_error():
    raise ValueError("private credential data")


def _exit_after_write(path):
    with Path(path).open("a") as stream:
        stream.write("one call\n")
    os._exit(7)


def _retain_connection(address):
    # A process-owned connection witnesses the lifetime of an idle interpreter,
    # just as the SDK's pooled HTTP connections outlive an individual call.
    global _connection
    _connection = socket.create_connection(address)


class TestTransferLifetimes(unittest.TestCase):
    def test_serial_requests_reuse_imports_and_frame_large_results(self):
        cancelled = threading.Event()
        with patch.dict(os.environ, {"LOOM_TRANSFER_TEST_CONTEXT": "reuse"}):
            first = run_transfer(os.getpid, (), cancelled)
            content = bytes(range(256)) * 1024
            assert run_transfer(_echo, (content,), cancelled) == content
            assert run_transfer(os.getpid, (), cancelled) == first
            assert first != os.getpid()

    def test_environment_and_directory_are_captured_per_call(self, tmp_path):
        cancelled = threading.Event()
        with patch.dict(os.environ, {"LOOM_TRANSFER_TEST_CONTEXT": "first"}):
            first = run_transfer(_context, (), cancelled)
            assert first[1:] == (os.getcwd(), "first")
            os.environ["LOOM_TRANSFER_TEST_CONTEXT"] = "second"
            with chdir(tmp_path):
                second = run_transfer(_context, (), cancelled)
                assert second[1:] == (os.getcwd(), "second")
            assert second[0] != first[0]

    def test_expected_failure_keeps_worker_but_unexpected_failure_retires_it(self):
        import pytest

        cancelled = threading.Event()
        with patch.dict(os.environ, {"LOOM_TRANSFER_TEST_CONTEXT": "failure"}):
            first = run_transfer(os.getpid, (), cancelled)
            with pytest.raises(FetchError, match="expected transfer failure"):
                run_transfer(_expected_error, (), cancelled)
            assert run_transfer(os.getpid, (), cancelled) == first
            with pytest.raises(
                RuntimeError, match="unexpected transfer failure"
            ) as failure:
                run_transfer(_unexpected_error, (), cancelled)
            assert "private credential data" not in str(failure.value)
            assert run_transfer(os.getpid, (), cancelled) != first

    def test_worker_exit_is_reported_without_replaying_side_effects(self, tmp_path):
        import pytest

        path = tmp_path / "calls"
        cancelled = threading.Event()
        with patch.dict(os.environ, {"LOOM_TRANSFER_TEST_CONTEXT": "exit"}):
            first = run_transfer(os.getpid, (), cancelled)
            with pytest.raises(FetchError, match=r"completing its request.*exit 7"):
                run_transfer(_exit_after_write, (str(path),), cancelled)
            assert path.read_text() == "one call\n"
            assert run_transfer(os.getpid, (), cancelled) != first

    def test_cancelled_call_never_reaches_the_worker(self, tmp_path):
        import pytest

        cancelled = threading.Event()
        cancelled.set()
        path = tmp_path / "calls"
        with pytest.raises(FetchError, match="cancelled"):
            run_transfer(_exit_after_write, (str(path),), cancelled)
        assert not path.exists()

    def test_forked_caller_does_not_reuse_or_stop_parent_workers(self):
        if not hasattr(os, "fork"):
            self.skipTest("os.fork is not available on this platform")
        script = """
import json, os, sys, threading
from loom.model.fetch_transfer import run_transfer
cancelled = threading.Event()
before = run_transfer(os.getpid, (), cancelled)
reader, writer = os.pipe()
child = os.fork()
if not child:
    os.close(reader)
    own = run_transfer(os.getpid, (), cancelled)
    os.write(writer, str(own).encode())
    os.close(writer)
    sys.exit(0)
os.close(writer)
with os.fdopen(reader, 'r') as stream:
    own = int(stream.read())
_, status = os.waitpid(child, 0)
after = run_transfer(os.getpid, (), cancelled)
print(json.dumps([before, own, after, status]))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            capture_output=True,
            text=True,
            check=True,
        )
        before, own, after, status = json.loads(result.stdout)
        assert status == 0
        assert before == after
        assert before != own

    def test_interpreter_exit_joins_its_idle_workers(self):
        script = """
import json, sys, threading
from loom.model.fetch_transfer import run_transfer
from loom.model.fetch_transfer_test import _retain_connection
run_transfer(_retain_connection, (tuple(json.loads(sys.argv[1])),), threading.Event())
input()
"""
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with subprocess.Popen(
                [sys.executable, "-c", script, json.dumps(listener.getsockname())],
                env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
                stdin=subprocess.PIPE,
            ) as caller:
                connection, _ = listener.accept()
                with connection:
                    caller.communicate(b"\n")
                    assert caller.returncode == 0
                    assert select.select([connection], [], [], 0)[0] == [connection]
                    with suppress(ConnectionResetError):
                        assert connection.recv(1) == b""
