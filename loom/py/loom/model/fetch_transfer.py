# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Owned, reusable interpreter lifetimes for blocking transfers and cache IO."""

from __future__ import annotations

import atexit
import os
import pickle
import signal
import subprocess
import sys
import threading
import traceback
from concurrent.futures import Future, wait
from contextlib import contextmanager

from loom.model.fetch_source import FetchError
from loom.model.manifest import ManifestError


# The framed pipe carries only trusted internal calls and results, never a
# model document or network response to deserialize. Unbuffered parent pipes
# allow a forked caller to close inherited handles without taking an IO lock
# that another thread held at fork. Frames can span multiple pipe writes.
def _read_bytes(stream, length):
    content = bytearray(length)
    position = 0
    while position < length:
        count = stream.readinto(memoryview(content)[position:])
        if not count:
            raise EOFError
        position += count
    return content


def _read_frame(stream):
    length = int.from_bytes(_read_bytes(stream, 8), "little")
    return _read_bytes(stream, length)


def _send(stream, value):
    content = pickle.dumps(value)
    for part in (len(content).to_bytes(8, "little"), content):
        remaining = memoryview(part)
        while remaining:
            count = stream.write(remaining)
            remaining = remaining[count:]


def _worker(directory):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    idle_directory = os.getcwd()
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0) as result_stream:
        # SDK logs can contain signed redirects. Keep their Python and native
        # output separate from the private result channel.
        with open(os.devnull, "wb") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
        while True:
            try:
                request = _read_frame(sys.stdin.buffer)
            except EOFError:
                return
            operation = arguments = None
            try:
                os.chdir(directory)
                operation, arguments = pickle.loads(request)
                result = (True, operation(*arguments))
            except (FetchError, ManifestError, OSError) as error:
                result = (False, error)
            except BaseException as error:
                location = traceback.extract_tb(error.__traceback__)[-1]
                result = (
                    False,
                    RuntimeError(
                        f"unexpected transfer failure {type(error).__name__} at "
                        f"{location.filename}:{location.lineno} ({location.name})"
                    ),
                )
            finally:
                # Windows keeps the current directory open. An idle worker must
                # not prevent its caller from removing a completed workspace.
                os.chdir(idle_directory)
            _send(result_stream, result)
            del request, operation, arguments, result


class _TransferWorker:
    def __init__(self, context):
        # Interpreter, working directory, and environment captured by this child.
        self.context = context
        executable, directory, environment = context
        # One child exclusively borrowed for one synchronous operation at a time.
        self.process = subprocess.Popen(
            [executable, "-m", "loom.model.fetch_transfer", directory],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            # Windows retains the process's initial directory handle even
            # after chdir. Only an executing operation owns the caller's cwd.
            cwd=os.path.abspath(os.sep),
            env=environment,
        )

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdin.close()
            process.stdout.close()

    def detach(self):
        process, self.process = self.process, None
        if process is not None:
            # Only this forked process's unbuffered descriptor copies are closed.
            # Its parent still owns the worker and may have an active exchange.
            process.stdin.close()
            process.stdout.close()

    def run(self, operation, arguments, cancelled):
        result = Future()
        process = self.process

        def exchange():
            try:
                _send(process.stdin, (operation, arguments))
                result.set_result(pickle.loads(_read_frame(process.stdout)))
            except BaseException as error:
                result.set_exception(error)

        # Both pipe directions belong to this exchange. Cancellation can stop a
        # child during a blocked request write as well as a blocked response.
        thread = threading.Thread(target=exchange)
        thread.start()
        try:
            while not wait((result,), timeout=0.1).done:
                if cancelled.is_set():
                    raise FetchError("model acquisition cancelled")
            if cancelled.is_set():
                raise FetchError("model acquisition cancelled")
            return result.result()
        except (EOFError, BrokenPipeError, pickle.UnpicklingError) as error:
            self.close()
            raise FetchError(
                "transfer process exited before completing its request "
                f"(exit {process.returncode})"
            ) from error
        except BaseException:
            self.close()
            raise
        finally:
            thread.join()


class _TransferPool:
    def __init__(self):
        # Protects borrowing and the bounded worker reservation count.
        self.condition = threading.Condition()
        # Workers available for another complete operation.
        self.idle = []
        # Live children, including those currently borrowed.
        self.workers = set()
        # Live children plus reservations whose startup is in progress.
        self.count = 0
        # Parent teardown rejects later attempts to start another child.
        self.closed = False

    def acquire(self, context, cancelled):
        retired = None
        with self.condition:
            while True:
                if self.closed:
                    raise RuntimeError("transfer workers are shut down")
                if cancelled.is_set():
                    raise FetchError("model acquisition cancelled")
                for index, worker in enumerate(self.idle):
                    if worker.context == context:
                        return self.idle.pop(index)
                if self.idle:
                    # An environment, import path, or working-directory change
                    # starts a new interpreter instead of reusing stale globals.
                    retired = self.idle.pop()
                    self.workers.remove(retired)
                    break
                if self.count < 4:
                    self.count += 1
                    break
                self.condition.wait(0.1)
        try:
            if retired is not None:
                retired.close()
            worker = _TransferWorker(context)
        except BaseException:
            with self.condition:
                self.count -= 1
                self.condition.notify()
            raise
        with self.condition:
            self.workers.add(worker)
        return worker

    def release(self, worker):
        with self.condition:
            if worker.process is None:
                self.workers.remove(worker)
                self.count -= 1
            else:
                self.idle.append(worker)
            self.condition.notify()

    def close(self):
        # Interpreter shutdown runs after the non-daemon callers and exchanges
        # join. Every remaining child is idle, but still has an explicit owner.
        self.closed = True
        for worker in self.workers:
            worker.close()

    @contextmanager
    def borrow(self, cancelled):
        directory = os.getcwd()
        context = (
            sys.executable,
            directory,
            {
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    os.path.join(directory, path) for path in sys.path
                ),
            },
        )
        worker = self.acquire(context, cancelled)
        try:
            yield worker
        except (FetchError, ManifestError, OSError):
            raise
        except BaseException:
            worker.close()
            raise
        finally:
            self.release(worker)


_pool = _TransferPool()


def _after_fork():
    global _pool
    for worker in _pool.workers:
        worker.detach()
    _pool = _TransferPool()


def _close_pool():
    _pool.close()


atexit.register(_close_pool)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def run_transfer(operation, arguments: tuple, cancelled: threading.Event):
    """Runs internal IO with reusable imports and joined writer cancellation."""
    with _pool.borrow(cancelled) as worker:
        succeeded, value = worker.run(operation, arguments, cancelled)
        if not succeeded:
            raise value
        return value


if __name__ == "__main__":
    _worker(sys.argv[1])
