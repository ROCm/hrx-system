# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Owned lifetimes for blocking network transfers and third-party cache IO."""

from __future__ import annotations

import os
import pickle
import signal
import subprocess
import sys
import threading
import traceback

from loom.model.fetch_source import FetchError
from loom.model.manifest import ManifestError


def _worker():
    # This private pipe carries only trusted calls from run_transfer, never a
    # model document or network response. The dedicated module entry point also
    # works from an interactive Python session or an unguarded caller script.
    operation, arguments = pickle.load(sys.stdin.buffer)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb") as result_stream:
        # SDK logging can carry signed redirects. Preserve the result pipe while
        # isolating both Python and native-library diagnostic output.
        with open(os.devnull, "wb") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
        try:
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
        pickle.dump(result, result_stream)


def run_transfer(operation, arguments: tuple, cancelled: threading.Event):
    """Runs an internal transfer and joins its sole writer before returning."""
    if cancelled.is_set():
        raise FetchError("model acquisition cancelled")
    request = pickle.dumps((operation, arguments))
    with subprocess.Popen(
        [sys.executable, "-m", "loom.model.fetch_transfer"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    ) as process:
        try:
            while True:
                if cancelled.is_set():
                    raise FetchError("model acquisition cancelled")
                try:
                    result, _ = process.communicate(request, timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    request = None
            if cancelled.is_set():
                raise FetchError("model acquisition cancelled")
            if process.returncode:
                raise FetchError(
                    f"transfer process failed with exit {process.returncode}"
                )
            succeeded, value = pickle.loads(result)
            if not succeeded:
                raise value
            return value
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()


if __name__ == "__main__":
    _worker()
