# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Graceful, isolated shutdown of subprocesses owned by serving tests."""

import os
import signal
import subprocess
import sys


def creation_options():
    """Give each Windows server its own hidden console for control delivery."""
    if os.name != "nt":
        return {}
    startup = subprocess.STARTUPINFO()
    startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    return dict(creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup)


def shutdown(process):
    """Request normal shutdown and join, including cleanup after delivery fails."""
    try:
        if process.poll() is None:
            try:
                if os.name == "nt":
                    # Attachment changes process-wide console state. A separate
                    # helper leaves the multithreaded test runner's console alone.
                    subprocess.run(
                        [sys.executable, "-I", __file__, str(process.pid)],
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                else:
                    process.send_signal(signal.SIGTERM)
            except (OSError, subprocess.CalledProcessError) as error:
                # The child can finish between poll and console attachment or
                # signal delivery. Its actual exit status still belongs to us.
                if process.poll() is None:
                    if isinstance(error, subprocess.CalledProcessError):
                        error.add_note(error.stderr)
                    raise
    except BaseException:
        if process.poll() is None:
            process.kill()
        raise
    finally:
        process.wait()
    return process.returncode


def _interrupt_console(process_id):
    import ctypes
    import traceback
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
    kernel.FreeConsole.argtypes = []
    kernel.FreeConsole.restype = wintypes.BOOL
    kernel.AttachConsole.argtypes = [wintypes.DWORD]
    kernel.AttachConsole.restype = wintypes.BOOL
    kernel.SetConsoleCtrlHandler.argtypes = [handler_type, wintypes.BOOL]
    kernel.SetConsoleCtrlHandler.restype = wintypes.BOOL
    kernel.GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.GenerateConsoleCtrlEvent.restype = wintypes.BOOL
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateProcess.restype = wintypes.BOOL
    handler = handler_type(lambda _: True)
    try:
        for function, arguments in (
            (kernel.FreeConsole, ()),
            (kernel.AttachConsole, (process_id,)),
            (kernel.SetConsoleCtrlHandler, (handler, True)),
            # Ctrl+C can be disabled by an inherited process attribute. Break
            # always reaches the server's native interrupt handler.
            (kernel.GenerateConsoleCtrlEvent, (1, 0)),
        ):
            if not function(*arguments):
                raise ctypes.WinError(ctypes.get_last_error())
        exit_code = 0
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        exit_code = 1
    # This one-shot sender owns no durable state. End its threads together so
    # Python finalization cannot release a still-running console callback. The
    # server receives normal console control and owns its own graceful shutdown.
    if not kernel.TerminateProcess(kernel.GetCurrentProcess(), exit_code):
        raise ctypes.WinError(ctypes.get_last_error())


if __name__ == "__main__":
    _interrupt_console(int(sys.argv[1]))
