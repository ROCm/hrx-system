# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercises Windows console events in a privately owned, hidden console."""

import os
import subprocess
import sys
import unittest


def _console_events():
    import ctypes
    import msvcrt
    import threading
    from ctypes import wintypes

    from loom.model.fetch_console import TerminalInput, terminal_screen
    from loom.model.fetch_windows import _InputRecord

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.WriteConsoleInputW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_InputRecord),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel.WriteConsoleInputW.restype = wintypes.BOOL
    kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetConsoleMode.restype = wintypes.BOOL

    def mode(stream):
        result = wintypes.DWORD()
        if not kernel.GetConsoleMode(
            msvcrt.get_osfhandle(stream.fileno()), ctypes.byref(result)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return result.value

    with open("CONIN$") as input, open("CONOUT$", "w") as output:
        original = mode(input), mode(output)
        records = []
        # Unicode U+00E0 collides with the CRT's extended-key prefix. The emoji
        # arrives as two UTF-16 events. Repeats and releases retain OS semantics.
        units = "à🦙".encode("utf-16-le")
        for offset in range(0, len(units), 2):
            record = _InputRecord(type=1)
            record.event.key.down = True
            record.event.key.repeat_count = 1
            record.event.key.character = chr(
                int.from_bytes(units[offset : offset + 2], "little")
            )
            records.append(record)
        for virtual_key in (0x26, 0x28, 0x24, 0x23, 0x21, 0x22):
            record = _InputRecord(type=1)
            record.event.key.down = True
            record.event.key.repeat_count = 1
            record.event.key.virtual_key = virtual_key
            records.append(record)
        repeat = _InputRecord(type=1)
        repeat.event.key.down = True
        repeat.event.key.repeat_count = 3
        repeat.event.key.character = "k"
        records.append(repeat)
        release = _InputRecord(type=1)
        release.event.key.repeat_count = 1
        release.event.key.character = "z"
        records.append(release)
        cancel = _InputRecord(type=1)
        cancel.event.key.down = True
        cancel.event.key.repeat_count = 1
        cancel.event.key.character = "\x1a"
        records.append(cancel)
        buffer = (_InputRecord * len(records))(*records)
        count = wintypes.DWORD()
        if not kernel.WriteConsoleInputW(
            msvcrt.get_osfhandle(input.fileno()),
            buffer,
            len(records),
            ctypes.byref(count),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        assert count.value == len(records)
        observed = []
        with TerminalInput(input) as terminal:
            try:
                while True:
                    key = terminal.read(0.1, threading.Event())
                    if key is not None and key != "<ignored>":
                        observed.append(key)
            except KeyboardInterrupt:
                pass
        assert observed == [
            "à",
            "🦙",
            "<up>",
            "<down>",
            "<home>",
            "<end>",
            "<page-up>",
            "<page-down>",
            "k",
            "k",
            "k",
        ], observed
        os.environ.pop("TERM", None)
        with terminal_screen(output) as supported:
            assert supported
        assert (mode(input), mode(output)) == original


def test_windows_console_events_and_modes():
    if os.name != "nt":
        raise unittest.SkipTest("requires the Windows console API")
    startup = subprocess.STARTUPINFO()
    startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    result = subprocess.run(
        [sys.executable, __file__, "console"],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        startupinfo=startup,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


if __name__ == "__main__":
    if sys.argv[1:] == ["console"]:
        _console_events()
    else:
        test_windows_console_events_and_modes()
