# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Windows console events retain the distinction between text and virtual keys."""

import codecs
import ctypes
from ctypes import wintypes


class _KeyEvent(ctypes.Structure):
    _fields_ = [
        # Key release events are ignored by the command's input consumer.
        ("down", wintypes.BOOL),
        # Repeated key-down events represented by this record.
        ("repeat_count", wintypes.WORD),
        # Virtual-key code identifies navigation independently of Unicode text.
        ("virtual_key", wintypes.WORD),
        # Hardware scan code retained for the Win32 structure layout.
        ("scan_code", wintypes.WORD),
        # UTF-16 code unit, including controls and surrogate pairs.
        ("character", wintypes.WCHAR),
        # Modifier and toggle flags supplied by the console.
        ("control_state", wintypes.DWORD),
    ]


class _Event(ctypes.Union):
    _fields_ = (
        # Keyboard event payload when the record type is KEY_EVENT.
        ("key", _KeyEvent),
        # The Win32 union also holds mouse, focus, menu, and resize events.
        ("other", ctypes.c_byte * 16),
    )


class _InputRecord(ctypes.Structure):
    _fields_ = [
        # Win32 INPUT_RECORD event discriminator.
        ("type", wintypes.WORD),
        # Discriminated event payload, aligned by the Win32 ABI.
        ("event", _Event),
    ]


class ConsoleInput:
    """Polls an exclusively read console without changing its input modes."""

    def __init__(self, stream):
        import msvcrt

        # Loaded API with explicit pointer-sized handle signatures.
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.GetNumberOfConsoleInputEvents.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.kernel.GetNumberOfConsoleInputEvents.restype = wintypes.BOOL
        self.kernel.ReadConsoleInputW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_InputRecord),
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.kernel.ReadConsoleInputW.restype = wintypes.BOOL
        # Borrowed caller-owned console input handle.
        self.handle = msvcrt.get_osfhandle(stream.fileno())
        # Incremental UTF-16 decoder joins adjacent surrogate key events.
        self.decoder = codecs.getincrementaldecoder("utf-16-le")(errors="replace")
        # Decoded key repeated by the most recent key-down event.
        self.key = None
        # Unconsumed repeats of that decoded key.
        self.remaining = 0

    def read(self):
        if self.remaining:
            self.remaining -= 1
            return self.key
        count = wintypes.DWORD()
        if not self.kernel.GetNumberOfConsoleInputEvents(
            self.handle, ctypes.byref(count)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not count.value:
            return None
        record = _InputRecord()
        if not self.kernel.ReadConsoleInputW(
            self.handle, ctypes.byref(record), 1, ctypes.byref(count)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if record.type != 1 or not record.event.key.down:
            return "<ignored>"
        event = record.event.key
        if event.character != "\0":
            key = self.decoder.decode(ord(event.character).to_bytes(2, "little"))
            if not key:
                return "<ignored>"
        else:
            key = {
                0x21: "<page-up>",
                0x22: "<page-down>",
                0x23: "<end>",
                0x24: "<home>",
                0x25: "<left>",
                0x26: "<up>",
                0x27: "<right>",
                0x28: "<down>",
                0x2E: "<delete>",
            }.get(event.virtual_key, "<ignored>")
        self.key = key
        self.remaining = event.repeat_count - 1
        return key
