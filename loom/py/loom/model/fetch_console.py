# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dependency-free console input and cancellable acquisition access waits."""

from __future__ import annotations

import codecs
import os
import threading
import time
import webbrowser
from contextlib import contextmanager
from typing import TextIO

from .fetch_hf import AccessRequired
from .fetch_source import FetchError


def display_text(text: str) -> str:
    """Renders metadata as one terminal line without interpreting controls."""
    return "".join(
        character if character.isprintable() else f"\\u{ord(character):04x}"
        for character in text
    )


class TerminalInput:
    """Owns temporary terminal input mode for one interactive operation."""

    def __init__(self, stream: TextIO):
        # The caller's input terminal; output remains independently owned.
        self.stream = stream
        # POSIX attributes restored on success, cancellation, or exception.
        self.saved_mode = None
        # Windows console events, loaded only on that platform.
        self.windows = None
        # POSIX reads may split a UTF-8 character across input events.
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def __enter__(self):
        if not self.stream.isatty():
            raise FetchError("interactive input requires a terminal; use --no-input")
        if os.name == "nt":
            from .fetch_windows import ConsoleInput

            self.windows = ConsoleInput(self.stream)
        else:
            import termios
            import tty

            self.saved_mode = termios.tcgetattr(self.stream.fileno())
            tty.setcbreak(self.stream.fileno(), termios.TCSANOW)
        return self

    def __exit__(self, *_):
        if self.saved_mode is not None:
            import termios

            termios.tcsetattr(self.stream.fileno(), termios.TCSANOW, self.saved_mode)

    def _read_character(self, timeout: float, cancelled: threading.Event) -> str | None:
        deadline = time.monotonic() + timeout
        while not cancelled.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if self.windows is not None:
                key = self.windows.read()
                if key is not None:
                    if key in ("\x03", "\x1a"):
                        raise KeyboardInterrupt
                    return key
                cancelled.wait(min(remaining, 0.1))
            else:
                import select

                ready, _, _ = select.select(
                    [self.stream.fileno()], [], [], min(remaining, 0.1)
                )
                if ready:
                    key = os.read(self.stream.fileno(), 1)
                    if not key or key in (b"\x03", b"\x04"):
                        raise KeyboardInterrupt
                    character = self.decoder.decode(key)
                    if character:
                        return character
        return None

    def read(self, timeout: float, cancelled: threading.Event) -> str | None:
        """Reads text or a named navigation key; cancellation always interrupts."""
        key = self._read_character(timeout, cancelled)
        if key != "\x1b" or self.windows is not None:
            return key
        # Escape is also a key. This short inter-character window disambiguates
        # it from a terminal's CSI/SS3 sequences without blocking an operation.
        suffix = self._read_character(0.1, cancelled)
        if suffix is None:
            return key
        if suffix not in ("[", "O"):
            return "<ignored>"
        sequence = suffix
        for _ in range(32):
            suffix = self._read_character(0.1, cancelled)
            if suffix is None or len(suffix) != 1:
                return "<ignored>"
            sequence += suffix
            if "@" <= suffix <= "~":
                return {
                    "[A": "<up>",
                    "OA": "<up>",
                    "[B": "<down>",
                    "OB": "<down>",
                    "[C": "<right>",
                    "OC": "<right>",
                    "[D": "<left>",
                    "OD": "<left>",
                    "[H": "<home>",
                    "OH": "<home>",
                    "[1~": "<home>",
                    "[F": "<end>",
                    "OF": "<end>",
                    "[4~": "<end>",
                    "[5~": "<page-up>",
                    "[6~": "<page-down>",
                    "[3~": "<delete>",
                }.get(sequence, "<ignored>")
        return "<ignored>"


@contextmanager
def terminal_screen(output: TextIO):
    """Enables an alternate VT screen, or yields False for plain line input."""
    restore = None
    supported = output.isatty() and os.environ.get("TERM") != "dumb"
    if supported and os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetConsoleMode.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel.GetConsoleMode.restype = wintypes.BOOL
        kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.SetConsoleMode.restype = wintypes.BOOL
        handle = msvcrt.get_osfhandle(output.fileno())
        mode = wintypes.DWORD()
        supported = bool(kernel.GetConsoleMode(handle, ctypes.byref(mode)))
        if supported:
            # ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING.
            supported = bool(kernel.SetConsoleMode(handle, mode.value | 0x5))
            if supported:

                def restore():
                    if not kernel.SetConsoleMode(handle, mode.value):
                        raise ctypes.WinError(ctypes.get_last_error())
    elif supported:
        supported = os.environ.get("TERM", "").startswith(
            (
                "xterm",
                "screen",
                "tmux",
                "vt",
                "ansi",
                "linux",
                "rxvt",
                "alacritty",
                "foot",
            )
        )
    try:
        if supported:
            output.write("\x1b[?1049h\x1b[?25l")
            output.flush()
        yield supported
    finally:
        try:
            if supported:
                output.write("\x1b[0m\x1b[?25h\x1b[?1049l")
                output.flush()
        finally:
            if restore is not None:
                restore()


class AccessWaiter:
    """Plain access interaction; provider decisions remain in the acquisition API."""

    def __init__(self, *, interactive: bool, input: TextIO, output: TextIO):
        # Headless waiting never reads keys or opens a browser.
        self.interactive = interactive
        # Caller-owned input terminal, used only for an interactive wait.
        self.input = input
        # Status output is independent of machine-readable command stdout.
        self.output = output
        # Repeated polls print status without repeating unchanged remediation.
        self.last_action = None

    def _line(self, text: str):
        self.output.write(display_text(text) + "\n")
        self.output.flush()

    def _show(self, action: AccessRequired):
        key = (
            action.repository,
            action.reason,
            action.url,
            action.account,
            action.credential_source,
        )
        if key == self.last_action:
            return
        self.last_action = key
        self._line(str(action))
        if action.account:
            self._line(f"Effective account: {action.account}")
        if action.reason == "gate":
            self._line(f"Open {action.url} and complete the access form if you agree.")
            self._line(
                "Use the downloader's account; the publisher may need "
                "to approve access."
            )
        if action.reason in ("gate", "credentials", "permissions"):
            self._line(
                "Run hf auth login in another terminal with an authorized account."
            )
            if action.credential_source in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                self._line(
                    f"{action.credential_source} overrides the saved login. "
                    "To change it, restart this command with that variable "
                    "updated or unset."
                )
            elif action.credential_source == "HF_HUB_DISABLE_IMPLICIT_TOKEN":
                self._line(
                    "HF_HUB_DISABLE_IMPLICIT_TOKEN disables the saved login; "
                    "unset it before restarting this command."
                )

    def __call__(
        self, action: AccessRequired, delay: float, cancelled: threading.Event
    ) -> None:
        self._show(action)
        self._line(f"Waiting for access; rechecking in {delay:g} seconds.")
        deadline = time.monotonic() + delay
        if not self.interactive:
            while not cancelled.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                cancelled.wait(min(remaining, 0.25))
            return
        minimum = time.monotonic() + action.retry_after
        with TerminalInput(self.input) as terminal:
            self._line("[o] Open page   [Enter] Check now   [q / Ctrl-C] Cancel")
            while not cancelled.is_set():
                key = terminal.read(max(0, deadline - time.monotonic()), cancelled)
                if key is None:
                    return
                if key.lower() == "q":
                    cancelled.set()
                    raise KeyboardInterrupt
                if key in ("\r", "\n"):
                    remaining = minimum - time.monotonic()
                    if remaining <= 0:
                        return
                    self._line(
                        f"Provider retry delay: {remaining:.0f} seconds remaining."
                    )
                elif key.lower() == "o":
                    if not webbrowser.open(action.url, new=2):
                        self._line(f"Open this URL in your browser: {action.url}")
