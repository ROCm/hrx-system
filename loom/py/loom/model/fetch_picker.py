# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Terminal views over validated catalog and model selections."""

from __future__ import annotations

import os
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .fetch_console import TerminalInput, display_text, terminal_screen
from .fetch_source import FetchError
from .manifest import ManifestError, ModelCard, Selection
from .memory import describe_memory


@dataclass(frozen=True)
class Choice:
    # Authored identifier returned to the same selection API used by flags.
    value: str
    # Human-readable name; terminal controls are escaped during rendering.
    label: str
    # Short supporting facts shown with the name and included in filtering.
    detail: str = ""


def _fit(text: str, width: int) -> str:
    result = ""
    for character in text:
        cells = (
            0
            if unicodedata.combining(character)
            else (2 if unicodedata.east_asian_width(character) in ("W", "F") else 1)
        )
        if cells > width:
            break
        result += character
        width -= cells
    return result


class Picker:
    """Owns interaction only; metadata and acquisition remain caller-owned."""

    def __init__(self, input: TextIO, output: TextIO):
        # Caller-owned terminal streams, independent of structured stdout.
        self.input = input
        # Screen and prompt output; mode is restored after each choice.
        self.output = output

    def _text(self, text: str) -> str:
        return (
            display_text(text)
            .encode(self.output.encoding or "utf-8", errors="backslashreplace")
            .decode(self.output.encoding or "utf-8")
        )

    def _line(self, text: str):
        self.output.write(self._text(text) + "\n")
        self.output.flush()

    def _readline(self, prompt: str) -> str:
        self.output.write(self._text(prompt))
        self.output.flush()
        value = self.input.readline()
        if not value:
            raise KeyboardInterrupt
        return value.rstrip("\r\n")

    def choose(
        self,
        title: str,
        choices: list[Choice],
        *,
        default: str | None = None,
        always: bool = False,
    ) -> str:
        if not choices:
            raise FetchError(f"{title}: no compatible choices")
        if len(choices) == 1 and not always:
            return choices[0].value
        with terminal_screen(self.output) as supported:
            if supported:
                selected = self._screen(title, choices, default)
            else:
                selected = self._plain(title, choices, default)
        self._line(f"{title}: {selected}")
        return selected

    def _plain(self, title, choices, default):
        self._line(title)
        self._line("Enter a name, /text to filter, or /cancel to cancel.")
        visible = choices
        while True:
            for choice in visible:
                self._line(f"  {choice.value}  {choice.label}  {choice.detail}")
            value = self._readline(
                f"Name [{default}]: " if default is not None else "Name: "
            )
            if value == "/cancel":
                raise KeyboardInterrupt
            if value.startswith("/"):
                visible = self._filter(choices, value[1:])
                if not visible:
                    self._line("No matches. Enter / to show all choices.")
            else:
                value = value or default
                if any(choice.value == value for choice in visible):
                    return value
                self._line("Choose one of the displayed names.")

    @staticmethod
    def _filter(choices, query):
        return [
            choice
            for choice in choices
            if query.casefold()
            in (choice.value + " " + choice.label + " " + choice.detail).casefold()
        ]

    def _screen(self, title, choices, default):
        query = ""
        visible = choices
        position = next(
            (index for index, choice in enumerate(visible) if choice.value == default),
            0,
        )
        previous = None
        cancelled = threading.Event()
        with TerminalInput(self.input) as terminal:
            while True:
                size = os.get_terminal_size(self.output.fileno())
                width, rows = max(0, size.columns - 1), size.lines
                page = max(1, rows - 10)
                state = (query, position, size)
                if state != previous:
                    previous = state
                    lines = [("  LOOM / FETCH", "1;36"), ("  " + title, "1"), ("", "")]
                    if width < 32 or rows < 12:
                        lines += [("Resize to at least 33 x 12; Esc cancels.", "")]
                    else:
                        lines += [
                            ("  Filter: " + (query or "type to search"), "36"),
                            ("", ""),
                        ]
                        start = (position // page) * page
                        for index in range(start, min(start + page, len(visible))):
                            choice = visible[index]
                            prefix = " > " if index == position else "   "
                            text = prefix + choice.value
                            if choice.label != choice.value:
                                text += "  " + choice.label
                            lines.append((text, "1;36" if index == position else ""))
                        if not visible:
                            lines.append(("   No matches", ""))
                        while len(lines) < page + 5:
                            lines.append(("", ""))
                        detail = visible[position].detail if visible else ""
                        lines += [
                            ("  " + detail, "2"),
                            (
                                f"  {position + 1 if visible else 0}"
                                f"/{len(visible)} choices",
                                "2",
                            ),
                            ("  Arrows / Home / End / PgUp / PgDn", ""),
                            ("  Enter: select   Ctrl-U: clear   Esc: cancel", ""),
                        ]
                    self.output.write("\x1b[H\x1b[2J")
                    for index, (text, style) in enumerate(lines[: max(0, rows - 1)]):
                        self.output.write(
                            f"\x1b[{index + 1};1H\x1b[{style or '0'}m"
                            + _fit(self._text(text), width)
                            + "\x1b[0m"
                        )
                    self.output.flush()
                key = terminal.read(0.1, cancelled)
                if key == "\x1b":
                    raise KeyboardInterrupt
                if key in ("\r", "\n") and visible and width >= 32 and rows >= 12:
                    return visible[position].value
                movement = {
                    "<up>": -1,
                    "<down>": 1,
                    "<page-up>": -page,
                    "<page-down>": page,
                    "<home>": -len(visible),
                    "<end>": len(visible),
                }.get(key)
                if movement is not None:
                    position = max(0, min(len(visible) - 1, position + movement))
                else:
                    new_query = query
                    if key in ("\x08", "\x7f"):
                        new_query = query[:-1]
                    elif key == "\x15":
                        new_query = ""
                    elif key is not None and len(key) == 1 and key.isprintable():
                        new_query += key
                    if new_query != query:
                        query = new_query
                        visible = self._filter(choices, query)
                        position = 0

    def directory(self) -> Path:
        while True:
            value = self._readline("Install directory (Ctrl-C cancels): ")
            if value:
                return Path(value).expanduser()
            self._line("Enter a directory; nothing has been installed.")

    def model(self, catalog: dict) -> str:
        return self.choose(
            "Model",
            [
                Choice(
                    identity,
                    entry["name"],
                    entry.get("description", "")
                    or ", ".join(
                        service["kind"] for service in entry["services"].values()
                    ),
                )
                for identity, entry in catalog["models"].items()
            ],
            always=True,
        )

    def selection(
        self,
        card: ModelCard,
        *,
        preset=None,
        service=None,
        variant=None,
        programs=None,
        partition=None,
    ) -> Selection:
        document = card.document
        presets = document.get("presets", {})
        declared = card.choices()
        tables = {
            "preset": presets,
            "service": document["services"],
            "variant": document["variants"],
            "programs": document["programs"],
            "partition": {
                choice["partition"] for choice in declared if "partition" in choice
            },
        }
        requested = dict(
            service=service, variant=variant, programs=programs, partition=partition
        )
        for label, value in dict(requested, preset=preset).items():
            if value is not None and value not in tables[label]:
                raise ManifestError(f"{label}: unknown name {value!r}")
        compatible = [
            choice
            for choice in declared
            if all(
                value is None or choice.get(key) == value
                for key, value in requested.items()
            )
        ]
        settings = presets.get(preset, {})
        if service is None:
            service = settings.get("service")
        if service is None:
            services = {choice["service"] for choice in compatible}
            service = self.choose(
                "Service",
                [
                    Choice(
                        name,
                        value["kind"],
                        f"ABI {value.get('version', 'development')}",
                    )
                    for name, value in document["services"].items()
                    if name in services
                ],
                default=settings.get("service"),
            )
        compatible = [choice for choice in compatible if choice["service"] == service]
        if preset is None:
            choices = [
                Choice(
                    name,
                    name,
                    value["variant"]
                    + " / "
                    + value["programs"]
                    + (
                        "; " + describe_memory(value["memory"])
                        if "memory" in value
                        else ""
                    ),
                )
                for name, value in presets.items()
                if value["service"] == service
            ]
            if choices:
                preset = self.choose("Preset", choices)
                settings = presets[preset]
        if variant is None:
            variant = settings.get("variant")
        if variant is None:
            variants = {choice["variant"] for choice in compatible}
            variant = self.choose(
                "Variant",
                [
                    Choice(name, name, value.get("description", ""))
                    for name, value in document["variants"].items()
                    if name in variants
                ],
                default=settings.get("variant"),
            )
        compatible = [choice for choice in compatible if choice["variant"] == variant]
        if partition is None:
            partition = settings.get("partition")
        if partition is None and "partitions" in document["variants"][variant]:
            partitions = {choice["partition"] for choice in compatible}
            partition = self.choose(
                "Partition",
                [
                    Choice(name, name, value.get("description", ""))
                    for name, value in document["variants"][variant][
                        "partitions"
                    ].items()
                    if name in partitions
                ],
                default=settings.get("partition"),
            )
        compatible = [
            choice for choice in compatible if choice.get("partition") == partition
        ]
        if programs is None:
            programs = settings.get("programs")
        if programs is None:
            available = {choice["programs"] for choice in compatible}
            if len(available) != 1:
                raise ManifestError(
                    "programs: choose a preset or specify --programs "
                    f"({len(available)} compatible choices)"
                )
            programs = next(iter(available))
        return card.select(
            preset,
            service=service,
            variant=variant,
            programs=programs,
            partition=partition,
        )
