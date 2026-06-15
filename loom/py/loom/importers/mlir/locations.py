# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""MLIR location conversion helpers."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from loom.ir import LOCATION_UNKNOWN, FileLocation, Module, OpaqueLocation

_FILE_LOCATION_RE = re.compile(
    r'^loc\((?P<source>"(?:\\.|[^"\\])*"):(?P<line>[0-9]+):(?P<col>[0-9]+)\)$'
)


@dataclass(slots=True)
class MlirLocationConverter:
    """Interns MLIR locations as Loom source locations."""

    module: Module
    tag: str = "mlir"
    _source_id: int | None = field(default=None, init=False)

    def unknown(self) -> int:
        return LOCATION_UNKNOWN

    def opaque(self, location: object | None) -> int:
        if location is None:
            return LOCATION_UNKNOWN
        text = str(location)
        if not text or text == "loc(unknown)":
            return LOCATION_UNKNOWN
        file_location_id = self._parse_file_location(text)
        if file_location_id != LOCATION_UNKNOWN:
            return file_location_id
        source_id = self._find_or_add_source(self.tag)
        return self.module.add_location(
            OpaqueLocation(source_id=source_id, data=text.encode("utf-8"))
        )

    def _parse_file_location(self, text: str) -> int:
        match = _FILE_LOCATION_RE.match(text)
        if match is None:
            return LOCATION_UNKNOWN
        source = ast.literal_eval(match.group("source"))
        line = int(match.group("line"))
        column = int(match.group("col"))
        source_id = self._find_or_add_source(source)
        return self.module.add_location(
            FileLocation(
                source_id=source_id,
                start_line=line,
                start_col=column,
                end_line=line,
                end_col=column,
            )
        )

    def _find_or_add_source(self, source: str) -> int:
        if source == self.tag and self._source_id is not None:
            return self._source_id
        for source_id, existing in enumerate(self.module.sources):
            if existing == source:
                if source == self.tag:
                    self._source_id = source_id
                return source_id
        source_id = len(self.module.sources)
        self.module.sources.append(source)
        if source == self.tag:
            self._source_id = source_id
        return source_id
