# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loom.builders import DialectBuilder
from loom.ir import Region

class PassBuilder(DialectBuilder):
    def pipeline(
        self,
        *,
        anchor: str,
        symbol: str,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def for_(
        self,
        *,
        anchor: str,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def where(
        self,
        *,
        predicate: str,
        attrs: Mapping[str, Any] | None = ...,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def repeat(
        self,
        *,
        mode: str,
        count: int | None = ...,
        max_iterations: int | None = ...,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def call(
        self,
        *,
        callee: str,
        location_id: int | None = ...,
    ) -> None: ...
    def run(
        self,
        *,
        key: str,
        options: Mapping[str, Any] | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def fail(
        self,
        *,
        message: str,
        location_id: int | None = ...,
    ) -> None: ...
    def halt(
        self,
        *,
        message: str,
        location_id: int | None = ...,
    ) -> None: ...
    def yield_(
        self,
        *,
        location_id: int | None = ...,
    ) -> None: ...
