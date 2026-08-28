# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from typing import Any

from loom.builders import DialectBuilder

class MetadataBuilder(DialectBuilder):
    def module(
        self,
        *,
        key: str,
        value: Any,
        location_id: int | None = ...,
    ) -> None: ...
