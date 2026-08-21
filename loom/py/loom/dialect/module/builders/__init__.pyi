# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Sequence

from loom.builders import DialectBuilder

class ModuleBuilder(DialectBuilder):
    def import_(
        self,
        *,
        provider: str,
        symbols: Sequence[str],
        location_id: int | None = ...,
    ) -> None: ...
