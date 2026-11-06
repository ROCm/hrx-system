# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from loom.builders import DialectBuilder

class TypeBuilder(DialectBuilder):
    def decl(
        self,
        *,
        namespace: str,
        type_name: str,
        name: str,
        location_id: int | None = ...,
    ) -> None: ...
