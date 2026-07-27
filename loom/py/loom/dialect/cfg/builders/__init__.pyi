# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from loom.builder import ValueRef
from loom.builders import DialectBuilder
from loom.ir import Block

class CfgBuilder(DialectBuilder):
    def br(
        self,
        *,
        dest: Block,
        args: list[ValueRef] = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def cond_br(
        self,
        *,
        condition: ValueRef,
        true_dest: Block,
        false_dest: Block,
        location_id: int | None = ...,
    ) -> None: ...
