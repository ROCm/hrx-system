# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass

import pytest

import loom
from loom.importers.core import SourceImportSession
from loom.ir import I32, INDEX, ShapedType, StaticDim, Type, TypeKind
from loom.verify import verify_module


@dataclass
class _UnhashableSource:
    name: str


class _TirLikeSource:
    def __init__(self, name: str) -> None:
        self.name = name

    def __hash__(self) -> int:
        return 0

    def __eq__(self, other: object) -> bool:
        return True


def test_maps_unhashable_source_objects_by_identity() -> None:
    _, builder = loom.module_builder()
    ref = builder.value("value", I32)
    source = _UnhashableSource("source-node")
    equal_source = _UnhashableSource("source-node")
    session = SourceImportSession(builder=builder)

    session.map_value(source, ref)

    assert session.mapped(source) is ref
    assert session.mapped(equal_source) is None
    assert session.mapped_value_type(source) == "i32"
    assert session.result_name(source) == "value"


def test_maps_hashable_foreign_source_objects_by_identity() -> None:
    _, builder = loom.module_builder()
    ref = builder.value("value", I32)
    source = _TirLikeSource("num_tokens")
    equalish_source = _TirLikeSource("num_tokens")
    session = SourceImportSession(builder=builder)

    session.map_value(source, ref)

    assert session.mapped(source) is ref
    assert session.mapped(equalish_source) is None


@pytest.mark.parametrize(
    ("target", "value_type"),
    [
        ("index.andi", INDEX),
        ("index.add", INDEX),
        ("scalar.addi", I32),
        ("vector.addi", ShapedType(TypeKind.VECTOR, I32, (StaticDim(4),))),
    ],
)
def test_binary_import_preserves_resolved_types(target: str, value_type: Type) -> None:
    module, builder = loom.module_builder()
    lhs = builder.value("lhs", value_type)
    rhs = builder.value("rhs", value_type)
    body = builder.region()
    builder.func.def_(callee="binary", args=[lhs, rhs], results=[], body=body)
    session = SourceImportSession(builder=builder)

    with builder.insertion_block(body.blocks[0]):
        result = session.build_binary(target, lhs, rhs, value_type, "result")
        builder.func.return_()

    assert result.type == value_type
    assert result.name == "result"
    assert body.blocks[0].ops[0].operands == [lhs.id, rhs.id]
    assert body.blocks[0].ops[0].results == [result.id]
    verify_module(module, ops=loom.default_ops()).raise_if_errors()
