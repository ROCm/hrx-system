# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for dynamic builder `.pyi` generation."""

from loom.assembly import Ref
from loom.dsl import ANY, Dialect, Op, OpCategory, Operand, Result
from loom.gen.python.builders_pyi import generate_builder_stub_files


def test_builders_pyi_uses_per_dialect_builders_dirs() -> None:
    test_dialect = Dialect("test", dialect_id=0x7F)
    generated = generate_builder_stub_files(
        [
            Op(
                "test.add",
                group=test_dialect,
                operands=[Operand("lhs", ANY), Operand("rhs", ANY)],
                results=[Result("result", ANY)],
                format=[Ref("lhs"), Ref("rhs")],
            )
        ]
    )
    root = generated["loom/py/loom/builders.pyi"]
    test_stub = generated["loom/py/loom/dialect/test/builders/__init__.pyi"]

    assert "class LoomBuilder:" in root
    assert "from loom.dialect.test.builders import TestBuilder" in root
    assert "def test(self) -> TestBuilder: ..." in root
    assert "class TestBuilder(DialectBuilder):" in test_stub
    assert "def add(" in test_stub
    assert "results: list[Type | TiedResultSpec]" in test_stub
    assert "name: str | None = ..." in test_stub


def test_builders_pyi_shards_category_grouped_dialects() -> None:
    vector_dialect = Dialect("vector", dialect_id=0x7E)
    arithmetic = OpCategory("arithmetic")
    memory = OpCategory("memory")
    add = Op(
        "vector.add",
        group=vector_dialect,
        operands=[Operand("lhs", ANY), Operand("rhs", ANY)],
        results=[Result("result", ANY)],
        format=[Ref("lhs"), Ref("rhs")],
    )
    load = Op(
        "vector.load",
        group=vector_dialect,
        operands=[Operand("source", ANY)],
        results=[Result("result", ANY)],
        format=[Ref("source")],
    )

    generated = generate_builder_stub_files(
        [add, load],
        category_groups={"vector": ((arithmetic, (add,)), (memory, (load,)))},
    )

    vector_init = generated["loom/py/loom/dialect/vector/builders/__init__.pyi"]
    arithmetic_stub = generated["loom/py/loom/dialect/vector/builders/arithmetic.pyi"]
    memory_stub = generated["loom/py/loom/dialect/vector/builders/memory.pyi"]

    assert "class VectorBuilder(" in vector_init
    assert "VectorArithmeticMixin" in vector_init
    assert "VectorMemoryMixin" in vector_init
    assert "class VectorArithmeticMixin:" in arithmetic_stub
    assert "def add(" in arithmetic_stub
    assert "class VectorMemoryMixin:" in memory_stub
    assert "def load(" in memory_stub
