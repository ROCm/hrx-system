# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for dynamic builder `.pyi` generation."""

from pathlib import Path
from unittest import mock

from loom.assembly import Attr, Ref
from loom.dsl import (
    ANY,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    OpCategory,
    Operand,
    Result,
)
from loom.gen.python import builders_pyi
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


def test_builders_pyi_imports_signed_enum_set_public_types() -> None:
    test_dialect = Dialect("test", dialect_id=0x7F)
    feature = EnumDef("Feature", [EnumCase("fast", 1)])
    generated = generate_builder_stub_files(
        [
            Op(
                "test.features",
                group=test_dialect,
                attrs=[AttrDef("features", "signed_enum_set", enum_def=feature)],
                format=[Attr("features")],
            )
        ]
    )
    test_stub = generated["loom/py/loom/dialect/test/builders/__init__.pyi"]

    assert "from collections.abc import Mapping" in test_stub
    assert "from loom.ir import SignedEnumSetAttr" in test_stub
    assert "features: SignedEnumSetAttr | Mapping[str | int, bool]" in test_stub


def test_checked_in_file_set_owns_only_generated_stubs(
    tmp_path: Path,
) -> None:
    expected_path = "loom/py/loom/dialect/test/builders/__init__.pyi"
    expected_file = tmp_path / expected_path
    expected_file.parent.mkdir(parents=True)
    expected_file.write_text("expected\n", encoding="utf-8")
    obsolete_path = "loom/py/loom/dialect/test/builders/obsolete.pyi"
    (tmp_path / obsolete_path).write_text("obsolete\n", encoding="utf-8")
    neighbor_path = expected_file.parent / "helpers.py"
    neighbor_path.write_text("authored\n", encoding="utf-8")
    with mock.patch.object(
        builders_pyi,
        "_output_files",
        return_value={expected_path: "expected\n"},
    ):
        generated_file_set = builders_pyi.checked_in_file_set(tmp_path)

    assert generated_file_set.output_paths == (expected_path,)
    assert generated_file_set.obsolete_paths == (obsolete_path,)
    assert neighbor_path.is_file()
