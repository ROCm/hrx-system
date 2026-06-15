# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for dynamic Loom Python builders."""

import re
from typing import Any, cast

import pytest

# Keep the core builder tests on the synthetic test dialect. Real production
# dialect imports belong in dialect-specific importer/builder coverage, not here.
import loom
import loom.dialect as dialect
from loom.builder import ValueRef, tied
from loom.builders import LoomBuilder, module_builder
from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect import vector
from loom.dialect.globals import ALL_GLOBAL_OPS
from loom.dialect.pass_ import ALL_PASS_OPS
from loom.dialect.scf import ALL_SCF_OPS
from loom.dialect.test import ALL_TEST_OPS
from loom.format.text.printer import Printer
from loom.ir import (
    F32,
    I32,
    INDEX,
    Block,
    OpaqueLocation,
    ShapedType,
    StaticDim,
    TypeKind,
)


def _builder() -> tuple[Block, LoomBuilder]:
    block = Block()
    _module, builder = module_builder(
        insertion_block=block,
        ops=ALL_TEST_OPS,
        types=ALL_BUILTIN_TYPES,
    )
    return block, builder


def test_root_package_exports_module_builder() -> None:
    assert loom.module_builder is module_builder
    assert len(loom.default_ops()) > 0
    assert len(loom.default_types()) > 0


def test_dialect_package_exports_dialects() -> None:
    assert dialect.vector is vector
    assert "vector" in dialect.__all__


def test_default_dynamic_builder_registers_all_dialects() -> None:
    block = Block()
    _module, builder = loom.module_builder(insertion_block=block)

    result = builder.index.constant(value=0, results=[INDEX], name="c0")

    assert isinstance(result, ValueRef)
    assert result.name == "c0"
    assert builder.vector.name == "vector"
    assert builder.scf.name == "scf"


def test_dynamic_builder_constructs_binary_op_with_result_name() -> None:
    block, builder = _builder()
    lhs = builder.value("lhs", I32)
    rhs = builder.value("rhs", I32)

    result = builder.test.addi(lhs=lhs, rhs=rhs, results=[I32], name="sum")

    assert isinstance(result, ValueRef)
    assert result.name == "sum"
    assert len(block.ops) == 1
    assert block.ops[0].name == "test.addi"
    assert block.ops[0].operands == [lhs.id, rhs.id]


def test_dynamic_builder_records_segmented_operand_counts() -> None:
    block, builder = _builder()
    root = builder.value("root", I32)
    guard = builder.value("guard", I32)
    lhs0 = builder.value("lhs0", I32)
    lhs1 = builder.value("lhs1", I32)
    rhs = builder.value("rhs", I32)

    result = builder.test.segmented(
        root=root,
        guard=guard,
        lhs=[lhs0, lhs1],
        rhs=[rhs],
        results=[I32],
        name="result",
    )

    assert isinstance(result, ValueRef)
    assert len(block.ops) == 1
    assert block.ops[0].name == "test.segmented"
    assert block.ops[0].operands == [root.id, guard.id, lhs0.id, lhs1.id, rhs.id]
    assert block.ops[0].operand_segment_counts == (1, 1, 2, 1)


def test_dynamic_builder_accepts_empty_variadic_terminator_operands() -> None:
    block, builder = _builder()

    builder.test.yield_()

    assert len(block.ops) == 1
    assert block.ops[0].name == "test.yield"
    assert block.ops[0].operands == []


def test_dynamic_builder_packs_index_lists_in_declared_storage_order() -> None:
    block, builder = _builder()
    tile_type = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
    tensor_type = ShapedType(TypeKind.TENSOR, F32, (StaticDim(4),))
    tile = builder.value("tile", tile_type)
    tensor = builder.value("tensor", tensor_type)
    offset = builder.value("offset", INDEX)

    result = builder.test.update(
        source=tile,
        target=tensor,
        offsets=[0, offset],
        results=[tied(tensor, tensor_type)],
        name="updated",
    )

    assert isinstance(result, ValueRef)
    assert result.name == "updated"
    op = block.ops[0]
    assert op.operands == [tile.id, tensor.id, offset.id]
    assert op.attributes["static_offsets"] == [0, -(2**63)]


def test_dynamic_builder_rejects_unknown_kwargs_with_op_name() -> None:
    _block, builder = _builder()
    lhs = builder.value("lhs", I32)
    rhs = builder.value("rhs", I32)
    test_builder = cast(Any, builder.test)

    with pytest.raises(
        TypeError, match=re.escape("test.addi: unexpected parameter 'lhz'")
    ):
        test_builder.addi(lhz=lhs, rhs=rhs, results=[I32])


def test_dynamic_builder_rejects_bad_single_result_name_shape() -> None:
    _block, builder = _builder()
    lhs = builder.value("lhs", I32)
    rhs = builder.value("rhs", I32)

    with pytest.raises(TypeError, match="pass only one"):
        builder.test.addi(lhs=lhs, rhs=rhs, results=[I32], name="x", names=["x"])


def test_dynamic_builder_exposes_keyword_dialect_names() -> None:
    _module, builder = module_builder(
        insertion_block=Block(),
        ops=(*ALL_GLOBAL_OPS, *ALL_PASS_OPS),
        types=ALL_BUILTIN_TYPES,
    )

    assert builder.global_.name == "global"
    assert builder.pass_.name == "pass"


def test_dynamic_builder_exposes_keyword_parameter_names() -> None:
    block = Block()
    _module, builder = module_builder(
        insertion_block=block,
        ops=ALL_GLOBAL_OPS,
        types=ALL_BUILTIN_TYPES,
    )

    results = builder.global_.load(global_="weights", results=[F32], names=["value"])

    assert isinstance(results, list)
    assert results[0].name == "value"
    assert block.ops[0].attributes["global"] == "weights"


def test_dynamic_builder_exposes_keyword_op_names() -> None:
    _module, builder = module_builder(
        insertion_block=Block(),
        ops=ALL_SCF_OPS,
        types=ALL_BUILTIN_TYPES,
    )

    builder.scf.yield_()


def test_region_helper_creates_block_arguments() -> None:
    _block, builder = _builder()

    region = builder.region(args=[("iv", INDEX)])

    assert len(region.blocks) == 1
    assert len(region.blocks[0].arg_ids) == 1
    value = builder.module.values[region.blocks[0].arg_ids[0]]
    assert value.name == "iv"
    assert value.type == INDEX
    assert value.is_block_arg


def test_dynamic_builder_constructs_region_bearing_scf_for() -> None:
    module, builder = loom.module_builder()
    function_body = builder.region()
    lower_bound = builder.value("lo", INDEX)
    upper_bound = builder.value("hi", INDEX)
    step = builder.value("step", INDEX)
    builder.func.def_(
        callee="main",
        args=[lower_bound, upper_bound, step],
        results=[],
        body=function_body,
    )
    loop_body = builder.region(args=[("iv", INDEX)])

    with builder.insertion_block(loop_body.blocks[0]):
        builder.scf.yield_()
    with builder.insertion_block(function_body.blocks[0]):
        builder.scf.for_(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            step=step,
            results=[],
            body=loop_body,
        )
        builder.func.return_()

    printer = Printer()
    printer.register_ops(loom.default_ops())
    printer.register_types(loom.default_types())

    assert printer.print_module(module) == (
        "func.def @main(%lo: index, %hi: index, %step: index) {\n"
        "  scf.for %iv = [%lo to %hi step %step] {\n"
        "  }\n"
        "  func.return\n"
        "}\n"
    )


def test_dynamic_builder_location_context_applies_to_operations() -> None:
    block, builder = _builder()
    source_id = len(builder.module.sources)
    builder.module.sources.append("mlir")
    location_id = builder.module.add_location(
        OpaqueLocation(source_id=source_id, data=b'loc("kernel.mlir":1:2)')
    )
    value = builder.value("value", I32)

    with builder.location(location_id):
        result = builder.test.neg(input=value, results=[I32], name="negated")
    builder.test.neg(input=result, results=[I32], name="again")

    assert block.ops[0].location_id == location_id
    assert block.ops[1].location_id == 0


def test_dynamic_builder_location_id_overrides_active_context() -> None:
    block, builder = _builder()
    source_id = len(builder.module.sources)
    builder.module.sources.append("mlir")
    active_location_id = builder.module.add_location(
        OpaqueLocation(source_id=source_id, data=b'loc("active.mlir":1:2)')
    )
    override_location_id = builder.module.add_location(
        OpaqueLocation(source_id=source_id, data=b'loc("override.mlir":3:4)')
    )
    value = builder.value("value", I32)

    with builder.location(active_location_id):
        builder.test.neg(
            input=value,
            results=[I32],
            name="negated",
            location_id=override_location_id,
        )

    assert block.ops[0].location_id == override_location_id
