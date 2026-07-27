# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import loom
from loom.importers.tilelang.buffers import TileLangBufferAccess
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import ExpressionOptions
from loom.importers.tilelang.ops import calls
from loom.ir import F32, EncodingInstance, ShapedType, StaticDim, TypeKind


class _Call:
    def __init__(self, *args: object) -> None:
        self.args = args

    def __str__(self) -> str:
        return "tl.tvm_mfma(...)"


def test_decodes_cdna_fp8_tvm_mfma_descriptor() -> None:
    _, builder = loom.module_builder()
    context = TileLangConversionContext(builder=builder)

    descriptor = calls._decode_tvm_mfma_descriptor(
        object(),
        context,
        "tl.tvm_mfma",
        "f32_16x16x32_fp8_fp8",
    )

    assert descriptor is not None
    assert descriptor.m == 16
    assert descriptor.n == 16
    assert descriptor.k == 32
    assert descriptor.lhs_family == "fp8"
    assert descriptor.rhs_family == "fp8"


def test_decodes_fnuz_tvm_mfma_operand_as_matrix_schema() -> None:
    _, builder = loom.module_builder()
    context = TileLangConversionContext(builder=builder)

    operand_format = calls._decode_tvm_mfma_operand_format(
        object(),
        context,
        "tl.tvm_mfma",
        "float8_e4m3fnuzx8",
    )
    assert operand_format is not None
    assert operand_format.element_format == "f8e4m3fnuz"
    assert str(operand_format.source_element_type) == "f8E4M3"
    assert str(operand_format.payload_type) == "vector<2xi32>"

    region = builder.region()
    with builder.insertion_block(region.blocks[0]):
        schema_value = calls._tvm_mfma_matrix_schema(context, operand_format)

    expected_schema = EncodingInstance(
        name="matrix_operand",
        params=(
            ("element_format", "f8e4m3fnuz"),
            ("payload_elements", 8),
            ("payload_registers", 2),
        ),
    )
    assert schema_value is context.storage_schema_values[expected_schema]
    assert str(schema_value.type) == "encoding<schema>"


def test_tvm_mfma_rejects_non_cdna_fp8_target() -> None:
    _, builder = loom.module_builder()
    context = TileLangConversionContext(
        builder=builder,
        target_preset="hip -mcpu=gfx1100",
    )
    expr = _Call(
        "f32_16x16x32_fp8_fp8",
        "row",
        "row",
        "float8_e4m3fnuzx8",
        "float8_e4m3fnuzx8",
        "float32x4",
        object(),
        0,
        object(),
        0,
        object(),
        0,
    )

    result = calls._convert_tvm_mfma_call(
        expr,
        context,
        object(),
        "tl.tvm_mfma",
        options=ExpressionOptions(effect=True),
    )

    assert result is None
    blocked = [target for record in context.blocked for target in record.target]
    assert "call `tl.tvm_mfma` requires an AMDGPU CDNA FP8 target" in blocked


def test_tvm_mfma_accumulator_chunk_origin_maps_flat_chunks_to_rows() -> None:
    _, builder = loom.module_builder()
    context = TileLangConversionContext(builder=builder)
    region = builder.region()
    view = builder.value(
        "acc",
        ShapedType(TypeKind.VIEW, F32, (StaticDim(64), StaticDim(64))),
    )

    with builder.insertion_block(region.blocks[0]):
        flat_chunk = context.ensure_constant("17", "index", "chunk")
        origin = calls._tvm_mfma_accumulator_chunk_origin(
            _Call(),
            context,
            "tl.tvm_mfma",
            TileLangBufferAccess(
                view=view,
                indices=(flat_chunk,),
                memory_scope="local.fragment",
            ),
            context.type_converter.vector_type(F32, 4),
        )

    assert origin is not None
    assert [value.name for value in origin] == ["acc_row", "acc_column"]
    origin_op_names = [
        op.name
        for op in region.blocks[0].ops
        if op.name in ("index.div", "index.rem", "index.mul")
    ]
    assert origin_op_names == [
        "index.div",
        "index.rem",
        "index.mul",
    ]
