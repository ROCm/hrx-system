# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native AIE2P block-floating operand encoding."""

from loom.dialect.encoding import defs as encoding
from loom.dialect.vector import defs as vector
from loom.dsl import EncodingOperandSummaryDef
from loom.target.arch.amd.xdna.aie2p.contracts.data_path import (
    BF16_CONVERSION_ROUNDING,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptors import AIE2P_CORE_DESCRIPTOR_SET
from loom.target.contracts import (
    ContractEmit,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    ValueRef,
    Vector,
    descriptor_by_key,
)

BFP16EBS8_SCHEMA = EncodingOperandSummaryDef(
    element_format=encoding.enum_fact(encoding.NumericFormat, "bfp16ebs8"),
    payload_packing=encoding.enum_fact(encoding.PayloadPacking, "target_fragment"),
    rounding_policy=encoding.enum_fact(encoding.RoundingPolicy, "flush_subnormal"),
    payload_register_count=18,
    payload_element_count=64,
)

BFP_ENCODE_GUARDS = (
    Guard.value_type("source", Vector("bf16", lanes=64)),
    Guard.value_type("result", Vector("i8", lanes=72)),
    Guard.operand_segment_count("auxiliary", 0),
    Guard.value_storage_operand_schema("schema", BFP16EBS8_SCHEMA),
)


def bfp_encode_emits(
    float_halves: tuple[ValueRef, ValueRef], result: ValueRef
) -> tuple[ContractEmit, ...]:
    """Encodes two ordered F32 accumulator halves into one BFP operand."""

    float_value = ValueRef.temporary("bfp_float_value")
    return (
        EmitRegisterConcat(
            sources=float_halves,
            result=float_value,
            result_type=Vector("f32", lanes=64),
        ),
        EmitDescriptorOp(
            descriptor=descriptor_by_key(
                AIE2P_CORE_DESCRIPTOR_SET, "amd.xdna.aie2p.state.rounding.immediate"
            ),
            immediates={"i": BF16_CONVERSION_ROUNDING},
            form=DescriptorEmitForm.OP,
        ),
        EmitDescriptorOp(
            descriptor=descriptor_by_key(
                AIE2P_CORE_DESCRIPTOR_SET, "amd.xdna.aie2p.convert.f32x64.bfp16ebs8"
            ),
            operands={"src": float_value},
            results={"dst": result},
            form=DescriptorEmitForm.OP,
        ),
    )


def _bfp_encode_rule() -> DescriptorRule:
    convert = descriptor_by_key(
        AIE2P_CORE_DESCRIPTOR_SET, "amd.xdna.aie2p.convert.bf16x32.to.f32x32"
    )
    float_halves = (
        ValueRef.temporary("float_half_0"),
        ValueRef.temporary("float_half_1"),
    )
    emits: list[ContractEmit] = []
    for half_index, float_half in enumerate(float_halves):
        source_half = ValueRef.temporary(f"source_half_{half_index}")
        emits.extend(
            (
                EmitRegisterSlice(
                    source=ValueRef.operand("source"),
                    result=source_half,
                    unit_offset=half_index * 2,
                    unit_count=2,
                ),
                EmitDescriptorOp(
                    descriptor=convert,
                    operands={"src": source_half},
                    results={"dst": float_half},
                    result_types={"dst": DescriptorResultType()},
                    form=DescriptorEmitForm.OP,
                ),
            )
        )
    return DescriptorRule(
        source_op=vector.vector_encode,
        descriptor=descriptor_by_key(
            AIE2P_CORE_DESCRIPTOR_SET, "amd.xdna.aie2p.convert.f32x64.bfp16ebs8"
        ),
        guards=BFP_ENCODE_GUARDS,
        emit=(*emits, *bfp_encode_emits(float_halves, ValueRef.result("result"))),
        report_key="encode.bf16.bfp16ebs8",
    )


AIE2P_BFP_ENCODE_RULE = _bfp_encode_rule()
