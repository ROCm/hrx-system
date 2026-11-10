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
from loom.target.arch.amd.xdna.aie2p.contracts.packet_program import PacketProgram
from loom.target.arch.amd.xdna.aie2p.core_descriptors import AIE2P_CORE_DESCRIPTOR_SET
from loom.target.contracts import (
    AttrProject,
    ContractEmit,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    SourceNode,
    ValueAliasRule,
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

AIE2P_BFP_ENCODE_RULES = (
    AIE2P_BFP_ENCODE_RULE,
    DescriptorRule(
        source_op=vector.vector_encode,
        descriptor=descriptor_by_key(
            AIE2P_CORE_DESCRIPTOR_SET, "amd.xdna.aie2p.convert.f32x64.bfp16ebs8"
        ),
        guards=(
            Guard.value_type("source", Vector("f32", lanes=64)),
            *BFP_ENCODE_GUARDS[1:],
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor_by_key(
                    AIE2P_CORE_DESCRIPTOR_SET,
                    "amd.xdna.aie2p.state.rounding.immediate",
                ),
                immediates={"i": BF16_CONVERSION_ROUNDING},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=descriptor_by_key(
                    AIE2P_CORE_DESCRIPTOR_SET,
                    "amd.xdna.aie2p.convert.f32x64.bfp16ebs8",
                ),
                operands={"src": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
        report_key="encode.f32.bfp16ebs8",
    ),
)


def bfp_unpack_emits(
    low_bytes: ValueRef, high_bytes: ValueRef, result: ValueRef
) -> tuple[ContractEmit, ...]:
    """Separates eight exponent/mantissa records without changing their bits.

    Memory stores eight repetitions of one exponent followed by eight
    mantissas. EX storage instead separates the 64 mantissas and eight
    exponents. Only the first eight bytes of the upper X carrier are read.
    Native FIFO loads perform this unpacking as part of the memory transfer;
    this register-only path also handles isolated records with exact bounds.
    """

    program = PacketProgram(32, "bfp_")
    mantissas = None
    for group in range(8):
        shifted = program.operation(
            f"shifted_{group}",
            "shift.bytes.x.configured",
            "d",
            s1=low_bytes,
            s2=high_bytes,
            shift=program.constant(f"byte_shift_{group}", group + 1),
        )
        if mantissas is None:
            mantissas = shifted
        else:
            mask_bits = program.constant(
                f"mantissa_mask_{group}",
                3 << (group * 2),
            )
            mask = program.operation(
                f"mantissa_selection_{group}",
                "select.mask.i32",
                "d0",
                s0=mask_bits,
                immediates={"imm": 0},
            )
            mantissas = program.operation(
                f"mantissas_{group}",
                "select.i32x16",
                "d",
                s1=mantissas,
                s2=shifted,
                sel=mask,
            )
    storage = program.operation(
        "mantissa_storage", "bfp.mantissas.from.vector512", "dst", src=mantissas
    )
    exponent_shifts = {
        byte_index: program.constant(f"exponent_shift_{byte_index}", byte_index * 8)
        for byte_index in range(1, 4)
    }
    for half_index, half in enumerate(("low32", "high32")):
        word = None
        for byte_index in range(4):
            exponent = program.operation(
                f"exponent_{half_index}_{byte_index}",
                "extract.i8.immediate",
                "dst",
                s1=low_bytes,
                immediates={"idx": (half_index * 4 + byte_index) * 9},
            )
            if word is None:
                word = exponent
            else:
                shifted = program.operation(
                    f"exponent_shifted_{half_index}_{byte_index}",
                    "lshl.i32",
                    "d0",
                    s0=exponent,
                    s1=exponent_shifts[byte_index],
                )
                word = program.operation(
                    f"exponent_word_{half_index}_{byte_index}",
                    "or.i32",
                    "d0",
                    s0=word,
                    s1=shifted,
                )
        destination = (
            program.temporary("mantissas_exponents_low32")
            if half_index == 0
            else result
        )
        program.emits.append(
            EmitDescriptorOp(
                descriptor=descriptor_by_key(
                    AIE2P_CORE_DESCRIPTOR_SET,
                    f"amd.xdna.aie2p.bfp.exponents.{half}.from.scalar",
                ),
                operands={"src": word, "storage": storage},
                results={"dst": destination},
                result_types={"dst": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            )
        )
        storage = destination
    return tuple(program.emits)


def _bfp_fragment_rule() -> DescriptorRule:
    """Interprets an exact packed byte payload in its native matrix carrier."""

    low_bytes = ValueRef.temporary("packed_low")
    high_bytes = ValueRef.temporary("packed_high")
    emits = (
        EmitRegisterSlice(
            source=ValueRef.operand("data"),
            result=low_bytes,
            unit_offset=0,
            unit_count=2,
        ),
        EmitRegisterSlice(
            source=ValueRef.operand("data"),
            result=high_bytes,
            unit_offset=2,
            unit_count=2,
        ),
        *bfp_unpack_emits(low_bytes, high_bytes, ValueRef.result("result")),
    )
    return DescriptorRule(
        source_op=vector.vector_fragment,
        descriptor=None,
        guards=(
            Guard.value_type("data", Vector("i8", lanes=72)),
            Guard.low_value_register_class("data", "aie2p.vec256"),
            Guard.low_value_register_class("result", "aie2p.mexa"),
        ),
        emit=emits,
        report_key="fragment.packed-bytes.bfp16ebs8",
    )


AIE2P_BFP_FRAGMENT_RULES = (
    _bfp_fragment_rule(),
    DescriptorRule(
        source_op=vector.vector_slice,
        source_nodes=(
            SourceNode.exclusive_user(
                "fragment",
                source_op=vector.vector_fragment,
                parent_result=ValueRef.result("result"),
                node_operand=ValueRef.operand("data"),
                guards=(Guard.low_value_register_class("result", "aie2p.mexa"),),
            ),
        ),
        guards=(
            Guard.value_type(
                "source", Vector("i8", minimum_lanes=72, maximum_lanes=72 * 65535)
            ),
            Guard.value_static_dim0_multiple("source", 72),
            Guard.low_value_register_class("source", "aie2p.mexa"),
            Guard.value_type("result", Vector("i8", lanes=72)),
            Guard.i64_array_count("static_offsets", 1),
            Guard.i64_array_element_range("static_offsets", 0, 0, 72 * 65534),
            Guard.i64_array_element_multiple("static_offsets", 0, 72),
        ),
        emit=(
            EmitRegisterSlice(
                source=ValueRef.operand("source"),
                result=ValueRef.result("result", source_node="fragment"),
                unit_offset=AttrProject.i64_array_element_quotient(
                    "static_offsets", element=0, divisor=72
                ),
            ),
        ),
        priority=1,
        report_key="fragment.bfp16ebs8.block",
    ),
    ValueAliasRule(
        source_op=vector.vector_fragment,
        source=ValueRef.operand("data"),
        result=ValueRef.result("result"),
    ),
)
