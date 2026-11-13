# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native widened BF16 fused arithmetic on AIE2P."""

from itertools import product

from loom.dialect.vector import defs as vector
from loom.scalar_type import ScalarTypeKind
from loom.target.arch.amd.xdna.aie2p.contracts.data_path import (
    BF16_CONVERSION_ROUNDING,
    F32_ACCUMULATOR_ADD_CONTROL,
)
from loom.target.arch.amd.xdna.aie2p.contracts.f32_accumulator import (
    emit_f32_full_accumulator_operand,
    emit_f32_full_accumulator_result,
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
from loom.target.low_descriptors import Descriptor


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, "amd.xdna.aie2p." + key)


def _operation(key: str, source: ValueRef, result: ValueRef) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=_descriptor(key),
        operands={"src": source},
        results={"dst": result},
        result_types={"dst": DescriptorResultType()},
        form=DescriptorEmitForm.OP,
    )


def _fma_rule(
    minimum: int,
    maximum: int,
    *,
    zero: bool,
    origins: tuple[bool, bool],
    seed_in_accumulator: bool,
    result_in_accumulator: bool,
    operands_in_accumulator: tuple[bool, bool],
) -> DescriptorRule:
    width = 64 if maximum > 32 else 32
    operation = "multiply" if zero else "accumulate"
    descriptor = _descriptor(f"{operation}.bf16x{width}.configured")
    f32_type = Vector(
        "f32", minimum_static_elements=minimum, maximum_static_elements=maximum
    )
    bf16_type = Vector(
        "bf16", minimum_static_elements=minimum, maximum_static_elements=maximum
    )
    emits: list[ContractEmit] = []
    guards = [
        *(Guard.value_type(name, f32_type) for name in ("a", "b", "c", "result")),
        *(
            Guard.instance_flags_has_all("fastmath", flag)
            for flag in ("afn", "nnan", "ninf", "nsz")
        ),
        *((Guard.value_float_equals("c", 0.0),) if zero else ()),
    ]
    operands = {"acc": ValueRef.temporary("control")}
    if maximum <= 16:
        guards.append(
            Guard.low_value_register_class(
                "result", "aie2p.ebmll" if result_in_accumulator else "aie2p.vec256"
            )
        )
        if not zero:
            guards.append(
                Guard.low_value_register_class(
                    "c", "aie2p.ebmll" if seed_in_accumulator else "aie2p.vec256"
                )
            )
    if not zero:
        operands["acc1"] = emit_f32_full_accumulator_operand(
            ValueRef.operand("c"), "seed", maximum, seed_in_accumulator, emits
        )
    if not all(origins):
        emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor("state.rounding.immediate"),
                immediates={"i": BF16_CONVERSION_ROUNDING},
                form=DescriptorEmitForm.OP,
            )
        )
    for name, field, origin, in_accumulator in zip(
        ("a", "b"), ("s1", "s2"), origins, operands_in_accumulator, strict=True
    ):
        if origin:
            guards.append(Guard.exact_lane_origin_type(name, bf16_type))
            operands[field] = ValueRef.exact_lane_origin_operand(name)
            continue
        guards.append(Guard.value_exact_float_type(name, ScalarTypeKind.BF16))
        if maximum <= 16:
            guards.append(
                Guard.low_value_register_class(
                    name, "aie2p.ebmll" if in_accumulator else "aie2p.vec256"
                )
            )
        wide = emit_f32_full_accumulator_operand(
            ValueRef.operand(name), name, maximum, in_accumulator, emits
        )
        chunks = []
        for half in range(width // 32):
            source = ValueRef.temporary(f"{name}_source_{half}")
            result = ValueRef.temporary(f"{name}_bf16_{half}")
            emits.append(
                EmitRegisterSlice(
                    source=wide, result=source, unit_offset=half * 2, unit_count=2
                )
            )
            emits.append(_operation("convert.f32x32.to.bf16x32", source, result))
            chunks.append(result)
        packed = ValueRef.temporary(f"{name}_bf16")
        emits.append(
            EmitRegisterConcat(
                sources=chunks, result=packed, result_type=Vector("bf16", lanes=width)
            )
        )
        operands[field] = packed
    native_result = (
        ValueRef.result("result")
        if width == 64
        else ValueRef.temporary("native_result")
    )
    emits.extend(
        (
            EmitDescriptorOp(
                descriptor=_descriptor("constant.i32.mova"),
                results={"dst": ValueRef.temporary("control")},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": F32_ACCUMULATOR_ADD_CONTROL},
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=descriptor,
                operands=operands,
                results={"dst": native_result},
                result_types=None if width == 64 else {"dst": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            ),
        )
    )
    if width == 32:
        emit_f32_full_accumulator_result(
            native_result,
            ValueRef.result("result"),
            maximum,
            result_in_accumulator,
            emits,
        )
    return DescriptorRule(
        source_op=vector.vector_fmaf,
        descriptor=descriptor,
        priority=sum(origins) + int(zero),
        guards=tuple(guards),
        emit=tuple(emits),
        report_key="native_bf16_widening_fma",
    )


AIE2P_FLOATING_FMA_RULES = (
    *(
        _fma_rule(
            minimum,
            maximum,
            zero=zero,
            origins=origins,
            seed_in_accumulator=seed,
            result_in_accumulator=result,
            operands_in_accumulator=carriers,
        )
        for minimum, maximum in ((1, 16), (17, 31), (32, 32), (33, 64))
        for zero in (False, True)
        for origins in product((False, True), repeat=2)
        for carriers in product(
            *(
                (False, True) if maximum <= 16 and not origin else (False,)
                for origin in origins
            )
        )
        for seed in ((False, True) if maximum <= 16 and not zero else (False,))
        for result in ((False, True) if maximum <= 16 else (False,))
    ),
)
