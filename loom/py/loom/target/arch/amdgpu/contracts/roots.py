# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Range-preserving square roots and explicitly approximate native roots."""

from loom.dialect.scalar import math as scalar_math
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amdgpu.contracts.materializers import REGISTERS_VGPR_MATERIALIZER
from loom.target.contracts import (
    ContractEmit,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    Scalar,
    TypePattern,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import DescriptorSet


def _sqrtf_scaled_f32_rule(
    descriptor_set: DescriptorSet, source_op: Op, type_pattern: TypePattern
) -> DescriptorRule:
    # V_SQRT_F32 ignores subnormal inputs. Exact power-of-two scaling brings
    # both signs into its normal domain without changing native precision.
    # The smallest scaled magnitude is 2^-125; every positive root is normal.
    # Classification leaves signed zeros, infinities, and NaNs untouched.
    descriptors = {
        name: descriptor_by_key(descriptor_set, f"amdgpu.{name}")
        for name in (
            "s_mov_b32",
            "v_cmp_class_f32",
            "v_mul_f32.lit",
            "v_cndmask_b32",
            "v_sqrt_f32",
        )
    }
    source = ValueRef.operand("input", materializer=REGISTERS_VGPR_MATERIALIZER.name)
    temp = ValueRef.temporary
    form = (
        DescriptorEmitForm.PER_LANE
        if type_pattern.kind == "vector"
        else DescriptorEmitForm.OP
    )
    emit: list[ContractEmit] = [
        EmitDescriptorOp(
            descriptor=descriptors["s_mov_b32"],
            immediates={"imm32": 0x90},  # Negative and positive subnormal classes.
            results={"dst": temp("classes")},
            result_types={"dst": DescriptorResultType()},
        ),
    ]

    def instruction(
        name: str,
        result: ValueRef,
        *,
        role: str = "dst",
        immediates: dict[str, int] | None = None,
        **operands: ValueRef,
    ) -> None:
        emit.append(
            EmitDescriptorOp(
                descriptor=descriptors[name],
                operands=operands,
                immediates=immediates or {},
                results={role: result},
                result_types={role: DescriptorResultType()},
                form=form,
            )
        )

    instruction(
        "v_cmp_class_f32",
        temp("subnormal"),
        role="mask",
        input=source,
        classes=temp("classes"),
    )
    instruction(
        "v_mul_f32.lit", temp("scaled"), rhs=source, immediates={"imm32": 0x4B800000}
    )  # 2^24
    instruction(
        "v_cndmask_b32",
        temp("radicand"),
        false_value=source,
        true_value=temp("scaled"),
        mask=temp("subnormal"),
    )
    instruction("v_sqrt_f32", temp("root"), input=temp("radicand"))
    instruction(
        "v_mul_f32.lit",
        temp("unscaled"),
        rhs=temp("root"),
        immediates={"imm32": 0x39800000},
    )  # 2^-12
    instruction(
        "v_cndmask_b32",
        ValueRef.result("result"),
        false_value=temp("root"),
        true_value=temp("unscaled"),
        mask=temp("subnormal"),
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptors["v_sqrt_f32"],
        report_key=f"amdgpu.{source_op.name.replace('.', '_')}.strategy.scaled_f32",
        guards=(
            *(Guard.value_type(field, type_pattern) for field in ("input", "result")),
            Guard.value_materializable("input", REGISTERS_VGPR_MATERIALIZER.name),
            *(
                Guard.descriptor_available(descriptor)
                for descriptor in descriptors.values()
            ),
        ),
        emit=tuple(emit),
    )


def _sqrtf_exact_f64_rule(descriptor_set: DescriptorSet) -> DescriptorRule:
    # OCML's MATH_SQRT uses the compiler builtin. Match LLVM AMDGPU's
    # lowerFSQRTF64 in llvm/lib/Target/AMDGPU/SIISelLowering.cpp: Goldschmidt
    # refinement, including its final rounding step.
    # V_SQRT_F64 alone has only approximate precision, even without fastmath.
    # Scaling below 2^-767 keeps the fused residuals out of the subnormal range.
    descriptors = {
        name: descriptor_by_key(descriptor_set, f"amdgpu.{name}")
        for name in (
            "s_mov_b32",
            "v_mov_b32",
            "v_cmp_olt_f64",
            "v_cndmask_b32",
            "v_ldexp_f64",
            "v_rsq_f64",
            "v_mul_f64",
            "v_fma_f64",
            "v_fma_f64.neg_a",
            "v_cmp_class_f64",
        )
    }
    source = ValueRef.operand("input", materializer=REGISTERS_VGPR_MATERIALIZER.name)
    temp = ValueRef.temporary
    emit: list[ContractEmit] = []

    def instruction(
        name: str, result: str, *, role: str = "dst", **operands: ValueRef
    ) -> ValueRef:
        value = temp(result)
        emit.append(
            EmitDescriptorOp(
                descriptor=descriptors[name],
                operands=operands,
                results={role: value},
                result_types={role: DescriptorResultType()},
            )
        )
        return value

    def constant(name: str, bits: int, *, move: str = "v_mov_b32") -> ValueRef:
        value = temp(name)
        emit.append(
            EmitDescriptorOp(
                descriptor=descriptors[move],
                immediates={"imm32": bits},
                results={"dst": value},
                result_types={"dst": DescriptorResultType()},
            )
        )
        return value

    zero = constant("zero", 0)
    threshold_high = constant("threshold_high", 0x10000000)
    half_high = constant("half_high", 0x3FE00000)
    for name, high in (("threshold", threshold_high), ("half", half_high)):
        emit.append(
            EmitRegisterConcat(
                sources=(zero, high),
                result=temp(name),
                result_type=ValueRef.result("result"),
            )
        )
    scaling = instruction(
        "v_cmp_olt_f64", "scaling", role="mask", lhs=source, rhs=temp("threshold")
    )
    up = constant("up", 256)
    up_exponent = instruction(
        "v_cndmask_b32", "up_exponent", false_value=zero, true_value=up, mask=scaling
    )
    x = instruction("v_ldexp_f64", "scaled_input", input=source, exponent=up_exponent)
    y = instruction("v_rsq_f64", "reciprocal_root", input=x)
    s0 = instruction("v_mul_f64", "s0", lhs=x, rhs=y)
    h0 = instruction("v_mul_f64", "h0", lhs=y, rhs=temp("half"))
    r0 = instruction("v_fma_f64.neg_a", "r0", a=h0, b=s0, c=temp("half"))
    h1 = instruction("v_fma_f64", "h1", a=h0, b=r0, c=h0)
    s1 = instruction("v_fma_f64", "s1", a=s0, b=r0, c=s0)
    d0 = instruction("v_fma_f64.neg_a", "d0", a=s1, b=s1, c=x)
    s2 = instruction("v_fma_f64", "s2", a=d0, b=h1, c=s1)
    d1 = instruction("v_fma_f64.neg_a", "d1", a=s2, b=s2, c=x)
    rounded = instruction("v_fma_f64", "rounded", a=d1, b=h1, c=s2)
    down = constant("down", 0xFFFFFF80)  # signed exponent -128
    down_exponent = instruction(
        "v_cndmask_b32",
        "down_exponent",
        false_value=zero,
        true_value=down,
        mask=scaling,
    )
    root = instruction("v_ldexp_f64", "root", input=rounded, exponent=down_exponent)
    classes = constant("classes", 0x260, move="s_mov_b32")  # -0, +0, +infinity
    special = instruction(
        "v_cmp_class_f64", "special", role="mask", input=x, classes=classes
    )
    selected = []
    for unit in range(2):
        for name, value in (("root", root), ("input", x)):
            emit.append(
                EmitRegisterSlice(
                    source=value,
                    result=temp(f"{name}_{unit}"),
                    unit_offset=unit,
                    unit_count=1,
                )
            )
        selected.append(
            instruction(
                "v_cndmask_b32",
                f"selected_{unit}",
                false_value=temp(f"root_{unit}"),
                true_value=temp(f"input_{unit}"),
                mask=special,
            )
        )
    emit.append(EmitRegisterConcat(sources=selected, result=ValueRef.result("result")))
    return DescriptorRule(
        source_op=scalar_math.scalar_sqrtf,
        descriptor=descriptors["v_rsq_f64"],
        report_key="amdgpu.scalar_sqrtf.strategy.exact_f64",
        guards=(
            *(Guard.value_type(field, Scalar("f64")) for field in ("input", "result")),
            Guard.instance_flags_has_none("fastmath", "afn"),
            Guard.value_materializable("input", REGISTERS_VGPR_MATERIALIZER.name),
            *(
                Guard.descriptor_available(descriptor)
                for descriptor in descriptors.values()
            ),
        ),
        emit=tuple(emit),
    )


def root_rules(descriptors: DescriptorSet) -> tuple[DescriptorRule, ...]:
    rules = []
    for source_op, type_pattern, width, reciprocal in (
        (scalar_math.scalar_sqrtf, Scalar("f32"), 32, False),
        (scalar_math.scalar_rsqrtf, Scalar("f32"), 32, True),
        (scalar_math.scalar_sqrtf, Scalar("f64"), 64, False),
        (scalar_math.scalar_rsqrtf, Scalar("f64"), 64, True),
        (
            vector.vector_sqrtf,
            Vector(
                "f32",
                minimum_static_elements=1,
                maximum_static_elements="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
            ),
            32,
            False,
        ),
        (
            vector.vector_rsqrtf,
            Vector(
                "f32",
                minimum_static_elements=1,
                maximum_static_elements="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
            ),
            32,
            True,
        ),
    ):
        scaled = width == 32 and not reciprocal
        descriptor = descriptor_by_key(
            descriptors, f"amdgpu.v_{'rsq' if reciprocal else 'sqrt'}_f{width}"
        )
        strategy = f"{'native' if scaled else 'afn'}_f{width}"
        rules.append(
            DescriptorRule(
                source_op=source_op,
                descriptor=descriptor,
                report_key=(
                    f"amdgpu.{source_op.name.replace('.', '_')}.strategy.{strategy}"
                ),
                guards=(
                    *(
                        Guard.value_type(field, type_pattern)
                        for field in ("input", "result")
                    ),
                    Guard.value_not_subnormal_or_instance_flags_has_all(
                        "input", "fastmath", "afn"
                    )
                    if scaled
                    else Guard.instance_flags_has_all("fastmath", "afn"),
                    Guard.descriptor_available(descriptor),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=descriptor,
                        operands={"input": ValueRef.operand("input")},
                        results={"dst": ValueRef.result("result")},
                        form=DescriptorEmitForm.PER_LANE
                        if type_pattern.kind == "vector"
                        else DescriptorEmitForm.OP,
                    ),
                ),
            )
        )
        if scaled:
            rules.append(_sqrtf_scaled_f32_rule(descriptors, source_op, type_pattern))
    rules.append(_sqrtf_exact_f64_rule(descriptors))
    return tuple(rules)
