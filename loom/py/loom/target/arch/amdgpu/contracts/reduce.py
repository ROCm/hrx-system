# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU vector reduction source-to-low contracts."""

from __future__ import annotations

from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.target.arch.amdgpu.contracts.materializers import I32_VGPR_MATERIALIZER
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import (
    ContractFragment,
    DescriptorAccumulatorSeed,
    DescriptorAccumulatorTree,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    Scalar,
    TypePattern,
    ValueProject,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_DESCRIPTOR_KEYS = (
    "amdgpu.v_add_u32",
    "amdgpu.v_mul_lo_u32",
    "amdgpu.v_min_i32",
    "amdgpu.v_max_i32",
    "amdgpu.v_min_u32",
    "amdgpu.v_max_u32",
    "amdgpu.v_and_b32",
    "amdgpu.v_or_b32",
    "amdgpu.v_xor_b32",
    "amdgpu.v_add_f32",
    "amdgpu.v_add_f32.lit",
    "amdgpu.v_mul_f32",
    "amdgpu.v_mul_f32.lit",
    "amdgpu.v_min_f32",
    "amdgpu.v_min_f32.lit",
    "amdgpu.v_max_f32",
    "amdgpu.v_max_f32.lit",
)

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.reduce",
    descriptor_keys=_DESCRIPTOR_KEYS,
)

_VEC_I32 = Vector(
    "i32",
    minimum_lanes=1,
    maximum_lanes="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
)
_VEC_F32 = Vector(
    "f32",
    minimum_lanes=1,
    maximum_lanes="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
)
_I32 = Scalar("i32")
_F32 = Scalar("f32")

_LITERAL_EXACT_F32_DIAGNOSTIC = GuardDiagnostic(
    subject_role="literal",
    subject_name="f32",
    constraint_key="amdgpu.literal.exact_f32",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(_DESCRIPTOR_SET, key)


def _vector_reduce_rule(
    *,
    kind: str,
    input_type: TypePattern,
    accumulator_type: TypePattern,
    descriptor: Descriptor,
    materialized_init: ValueRef | None = None,
    accumulator_seed: DescriptorAccumulatorSeed = DescriptorAccumulatorSeed.OPERAND,
    accumulator_tree: DescriptorAccumulatorTree = DescriptorAccumulatorTree.CHAIN,
    extra_guards: tuple[Guard, ...] = (),
) -> DescriptorRule:
    init = (
        ValueRef.operand("input")
        if accumulator_seed == DescriptorAccumulatorSeed.FIRST_LANE
        else materialized_init or ValueRef.operand("init")
    )
    guards = [
        Guard.enum_attr_equals("kind", kind),
        Guard.value_type("input", input_type),
        Guard.value_type("init", accumulator_type),
        Guard.value_type("result", accumulator_type),
        Guard.low_value_register_class("input", "amdgpu.vgpr"),
        Guard.low_value_register_class("result", "amdgpu.vgpr"),
    ]
    guards.extend(extra_guards)
    if materialized_init is not None:
        materializer = materialized_init.materializer
        if materializer is None:
            raise ValueError("materialized init requires a materializer name")
        guards.append(Guard.value_materializable("init", materializer))
    guards.append(Guard.descriptor_available(descriptor))
    return DescriptorRule(
        source_op=vector.vector_reduce,
        descriptor=descriptor,
        guards=guards,
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": init,
                    "rhs": ValueRef.operand("input"),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.ACCUMULATE_LANES,
                accumulator="lhs",
                accumulator_seed=accumulator_seed,
                accumulator_tree=accumulator_tree,
            ),
        ),
    )


def _i32_rule(kind: str, descriptor_key: str) -> DescriptorRule:
    return _vector_reduce_rule(
        kind=kind,
        input_type=_VEC_I32,
        accumulator_type=_I32,
        descriptor=_descriptor(descriptor_key),
        materialized_init=ValueRef.operand(
            "init",
            materializer=I32_VGPR_MATERIALIZER.name,
        ),
    )


def _i32_add_zero_rule() -> DescriptorRule:
    return _vector_reduce_rule(
        kind="addi",
        input_type=_VEC_I32,
        accumulator_type=_I32,
        descriptor=_descriptor("amdgpu.v_add_u32"),
        accumulator_seed=DescriptorAccumulatorSeed.FIRST_LANE,
        extra_guards=(Guard.value_i64_range("init", 0, 0),),
    )


def _i32_signed_extremum_seed_rule(kind: str, descriptor_key: str) -> DescriptorRule:
    range_guard = (
        Guard.value_i64_range_le("input", "init")
        if kind == "minsi"
        else Guard.value_i64_range_ge("input", "init")
    )
    return _vector_reduce_rule(
        kind=kind,
        input_type=_VEC_I32,
        accumulator_type=_I32,
        descriptor=_descriptor(descriptor_key),
        accumulator_seed=DescriptorAccumulatorSeed.FIRST_LANE,
        extra_guards=(range_guard,),
    )


def _i32_unsigned_extremum_seed_rule(kind: str, descriptor_key: str) -> DescriptorRule:
    range_guard = (
        Guard.value_i64_range_le("input", "init")
        if kind == "minui"
        else Guard.value_i64_range_ge("input", "init")
    )
    return _vector_reduce_rule(
        kind=kind,
        input_type=_VEC_I32,
        accumulator_type=_I32,
        descriptor=_descriptor(descriptor_key),
        accumulator_seed=DescriptorAccumulatorSeed.FIRST_LANE,
        extra_guards=(
            Guard.value_i64_range("input", 0, 0x7FFF_FFFF_FFFF_FFFF),
            Guard.value_i64_range("init", 0, 0x7FFF_FFFF_FFFF_FFFF),
            range_guard,
        ),
    )


def _f32_rule(kind: str, descriptor_key: str) -> DescriptorRule:
    return _vector_reduce_rule(
        kind=kind,
        input_type=_VEC_F32,
        accumulator_type=_F32,
        descriptor=_descriptor(descriptor_key),
    )


def _f32_add_assumed_zero_rule() -> DescriptorRule:
    return _vector_reduce_rule(
        kind="addf",
        input_type=_VEC_F32,
        accumulator_type=_F32,
        descriptor=_descriptor("amdgpu.v_add_f32"),
        accumulator_seed=DescriptorAccumulatorSeed.FIRST_LANE,
        extra_guards=(
            Guard.instance_flags_has_all("fastmath", "nnan"),
            Guard.instance_flags_has_all("fastmath", "nsz"),
            Guard.value_f64_equals("init", 0.0),
        ),
    )


def _f32_add_reassociated_zero_rule() -> DescriptorRule:
    return _vector_reduce_rule(
        kind="addf",
        input_type=_VEC_F32,
        accumulator_type=_F32,
        descriptor=_descriptor("amdgpu.v_add_f32"),
        accumulator_seed=DescriptorAccumulatorSeed.FIRST_LANE,
        accumulator_tree=DescriptorAccumulatorTree.BALANCED,
        extra_guards=(
            Guard.instance_flags_has_all("fastmath", "reassoc"),
            Guard.instance_flags_has_all("fastmath", "nnan"),
            Guard.instance_flags_has_all("fastmath", "nsz"),
            Guard.value_f64_equals("init", 0.0),
        ),
    )


def _f32_extremum_reassociated_rule(kind: str, descriptor_key: str) -> DescriptorRule:
    return _vector_reduce_rule(
        kind=kind,
        input_type=_VEC_F32,
        accumulator_type=_F32,
        descriptor=_descriptor(descriptor_key),
        accumulator_tree=DescriptorAccumulatorTree.BALANCED,
        extra_guards=(
            Guard.instance_flags_has_all("fastmath", "reassoc"),
            Guard.instance_flags_has_all("fastmath", "nnan"),
            Guard.instance_flags_has_all("fastmath", "nsz"),
        ),
    )


def _f32_literal_seed_rule(
    kind: str,
    seed_descriptor_key: str,
    accumulate_descriptor_key: str,
    *,
    reassociated: bool = False,
) -> DescriptorRule:
    seed_descriptor = _descriptor(seed_descriptor_key)
    accumulate_descriptor = _descriptor(accumulate_descriptor_key)
    extra_guards: tuple[Guard, ...] = ()
    if reassociated:
        extra_guards = (
            Guard.instance_flags_has_all("fastmath", "reassoc"),
            Guard.instance_flags_has_all("fastmath", "nnan"),
            Guard.instance_flags_has_all("fastmath", "nsz"),
        )
    return DescriptorRule(
        source_op=vector.vector_reduce,
        descriptor=seed_descriptor,
        guards=(
            Guard.enum_attr_equals("kind", kind),
            Guard.value_type("input", _VEC_F32),
            Guard.value_type("init", _F32),
            Guard.value_type("result", _F32),
            Guard.low_value_register_class("input", "amdgpu.vgpr"),
            Guard.low_value_register_class("result", "amdgpu.vgpr"),
            Guard.value_exact_f64(
                "init",
                diagnostic=_LITERAL_EXACT_F32_DIAGNOSTIC,
            ),
            *extra_guards,
            Guard.descriptor_available(seed_descriptor),
            Guard.descriptor_available(accumulate_descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=seed_descriptor,
                operands={"rhs": ValueRef.operand("input")},
                results={"dst": ValueRef.temporary("seed")},
                result_types={"dst": ValueRef.operand("init")},
                immediates={"imm32": ValueProject.f64_as_f32_bits("init")},
                form=DescriptorEmitForm.FIRST_LANE,
            ),
            EmitDescriptorOp(
                descriptor=accumulate_descriptor,
                operands={
                    "lhs": ValueRef.temporary("seed"),
                    "rhs": ValueRef.operand("input"),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.ACCUMULATE_LANES,
                accumulator="lhs",
                accumulator_tree=(
                    DescriptorAccumulatorTree.BALANCED
                    if reassociated
                    else DescriptorAccumulatorTree.CHAIN
                ),
                skip_first_lane=True,
            ),
        ),
    )


AMDGPU_REDUCE_CONTRACT_DIALECT_OPS = {
    "vector": ALL_VECTOR_OPS,
}

AMDGPU_REDUCE_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.reduce",
    descriptor_set=_DESCRIPTOR_SET,
    c_source_includes=("loom/target/arch/amdgpu/lower/kinds.h",),
    materializers=(I32_VGPR_MATERIALIZER,),
    cases=[
        _i32_add_zero_rule(),
        _i32_rule("addi", "amdgpu.v_add_u32"),
        _i32_rule("muli", "amdgpu.v_mul_lo_u32"),
        _i32_signed_extremum_seed_rule("minsi", "amdgpu.v_min_i32"),
        _i32_rule("minsi", "amdgpu.v_min_i32"),
        _i32_signed_extremum_seed_rule("maxsi", "amdgpu.v_max_i32"),
        _i32_rule("maxsi", "amdgpu.v_max_i32"),
        _i32_unsigned_extremum_seed_rule("minui", "amdgpu.v_min_u32"),
        _i32_rule("minui", "amdgpu.v_min_u32"),
        _i32_unsigned_extremum_seed_rule("maxui", "amdgpu.v_max_u32"),
        _i32_rule("maxui", "amdgpu.v_max_u32"),
        _i32_rule("andi", "amdgpu.v_and_b32"),
        _i32_rule("ori", "amdgpu.v_or_b32"),
        _i32_rule("xori", "amdgpu.v_xor_b32"),
        _f32_add_reassociated_zero_rule(),
        _f32_add_assumed_zero_rule(),
        _f32_literal_seed_rule(
            "addf",
            "amdgpu.v_add_f32.lit",
            "amdgpu.v_add_f32",
            reassociated=True,
        ),
        _f32_literal_seed_rule("addf", "amdgpu.v_add_f32.lit", "amdgpu.v_add_f32"),
        _f32_literal_seed_rule("mulf", "amdgpu.v_mul_f32.lit", "amdgpu.v_mul_f32"),
        _f32_literal_seed_rule(
            "minnumf",
            "amdgpu.v_min_f32.lit",
            "amdgpu.v_min_f32",
            reassociated=True,
        ),
        _f32_literal_seed_rule(
            "maxnumf",
            "amdgpu.v_max_f32.lit",
            "amdgpu.v_max_f32",
            reassociated=True,
        ),
        _f32_literal_seed_rule(
            "minnumf",
            "amdgpu.v_min_f32.lit",
            "amdgpu.v_min_f32",
        ),
        _f32_literal_seed_rule(
            "maxnumf",
            "amdgpu.v_max_f32.lit",
            "amdgpu.v_max_f32",
        ),
        _f32_rule("addf", "amdgpu.v_add_f32"),
        _f32_rule("mulf", "amdgpu.v_mul_f32"),
        _f32_extremum_reassociated_rule("minnumf", "amdgpu.v_min_f32"),
        _f32_extremum_reassociated_rule("maxnumf", "amdgpu.v_max_f32"),
        _f32_rule("minnumf", "amdgpu.v_min_f32"),
        _f32_rule("maxnumf", "amdgpu.v_max_f32"),
    ],
)
