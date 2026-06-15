# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 AVX2 source-to-low contract fragment."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import math as scalar_math
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dialect.view import ALL_VIEW_OPS
from loom.dsl import Op
from loom.target.arch.x86.contracts.scalar import x86_scalar_core_cases
from loom.target.arch.x86.descriptors import X86_AVX2_DESCRIPTOR_SET
from loom.target.contracts import (
    AttrProject,
    ContractCase,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    Scalar,
    SourceMemoryConstraint,
    SourceMemoryDynamicIndexSource,
    SourceMemoryOperation,
    SourceMemoryProject,
    SourceMemoryRootKind,
    TypePattern,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.contracts.templates import (
    ReductionDescriptorCase,
    reduction_descriptor_rules,
)
from loom.target.low_descriptors import Descriptor

_DescriptorLookup = Callable[[str], Descriptor]

_I32 = Scalar("i32")
_F32 = Scalar("f32")
_V4I32 = Vector("i32", lanes=4)
_V4F32 = Vector("f32", lanes=4)

_DISP32_MIN = -(2**31)
_DISP32_MAX = (2**31) - 1

_SOURCE_MEMORY_DIAGNOSTIC = GuardDiagnostic(
    subject_role="source-memory",
    subject_name="x86-avx2",
    constraint_key="x86.avx2.source_memory",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(X86_AVX2_DESCRIPTOR_SET, key)


def _op_emit(
    *,
    descriptor: Descriptor,
    operands: dict[str, ValueRef] | None = None,
    results: dict[str, ValueRef] | None = None,
    result_types: dict[str, TypePattern] | None = None,
    immediates: Mapping[str, AttrProject | SourceMemoryProject | int] | None = None,
    source_memory: SourceMemoryConstraint | None = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        immediates={} if immediates is None else immediates,
        form=DescriptorEmitForm.OP,
        source_memory=source_memory,
    )


def _typed_guards(
    fields: Iterable[str],
    type_pattern: TypePattern,
) -> tuple[Guard, ...]:
    return tuple(Guard.value_type(field, type_pattern) for field in fields)


def _binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _fma_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_typed_guards(("a", "b", "c", "result"), type_pattern),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "acc": ValueRef.operand("c"),
                    "lhs": ValueRef.operand("a"),
                    "rhs": ValueRef.operand("b"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _splat_rule(
    scalar_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=descriptor,
        guards=(
            Guard.value_type("scalar", scalar_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"value": ValueRef.operand("scalar")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _extract_rule(
    source_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_extract,
        descriptor=descriptor,
        guards=(
            Guard.value_type("source", source_type),
            Guard.value_type("result", result_type),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range("static_indices", 0, 0, 3),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"source": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "lane" if result_type == _I32 else "control": (
                        AttrProject.i64_array_element(
                            "static_indices",
                            element=0,
                        )
                    )
                },
            ),
        ),
    )


def _insert_rule(
    value_type: TypePattern,
    dest_type: TypePattern,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    immediate = (
        AttrProject.i64_array_element("static_indices", element=0)
        if value_type == _I32
        else AttrProject.i64_array_element(
            "static_indices",
            element=0,
            target_bit_offset=4,
        )
    )
    return DescriptorRule(
        source_op=vector.vector_insert,
        descriptor=descriptor,
        guards=(
            Guard.value_type("value", value_type),
            Guard.value_type("dest", dest_type),
            Guard.value_type("result", dest_type),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range("static_indices", 0, 0, 3),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "dest": ValueRef.operand("dest"),
                    "value": ValueRef.operand("value"),
                },
                results={"dst": ValueRef.result("result")},
                immediates={
                    "lane" if value_type == _I32 else "control": immediate,
                },
            ),
        ),
    )


def _shuffle_rule(
    type_pattern: TypePattern,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_shuffle,
        descriptor=descriptor,
        guards=(
            Guard.value_type("source", type_pattern),
            Guard.value_type("result", type_pattern),
            Guard.i64_array_count("source_lanes", 4),
            Guard.i64_array_elements_range("source_lanes", 0, 3),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"source": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "control": AttrProject.i64_array_pack_elements(
                        "source_lanes",
                        element=0,
                        count=4,
                        bit_width=2,
                    )
                },
            ),
        ),
    )


def _source_memory_constraint(
    operation: SourceMemoryOperation,
    *,
    lanes: int,
    dynamic: bool,
) -> SourceMemoryConstraint:
    return SourceMemoryConstraint(
        operation=operation,
        root_kind=SourceMemoryRootKind.BLOCK_ARGUMENT,
        memory_spaces=("unknown", "generic", "global"),
        element_byte_count=4,
        vector_lane_count=lanes,
        vector_lane_byte_stride=4,
        static_byte_offset_minimum=_DISP32_MIN,
        static_byte_offset_maximum=_DISP32_MAX,
        dynamic_term_count=1 if dynamic else 0,
        dynamic_index_source=(
            SourceMemoryDynamicIndexSource.VALUE
            if dynamic
            else SourceMemoryDynamicIndexSource.NONE
        ),
        dynamic_byte_stride=4 if dynamic else 0,
        diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
    )


def _memory_immediates(
    dynamic: bool,
) -> dict[str, SourceMemoryProject | int]:
    immediates: dict[str, SourceMemoryProject | int] = {
        "disp32": SourceMemoryProject.static_byte_offset()
    }
    if dynamic:
        immediates["scale"] = SourceMemoryProject.dynamic_byte_stride()
    return immediates


def _vector_load_rule(
    result_type: TypePattern,
    *,
    lanes: int,
    dynamic: bool,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    operands = {"base": ValueRef.operand("view")}
    if dynamic:
        operands["index"] = ValueRef.operand("indices")
    return DescriptorRule(
        source_op=vector.vector_load,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", 1 if dynamic else 0),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands=operands,
                results={"dst": ValueRef.result("result")},
                immediates=_memory_immediates(dynamic),
                source_memory=_source_memory_constraint(
                    SourceMemoryOperation.LOAD,
                    lanes=lanes,
                    dynamic=dynamic,
                ),
            ),
        ),
    )


def _vector_store_rule(
    value_type: TypePattern,
    *,
    lanes: int,
    dynamic: bool,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    operands = {
        "value": ValueRef.operand("value"),
        "base": ValueRef.operand("view"),
    }
    if dynamic:
        operands["index"] = ValueRef.operand("indices")
    return DescriptorRule(
        source_op=vector.vector_store,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", 1 if dynamic else 0),
            Guard.value_type("value", value_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands=operands,
                immediates=_memory_immediates(dynamic),
                source_memory=_source_memory_constraint(
                    SourceMemoryOperation.STORE,
                    lanes=lanes,
                    dynamic=dynamic,
                ),
            ),
        ),
    )


def _memory_descriptor_key(
    operation: str,
    *,
    dynamic: bool,
) -> str:
    indexed = ".indexed" if dynamic else ""
    return f"x86.avx2.vmovdqu32.{operation}{indexed}.xmm"


def _memory_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for value_type, lanes in (
        (_V4I32, 4),
        (_V4F32, 4),
    ):
        for dynamic in (False, True):
            rules.append(
                _vector_load_rule(
                    value_type,
                    lanes=lanes,
                    dynamic=dynamic,
                    descriptor_key=_memory_descriptor_key(
                        "load",
                        dynamic=dynamic,
                    ),
                    descriptor_lookup=descriptor_lookup,
                )
            )
            rules.append(
                _vector_store_rule(
                    value_type,
                    lanes=lanes,
                    dynamic=dynamic,
                    descriptor_key=_memory_descriptor_key(
                        "store",
                        dynamic=dynamic,
                    ),
                    descriptor_lookup=descriptor_lookup,
                )
            )
    return tuple(rules)


def _f32x4_reduce_emit_chain(
    input_value: ValueRef,
    descriptor_lookup: _DescriptorLookup,
    *,
    temporary_prefix: str = "",
) -> tuple[EmitDescriptorOp, ...]:
    shuffle = descriptor_lookup("x86.avx2.vpermilps.xmm")
    addps = descriptor_lookup("x86.avx2.vaddps.xmm")
    addss = descriptor_lookup("x86.avx2.vaddss.xmm")

    def temp(name: str) -> ValueRef:
        return ValueRef.temporary(f"{temporary_prefix}{name}")

    return (
        _op_emit(
            descriptor=shuffle,
            operands={"source": input_value},
            results={"dst": temp("shuffle0")},
            result_types={"dst": _V4F32},
            immediates={"control": 78},
        ),
        _op_emit(
            descriptor=addps,
            operands={"lhs": input_value, "rhs": temp("shuffle0")},
            results={"dst": temp("pair_sum")},
            result_types={"dst": _V4F32},
        ),
        _op_emit(
            descriptor=shuffle,
            operands={"source": temp("pair_sum")},
            results={"dst": temp("shuffle1")},
            result_types={"dst": _V4F32},
            immediates={"control": 177},
        ),
        _op_emit(
            descriptor=addps,
            operands={"lhs": temp("pair_sum"), "rhs": temp("shuffle1")},
            results={"dst": temp("vector_sum")},
            result_types={"dst": _V4F32},
        ),
        _op_emit(
            descriptor=addss,
            operands={
                "lhs": ValueRef.operand("init"),
                "rhs": temp("vector_sum"),
            },
            results={"dst": ValueRef.result("result")},
            result_types={"dst": _F32},
        ),
    )


def _reduce_f32x4_rule(
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    addss = descriptor_lookup("x86.avx2.vaddss.xmm")
    vpermilps = descriptor_lookup("x86.avx2.vpermilps.xmm")
    vaddps = descriptor_lookup("x86.avx2.vaddps.xmm")
    return DescriptorRule(
        source_op=vector.vector_reduce,
        descriptor=addss,
        guards=(
            Guard.enum_attr_equals("kind", "addf"),
            Guard.value_type("input", _V4F32),
            Guard.value_type("init", _F32),
            Guard.value_type("result", _F32),
            Guard.descriptor_available(vpermilps),
            Guard.descriptor_available(vaddps),
            Guard.descriptor_available(addss),
        ),
        emit=_f32x4_reduce_emit_chain(
            ValueRef.operand("input"),
            descriptor_lookup,
        ),
    )


def x86_avx2_core_cases(
    descriptor_lookup: _DescriptorLookup,
) -> Sequence[ContractCase]:
    return (
        *x86_scalar_core_cases(descriptor_lookup),
        _binary_rule(
            scalar_arithmetic.scalar_addf,
            _F32,
            "x86.avx2.vaddss.xmm",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_arithmetic.scalar_subf,
            _F32,
            "x86.avx2.vsubss.xmm",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_arithmetic.scalar_mulf,
            _F32,
            "x86.avx2.vmulss.xmm",
            descriptor_lookup,
        ),
        _fma_rule(
            scalar_math.scalar_fmaf,
            _F32,
            "x86.avx2.vfmadd231ss.xmm",
            descriptor_lookup,
        ),
        _splat_rule(_I32, _V4I32, "x86.avx2.vpbroadcastd.xmm", descriptor_lookup),
        _splat_rule(_F32, _V4F32, "x86.avx2.vbroadcastss.xmm", descriptor_lookup),
        _extract_rule(
            _V4I32,
            _I32,
            "x86.avx2.vpextrd.gpr32.xmm",
            descriptor_lookup,
        ),
        _extract_rule(_V4F32, _F32, "x86.avx2.vpermilps.xmm", descriptor_lookup),
        _insert_rule(_I32, _V4I32, "x86.avx2.vpinsrd.xmm", descriptor_lookup),
        _insert_rule(_F32, _V4F32, "x86.avx2.vinsertps.xmm", descriptor_lookup),
        _shuffle_rule(_V4I32, "x86.avx2.vpshufd.xmm", descriptor_lookup),
        _shuffle_rule(_V4F32, "x86.avx2.vpermilps.xmm", descriptor_lookup),
        _binary_rule(
            vector.vector_addf,
            _V4F32,
            "x86.avx2.vaddps.xmm",
            descriptor_lookup,
        ),
        _binary_rule(
            vector.vector_subf,
            _V4F32,
            "x86.avx2.vsubps.xmm",
            descriptor_lookup,
        ),
        _binary_rule(
            vector.vector_mulf,
            _V4F32,
            "x86.avx2.vmulps.xmm",
            descriptor_lookup,
        ),
        _binary_rule(
            vector.vector_addi,
            _V4I32,
            "x86.avx2.vpaddd.xmm",
            descriptor_lookup,
        ),
        _binary_rule(
            vector.vector_subi,
            _V4I32,
            "x86.avx2.vpsubd.xmm",
            descriptor_lookup,
        ),
        _binary_rule(
            vector.vector_muli,
            _V4I32,
            "x86.avx2.vpmulld.xmm",
            descriptor_lookup,
        ),
        *_memory_rules(descriptor_lookup),
        *reduction_descriptor_rules(
            vector.vector_reduce,
            (
                ReductionDescriptorCase(
                    kind="addi",
                    input_type=_V4I32,
                    accumulator_type=_I32,
                    extract_descriptor=descriptor_lookup("x86.avx2.vpextrd.gpr32.xmm"),
                    combine_descriptor=descriptor_lookup("x86.scalar.add.gpr32"),
                ),
            ),
            lane_count=4,
        ),
        _reduce_f32x4_rule(descriptor_lookup),
    )


X86_AVX2_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
    "vector": ALL_VECTOR_OPS,
    "view": ALL_VIEW_OPS,
}

X86_AVX2_CONTRACT_FRAGMENT = ContractFragment(
    name="x86.avx2",
    descriptor_set=X86_AVX2_DESCRIPTOR_SET,
    cases=x86_avx2_core_cases(_descriptor),
)
