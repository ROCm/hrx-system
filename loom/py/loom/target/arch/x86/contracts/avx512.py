# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 AVX512 source-to-low contract fragment."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dialect.view import ALL_VIEW_OPS
from loom.dsl import Op
from loom.target.arch.x86.contracts.avx2 import x86_avx2_core_cases
from loom.target.arch.x86.descriptors import X86_AVX512_CORE_DESCRIPTOR_SET
from loom.target.contracts import (
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
from loom.target.low_descriptors import Descriptor

_I32 = Scalar("i32")
_F32 = Scalar("f32")
_V4I1 = Vector("i1", lanes=4)
_V4I32 = Vector("i32", lanes=4)
_V4F32 = Vector("f32", lanes=4)
_V16I1 = Vector("i1", lanes=16)
_V16I32 = Vector("i32", lanes=16)
_V16F32 = Vector("f32", lanes=16)

_DISP32_MIN = -(2**31)
_DISP32_MAX = (2**31) - 1

_SOURCE_MEMORY_DIAGNOSTIC = GuardDiagnostic(
    subject_role="source-memory",
    subject_name="x86-avx512",
    constraint_key="x86.avx512.source_memory",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(X86_AVX512_CORE_DESCRIPTOR_SET, key)


def _op_emit(
    *,
    descriptor: Descriptor,
    operands: dict[str, ValueRef] | None = None,
    results: dict[str, ValueRef] | None = None,
    result_types: dict[str, TypePattern] | None = None,
    immediates: dict[str, SourceMemoryProject | int] | None = None,
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
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
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
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
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
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
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


def _select_rule(
    condition_type: TypePattern,
    value_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_select,
        descriptor=descriptor,
        guards=(
            Guard.value_type("condition", condition_type),
            Guard.value_type("true_value", value_type),
            Guard.value_type("false_value", value_type),
            Guard.value_type("result", value_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "mask": ValueRef.operand("condition"),
                    "true_value": ValueRef.operand("true_value"),
                    "false_value": ValueRef.operand("false_value"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _compare_rule(
    source_op: Op,
    predicate: str,
    operand_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            Guard.value_type("lhs", operand_type),
            Guard.value_type("rhs", operand_type),
            Guard.value_type("result", result_type),
        ),
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
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
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
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
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
    register_suffix: str,
) -> str:
    indexed = ".indexed" if dynamic else ""
    return f"x86.avx512.vmovdqu32.{operation}{indexed}.{register_suffix}"


def _memory_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for value_type, lanes, register_suffix in (
        (_V16I32, 16, "zmm"),
        (_V16F32, 16, "zmm"),
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
                        register_suffix=register_suffix,
                    ),
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
                        register_suffix=register_suffix,
                    ),
                )
            )
    return tuple(rules)


def _f32x4_reduce_emit_chain(
    input_value: ValueRef,
    *,
    temporary_prefix: str = "",
) -> tuple[EmitDescriptorOp, ...]:
    shuffle = _descriptor("x86.avx2.vpermilps.xmm")
    addps = _descriptor("x86.avx2.vaddps.xmm")
    addss = _descriptor("x86.avx2.vaddss.xmm")

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


def _reduce_f32x16_rule() -> DescriptorRule:
    extract = _descriptor("x86.avx512.vextractf32x4.xmm.zmm")
    addps = _descriptor("x86.avx2.vaddps.xmm")
    addss = _descriptor("x86.avx2.vaddss.xmm")
    vpermilps = _descriptor("x86.avx2.vpermilps.xmm")

    extract_emits = tuple(
        _op_emit(
            descriptor=extract,
            operands={"source": ValueRef.operand("input")},
            results={"dst": ValueRef.temporary(f"q{lane}")},
            result_types={"dst": _V4F32},
            immediates={"lane": lane},
        )
        for lane in range(4)
    )
    quarter_sum_emits = (
        _op_emit(
            descriptor=addps,
            operands={
                "lhs": ValueRef.temporary("q0"),
                "rhs": ValueRef.temporary("q1"),
            },
            results={"dst": ValueRef.temporary("sum01")},
            result_types={"dst": _V4F32},
        ),
        _op_emit(
            descriptor=addps,
            operands={
                "lhs": ValueRef.temporary("q2"),
                "rhs": ValueRef.temporary("q3"),
            },
            results={"dst": ValueRef.temporary("sum23")},
            result_types={"dst": _V4F32},
        ),
        _op_emit(
            descriptor=addps,
            operands={
                "lhs": ValueRef.temporary("sum01"),
                "rhs": ValueRef.temporary("sum23"),
            },
            results={"dst": ValueRef.temporary("xmm_sum")},
            result_types={"dst": _V4F32},
        ),
    )
    return DescriptorRule(
        source_op=vector.vector_reduce,
        descriptor=addss,
        guards=(
            Guard.enum_attr_equals("kind", "addf"),
            Guard.value_type("input", _V16F32),
            Guard.value_type("init", _F32),
            Guard.value_type("result", _F32),
            Guard.descriptor_available(extract),
            Guard.descriptor_available(vpermilps),
            Guard.descriptor_available(addps),
            Guard.descriptor_available(addss),
        ),
        emit=(
            *extract_emits,
            *quarter_sum_emits,
            *_f32x4_reduce_emit_chain(
                ValueRef.temporary("xmm_sum"),
                temporary_prefix="horizontal_",
            ),
        ),
    )


def _cases() -> Sequence[ContractCase]:
    return (
        *x86_avx2_core_cases(_descriptor),
        _splat_rule(_I32, _V16I32, "x86.avx512.vpbroadcastd.zmm"),
        _splat_rule(_F32, _V16F32, "x86.avx512.vbroadcastss.zmm"),
        _select_rule(_V16I1, _V16I32, "x86.avx512.vpblendmd.zmm"),
        _select_rule(_V16I1, _V16F32, "x86.avx512.vblendmps.zmm"),
        _select_rule(_V4I1, _V4I32, "x86.avx512.vpblendmd.xmm"),
        _select_rule(_V4I1, _V4F32, "x86.avx512.vblendmps.xmm"),
        _compare_rule(
            vector.vector_cmpi, "eq", _V16I32, _V16I1, "x86.avx512.vpcmpd.eq.zmm"
        ),
        _compare_rule(
            vector.vector_cmpi, "ne", _V16I32, _V16I1, "x86.avx512.vpcmpd.ne.zmm"
        ),
        _compare_rule(
            vector.vector_cmpi, "slt", _V16I32, _V16I1, "x86.avx512.vpcmpd.slt.zmm"
        ),
        _compare_rule(
            vector.vector_cmpi, "sle", _V16I32, _V16I1, "x86.avx512.vpcmpd.sle.zmm"
        ),
        _compare_rule(
            vector.vector_cmpi, "sgt", _V16I32, _V16I1, "x86.avx512.vpcmpd.sgt.zmm"
        ),
        _compare_rule(
            vector.vector_cmpi, "sge", _V16I32, _V16I1, "x86.avx512.vpcmpd.sge.zmm"
        ),
        _compare_rule(
            vector.vector_cmpi, "ult", _V16I32, _V16I1, "x86.avx512.vpcmpud.ult.zmm"
        ),
        _compare_rule(
            vector.vector_cmpi, "ule", _V16I32, _V16I1, "x86.avx512.vpcmpud.ule.zmm"
        ),
        _compare_rule(
            vector.vector_cmpi, "ugt", _V16I32, _V16I1, "x86.avx512.vpcmpud.ugt.zmm"
        ),
        _compare_rule(
            vector.vector_cmpi, "uge", _V16I32, _V16I1, "x86.avx512.vpcmpud.uge.zmm"
        ),
        _compare_rule(
            vector.vector_cmpi, "slt", _V4I32, _V4I1, "x86.avx512.vpcmpd.slt.xmm"
        ),
        _compare_rule(
            vector.vector_cmpf, "oeq", _V16F32, _V16I1, "x86.avx512.vcmpps.oeq.zmm"
        ),
        _compare_rule(
            vector.vector_cmpf, "ogt", _V16F32, _V16I1, "x86.avx512.vcmpps.ogt.zmm"
        ),
        _compare_rule(
            vector.vector_cmpf, "oge", _V16F32, _V16I1, "x86.avx512.vcmpps.oge.zmm"
        ),
        _compare_rule(
            vector.vector_cmpf, "olt", _V16F32, _V16I1, "x86.avx512.vcmpps.olt.zmm"
        ),
        _compare_rule(
            vector.vector_cmpf, "ole", _V16F32, _V16I1, "x86.avx512.vcmpps.ole.zmm"
        ),
        _compare_rule(
            vector.vector_cmpf, "one", _V16F32, _V16I1, "x86.avx512.vcmpps.one.zmm"
        ),
        _compare_rule(
            vector.vector_cmpf, "ord", _V16F32, _V16I1, "x86.avx512.vcmpps.ord.zmm"
        ),
        _compare_rule(
            vector.vector_cmpf, "ueq", _V16F32, _V16I1, "x86.avx512.vcmpps.ueq.zmm"
        ),
        _compare_rule(
            vector.vector_cmpf, "ugt", _V16F32, _V16I1, "x86.avx512.vcmpps.ugt.zmm"
        ),
        _compare_rule(
            vector.vector_cmpf, "uge", _V16F32, _V16I1, "x86.avx512.vcmpps.uge.zmm"
        ),
        _compare_rule(
            vector.vector_cmpf, "ult", _V16F32, _V16I1, "x86.avx512.vcmpps.ult.zmm"
        ),
        _compare_rule(
            vector.vector_cmpf, "ule", _V16F32, _V16I1, "x86.avx512.vcmpps.ule.zmm"
        ),
        _compare_rule(
            vector.vector_cmpf, "une", _V16F32, _V16I1, "x86.avx512.vcmpps.une.zmm"
        ),
        _compare_rule(
            vector.vector_cmpf, "uno", _V16F32, _V16I1, "x86.avx512.vcmpps.uno.zmm"
        ),
        _compare_rule(
            vector.vector_cmpf, "olt", _V4F32, _V4I1, "x86.avx512.vcmpps.olt.xmm"
        ),
        _binary_rule(vector.vector_addf, _V16F32, "x86.avx512.vaddps.zmm"),
        _binary_rule(vector.vector_subf, _V16F32, "x86.avx512.vsubps.zmm"),
        _binary_rule(vector.vector_mulf, _V16F32, "x86.avx512.vmulps.zmm"),
        _fma_rule(vector.vector_fmaf, _V16F32, "x86.avx512.vfmadd231ps.zmm"),
        _binary_rule(vector.vector_addi, _V16I32, "x86.avx512.vpaddd.zmm"),
        _binary_rule(vector.vector_subi, _V16I32, "x86.avx512.vpsubd.zmm"),
        _binary_rule(vector.vector_muli, _V16I32, "x86.avx512.vpmulld.zmm"),
        _binary_rule(vector.vector_minsi, _V16I32, "x86.avx512.vpminsd.zmm"),
        _binary_rule(vector.vector_maxsi, _V16I32, "x86.avx512.vpmaxsd.zmm"),
        _binary_rule(vector.vector_minui, _V16I32, "x86.avx512.vpminud.zmm"),
        _binary_rule(vector.vector_maxui, _V16I32, "x86.avx512.vpmaxud.zmm"),
        _binary_rule(vector.vector_andi, _V16I32, "x86.avx512.vpandd.zmm"),
        _binary_rule(vector.vector_andi, _V16I1, "x86.avx512.kandq"),
        _binary_rule(vector.vector_ori, _V16I32, "x86.avx512.vpord.zmm"),
        _binary_rule(vector.vector_ori, _V16I1, "x86.avx512.korq"),
        _binary_rule(vector.vector_xori, _V16I32, "x86.avx512.vpxord.zmm"),
        _binary_rule(vector.vector_xori, _V16I1, "x86.avx512.kxorq"),
        _binary_rule(vector.vector_shli, _V16I32, "x86.avx512.vpsllvd.zmm"),
        _binary_rule(vector.vector_shrsi, _V16I32, "x86.avx512.vpsravd.zmm"),
        _binary_rule(vector.vector_shrui, _V16I32, "x86.avx512.vpsrlvd.zmm"),
        *_memory_rules(),
        _reduce_f32x16_rule(),
    )


X86_AVX512_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
    "vector": ALL_VECTOR_OPS,
    "view": ALL_VIEW_OPS,
}

X86_AVX512_CONTRACT_FRAGMENT = ContractFragment(
    name="x86.avx512",
    descriptor_set=X86_AVX512_CORE_DESCRIPTOR_SET,
    cases=_cases(),
)
