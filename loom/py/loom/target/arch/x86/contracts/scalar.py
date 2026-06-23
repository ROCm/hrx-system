# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 scalar source-to-low contract fragment."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.buffer import defs as buffer
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.view import ALL_VIEW_OPS
from loom.dialect.view import defs as view
from loom.dsl import Op
from loom.target.arch.x86.descriptors import X86_SCALAR_DESCRIPTOR_SET
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
    SourceMemoryByteOffsetMaterializer,
    SourceMemoryConstraint,
    SourceMemoryDynamicIndexSource,
    SourceMemoryOperation,
    SourceMemoryProject,
    SourceMemoryRootKind,
    TypePattern,
    ValueAliasRule,
    ValueProject,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I1 = Scalar("i1")
_I32 = Scalar("i32")
_I64 = Scalar("i64")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")

_I32_MIN = -(2**31)
_I32_MAX = (2**31) - 1
_I64_MIN = -(2**63) + 1
_I64_MAX = (2**63) - 1
_DISP32_MIN = -(2**31)
_DISP32_MAX = (2**31) - 1

_ADDRESS_FORM_DIAGNOSTIC = GuardDiagnostic(
    subject_role="address-form",
    subject_name="x86-scalar",
    constraint_key="x86.scalar.address_form",
)
_SOURCE_MEMORY_DIAGNOSTIC = GuardDiagnostic(
    subject_role="source-memory",
    subject_name="x86-scalar",
    constraint_key="x86.scalar.source_memory",
)
_INDEX_CAST_DIAGNOSTIC = GuardDiagnostic(
    subject_role="index-cast",
    subject_name="x86-scalar",
    constraint_key="x86.scalar.index_cast",
)

_DescriptorLookup = Callable[[str], Descriptor]


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(X86_SCALAR_DESCRIPTOR_SET, key)


def x86_source_memory_byte_offset_materializer(
    descriptor_lookup: _DescriptorLookup,
) -> SourceMemoryByteOffsetMaterializer:
    return SourceMemoryByteOffsetMaterializer(
        const_i64=descriptor_lookup("x86.scalar.movimm.gpr64"),
        add_i64=descriptor_lookup("x86.scalar.add.gpr64"),
        mul_i64=descriptor_lookup("x86.scalar.imul.gpr64"),
        shl_i64=None,
        const_i64_immediate="imm64",
    )


def _op_emit(
    *,
    descriptor: Descriptor,
    operands: dict[str, ValueRef] | None = None,
    results: dict[str, ValueRef] | None = None,
    result_types: dict[str, TypePattern] | None = None,
    immediates: Mapping[str, AttrProject | SourceMemoryProject | ValueProject | int]
    | None = None,
    source_memory: SourceMemoryConstraint | None = None,
    source_memory_byte_offset_materializer: (
        SourceMemoryByteOffsetMaterializer | None
    ) = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        immediates={} if immediates is None else immediates,
        source_memory=source_memory,
        source_memory_byte_offset_materializer=source_memory_byte_offset_materializer,
        form=DescriptorEmitForm.OP,
    )


def _typed_guards(
    fields: Iterable[str],
    type_pattern: TypePattern,
) -> tuple[Guard, ...]:
    return tuple(Guard.value_type(field, type_pattern) for field in fields)


def _exact_i64_literal_guards(field: str, value: int) -> tuple[Guard, ...]:
    return (
        Guard.value_exact_i64(field, diagnostic=_ADDRESS_FORM_DIAGNOSTIC),
        Guard.value_i64_range(
            field,
            value,
            value,
            diagnostic=_ADDRESS_FORM_DIAGNOSTIC,
        ),
    )


def _disp32_guards(field: str) -> tuple[Guard, ...]:
    return (
        Guard.value_exact_i64(field, diagnostic=_ADDRESS_FORM_DIAGNOSTIC),
        Guard.value_i64_range(
            field,
            _DISP32_MIN,
            _DISP32_MAX,
            diagnostic=_ADDRESS_FORM_DIAGNOSTIC,
        ),
    )


def _negated_disp32_guards(field: str) -> tuple[Guard, ...]:
    return (
        Guard.value_exact_i64(field, diagnostic=_ADDRESS_FORM_DIAGNOSTIC),
        Guard.value_i64_range(
            field,
            -_DISP32_MAX,
            -_DISP32_MIN,
            diagnostic=_ADDRESS_FORM_DIAGNOSTIC,
        ),
    )


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


def _index_const_i64_rule(
    result_type: TypePattern,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.movimm.gpr64")
    return DescriptorRule(
        source_op=index.index_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", result_type),
            Guard.i64_range("value", -(2**63) + 1, (2**63) - 1),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"imm64": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _const_scalar_i64_rule(
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.movimm.gpr64")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", _I64),
            Guard.i64_range("value", _I64_MIN, _I64_MAX),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"imm64": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _const_i32_rule(
    result_type: TypePattern,
    descriptor_lookup: _DescriptorLookup,
    *,
    minimum: int = _I32_MIN,
    maximum: int = _I32_MAX,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.movimm.gpr32")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", result_type),
            Guard.i64_range("value", minimum, maximum),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _shift_imm_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
    *,
    maximum: int,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            Guard.value_exact_i64("rhs"),
            Guard.value_i64_range("rhs", 0, maximum),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"lhs": ValueRef.operand("lhs")},
                results={"dst": ValueRef.result("result")},
                immediates={"shift": ValueProject.exact_i64("rhs")},
            ),
        ),
    )


def _cmpi_rule(
    predicate: str,
    type_pattern: TypePattern,
    descriptor_suffix: str,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup(f"x86.scalar.cmp.{predicate}.{descriptor_suffix}")
    return DescriptorRule(
        source_op=scalar_comparison.scalar_cmpi,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            *_typed_guards(("lhs", "rhs"), type_pattern),
            Guard.value_type("result", _I1),
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
    dynamic: bool,
    element_byte_count: int,
    dynamic_byte_stride_factor: int = 1,
    materialize_byte_offset: bool = False,
) -> SourceMemoryConstraint:
    return SourceMemoryConstraint(
        operation=operation,
        root_kind=SourceMemoryRootKind.BLOCK_ARGUMENT,
        memory_spaces=("unknown", "generic", "global"),
        element_byte_count=element_byte_count,
        vector_lane_count=1,
        vector_lane_byte_stride=element_byte_count,
        static_byte_offset_minimum=_DISP32_MIN,
        static_byte_offset_maximum=_DISP32_MAX,
        dynamic_term_count=1 if dynamic else 0,
        dynamic_index_source=(
            SourceMemoryDynamicIndexSource.VALUE
            if dynamic
            else SourceMemoryDynamicIndexSource.NONE
        ),
        dynamic_byte_stride=(
            None
            if dynamic and materialize_byte_offset
            else element_byte_count * dynamic_byte_stride_factor
        )
        if dynamic
        else 0,
        diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
    )


def x86_factored_memory_immediates(
    *,
    element_byte_count: int,
) -> dict[str, SourceMemoryProject | int]:
    return {
        "disp32": SourceMemoryProject.static_byte_offset_remainder(element_byte_count),
        "scale": element_byte_count,
    }


def _memory_immediates(
    dynamic: bool,
    *,
    materialize_byte_offset: bool = False,
) -> dict[str, SourceMemoryProject | int]:
    immediates: dict[str, SourceMemoryProject | int] = {
        "disp32": SourceMemoryProject.static_byte_offset()
    }
    if dynamic:
        immediates["scale"] = (
            1 if materialize_byte_offset else SourceMemoryProject.dynamic_byte_stride()
        )
    return immediates


def x86_factored_index_emit(
    *,
    descriptor_lookup: _DescriptorLookup,
    element_byte_count: int,
    dynamic_byte_stride_factor: int,
    source_memory: SourceMemoryConstraint,
) -> EmitDescriptorOp:
    return _op_emit(
        descriptor=descriptor_lookup("x86.scalar.lea.scale.gpr64"),
        operands={"index": ValueRef.source_memory_dynamic_term()},
        results={"dst": ValueRef.temporary("factored_index")},
        result_types={"dst": _I64},
        immediates={
            "disp32": SourceMemoryProject.static_byte_offset_quotient(
                element_byte_count
            ),
            "scale": dynamic_byte_stride_factor,
        },
        source_memory=source_memory,
    )


def _view_load_rule(
    result_type: TypePattern,
    *,
    dynamic: bool,
    element_byte_count: int,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
    dynamic_byte_stride_factor: int = 1,
    materialize_byte_offset: bool = False,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    operands = {"base": ValueRef.operand("view")}
    if dynamic:
        if materialize_byte_offset:
            operands["index"] = ValueRef.source_memory_dynamic_byte_offset()
        elif dynamic_byte_stride_factor != 1:
            operands["index"] = ValueRef.temporary("factored_index")
        else:
            operands["index"] = ValueRef.operand("indices")
    source_memory = _source_memory_constraint(
        SourceMemoryOperation.LOAD,
        dynamic=dynamic,
        element_byte_count=element_byte_count,
        dynamic_byte_stride_factor=dynamic_byte_stride_factor,
        materialize_byte_offset=materialize_byte_offset,
    )
    emit: tuple[EmitDescriptorOp, ...]
    memory_emit = _op_emit(
        descriptor=descriptor,
        operands=operands,
        results={"dst": ValueRef.result("result")},
        immediates=(
            x86_factored_memory_immediates(element_byte_count=element_byte_count)
            if dynamic and dynamic_byte_stride_factor != 1
            else _memory_immediates(
                dynamic,
                materialize_byte_offset=materialize_byte_offset,
            )
        ),
        source_memory=source_memory,
        source_memory_byte_offset_materializer=(
            x86_source_memory_byte_offset_materializer(descriptor_lookup)
            if materialize_byte_offset
            else None
        ),
    )
    if dynamic and dynamic_byte_stride_factor != 1:
        emit = (
            x86_factored_index_emit(
                descriptor_lookup=descriptor_lookup,
                element_byte_count=element_byte_count,
                dynamic_byte_stride_factor=dynamic_byte_stride_factor,
                source_memory=source_memory,
            ),
            memory_emit,
        )
    else:
        emit = (memory_emit,)
    return DescriptorRule(
        source_op=view.view_load,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", 1 if dynamic else 0),
            Guard.value_type("result", result_type),
        ),
        emit=emit,
    )


def _view_store_rule(
    value_type: TypePattern,
    *,
    dynamic: bool,
    element_byte_count: int,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
    dynamic_byte_stride_factor: int = 1,
    materialize_byte_offset: bool = False,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    operands = {
        "value": ValueRef.operand("value"),
        "base": ValueRef.operand("view"),
    }
    if dynamic:
        if materialize_byte_offset:
            operands["index"] = ValueRef.source_memory_dynamic_byte_offset()
        elif dynamic_byte_stride_factor != 1:
            operands["index"] = ValueRef.temporary("factored_index")
        else:
            operands["index"] = ValueRef.operand("indices")
    source_memory = _source_memory_constraint(
        SourceMemoryOperation.STORE,
        dynamic=dynamic,
        element_byte_count=element_byte_count,
        dynamic_byte_stride_factor=dynamic_byte_stride_factor,
        materialize_byte_offset=materialize_byte_offset,
    )
    memory_emit = _op_emit(
        descriptor=descriptor,
        operands=operands,
        immediates=(
            x86_factored_memory_immediates(element_byte_count=element_byte_count)
            if dynamic and dynamic_byte_stride_factor != 1
            else _memory_immediates(
                dynamic,
                materialize_byte_offset=materialize_byte_offset,
            )
        ),
        source_memory=source_memory,
        source_memory_byte_offset_materializer=(
            x86_source_memory_byte_offset_materializer(descriptor_lookup)
            if materialize_byte_offset
            else None
        ),
    )
    if dynamic and dynamic_byte_stride_factor != 1:
        emit = (
            x86_factored_index_emit(
                descriptor_lookup=descriptor_lookup,
                element_byte_count=element_byte_count,
                dynamic_byte_stride_factor=dynamic_byte_stride_factor,
                source_memory=source_memory,
            ),
            memory_emit,
        )
    else:
        emit = (memory_emit,)
    return DescriptorRule(
        source_op=view.view_store,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", 1 if dynamic else 0),
            Guard.value_type("value", value_type),
        ),
        emit=emit,
    )


def _memory_descriptor_key(
    operation: str,
    *,
    dynamic: bool,
    register_suffix: str,
) -> str:
    indexed = ".indexed" if dynamic else ""
    return f"x86.scalar.mov.{operation}{indexed}.{register_suffix}"


def _memory_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for value_type, element_byte_count, register_suffix in (
        (_I32, 4, "gpr32"),
        (_I64, 8, "gpr64"),
    ):
        for dynamic in (False, True):
            descriptor_key = _memory_descriptor_key(
                "load",
                dynamic=dynamic,
                register_suffix=register_suffix,
            )
            rules.append(
                _view_load_rule(
                    value_type,
                    dynamic=dynamic,
                    element_byte_count=element_byte_count,
                    descriptor_key=descriptor_key,
                    descriptor_lookup=descriptor_lookup,
                )
            )
            if dynamic:
                rules.extend(
                    _view_load_rule(
                        value_type,
                        dynamic=True,
                        element_byte_count=element_byte_count,
                        descriptor_key=descriptor_key,
                        descriptor_lookup=descriptor_lookup,
                        dynamic_byte_stride_factor=dynamic_byte_stride_factor,
                    )
                    for dynamic_byte_stride_factor in (2, 4, 8)
                )
                rules.append(
                    _view_load_rule(
                        value_type,
                        dynamic=True,
                        element_byte_count=element_byte_count,
                        descriptor_key=descriptor_key,
                        descriptor_lookup=descriptor_lookup,
                        materialize_byte_offset=True,
                    )
                )
            descriptor_key = _memory_descriptor_key(
                "store",
                dynamic=dynamic,
                register_suffix=register_suffix,
            )
            rules.append(
                _view_store_rule(
                    value_type,
                    dynamic=dynamic,
                    element_byte_count=element_byte_count,
                    descriptor_key=descriptor_key,
                    descriptor_lookup=descriptor_lookup,
                )
            )
            if dynamic:
                rules.extend(
                    _view_store_rule(
                        value_type,
                        dynamic=True,
                        element_byte_count=element_byte_count,
                        descriptor_key=descriptor_key,
                        descriptor_lookup=descriptor_lookup,
                        dynamic_byte_stride_factor=dynamic_byte_stride_factor,
                    )
                    for dynamic_byte_stride_factor in (2, 4, 8)
                )
                rules.append(
                    _view_store_rule(
                        value_type,
                        dynamic=True,
                        element_byte_count=element_byte_count,
                        descriptor_key=descriptor_key,
                        descriptor_lookup=descriptor_lookup,
                        materialize_byte_offset=True,
                    )
                )
    return tuple(rules)


def _buffer_view_rule() -> ValueAliasRule:
    return ValueAliasRule(
        source_op=buffer.buffer_view,
        source=ValueRef.operand("buffer"),
        result=ValueRef.result("result"),
    )


def _index_cast_alias_rule(
    input_type: TypePattern,
    result_type: TypePattern,
) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=index.index_cast,
        source=ValueRef.operand("input"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("input", input_type),
            Guard.value_type("result", result_type),
        ),
    )


def _index_cast_i32_extend_rule(
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
    *,
    unsigned: bool,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    value_guard = (
        Guard.value_unsigned_bit_count("input", 32, diagnostic=_INDEX_CAST_DIAGNOSTIC)
        if unsigned
        else Guard.value_signed_bit_count(
            "input", 32, diagnostic=_INDEX_CAST_DIAGNOSTIC
        )
    )
    return DescriptorRule(
        source_op=index.index_cast,
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", _I32),
            Guard.value_type("result", _INDEX),
            value_guard,
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"src": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _add_disp_rule(
    type_pattern: TypePattern,
    *,
    base_field: str,
    displacement_field: str,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.lea.disp.gpr64")
    return DescriptorRule(
        source_op=index.index_add,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            *_disp32_guards(displacement_field),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"base": ValueRef.operand(base_field)},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "disp32": ValueProject.exact_i64(displacement_field),
                },
            ),
        ),
    )


def _sub_disp_rule(
    type_pattern: TypePattern,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.lea.disp.gpr64")
    return DescriptorRule(
        source_op=index.index_sub,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            *_negated_disp32_guards("rhs"),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"base": ValueRef.operand("lhs")},
                results={"dst": ValueRef.result("result")},
                immediates={"disp32": ValueProject.exact_i64_negate("rhs")},
            ),
        ),
    )


def _mul_scale_rule(
    *,
    index_field: str,
    scale_field: str,
    scale: int,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.lea.scale.gpr64")
    return DescriptorRule(
        source_op=index.index_mul,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), _INDEX),
            *_exact_i64_literal_guards(scale_field, scale),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"index": ValueRef.operand(index_field)},
                results={"dst": ValueRef.result("result")},
                immediates={"disp32": 0, "scale": scale},
            ),
        ),
    )


def _index_scale_literal_rule(
    *,
    scale: int,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.lea.scale.gpr64")
    return DescriptorRule(
        source_op=index.index_scale,
        descriptor=descriptor,
        guards=(
            Guard.value_type("index", _INDEX),
            Guard.value_type("stride", _OFFSET),
            Guard.value_type("result", _OFFSET),
            *_exact_i64_literal_guards("stride", scale),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"index": ValueRef.operand("index")},
                results={"dst": ValueRef.result("result")},
                immediates={"disp32": 0, "scale": scale},
            ),
        ),
    )


def _index_scale_rule(descriptor_lookup: _DescriptorLookup) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.imul.gpr64")
    return DescriptorRule(
        source_op=index.index_scale,
        descriptor=descriptor,
        guards=(
            Guard.value_type("index", _INDEX),
            Guard.value_type("stride", _OFFSET),
            Guard.value_type("result", _OFFSET),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("index"),
                    "rhs": ValueRef.operand("stride"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _shli_scale_rule(
    *,
    shift_amount: int,
    scale: int,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.lea.scale.gpr64")
    return DescriptorRule(
        source_op=index.index_shli,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), _INDEX),
            *_exact_i64_literal_guards("rhs", shift_amount),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"index": ValueRef.operand("lhs")},
                results={"dst": ValueRef.result("result")},
                immediates={"disp32": 0, "scale": scale},
            ),
        ),
    )


def _madd_scale_disp_rule(
    *,
    index_field: str,
    scale_field: str,
    scale: int,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.lea.scale.gpr64")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _INDEX),
            *_exact_i64_literal_guards(scale_field, scale),
            *_disp32_guards("c"),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"index": ValueRef.operand(index_field)},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "disp32": ValueProject.exact_i64("c"),
                    "scale": scale,
                },
            ),
        ),
    )


def _madd_add_scale_rule(
    *,
    index_field: str,
    scale_field: str,
    scale: int,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.lea.add_scale.gpr64")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _INDEX),
            *_exact_i64_literal_guards(scale_field, scale),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "base": ValueRef.operand("c"),
                    "index": ValueRef.operand(index_field),
                },
                results={"dst": ValueRef.result("result")},
                immediates={"disp32": 0, "scale": scale},
            ),
        ),
    )


def _madd_product_disp_rule(descriptor_lookup: _DescriptorLookup) -> DescriptorRule:
    multiply = descriptor_lookup("x86.scalar.imul.gpr64")
    add = descriptor_lookup("x86.scalar.lea.disp.gpr64")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=add,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _INDEX),
            *_disp32_guards("c"),
        ),
        emit=(
            _op_emit(
                descriptor=multiply,
                operands={
                    "lhs": ValueRef.operand("a"),
                    "rhs": ValueRef.operand("b"),
                },
                results={"dst": ValueRef.temporary("product")},
                result_types={"dst": _INDEX},
            ),
            _op_emit(
                descriptor=add,
                operands={"base": ValueRef.temporary("product")},
                results={"dst": ValueRef.result("result")},
                immediates={"disp32": ValueProject.exact_i64("c")},
            ),
        ),
    )


def _index_madd_rule(descriptor_lookup: _DescriptorLookup) -> DescriptorRule:
    multiply = descriptor_lookup("x86.scalar.imul.gpr64")
    add = descriptor_lookup("x86.scalar.lea.add.gpr64")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=add,
        guards=_typed_guards(("a", "b", "c", "result"), _INDEX),
        emit=(
            _op_emit(
                descriptor=multiply,
                operands={
                    "lhs": ValueRef.operand("a"),
                    "rhs": ValueRef.operand("b"),
                },
                results={"dst": ValueRef.temporary("product")},
                result_types={"dst": _INDEX},
            ),
            _op_emit(
                descriptor=add,
                operands={
                    "lhs": ValueRef.temporary("product"),
                    "rhs": ValueRef.operand("c"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _scale_rules(descriptor_lookup: _DescriptorLookup) -> tuple[DescriptorRule, ...]:
    rules = [
        _index_scale_literal_rule(
            scale=scale,
            descriptor_lookup=descriptor_lookup,
        )
        for scale in (1, 2, 4, 8)
    ]
    rules.append(_index_scale_rule(descriptor_lookup))
    for scale in (2, 4, 8):
        rules.append(
            _mul_scale_rule(
                index_field="lhs",
                scale_field="rhs",
                scale=scale,
                descriptor_lookup=descriptor_lookup,
            )
        )
        rules.append(
            _mul_scale_rule(
                index_field="rhs",
                scale_field="lhs",
                scale=scale,
                descriptor_lookup=descriptor_lookup,
            )
        )
    for shift_amount, scale in ((1, 2), (2, 4), (3, 8)):
        rules.append(
            _shli_scale_rule(
                shift_amount=shift_amount,
                scale=scale,
                descriptor_lookup=descriptor_lookup,
            )
        )
    return tuple(rules)


def _madd_address_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for scale in (1, 2, 4, 8):
        for index_field, scale_field in (("a", "b"), ("b", "a")):
            rules.append(
                _madd_scale_disp_rule(
                    index_field=index_field,
                    scale_field=scale_field,
                    scale=scale,
                    descriptor_lookup=descriptor_lookup,
                )
            )
        for index_field, scale_field in (("a", "b"), ("b", "a")):
            rules.append(
                _madd_add_scale_rule(
                    index_field=index_field,
                    scale_field=scale_field,
                    scale=scale,
                    descriptor_lookup=descriptor_lookup,
                )
            )
    rules.append(_madd_product_disp_rule(descriptor_lookup))
    rules.append(_index_madd_rule(descriptor_lookup))
    return tuple(rules)


def x86_scalar_core_cases(
    descriptor_lookup: _DescriptorLookup,
) -> Sequence[ContractCase]:
    return (
        _buffer_view_rule(),
        _binary_rule(
            scalar_arithmetic.scalar_addi,
            _I32,
            "x86.scalar.add.gpr32",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_arithmetic.scalar_subi,
            _I32,
            "x86.scalar.sub.gpr32",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_arithmetic.scalar_muli,
            _I32,
            "x86.scalar.imul.gpr32",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_arithmetic.scalar_addi,
            _I64,
            "x86.scalar.add.gpr64",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_arithmetic.scalar_subi,
            _I64,
            "x86.scalar.sub.gpr64",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_arithmetic.scalar_muli,
            _I64,
            "x86.scalar.imul.gpr64",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_bitwise.scalar_andi,
            _I32,
            "x86.scalar.and.gpr32",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_bitwise.scalar_ori,
            _I32,
            "x86.scalar.or.gpr32",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_bitwise.scalar_xori,
            _I32,
            "x86.scalar.xor.gpr32",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_bitwise.scalar_andi,
            _I64,
            "x86.scalar.and.gpr64",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_bitwise.scalar_ori,
            _I64,
            "x86.scalar.or.gpr64",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_bitwise.scalar_xori,
            _I64,
            "x86.scalar.xor.gpr64",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_bitwise.scalar_andi,
            _I1,
            "x86.scalar.and.gpr32",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_bitwise.scalar_ori,
            _I1,
            "x86.scalar.or.gpr32",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_bitwise.scalar_xori,
            _I1,
            "x86.scalar.xor.gpr32",
            descriptor_lookup,
        ),
        _shift_imm_rule(
            scalar_bitwise.scalar_shli,
            _I32,
            "x86.scalar.shl.imm.gpr32",
            descriptor_lookup,
            maximum=31,
        ),
        _shift_imm_rule(
            scalar_bitwise.scalar_shrsi,
            _I32,
            "x86.scalar.sar.imm.gpr32",
            descriptor_lookup,
            maximum=31,
        ),
        _shift_imm_rule(
            scalar_bitwise.scalar_shrui,
            _I32,
            "x86.scalar.shr.imm.gpr32",
            descriptor_lookup,
            maximum=31,
        ),
        _shift_imm_rule(
            scalar_bitwise.scalar_shli,
            _I64,
            "x86.scalar.shl.imm.gpr64",
            descriptor_lookup,
            maximum=63,
        ),
        _shift_imm_rule(
            scalar_bitwise.scalar_shrsi,
            _I64,
            "x86.scalar.sar.imm.gpr64",
            descriptor_lookup,
            maximum=63,
        ),
        _shift_imm_rule(
            scalar_bitwise.scalar_shrui,
            _I64,
            "x86.scalar.shr.imm.gpr64",
            descriptor_lookup,
            maximum=63,
        ),
        *(
            _cmpi_rule(predicate, _I32, "gpr32", descriptor_lookup)
            for predicate in (
                "eq",
                "ne",
                "slt",
                "sle",
                "sgt",
                "sge",
                "ult",
                "ule",
                "ugt",
                "uge",
            )
        ),
        *(
            _cmpi_rule(predicate, _I64, "gpr64", descriptor_lookup)
            for predicate in (
                "eq",
                "ne",
                "slt",
                "sle",
                "sgt",
                "sge",
                "ult",
                "ule",
                "ugt",
                "uge",
            )
        ),
        _const_i32_rule(_I32, descriptor_lookup),
        _const_i32_rule(_I1, descriptor_lookup, minimum=0, maximum=1),
        _const_scalar_i64_rule(descriptor_lookup),
        _index_const_i64_rule(_INDEX, descriptor_lookup),
        _index_const_i64_rule(_OFFSET, descriptor_lookup),
        _index_cast_i32_extend_rule(
            "x86.scalar.movzx.gpr64.gpr32",
            descriptor_lookup,
            unsigned=True,
        ),
        _index_cast_i32_extend_rule(
            "x86.scalar.movsxd.gpr64.gpr32",
            descriptor_lookup,
            unsigned=False,
        ),
        _index_cast_alias_rule(_I64, _INDEX),
        _add_disp_rule(
            _INDEX,
            base_field="lhs",
            displacement_field="rhs",
            descriptor_lookup=descriptor_lookup,
        ),
        _add_disp_rule(
            _INDEX,
            base_field="rhs",
            displacement_field="lhs",
            descriptor_lookup=descriptor_lookup,
        ),
        _add_disp_rule(
            _OFFSET,
            base_field="lhs",
            displacement_field="rhs",
            descriptor_lookup=descriptor_lookup,
        ),
        _add_disp_rule(
            _OFFSET,
            base_field="rhs",
            displacement_field="lhs",
            descriptor_lookup=descriptor_lookup,
        ),
        _sub_disp_rule(_INDEX, descriptor_lookup),
        _sub_disp_rule(_OFFSET, descriptor_lookup),
        _binary_rule(
            index.index_add, _INDEX, "x86.scalar.lea.add.gpr64", descriptor_lookup
        ),
        _binary_rule(
            index.index_add, _OFFSET, "x86.scalar.lea.add.gpr64", descriptor_lookup
        ),
        *_scale_rules(descriptor_lookup),
        _binary_rule(
            index.index_mul, _INDEX, "x86.scalar.imul.gpr64", descriptor_lookup
        ),
        *_madd_address_rules(descriptor_lookup),
        *_memory_rules(descriptor_lookup),
    )


X86_SCALAR_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
    "view": ALL_VIEW_OPS,
}

X86_SCALAR_CONTRACT_FRAGMENT = ContractFragment(
    name="x86.scalar",
    descriptor_set=X86_SCALAR_DESCRIPTOR_SET,
    public_header="loom/target/arch/x86/contracts/scalar.h",
    cases=x86_scalar_core_cases(_descriptor),
)
