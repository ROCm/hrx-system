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
from loom.dialect.cfg import ALL_CFG_OPS, cfg_cond_br
from loom.dialect.globals import ALL_GLOBAL_OPS
from loom.dialect.globals import defs as globals
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scf import ALL_SCF_OPS
from loom.dialect.scf import defs as scf
from loom.dialect.view import ALL_VIEW_OPS
from loom.dialect.view import defs as view
from loom.dsl import Op
from loom.target.arch.x86.contracts.constants import (
    floating_storage_scalar_constant_rule,
)
from loom.target.arch.x86.contracts.integer_bits import (
    scalar_integer_bit_count_rules,
)
from loom.target.arch.x86.contracts.integer_division import (
    runtime_integer_division_rules,
    unsigned_constant_division_rules,
)
from loom.target.arch.x86.contracts.integer_rotation import (
    scalar_integer_rotation_rules,
)
from loom.target.arch.x86.contracts.memory import (
    x86_scalar_memory_rules,
    x86_view_carrier_rules,
)
from loom.target.arch.x86.descriptors import (
    X86_INTEGER_CONDITION_CODES,
    X86_SCALAR_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    AttrProject,
    Buffer,
    ContractCase,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    DirectDescriptorCase,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    Scalar,
    SourceNode,
    TypePattern,
    ValueAliasRule,
    ValueProject,
    ValueRef,
    binary_descriptor_rules,
    descriptor_by_key,
    unary_descriptor_rules,
)
from loom.target.low_descriptors import Descriptor

_I1 = Scalar("i1")
_I8 = Scalar("i8")
_I16 = Scalar("i16")
_I32 = Scalar("i32")
_I64 = Scalar("i64")
_F8E4M3 = Scalar("f8E4M3")
_F8E5M2 = Scalar("f8E5M2")
_F16 = Scalar("f16")
_BF16 = Scalar("bf16")
_STORAGE_FLOATS = Scalar(("f8E4M3", "f8E5M2", "f16", "bf16"))
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")
_GPR32_SELECT_TYPES = Scalar(
    ("i1", "i8", "i16", "i32", "f8E4M3", "f8E5M2", "f16", "bf16")
)
_GPR64_SELECT_TYPES = Scalar(("i64", "index", "offset"))
_INTEGER_CONDITION_CODE_MAP = dict(X86_INTEGER_CONDITION_CODES)

_I8_MIN = -(2**7)
_I8_MAX = (2**7) - 1
_I16_MIN = -(2**15)
_I16_MAX = (2**15) - 1
_I32_MIN = -(2**31)
_I32_MAX = (2**31) - 1
_I64_MIN = -(2**63)
_I64_MAX = (2**63) - 1
_DISP32_MIN = -(2**31)
_DISP32_MAX = (2**31) - 1

# Immediate arithmetic retains a separately loaded value and avoids
# materializing the constant in a register. Source-graph scores weight priority
# by covered node count, so this must exceed the two-node read-memory score.
_SCALAR_INTEGER_IMMEDIATE_PRIORITY = 3

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


def _integer_compare_immediate(bit_count: int) -> tuple[str, int, int]:
    if bit_count == 8:
        return "imm8", _I8_MIN, _I8_MAX
    if bit_count == 16:
        return "imm16", _I16_MIN, _I16_MAX
    return "imm32", _I32_MIN, _I32_MAX


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(X86_SCALAR_DESCRIPTOR_SET, key)


def _op_emit(
    *,
    descriptor: Descriptor,
    operands: dict[str, ValueRef] | None = None,
    results: dict[str, ValueRef] | None = None,
    result_types: dict[str, TypePattern] | None = None,
    immediates: Mapping[str, AttrProject | ValueProject | int] | None = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        immediates={} if immediates is None else immediates,
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


def _integer_binary_rules(
    source_op: Op,
    type_pattern: TypePattern,
    operation: str,
    register_width: int,
    descriptor_lookup: _DescriptorLookup,
    *,
    constant_operands: Sequence[str] = ("rhs", "lhs"),
    immediate_priority: int = 0,
) -> Iterable[DescriptorRule]:
    descriptor = descriptor_lookup(f"x86.scalar.{operation}.imm.gpr{register_width}")
    for constant_field in constant_operands:
        input_field = "lhs" if constant_field == "rhs" else "rhs"
        yield DescriptorRule(
            source_op=source_op,
            descriptor=descriptor,
            guards=(
                *_typed_guards(("lhs", "rhs", "result"), type_pattern),
                Guard.value_exact_i64(constant_field),
                # Wide operations sign-extend imm32; positive full-word masks
                # outside this range retain their full-width register form.
                Guard.value_i64_range(constant_field, _I32_MIN, _I32_MAX),
            ),
            emit=(
                _op_emit(
                    descriptor=descriptor,
                    operands={"lhs": ValueRef.operand(input_field)},
                    results={"dst": ValueRef.result("result")},
                    immediates={"imm32": ValueProject.exact_i64(constant_field)},
                ),
            ),
            priority=immediate_priority,
        )
    yield from binary_descriptor_rules(
        (
            DirectDescriptorCase(
                source_op,
                descriptor_lookup(f"x86.scalar.{operation}.gpr{register_width}"),
                type_pattern,
            ),
        ),
        form=DescriptorEmitForm.OP,
    )


def _select_rule(
    type_pattern: TypePattern,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    return DescriptorRule(
        source_op=scf.scf_select,
        descriptor=descriptor,
        guards=(
            Guard.value_type("condition", _I1),
            *_typed_guards(("true_value", "false_value", "result"), type_pattern),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "condition": ValueRef.operand("condition"),
                    "true_value": ValueRef.operand("true_value"),
                    "false_value": ValueRef.operand("false_value"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _integer_compare_select_rules(
    compare_op: Op,
    compare_type: TypePattern,
    compare_width: int,
    descriptor_lookup: _DescriptorLookup,
) -> Iterable[DescriptorRule]:
    immediate_name, immediate_min, immediate_max = _integer_compare_immediate(
        compare_width
    )
    for result_type, result_width in (
        (_GPR32_SELECT_TYPES, 32),
        (_GPR64_SELECT_TYPES, 64),
        (Buffer(), 64),
    ):
        select = SourceNode.exclusive_user(
            "select",
            source_op=scf.scf_select,
            parent_result=ValueRef.result("result"),
            node_operand=ValueRef.operand("condition"),
            guards=(
                Guard.value_type("condition", _I1),
                *_typed_guards(("true_value", "false_value", "result"), result_type),
            ),
        )
        for immediate in (True, False):
            form_name = "imm." if immediate else ""
            descriptor = descriptor_lookup(
                f"x86.scalar.cmp.select.{form_name}gpr{compare_width}.gpr{result_width}"
            )
            guards = [
                *_typed_guards(("lhs", "rhs"), compare_type),
                Guard.value_type("result", _I1),
            ]
            operands = {
                "lhs": ValueRef.operand("lhs"),
                "true_value": ValueRef.operand("true_value", source_node="select"),
                "false_value": ValueRef.operand("false_value", source_node="select"),
            }
            immediates: dict[str, AttrProject | ValueProject | int] = {
                "predicate": AttrProject.enum_remap(
                    "predicate", _INTEGER_CONDITION_CODE_MAP
                )
            }
            if immediate:
                guards.extend(
                    (
                        Guard.value_exact_i64("rhs"),
                        Guard.value_i64_range("rhs", immediate_min, immediate_max),
                    )
                )
                immediates[immediate_name] = ValueProject.exact_i64("rhs")
            else:
                operands["rhs"] = ValueRef.operand("rhs")
            yield DescriptorRule(
                source_op=compare_op,
                descriptor=descriptor,
                source_nodes=(select,),
                guards=guards,
                emit=(
                    _op_emit(
                        descriptor=descriptor,
                        operands=operands,
                        results={
                            "dst": ValueRef.result("result", source_node="select")
                        },
                        immediates=immediates,
                    ),
                ),
                priority=_SCALAR_INTEGER_IMMEDIATE_PRIORITY if immediate else 0,
                target_query=False,
            )


def _conversion_rule(
    source_op: Op,
    source_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"src": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _signed_boolean_extension_rule(
    result_type: TypePattern,
    register_width: int,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    # Scalar predicates are canonical 0/1 GPR32 carriers. Negation therefore
    # produces the signed all-zero or all-one extension without constants or a
    # conditional move. Wide results first establish a complete GPR64 value.
    negate = descriptor_lookup(f"x86.scalar.neg.gpr{register_width}")
    emit: list[EmitDescriptorOp] = []
    source = ValueRef.operand("input")
    if register_width == 64:
        widen = descriptor_lookup("x86.scalar.movzx.gpr64.gpr32")
        emit.append(
            _op_emit(
                descriptor=widen,
                operands={"src": source},
                results={"dst": ValueRef.temporary("unsigned")},
                result_types={"dst": _I64},
            )
        )
        source = ValueRef.temporary("unsigned")
    emit.append(
        _op_emit(
            descriptor=negate,
            operands={"input": source},
            results={"dst": ValueRef.result("result")},
        )
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_extsi,
        descriptor=negate,
        guards=(
            Guard.value_type("input", _I1),
            Guard.value_type("result", result_type),
        ),
        emit=tuple(emit),
    )


def _predicate_truncation_rule(
    input_type: TypePattern,
    input_width: int,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    # Integer truncation to i1 selects the low bit; it is not a nonzero test.
    # The mask also preserves the canonical 0/1 predicate carrier invariant.
    mask = descriptor_lookup("x86.scalar.and.imm.gpr32")
    emit: list[EmitDescriptorOp] = []
    source = ValueRef.operand("input")
    if input_width == 64:
        truncate = descriptor_lookup("x86.scalar.mov.trunc.gpr32.gpr64")
        emit.append(
            _op_emit(
                descriptor=truncate,
                operands={"src": source},
                results={"dst": ValueRef.temporary("word")},
                result_types={"dst": _I32},
            )
        )
        source = ValueRef.temporary("word")
    emit.append(
        _op_emit(
            descriptor=mask,
            operands={"lhs": source},
            results={"dst": ValueRef.result("result")},
            immediates={"imm32": 1},
        )
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_trunci,
        descriptor=mask,
        guards=(
            Guard.value_type("input", input_type),
            Guard.value_type("result", _I1),
        ),
        emit=tuple(emit),
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
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.movimm.gpr32")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", result_type),
            Guard.i64_range("value", _I32_MIN, _I32_MAX),
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


def _const_i1_rule(descriptor_lookup: _DescriptorLookup) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.movimm.gpr32")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.value_type("result", _I1),
            Guard.value_exact_i64("result"),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": ValueProject.exact_i64("result")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _shift_rules(
    source_op: Op,
    type_pattern: TypePattern,
    operation: str,
    bit_count: int,
    descriptor_lookup: _DescriptorLookup,
) -> Iterable[DescriptorRule]:
    descriptor = descriptor_lookup(f"x86.scalar.{operation}.imm.gpr{bit_count}")
    yield DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        priority=1,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            Guard.value_exact_i64("rhs"),
            Guard.value_i64_range("rhs", 0, bit_count - 1),
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
    descriptor = descriptor_lookup(f"x86.scalar.{operation}.cl.gpr{bit_count}")
    yield DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
                # Bind only this use to the fixed count register. Other uses
                # retain the original value's unconstrained register class.
                copy_operands=("rhs",),
            ),
        ),
    )


def _integer_compare_rule(
    source_op: Op,
    type_pattern: TypePattern,
    bit_count: int,
    immediate: bool,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    form_name = "imm." if immediate else ""
    descriptor = descriptor_lookup(f"x86.scalar.cmp.{form_name}gpr{bit_count}")
    immediate_name, immediate_min, immediate_max = _integer_compare_immediate(bit_count)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs"), type_pattern),
            Guard.value_type("result", _I1),
            *(
                (
                    Guard.value_exact_i64("rhs"),
                    Guard.value_i64_range("rhs", immediate_min, immediate_max),
                )
                if immediate
                else ()
            ),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    **({} if immediate else {"rhs": ValueRef.operand("rhs")}),
                },
                results={"dst": ValueRef.result("result")},
                immediates={
                    **(
                        {immediate_name: ValueProject.exact_i64("rhs")}
                        if immediate
                        else {}
                    ),
                    "predicate": AttrProject.enum_remap(
                        "predicate", _INTEGER_CONDITION_CODE_MAP
                    ),
                },
            ),
        ),
    )


def _integer_compare_branch_rule(
    compare_op: Op,
    compare_type: TypePattern,
    compare_width: int,
    immediate: bool,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    form_name = "imm." if immediate else ""
    descriptor = descriptor_lookup(
        f"x86.scalar.cmp.branch.{form_name}gpr{compare_width}"
    )
    immediate_name, immediate_min, immediate_max = _integer_compare_immediate(
        compare_width
    )
    compare_guards = [
        *_typed_guards(("lhs", "rhs"), compare_type),
        Guard.value_type("result", _I1),
    ]
    operands = {
        "lhs": ValueRef.operand("lhs", source_node="compare"),
    }
    immediates: dict[str, AttrProject | ValueProject | int] = {
        "predicate": AttrProject.enum_remap(
            "predicate", _INTEGER_CONDITION_CODE_MAP
        ).in_source_node("compare")
    }
    if immediate:
        compare_guards.extend(
            (
                Guard.value_exact_i64("rhs"),
                Guard.value_i64_range("rhs", immediate_min, immediate_max),
            )
        )
        immediates[immediate_name] = ValueProject.exact_i64("rhs").in_source_node(
            "compare"
        )
    else:
        operands["rhs"] = ValueRef.operand("rhs", source_node="compare")
    return DescriptorRule(
        source_op=cfg_cond_br,
        descriptor=descriptor,
        source_nodes=(
            SourceNode.exclusive_definition(
                "compare",
                source_op=compare_op,
                parent_operand=ValueRef.operand("condition"),
                node_result=ValueRef.result("result"),
                guards=compare_guards,
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                form=DescriptorEmitForm.COND_BR,
                operands=operands,
                immediates=immediates,
            ),
        ),
        priority=_SCALAR_INTEGER_IMMEDIATE_PRIORITY if immediate else 0,
        target_query=False,
    )


def _view_alias_rules() -> Iterable[ValueAliasRule]:
    yield ValueAliasRule(
        source_op=view.view_refine,
        source=ValueRef.operand("source"),
        result=ValueRef.result("result"),
    )


def _buffer_load_i8_u_rule(
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.movzx.load.indexed.u8.gpr32")
    return DescriptorRule(
        source_op=buffer.buffer_load_i8_u,
        descriptor=descriptor,
        guards=(
            Guard.value_type("byte_offset", _OFFSET),
            Guard.value_type("result", _I32),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "base": ValueRef.operand("source"),
                    "index": ValueRef.operand("byte_offset"),
                },
                results={"dst": ValueRef.result("result")},
                immediates={"disp32": 0, "scale": 1},
            ),
        ),
    )


def _buffer_store_i8_rule(
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.mov.store.indexed.u8.gpr32")
    return DescriptorRule(
        source_op=buffer.buffer_store_i8,
        descriptor=descriptor,
        guards=(
            Guard.value_type("value", _I32),
            Guard.value_type("byte_offset", _OFFSET),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "value": ValueRef.operand("value"),
                    "base": ValueRef.operand("target"),
                    "index": ValueRef.operand("byte_offset"),
                },
                immediates={"disp32": 0, "scale": 1},
            ),
        ),
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


def _conversion_alias_rule(
    source_op: Op,
    input_type: TypePattern,
    result_type: TypePattern,
) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=source_op,
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
    value_guards = (
        (
            Guard.value_unsigned_bit_count(
                "input", 32, diagnostic=_INDEX_CAST_DIAGNOSTIC
            ),
        )
        if unsigned
        else ()
    )
    return DescriptorRule(
        source_op=index.index_cast,
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", _I32),
            Guard.value_type("result", _INDEX),
            *value_guards,
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


def _index_scale_immediate_rule(
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.imul.imm.gpr64")
    return DescriptorRule(
        source_op=index.index_scale,
        descriptor=descriptor,
        guards=(
            Guard.value_type("index", _INDEX),
            Guard.value_type("stride", _OFFSET),
            Guard.value_type("result", _OFFSET),
            Guard.value_exact_i64("stride"),
            Guard.value_i64_range("stride", _I32_MIN, _I32_MAX),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"lhs": ValueRef.operand("index")},
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": ValueProject.exact_i64("stride")},
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
        priority=2,
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
    rules.append(_index_scale_immediate_rule(descriptor_lookup))
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


def _cases() -> Sequence[ContractCase]:
    descriptor_lookup = _descriptor
    return (
        *x86_view_carrier_rules(
            descriptor_lookup, diagnostic=_SOURCE_MEMORY_DIAGNOSTIC
        ),
        *_view_alias_rules(),
        _buffer_load_i8_u_rule(descriptor_lookup),
        _buffer_store_i8_rule(descriptor_lookup),
        *(
            rule
            for type_pattern, register_width in ((_I32, 32), (_I64, 64))
            for source_op, operation in (
                (scalar_arithmetic.scalar_addi, "add"),
                (scalar_arithmetic.scalar_subi, "sub"),
                (scalar_arithmetic.scalar_muli, "imul"),
            )
            for rule in _integer_binary_rules(
                source_op,
                type_pattern,
                operation,
                register_width,
                descriptor_lookup,
                constant_operands=("rhs",) if operation == "sub" else ("rhs", "lhs"),
                immediate_priority=_SCALAR_INTEGER_IMMEDIATE_PRIORITY,
            )
        ),
        *unary_descriptor_rules(
            tuple(
                DirectDescriptorCase(
                    scalar_arithmetic.scalar_negi,
                    descriptor_lookup(f"x86.scalar.neg.gpr{register_width}"),
                    type_pattern,
                )
                for type_pattern, register_width in ((_I32, 32), (_I64, 64))
            ),
            form=DescriptorEmitForm.OP,
        ),
        *(
            rule
            for type_pattern, register_width in (
                (_I1, 32),
                (_I8, 32),
                (_I16, 32),
                (_I32, 32),
                (_I64, 64),
            )
            for source_op, operation in (
                (scalar_bitwise.scalar_andi, "and"),
                (scalar_bitwise.scalar_ori, "or"),
                (scalar_bitwise.scalar_xori, "xor"),
            )
            for rule in _integer_binary_rules(
                source_op,
                type_pattern,
                operation,
                register_width,
                descriptor_lookup,
                immediate_priority=_SCALAR_INTEGER_IMMEDIATE_PRIORITY,
            )
        ),
        _select_rule(_GPR32_SELECT_TYPES, "x86.scalar.select.gpr32", descriptor_lookup),
        _select_rule(_GPR64_SELECT_TYPES, "x86.scalar.select.gpr64", descriptor_lookup),
        _select_rule(Buffer(), "x86.scalar.select.gpr64", descriptor_lookup),
        *(
            rule
            for type_pattern, bit_count in ((_I32, 32), (_I64, 64))
            for source_op, operation in (
                (scalar_bitwise.scalar_shli, "shl"),
                (scalar_bitwise.scalar_shrsi, "sar"),
                (scalar_bitwise.scalar_shrui, "shr"),
            )
            for rule in _shift_rules(
                source_op, type_pattern, operation, bit_count, descriptor_lookup
            )
        ),
        *(
            rule
            for source_op, operation in (
                (index.index_shli, "shl"),
                (index.index_shrsi, "sar"),
                (index.index_shrui, "shr"),
            )
            for rule in _shift_rules(
                source_op, _INDEX, operation, 64, descriptor_lookup
            )
        ),
        *(
            rule
            for source_op, compare_type, compare_width in (
                (scalar_comparison.scalar_cmpi, _I8, 8),
                (scalar_comparison.scalar_cmpi, _I16, 16),
                (scalar_comparison.scalar_cmpi, _I32, 32),
                (scalar_comparison.scalar_cmpi, _I64, 64),
                (index.index_cmp, Scalar(("index", "offset")), 64),
            )
            for rule in _integer_compare_select_rules(
                source_op, compare_type, compare_width, descriptor_lookup
            )
        ),
        *(
            _integer_compare_branch_rule(
                source_op,
                compare_type,
                compare_width,
                immediate,
                descriptor_lookup,
            )
            for source_op, compare_type, compare_width in (
                (scalar_comparison.scalar_cmpi, _I8, 8),
                (scalar_comparison.scalar_cmpi, _I16, 16),
                (scalar_comparison.scalar_cmpi, _I32, 32),
                (scalar_comparison.scalar_cmpi, _I64, 64),
                (index.index_cmp, Scalar(("index", "offset")), 64),
            )
            for immediate in (True, False)
        ),
        *(
            _integer_compare_rule(
                scalar_comparison.scalar_cmpi,
                type_pattern,
                bit_count,
                immediate,
                descriptor_lookup,
            )
            for type_pattern, bit_count in (
                (_I8, 8),
                (_I16, 16),
                (_I32, 32),
                (_I64, 64),
            )
            for immediate in (True, False)
        ),
        *(
            _const_i32_rule(result_type, descriptor_lookup)
            for result_type in (_I8, _I16, _I32)
        ),
        floating_storage_scalar_constant_rule(
            _STORAGE_FLOATS,
            descriptor_lookup("x86.scalar.movimm.gpr32"),
        ),
        _const_i1_rule(descriptor_lookup),
        _const_scalar_i64_rule(descriptor_lookup),
        _index_const_i64_rule(_INDEX, descriptor_lookup),
        _index_const_i64_rule(_OFFSET, descriptor_lookup),
        DescriptorRule(
            source_op=globals.global_load,
            guards=(
                Guard.value_type("result", Buffer()),
                # Readonly byte symbols provide constant-memory buffer roots.
                # Loading a value global has no such address provenance.
                Guard.value_memory_space("result", ("constant",)),
            ),
            emit=(
                _op_emit(
                    descriptor=descriptor_lookup("x86.scalar.lea.symbol.gpr64"),
                    results={"dst": ValueRef.result("result")},
                    immediates={"symbol": AttrProject.direct("global")},
                ),
            ),
        ),
        *(
            _integer_compare_rule(
                index.index_cmp,
                Scalar(("index", "offset")),
                64,
                immediate,
                descriptor_lookup,
            )
            for immediate in (True, False)
        ),
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
        _index_cast_alias_rule(_INDEX, _I64),
        _index_cast_alias_rule(_OFFSET, _I64),
        _index_cast_alias_rule(_OFFSET, _INDEX),
        _index_cast_alias_rule(_I64, _OFFSET),
        _index_cast_alias_rule(_INDEX, _OFFSET),
        _conversion_rule(
            index.index_cast,
            _I32,
            _OFFSET,
            "x86.scalar.movzx.gpr64.gpr32",
            descriptor_lookup,
        ),
        *(
            _conversion_rule(
                index.index_cast,
                input_type,
                _I32,
                "x86.scalar.mov.trunc.gpr32.gpr64",
                descriptor_lookup,
            )
            for input_type in (_INDEX, _OFFSET)
        ),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _F8E4M3, _I8),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _I8, _F8E4M3),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _F8E5M2, _I8),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _I8, _F8E5M2),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _F16, _I16),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _I16, _F16),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _BF16, _I16),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _I16, _BF16),
        *(
            _conversion_alias_rule(
                scalar_conversion.scalar_trunci, input_type, result_type
            )
            for input_type, result_type in (
                (_I16, _I8),
                (_I32, _I8),
                (_I32, _I16),
            )
        ),
        *(
            _conversion_rule(
                scalar_conversion.scalar_trunci,
                _I64,
                result_type,
                "x86.scalar.mov.trunc.gpr32.gpr64",
                descriptor_lookup,
            )
            for result_type in (_I8, _I16, _I32)
        ),
        *(
            _predicate_truncation_rule(input_type, input_width, descriptor_lookup)
            for input_type, input_width in (
                (_I8, 8),
                (_I16, 16),
                (_I32, 32),
                (_I64, 64),
            )
        ),
        *(
            _conversion_alias_rule(scalar_conversion.scalar_extui, _I1, result_type)
            for result_type in (_I8, _I16, _I32)
        ),
        _conversion_rule(
            scalar_conversion.scalar_extui,
            _I1,
            _I64,
            "x86.scalar.movzx.gpr64.gpr32",
            descriptor_lookup,
        ),
        *(
            _signed_boolean_extension_rule(
                result_type, register_width, descriptor_lookup
            )
            for result_type, register_width in (
                (_I8, 32),
                (_I16, 32),
                (_I32, 32),
                (_I64, 64),
            )
        ),
        *(
            _conversion_rule(
                scalar_conversion.scalar_extui,
                input_type,
                result_type,
                f"x86.scalar.movzx.u{input_width}.gpr{carrier_width}",
                descriptor_lookup,
            )
            for input_type, input_width, result_type, carrier_width in (
                (_I8, 8, _I16, 32),
                (_I8, 8, _I32, 32),
                (_I8, 8, _I64, 64),
                (_I16, 16, _I32, 32),
                (_I16, 16, _I64, 64),
            )
        ),
        *(
            _conversion_rule(
                scalar_conversion.scalar_extsi,
                input_type,
                result_type,
                f"x86.scalar.movsx.i{input_width}.gpr{carrier_width}",
                descriptor_lookup,
            )
            for input_type, input_width, result_type, carrier_width in (
                (_I8, 8, _I16, 32),
                (_I8, 8, _I32, 32),
                (_I8, 8, _I64, 64),
                (_I16, 16, _I32, 32),
                (_I16, 16, _I64, 64),
            )
        ),
        _conversion_rule(
            scalar_conversion.scalar_extsi,
            _I32,
            _I64,
            "x86.scalar.movsxd.gpr64.gpr32",
            descriptor_lookup,
        ),
        _conversion_rule(
            scalar_conversion.scalar_extui,
            _I32,
            _I64,
            "x86.scalar.movzx.gpr64.gpr32",
            descriptor_lookup,
        ),
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
        *binary_descriptor_rules(
            tuple(
                DirectDescriptorCase(
                    source_op,
                    descriptor_lookup(f"x86.scalar.{operation}.gpr64"),
                    type_pattern,
                )
                for source_op, operation in (
                    (index.index_sub, "sub"),
                    (index.index_add, "lea.add"),
                )
                for type_pattern in (_INDEX, _OFFSET)
            ),
            form=DescriptorEmitForm.OP,
        ),
        *_scale_rules(descriptor_lookup),
        *_integer_binary_rules(
            index.index_mul,
            _INDEX,
            "imul",
            64,
            descriptor_lookup,
        ),
        *(
            rule
            for source_op, operation in (
                (index.index_andi, "and"),
                (index.index_ori, "or"),
                (index.index_xori, "xor"),
            )
            for rule in _integer_binary_rules(
                source_op, _INDEX, operation, 64, descriptor_lookup
            )
        ),
        *_madd_address_rules(descriptor_lookup),
        *x86_scalar_memory_rules(
            descriptor_lookup, diagnostic=_SOURCE_MEMORY_DIAGNOSTIC
        ),
        *unsigned_constant_division_rules(descriptor_lookup),
        *runtime_integer_division_rules(descriptor_lookup),
        *scalar_integer_rotation_rules(descriptor_lookup),
        *scalar_integer_bit_count_rules(descriptor_lookup),
    )


X86_SCALAR_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "cfg": ALL_CFG_OPS,
    "global": ALL_GLOBAL_OPS,
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
    "scf": ALL_SCF_OPS,
    "view": ALL_VIEW_OPS,
}

X86_SCALAR_CONTRACT_FRAGMENT = ContractFragment(
    name="x86.scalar",
    descriptor_set=X86_SCALAR_DESCRIPTOR_SET,
    public_header="loom/target/arch/x86/contracts/scalar.h",
    cases=_cases(),
)
