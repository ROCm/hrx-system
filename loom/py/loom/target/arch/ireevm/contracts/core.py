# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IREE VM source-to-low contract fragment."""

from __future__ import annotations

from collections.abc import Iterable

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scalar import math as scalar_math
from loom.dsl import Op
from loom.target.arch.ireevm.descriptors import IREEVM_CORE_DESCRIPTOR_SET
from loom.target.contracts import (
    AttrProject,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    Scalar,
    TypePattern,
    ValueProject,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I1 = Scalar("i1")
_I32 = Scalar("i32")
_I64 = Scalar("i64")
_F32 = Scalar("f32")
_F64 = Scalar("f64")

_SUPPORTED_SCALAR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="source",
    constraint_key="ireevm.scalar.supported",
)
_I32_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="i32",
    constraint_key="ireevm.scalar.i32",
)
_I64_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="i64",
    constraint_key="ireevm.scalar.i64",
)
_F32_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="f32",
    constraint_key="ireevm.scalar.f32",
)
_F64_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="f64",
    constraint_key="ireevm.scalar.f64",
)
_I64_ATTR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="attr",
    subject_name="value",
    constraint_key="ireevm.constant.i64_attr",
)
_F64_ATTR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="attr",
    subject_name="value",
    constraint_key="ireevm.constant.f64_attr",
)
_I1_CONSTANT_RANGE_DIAGNOSTIC = GuardDiagnostic(
    subject_role="attr",
    subject_name="value",
    constraint_key="ireevm.constant.i1_range",
)
_I32_CONSTANT_RANGE_DIAGNOSTIC = GuardDiagnostic(
    subject_role="attr",
    subject_name="value",
    constraint_key="ireevm.constant.i32_range",
)
_FLOAT_CONSTANT_EXACT_DIAGNOSTIC = GuardDiagnostic(
    subject_role="value",
    subject_name="result",
    constraint_key="ireevm.constant.float_exact",
)
_PREDICATE_DIAGNOSTIC = GuardDiagnostic(
    subject_role="attr",
    subject_name="predicate",
    constraint_key="ireevm.compare.predicate",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(IREEVM_CORE_DESCRIPTOR_SET, key)


def _type_diagnostic(type_pattern: TypePattern) -> GuardDiagnostic:
    if type_pattern == _I32:
        return _I32_DIAGNOSTIC
    if type_pattern == _I64:
        return _I64_DIAGNOSTIC
    if type_pattern == _F32:
        return _F32_DIAGNOSTIC
    if type_pattern == _F64:
        return _F64_DIAGNOSTIC
    return _SUPPORTED_SCALAR_DIAGNOSTIC


def _value_type_guard(field_name: str, type_pattern: TypePattern) -> Guard:
    return Guard.value_type(
        field_name,
        type_pattern,
        diagnostic=_type_diagnostic(type_pattern),
    )


def _same_type_guards(
    field_names: Iterable[str],
    type_pattern: TypePattern,
) -> tuple[Guard, ...]:
    return tuple(
        _value_type_guard(field_name, type_pattern) for field_name in field_names
    )


def _const_i32_rule(
    result_type: TypePattern,
    range_diagnostic: GuardDiagnostic,
) -> DescriptorRule:
    descriptor = _descriptor("ireevm.const.i32")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64", diagnostic=_I64_ATTR_DIAGNOSTIC),
            Guard.value_type(
                "result",
                result_type,
                diagnostic=_SUPPORTED_SCALAR_DIAGNOSTIC,
            ),
            Guard.i64_range(
                "value",
                0 if result_type == _I1 else -(2**31),
                1 if result_type == _I1 else (2**31) - 1,
                diagnostic=range_diagnostic,
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"i32_value": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _const_i64_rule() -> DescriptorRule:
    descriptor = _descriptor("ireevm.const.i64")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64", diagnostic=_I64_ATTR_DIAGNOSTIC),
            _value_type_guard("result", _I64),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"i64_value": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _const_float_rule(result_type: TypePattern, descriptor_key: str) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    bits_immediate = "f32_bits" if result_type == _F32 else "f64_bits"
    bits_project = (
        ValueProject.f64_as_f32_bits("result")
        if result_type == _F32
        else ValueProject.f64_as_f64_bits("result")
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "f64", diagnostic=_F64_ATTR_DIAGNOSTIC),
            _value_type_guard("result", result_type),
            Guard.value_exact_f64(
                "result",
                diagnostic=_FLOAT_CONSTANT_EXACT_DIAGNOSTIC,
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={bits_immediate: bits_project},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _unary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_same_type_guards(("input", "result"), type_pattern),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"input": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_same_type_guards(("lhs", "rhs", "result"), type_pattern),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _ternary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_same_type_guards(("a", "b", "c", "result"), type_pattern),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "a": ValueRef.operand("a"),
                    "b": ValueRef.operand("b"),
                    "c": ValueRef.operand("c"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _shift_i32_rule(source_op: Op, descriptor_key: str) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_same_type_guards(("lhs", "rhs", "result"), _I32),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "input": ValueRef.operand("lhs"),
                    "amount": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _conversion_rule(
    source_op: Op,
    source_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            _value_type_guard("input", source_type),
            _value_type_guard("result", result_type),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"input": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _cmpi_rule(
    predicate: str,
    operand_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
    *,
    swap_operands: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=scalar_comparison.scalar_cmpi,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals(
                "predicate",
                predicate,
                diagnostic=_PREDICATE_DIAGNOSTIC,
            ),
            _value_type_guard("lhs", operand_type),
            _value_type_guard("rhs", operand_type),
            Guard.value_type(
                "result",
                result_type,
                diagnostic=_SUPPORTED_SCALAR_DIAGNOSTIC,
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
                swap_first_two_operands=swap_operands,
            ),
        ),
    )


def _cmpf_rule(
    predicate: str,
    operand_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
    *,
    swap_operands: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=scalar_comparison.scalar_cmpf,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals(
                "predicate",
                predicate,
                diagnostic=_PREDICATE_DIAGNOSTIC,
            ),
            _value_type_guard("lhs", operand_type),
            _value_type_guard("rhs", operand_type),
            Guard.value_type(
                "result",
                result_type,
                diagnostic=_SUPPORTED_SCALAR_DIAGNOSTIC,
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
                swap_first_two_operands=swap_operands,
            ),
        ),
    )


def _isnan_rule(
    operand_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=scalar_comparison.scalar_isnanf,
        descriptor=descriptor,
        guards=(
            _value_type_guard("input", operand_type),
            Guard.value_type(
                "result",
                result_type,
                diagnostic=_SUPPORTED_SCALAR_DIAGNOSTIC,
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"input": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


_INTEGER_BINARY_CASES = (
    (scalar_arithmetic.scalar_addi, "add"),
    (scalar_arithmetic.scalar_subi, "sub"),
    (scalar_arithmetic.scalar_muli, "mul"),
    (scalar_arithmetic.scalar_divsi, "div.s"),
    (scalar_arithmetic.scalar_divui, "div.u"),
    (scalar_arithmetic.scalar_remsi, "rem.s"),
    (scalar_arithmetic.scalar_remui, "rem.u"),
    (scalar_arithmetic.scalar_minsi, "min.s"),
    (scalar_arithmetic.scalar_minui, "min.u"),
    (scalar_arithmetic.scalar_maxsi, "max.s"),
    (scalar_arithmetic.scalar_maxui, "max.u"),
    (scalar_bitwise.scalar_andi, "and"),
    (scalar_bitwise.scalar_ori, "or"),
    (scalar_bitwise.scalar_xori, "xor"),
)

_INTEGER_UNARY_CASES = (
    (scalar_arithmetic.scalar_absi, "abs"),
    (scalar_bitwise.scalar_ctlzi, "ctlz"),
)

_FLOAT_BINARY_CASES = (
    (scalar_arithmetic.scalar_addf, "add"),
    (scalar_arithmetic.scalar_subf, "sub"),
    (scalar_arithmetic.scalar_mulf, "mul"),
    (scalar_arithmetic.scalar_divf, "div"),
    (scalar_arithmetic.scalar_remf, "rem"),
    (scalar_math.scalar_atan2f, "atan2"),
    (scalar_math.scalar_powf, "pow"),
)

_FLOAT_UNARY_CASES = (
    (scalar_arithmetic.scalar_negf, "neg"),
    (scalar_arithmetic.scalar_absf, "abs"),
    (scalar_math.scalar_ceilf, "ceil"),
    (scalar_math.scalar_floorf, "floor"),
    (scalar_math.scalar_roundf, "round"),
    (scalar_math.scalar_roundevenf, "roundeven"),
    (scalar_math.scalar_atanf, "atan"),
    (scalar_math.scalar_cosf, "cos"),
    (scalar_math.scalar_sinf, "sin"),
    (scalar_math.scalar_expf, "exp"),
    (scalar_math.scalar_exp2f, "exp2"),
    (scalar_math.scalar_expm1f, "expm1"),
    (scalar_math.scalar_logf, "log"),
    (scalar_math.scalar_log10f, "log10"),
    (scalar_math.scalar_log1pf, "log1p"),
    (scalar_math.scalar_log2f, "log2"),
    (scalar_math.scalar_rsqrtf, "rsqrt"),
    (scalar_math.scalar_sqrtf, "sqrt"),
    (scalar_math.scalar_tanhf, "tanh"),
    (scalar_math.scalar_erff, "erf"),
)

_CONVERSION_CASES = (
    (scalar_conversion.scalar_extsi, _I32, _I64, "ireevm.ext.s.i32.i64"),
    (scalar_conversion.scalar_extui, _I32, _I64, "ireevm.ext.u.i32.i64"),
    (scalar_conversion.scalar_trunci, _I64, _I32, "ireevm.trunc.i64.i32"),
    (scalar_conversion.scalar_extf, _F32, _F64, "ireevm.ext.f32.f64"),
    (scalar_conversion.scalar_fptrunc, _F64, _F32, "ireevm.trunc.f64.f32"),
    (scalar_conversion.scalar_sitofp, _I32, _F32, "ireevm.cast.s.i32.f32"),
    (scalar_conversion.scalar_uitofp, _I32, _F32, "ireevm.cast.u.i32.f32"),
    (scalar_conversion.scalar_sitofp, _I32, _F64, "ireevm.cast.s.i32.f64"),
    (scalar_conversion.scalar_uitofp, _I32, _F64, "ireevm.cast.u.i32.f64"),
    (scalar_conversion.scalar_sitofp, _I64, _F32, "ireevm.cast.s.i64.f32"),
    (scalar_conversion.scalar_uitofp, _I64, _F32, "ireevm.cast.u.i64.f32"),
    (scalar_conversion.scalar_sitofp, _I64, _F64, "ireevm.cast.s.i64.f64"),
    (scalar_conversion.scalar_uitofp, _I64, _F64, "ireevm.cast.u.i64.f64"),
    (scalar_conversion.scalar_fptosi, _F32, _I32, "ireevm.cast.s.f32.i32"),
    (scalar_conversion.scalar_fptoui, _F32, _I32, "ireevm.cast.u.f32.i32"),
    (scalar_conversion.scalar_fptosi, _F32, _I64, "ireevm.cast.s.f32.i64"),
    (scalar_conversion.scalar_fptoui, _F32, _I64, "ireevm.cast.u.f32.i64"),
    (scalar_conversion.scalar_fptosi, _F64, _I32, "ireevm.cast.s.f64.i32"),
    (scalar_conversion.scalar_fptoui, _F64, _I32, "ireevm.cast.u.f64.i32"),
    (scalar_conversion.scalar_fptosi, _F64, _I64, "ireevm.cast.s.f64.i64"),
    (scalar_conversion.scalar_fptoui, _F64, _I64, "ireevm.cast.u.f64.i64"),
    (scalar_conversion.scalar_bitcast, _I32, _F32, "ireevm.bitcast.i32.f32"),
    (scalar_conversion.scalar_bitcast, _F32, _I32, "ireevm.bitcast.f32.i32"),
    (scalar_conversion.scalar_bitcast, _I64, _F64, "ireevm.bitcast.i64.f64"),
    (scalar_conversion.scalar_bitcast, _F64, _I64, "ireevm.bitcast.f64.i64"),
)

_CMPI_CASES = (
    ("eq", "eq", False),
    ("ne", "ne", False),
    ("slt", "lt.s", False),
    ("sgt", "lt.s", True),
    ("ult", "lt.u", False),
    ("ugt", "lt.u", True),
)

_CMPF_CASES = (
    ("oeq", "eq.o", False),
    ("ueq", "eq.u", False),
    ("one", "ne.o", False),
    ("une", "ne.u", False),
    ("olt", "lt.o", False),
    ("ult", "lt.u", False),
    ("ole", "le.o", False),
    ("ule", "le.u", False),
    ("ogt", "lt.o", True),
    ("ugt", "lt.u", True),
    ("oge", "le.o", True),
    ("uge", "le.u", True),
)


def _integer_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for type_pattern, suffix in ((_I32, "i32"), (_I64, "i64")):
        for source_op, stem in _INTEGER_BINARY_CASES:
            rules.append(
                _binary_rule(source_op, type_pattern, f"ireevm.{stem}.{suffix}")
            )
        for source_op, stem in _INTEGER_UNARY_CASES:
            rules.append(
                _unary_rule(source_op, type_pattern, f"ireevm.{stem}.{suffix}")
            )
        rules.append(
            _ternary_rule(
                scalar_arithmetic.scalar_fmai, type_pattern, f"ireevm.fma.{suffix}"
            )
        )
        for result_type in (_I1, _I32):
            for predicate, stem, swap_operands in _CMPI_CASES:
                rules.append(
                    _cmpi_rule(
                        predicate,
                        type_pattern,
                        result_type,
                        f"ireevm.cmp.{stem}.{suffix}",
                        swap_operands=swap_operands,
                    )
                )
    rules.extend(
        (
            _shift_i32_rule(scalar_bitwise.scalar_shli, "ireevm.shl.i32"),
            _shift_i32_rule(scalar_bitwise.scalar_shrsi, "ireevm.shr.s.i32"),
            _shift_i32_rule(scalar_bitwise.scalar_shrui, "ireevm.shr.u.i32"),
        )
    )
    return tuple(rules)


def _float_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for type_pattern, suffix in ((_F32, "f32"), (_F64, "f64")):
        for source_op, stem in _FLOAT_BINARY_CASES:
            rules.append(
                _binary_rule(source_op, type_pattern, f"ireevm.{stem}.{suffix}")
            )
        for source_op, stem in _FLOAT_UNARY_CASES:
            rules.append(
                _unary_rule(source_op, type_pattern, f"ireevm.{stem}.{suffix}")
            )
        rules.append(
            _ternary_rule(scalar_math.scalar_fmaf, type_pattern, f"ireevm.fma.{suffix}")
        )
        for result_type in (_I1, _I32):
            rules.append(
                _isnan_rule(type_pattern, result_type, f"ireevm.cmp.nan.{suffix}")
            )
            for predicate, stem, swap_operands in _CMPF_CASES:
                rules.append(
                    _cmpf_rule(
                        predicate,
                        type_pattern,
                        result_type,
                        f"ireevm.cmp.{stem}.{suffix}",
                        swap_operands=swap_operands,
                    )
                )
    return tuple(rules)


def _conversion_rules() -> tuple[DescriptorRule, ...]:
    return tuple(
        _conversion_rule(source_op, source_type, result_type, descriptor_key)
        for source_op, source_type, result_type, descriptor_key in _CONVERSION_CASES
    )


IREEVM_CORE_CONTRACT_DIALECT_OPS = {
    "scalar": ALL_SCALAR_OPS,
}

IREEVM_CORE_CONTRACT_FRAGMENT = ContractFragment(
    name="ireevm.core",
    descriptor_set=IREEVM_CORE_DESCRIPTOR_SET,
    public_header="loom/target/arch/ireevm/contracts/core.h",
    cases=(
        *_integer_rules(),
        *_float_rules(),
        *_conversion_rules(),
        _const_i32_rule(_I32, _I32_CONSTANT_RANGE_DIAGNOSTIC),
        _const_i32_rule(_I1, _I1_CONSTANT_RANGE_DIAGNOSTIC),
        _const_i64_rule(),
        _const_float_rule(_F32, "ireevm.const.f32"),
        _const_float_rule(_F64, "ireevm.const.f64"),
    ),
)
