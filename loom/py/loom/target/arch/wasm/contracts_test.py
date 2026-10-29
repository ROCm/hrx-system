# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from loom.dialect.index import defs as index
from loom.dialect.scalar import arithmetic, bitwise, comparison, conversion, math
from loom.dialect.scf import defs as scf
from loom.dialect.vector import defs as vector
from loom.ir import ScalarType
from loom.scalar_type import ScalarTypeKind, scalar_type_name
from loom.target.arch.wasm.contracts import WASM_CORE_SIMD128_CONTRACT_FRAGMENT
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorRule,
    GuardKind,
    RecipeRule,
    UnsupportedRule,
    ValueAliasRule,
    ValueProject,
    ValueRef,
    Vector,
)

_FLOAT_KINDS = frozenset(
    {
        ScalarTypeKind.F8E4M3,
        ScalarTypeKind.F8E5M2,
        ScalarTypeKind.F16,
        ScalarTypeKind.BF16,
        ScalarTypeKind.F32,
        ScalarTypeKind.F64,
    }
)
_NON_FLOAT_KINDS = frozenset(
    {
        ScalarTypeKind.INDEX,
        ScalarTypeKind.OFFSET,
        ScalarTypeKind.I1,
        ScalarTypeKind.I8,
        ScalarTypeKind.I16,
        ScalarTypeKind.I32,
        ScalarTypeKind.I64,
    }
)


def test_uniform_shifts_cover_constant_and_runtime_counts_for_all_lane_widths():
    operations = {
        vector.vector_shli: "shl",
        vector.vector_shrsi: "shr_s",
        vector.vector_shrui: "shr_u",
    }
    actual = set()
    for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases:
        if (
            not isinstance(rule, DescriptorRule)
            or rule.source_op not in operations
            or rule.report_key != "wasm.integer_shift.uniform.native"
        ):
            continue
        type_guards = {
            guard.field: guard.type_pattern
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE and guard.value_ref is None
        }
        result_type = type_guards["result"]
        assert type_guards["lhs"] == type_guards["rhs"] == result_type
        (element,) = result_type.elements
        bit_count = int(element[1:])
        assert (result_type.minimum_lanes, result_type.maximum_lanes) == (
            1,
            128 // bit_count,
        )
        range_guard = next(
            guard for guard in rule.guards if guard.kind is GuardKind.VALUE_I64_RANGE
        )
        assert range_guard.field == "rhs"
        assert (range_guard.minimum, range_guard.maximum) == (0, bit_count - 1)
        constant = any(guard.kind is GuardKind.VALUE_EXACT_I64 for guard in rule.guards)
        if constant:
            assert rule.guards[0].kind is GuardKind.VALUE_EXACT_I64
            assert rule.priority == 1
        else:
            assert rule.guards[0].value_ref == ValueRef.uniform_element_origin_operand(
                "rhs"
            )
            assert rule.priority == 2
        actual.add((bit_count, operations[rule.source_op], constant))
        final = rule.emit[-1]
        assert (
            final.descriptor.key
            == f"wasm.i{bit_count}x{128 // bit_count}.{operations[rule.source_op]}"
        )
        assert final.operands["value"] == ValueRef.operand("lhs")
        if constant:
            assert len(rule.emit) == 2
            assert rule.emit[0].descriptor.key == "wasm.i32.const"
            assert final.operands["count"] == ValueRef.temporary("count")
        elif bit_count == 64:
            assert len(rule.emit) == 2
            assert rule.emit[0].descriptor.key == "wasm.i32.wrap_i64"
            assert rule.emit[0].operands[
                "input"
            ] == ValueRef.uniform_element_origin_operand("rhs")
        else:
            assert len(rule.emit) == 1
            assert final.operands["count"] == ValueRef.uniform_element_origin_operand(
                "rhs"
            )
    assert actual == {
        (bit_count, operation, constant)
        for bit_count in (8, 16, 32, 64)
        for operation in ("shl", "shr_s", "shr_u")
        for constant in (False, True)
    }


def _narrowing_pair(rule: DescriptorRule | ValueAliasRule) -> tuple[str, str]:
    type_guards = {
        guard.field: guard.type_pattern
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE
    }
    input_type = type_guards["input"]
    result_type = type_guards["result"]
    assert input_type is not None and input_type.kind == "scalar"
    assert result_type is not None and result_type.kind == "scalar"
    assert len(input_type.elements) == 1
    assert len(result_type.elements) == 1
    return input_type.elements[0], result_type.elements[0]


def _expected_narrowing_pairs() -> frozenset[tuple[str, str]]:
    return frozenset(
        (scalar_type_name(source), scalar_type_name(result))
        for source in _FLOAT_KINDS
        for result in _FLOAT_KINDS
        if ScalarType(source).bitwidth > ScalarType(result).bitwidth
    )


def test_scalar_float_narrowing_rules_cover_the_complete_type_matrix() -> None:
    assert _FLOAT_KINDS | _NON_FLOAT_KINDS == frozenset(ScalarTypeKind)
    assert _FLOAT_KINDS.isdisjoint(_NON_FLOAT_KINDS)

    expected_pairs = _expected_narrowing_pairs()
    actual_pairs = {
        _narrowing_pair(rule)
        for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases
        if isinstance(rule, DescriptorRule)
        and rule.source_op is conversion.scalar_fptrunc
    }

    assert len(expected_pairs) == 13
    assert actual_pairs == expected_pairs


@pytest.mark.parametrize("preserve_nan", [False, True])
def test_exact_bfloat_value_rule_separates_rounding_from_nan_handling(
    preserve_nan: bool,
) -> None:
    report_key = "exactly_representable_binary32_to_bfloat16" + (
        "_preserve_nan" if preserve_nan else ""
    )
    rule = next(
        rule
        for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases
        if isinstance(rule, DescriptorRule) and rule.report_key == report_key
    )

    assert any(
        guard.kind is GuardKind.VALUE_FLOAT_EXACTLY_REPRESENTABLE_AS
        for guard in rule.guards
    )
    assert (
        any(guard.kind is GuardKind.VALUE_NOT_NAN for guard in rule.guards)
        != preserve_nan
    )
    assert tuple(emit.descriptor.key for emit in rule.emit) == (
        "wasm.i32.reinterpret_f32",
        "wasm.i32.const",
        "wasm.i32.shr_u",
    ) + (
        (
            "wasm.i32.const",
            "wasm.i32.and",
            "wasm.i32.const",
            "wasm.i32.gt_u",
            "wasm.i32.const",
            "wasm.i32.or",
            "wasm.i32.select",
        )
        if preserve_nan
        else ()
    )
    assert rule.emit[1].immediates == {"i32_value": 16}


def test_scalar_integer_constants_cover_every_fixed_width_type() -> None:
    actual = {}
    for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases:
        if (
            not isinstance(rule, DescriptorRule)
            or rule.source_op is not conversion.scalar_constant
        ):
            continue
        result_guard = next(
            guard
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE and guard.field == "result"
        )
        result_type = result_guard.type_pattern
        if result_type.kind != "scalar" or result_type.elements[0] not in (
            "i1",
            "i8",
            "i16",
            "i32",
            "i64",
        ):
            continue
        assert len(result_type.elements) == 1
        assert len(rule.emit) == 1
        assert rule.emit[0].descriptor == rule.descriptor
        actual[result_type.elements[0]] = rule.descriptor.key

    assert actual == {
        "i1": "wasm.i32.const",
        "i8": "wasm.i32.const",
        "i16": "wasm.i32.const",
        "i32": "wasm.i32.const",
        "i64": "wasm.i64.const",
    }


def test_scalar_float_constants_preserve_every_source_format() -> None:
    expected = {
        "f8E4M3": "wasm.i32.const",
        "f8E5M2": "wasm.i32.const",
        "f16": "wasm.i32.const",
        "bf16": "wasm.i32.const",
        "f32": "wasm.f32.const",
        "f64": "wasm.f64.const",
    }
    assert set(expected) == {scalar_type_name(kind) for kind in _FLOAT_KINDS}
    actual = {}
    for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases:
        if (
            not isinstance(rule, DescriptorRule)
            or rule.source_op is not conversion.scalar_constant
        ):
            continue
        result_type = next(
            guard.type_pattern
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE and guard.field == "result"
        )
        (element,) = result_type.elements
        if element not in expected:
            continue
        assert element not in actual
        assert len(rule.emit) == 1
        immediate = "bits" if element in ("f32", "f64") else "i32_value"
        assert rule.emit[0].immediates == {immediate: ValueProject.float_bits("result")}
        actual[element] = rule.emit[0].descriptor.key
    assert actual == expected


def test_scalar_integer_truncation_covers_every_carrier_pair() -> None:
    actual = {}
    for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases:
        if rule.source_op is not conversion.scalar_trunci:
            continue
        input_type, result_type = _narrowing_pair(rule)
        if result_type == "i1":
            continue
        pair = (input_type, result_type)
        assert pair not in actual
        if input_type == "i64":
            assert isinstance(rule, DescriptorRule)
            assert len(rule.emit) == 1
            assert rule.emit[0].descriptor.key == "wasm.i32.wrap_i64"
            assert rule.emit[0].operands == {"input": ValueRef.operand("input")}
            assert rule.emit[0].results == {"dst": ValueRef.result("result")}
        else:
            assert isinstance(rule, ValueAliasRule)
            assert rule.source == ValueRef.operand("input")
            assert rule.result == ValueRef.result("result")
        actual[pair] = rule

    assert set(actual) == {
        (f"i{source_bits}", f"i{result_bits}")
        for source_bits in (16, 32, 64)
        for result_bits in (8, 16, 32)
        if result_bits < source_bits
    }


def test_scalar_integer_to_predicate_rules_preserve_the_low_bit() -> None:
    actual = {source_type: [] for source_type in ("i8", "i16", "i32", "i64")}
    for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases:
        if rule.source_op is not conversion.scalar_trunci:
            continue
        input_type, result_type = _narrowing_pair(rule)
        if result_type != "i1":
            continue
        actual[input_type].append(rule)

    for source_type, rules in actual.items():
        assert len(rules) == 2
        normalized_rule, general_rule = rules
        normalized_guard = next(
            guard
            for guard in normalized_rule.guards
            if guard.kind is GuardKind.VALUE_I64_RANGE
        )
        assert (
            normalized_guard.field,
            normalized_guard.minimum,
            normalized_guard.maximum,
        ) == (
            "input",
            0,
            1,
        )
        if source_type == "i64":
            assert isinstance(normalized_rule, DescriptorRule)
            assert tuple(emit.descriptor.key for emit in normalized_rule.emit) == (
                "wasm.i32.wrap_i64",
            )
        else:
            assert isinstance(normalized_rule, ValueAliasRule)

        assert isinstance(general_rule, DescriptorRule)
        assert tuple(emit.descriptor.key for emit in general_rule.emit) == (
            *(("wasm.i32.wrap_i64",) if source_type == "i64" else ()),
            "wasm.i32.const",
            "wasm.i32.and",
        )


def test_scalar_float_to_integer_rules_cover_every_native_carrier_pair() -> None:
    actual = {}
    for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases:
        if (
            not isinstance(rule, DescriptorRule)
            or rule.source_op
            not in (conversion.scalar_fptosi, conversion.scalar_fptoui)
            or len(rule.emit) != 1
        ):
            continue
        input_type, result_type = _narrowing_pair(rule)
        if input_type not in ("f32", "f64") or result_type not in ("i32", "i64"):
            continue
        signedness = "s" if rule.source_op is conversion.scalar_fptosi else "u"
        actual[(input_type, result_type, signedness)] = rule.descriptor.key

    assert actual == {
        (float_type, integer_type, signedness): (
            f"wasm.{integer_type}.trunc_{float_type}_{signedness}"
        )
        for float_type in ("f32", "f64")
        for integer_type in ("i32", "i64")
        for signedness in ("s", "u")
    }


def test_scalar_float_to_narrow_integer_rules_use_i32_carriers() -> None:
    actual = {}
    for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases:
        if not isinstance(rule, DescriptorRule) or rule.source_op not in (
            conversion.scalar_fptosi,
            conversion.scalar_fptoui,
        ):
            continue
        input_type, result_type = _narrowing_pair(rule)
        if input_type not in ("f32", "f64") or result_type not in (
            "i1",
            "i8",
            "i16",
        ):
            continue
        signedness = "s" if rule.source_op is conversion.scalar_fptosi else "u"
        actual[(input_type, result_type, signedness)] = tuple(
            emit.descriptor.key for emit in rule.emit
        )

    assert actual == {
        (float_type, integer_type, signedness): (
            (
                f"wasm.i32.trunc_{float_type}_{signedness}",
                "wasm.i32.const",
                "wasm.i32.and",
            )
            if integer_type == "i1"
            else (f"wasm.i32.trunc_{float_type}_{signedness}",)
        )
        for float_type in ("f32", "f64")
        for integer_type in ("i1", "i8", "i16")
        for signedness in ("s", "u")
    }


def test_exact_recipes_cover_every_non_native_pair() -> None:
    native_pairs = frozenset({("f64", "f32")})
    exact_recipe_pairs = {
        _narrowing_pair(rule)
        for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases
        if isinstance(rule, DescriptorRule)
        and rule.source_op is conversion.scalar_fptrunc
        and rule.report_key.startswith("exact_")
    }

    assert exact_recipe_pairs == _expected_narrowing_pairs() - native_pairs


def test_bit_operations_use_native_instructions_for_each_integer_carrier() -> None:
    for scalar_op, index_op, instruction in (
        (bitwise.scalar_rotli, index.index_rotli, "rotl"),
        (bitwise.scalar_rotri, index.index_rotri, "rotr"),
        (bitwise.scalar_ctlzi, index.index_ctlzi, "clz"),
        (bitwise.scalar_cttzi, index.index_cttzi, "ctz"),
        (bitwise.scalar_ctpopi, index.index_ctpopi, "popcnt"),
    ):
        actual = {}
        for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases:
            if not isinstance(rule, DescriptorRule):
                continue
            if rule.source_op not in (scalar_op, index_op):
                continue
            result_guard = next(
                guard
                for guard in rule.guards
                if guard.kind is GuardKind.VALUE_TYPE and guard.field == "result"
            )
            assert len(rule.emit) == 1
            actual[(rule.source_op, result_guard.type_pattern.elements[0])] = rule.emit[
                0
            ].descriptor.key
        assert actual == {
            (scalar_op, "i32"): f"wasm.i32.{instruction}",
            (scalar_op, "i64"): f"wasm.i64.{instruction}",
            (index_op, "index"): f"wasm.i32.{instruction}",
        }


def _numeric_vector_shapes(rule, field):
    for guard in rule.guards:
        if guard.kind is not GuardKind.VALUE_TYPE or guard.field != field:
            continue
        pattern = guard.type_pattern
        if pattern.kind == "vector" and pattern.lanes is not None:
            yield from (
                (element, pattern.lanes)
                for element in pattern.elements
                if element != "i1"
            )


def _numeric_vector_ranges(rule, field):
    for guard in rule.guards:
        if guard.kind is not GuardKind.VALUE_TYPE or guard.field != field:
            continue
        pattern = guard.type_pattern
        if (
            pattern.kind == "vector"
            and isinstance(pattern.minimum_lanes, int)
            and isinstance(pattern.maximum_lanes, int)
        ):
            yield from (
                (element, pattern.minimum_lanes, pattern.maximum_lanes)
                for element in pattern.elements
                if element != "i1"
            )


def _type_guard(rule: DescriptorRule, field: str):
    return next(
        guard.type_pattern
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE and guard.field == field
    )


def test_native_scalar_float_contracts_cover_both_widths() -> None:
    operations = {
        arithmetic.scalar_addf: "add",
        arithmetic.scalar_subf: "sub",
        arithmetic.scalar_mulf: "mul",
        arithmetic.scalar_divf: "div",
        arithmetic.scalar_minimumf: "min",
        arithmetic.scalar_maximumf: "max",
        arithmetic.scalar_copysignf: "copysign",
        arithmetic.scalar_absf: "abs",
        arithmetic.scalar_negf: "neg",
        math.scalar_ceilf: "ceil",
        math.scalar_floorf: "floor",
        math.scalar_truncf: "trunc",
        math.scalar_roundevenf: "nearest",
        math.scalar_sqrtf: "sqrt",
    }
    actual = {
        (rule.source_op, _type_guard(rule, "result").elements[0]): rule.descriptor.key
        for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases
        if isinstance(rule, DescriptorRule)
        and rule.source_op in operations
        and _type_guard(rule, "result").elements[0] in ("f32", "f64")
    }
    assert actual == {
        (source_op, type_name): f"wasm.{type_name}.{operation}"
        for source_op, operation in operations.items()
        for type_name in ("f32", "f64")
    }

    actual_comparisons = {}
    for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases:
        if (
            not isinstance(rule, DescriptorRule)
            or rule.source_op is not comparison.scalar_cmpf
        ):
            continue
        operand_type = _type_guard(rule, "lhs")
        if operand_type.elements[0] not in ("f32", "f64"):
            continue
        predicate = next(
            guard.enum_keyword
            for guard in rule.guards
            if guard.kind is GuardKind.ENUM_ATTR_EQUALS and guard.field == "predicate"
        )
        no_nan = any(
            guard.kind is GuardKind.INSTANCE_FLAGS_HAS_ALL and guard.field == "fastmath"
            for guard in rule.guards
        )
        key = (operand_type.elements[0], predicate, no_nan)
        assert key not in actual_comparisons
        actual_comparisons[key] = tuple(emit.descriptor.key for emit in rule.emit)
        mechanism = "native" if len(rule.emit) == 1 else "composed"
        assert rule.report_key == (
            f"wasm.scalar_float_compare.{mechanism}" + (".nnan" if no_nan else "")
        )
        assert rule.priority == (1 if no_nan else 0)

    expected_comparisons = {}
    for type_name in ("f32", "f64"):
        native_prefix = f"wasm.{type_name}."

        strict = {
            "oeq": (native_prefix + "eq",),
            "ogt": (native_prefix + "gt",),
            "oge": (native_prefix + "ge",),
            "olt": (native_prefix + "lt",),
            "ole": (native_prefix + "le",),
            "one": (native_prefix + "lt", native_prefix + "gt", "wasm.i32.or"),
            "ord": (native_prefix + "eq", native_prefix + "eq", "wasm.i32.and"),
            "ueq": (
                native_prefix + "lt",
                native_prefix + "gt",
                "wasm.i32.or",
                "wasm.i32.eqz",
            ),
            "ugt": (native_prefix + "le", "wasm.i32.eqz"),
            "uge": (native_prefix + "lt", "wasm.i32.eqz"),
            "ult": (native_prefix + "ge", "wasm.i32.eqz"),
            "ule": (native_prefix + "gt", "wasm.i32.eqz"),
            "une": (native_prefix + "ne",),
            "uno": (native_prefix + "ne", native_prefix + "ne", "wasm.i32.or"),
        }
        for predicate, descriptors in strict.items():
            expected_comparisons[(type_name, predicate, False)] = descriptors
        for predicate, operation in {
            "one": "ne",
            "ueq": "eq",
            "ugt": "gt",
            "uge": "ge",
            "ult": "lt",
            "ule": "le",
        }.items():
            expected_comparisons[(type_name, predicate, True)] = (
                native_prefix + operation,
            )
    assert actual_comparisons == expected_comparisons


def test_native_vector_float_contracts_cover_every_carrier_lane() -> None:
    operations = {
        vector.vector_addf: "add",
        vector.vector_subf: "sub",
        vector.vector_mulf: "mul",
        vector.vector_divf: "div",
        vector.vector_minimumf: "min",
        vector.vector_maximumf: "max",
        vector.vector_absf: "abs",
        vector.vector_negf: "neg",
        vector.vector_ceilf: "ceil",
        vector.vector_floorf: "floor",
        vector.vector_truncf: "trunc",
        vector.vector_roundevenf: "nearest",
        vector.vector_sqrtf: "sqrt",
    }
    actual = {
        (rule.source_op, element, minimum, maximum): (
            rule.descriptor.key,
            rule.emit[0].form,
        )
        for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases
        if isinstance(rule, DescriptorRule) and rule.source_op in operations
        for element, minimum, maximum in _numeric_vector_ranges(rule, "result")
        if element in ("f32", "f64")
    }
    assert actual == {
        (source_op, element, 1, lane_count): (
            f"wasm.{shape}.{operation}",
            DescriptorEmitForm.OP,
        )
        for source_op, operation in operations.items()
        for element, lane_count, shape in (
            ("f32", 4, "f32x4"),
            ("f64", 2, "f64x2"),
        )
    }


def _integer_vector_ranges(rule, field):
    yield from (
        shape
        for shape in _numeric_vector_ranges(rule, field)
        if shape[0] in ("i8", "i16", "i32", "i64")
    )


def _expected_numeric_vector_shapes():
    return {
        (scalar_type_name(kind), 128 // ScalarType(kind).bitwidth)
        for kind in ScalarTypeKind
        if kind not in (ScalarTypeKind.I1, ScalarTypeKind.INDEX, ScalarTypeKind.OFFSET)
    }


def _expected_partial_numeric_vector_ranges():
    return {
        (element, 1, maximum - 1)
        for element, maximum in _expected_numeric_vector_shapes()
    }


def test_numeric_constants_cover_every_native_and_partial_carrier() -> None:
    actual = set()
    for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases:
        if not isinstance(rule, DescriptorRule) or rule.source_op not in (
            vector.vector_splati,
            vector.vector_splatf,
        ):
            continue
        actual.update(_numeric_vector_shapes(rule, "result"))
        for element, minimum, maximum in _numeric_vector_ranges(rule, "result"):
            actual.update((element, count) for count in range(minimum, maximum + 1))
    assert actual == {
        (element, count)
        for element, maximum in _expected_numeric_vector_shapes()
        for count in range(1, maximum + 1)
    }


def test_full_numeric_carriers_have_complete_structural_rules() -> None:
    expected = _expected_numeric_vector_shapes()
    for source_ops, field in (
        ((vector.vector_splati, vector.vector_splatf), "result"),
        ((vector.vector_splat,), "result"),
        ((vector.vector_extract,), "source"),
        ((vector.vector_insert,), "dest"),
        ((vector.vector_shuffle,), "result"),
    ):
        actual = {
            shape
            for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases
            if isinstance(rule, DescriptorRule) and rule.source_op in source_ops
            for shape in _numeric_vector_shapes(rule, field)
        }
        assert actual == expected, tuple(source_op.name for source_op in source_ops)


def test_partial_numeric_carriers_have_complete_direct_structural_rules() -> None:
    expected = _expected_partial_numeric_vector_ranges()
    for source_ops, field, rules_per_shape in (
        ((vector.vector_splati, vector.vector_splatf), "result", 1),
        ((vector.vector_splat,), "result", 1),
        ((vector.vector_extract,), "source", 1),
        ((vector.vector_insert,), "dest", 2),
    ):
        counts = {}
        for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases:
            if not isinstance(rule, DescriptorRule) or rule.source_op not in source_ops:
                continue
            for shape in _numeric_vector_ranges(rule, field):
                counts[shape] = counts.get(shape, 0) + 1
        assert counts == {shape: rules_per_shape for shape in expected}, tuple(
            source_op.name for source_op in source_ops
        )


def test_partial_integer_carriers_have_complete_bitwise_rules() -> None:
    expected = {
        ("i8", 1, 15),
        ("i16", 1, 7),
        ("i32", 1, 3),
        ("i64", 1, 1),
    }
    for source_op in (
        vector.vector_andi,
        vector.vector_ori,
        vector.vector_xori,
    ):
        actual = {
            shape
            for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases
            if isinstance(rule, DescriptorRule) and rule.source_op is source_op
            for shape in _integer_vector_ranges(rule, "result")
        }
        assert actual == expected, source_op.name


def test_numeric_memory_shapes_share_exact_v128_accesses() -> None:
    for source_op, field, descriptor_key in (
        (vector.vector_load, "result", "wasm.v128.load"),
        (vector.vector_store, "value", "wasm.v128.store"),
    ):
        counts = {}
        for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases:
            if not isinstance(rule, DescriptorRule) or rule.source_op is not source_op:
                continue
            assert rule.descriptor.key == descriptor_key
            memory = rule.emit[-1].source_memory
            assert memory.element_byte_count * memory.vector_lane_count == 16
            assert memory.vector_lane_byte_stride == memory.element_byte_count
            for shape in _numeric_vector_shapes(rule, field):
                counts[shape] = counts.get(shape, 0) + 1
        # Static, dynamic with immediate bias, and dynamic with register bias.
        assert counts == {shape: 3 for shape in _expected_numeric_vector_shapes()}


def test_static_and_dynamic_insert_cover_every_native_numeric_shape() -> None:
    shapes_by_index_count = {0: {}, 1: {}}
    for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases:
        if (
            not isinstance(rule, DescriptorRule)
            or rule.source_op is not vector.vector_insert
        ):
            continue
        index_count = next(
            guard.count
            for guard in rule.guards
            if guard.kind is GuardKind.OPERAND_SEGMENT_COUNT
            and guard.field == "indices"
        )
        shapes = shapes_by_index_count[index_count]
        for shape in _numeric_vector_shapes(rule, "result"):
            assert shape not in shapes
            shapes[shape] = rule
            assert len(rule.emit) == (1 if index_count == 0 else 5)
            if index_count == 0:
                assert rule.descriptor.key.endswith(".replace_lane")
    for shapes in shapes_by_index_count.values():
        assert shapes.keys() == _expected_numeric_vector_shapes()


def test_dynamic_insert_masks_select_exactly_one_complete_physical_lane() -> None:
    lane_counts = set()
    for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases:
        if (
            not isinstance(rule, DescriptorRule)
            or rule.source_op is not vector.vector_insert
        ):
            continue
        if not any(
            guard.kind is GuardKind.OPERAND_SEGMENT_COUNT and guard.count == 1
            for guard in rule.guards
        ):
            continue
        result_type = next(
            guard.type_pattern
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE and guard.field == "result"
        )
        if result_type.elements == ("i1",):
            # Predicate representations have their own exhaustive family test.
            continue
        if result_type.lanes is not None:
            physical_lane_count = result_type.lanes
        else:
            assert result_type.maximum_lanes is not None
            physical_lane_count = result_type.maximum_lanes + 1
        lane_counts.add(physical_lane_count)
        constant, index, compare, splat, select = rule.emit[-5:]
        assert constant.descriptor.key == "wasm.v128.const"
        assert index.descriptor.key == "wasm.i8x16.splat"
        assert compare.descriptor.key == "wasm.i8x16.eq"
        assert splat.descriptor.key.endswith(".splat")
        assert select.descriptor.key == "wasm.v128.bitselect"
        ordinals = constant.immediates["lo64"].to_bytes(8, "little")
        ordinals += constant.immediates["hi64"].to_bytes(8, "little")
        lane_bits = 128 // physical_lane_count
        for lane in range(physical_lane_count):
            selected_bytes = bytes(
                255 if ordinal == lane else 0 for ordinal in ordinals
            )
            assert int.from_bytes(selected_bytes, "little") == (
                ((1 << lane_bits) - 1) << (lane * lane_bits)
            )
    assert lane_counts == {2, 4, 8, 16}


def test_structural_planner_exclusively_owns_bitcast_and_numeric_select() -> None:
    assert not any(
        isinstance(rule, ValueAliasRule) and rule.source_op is vector.vector_bitcast
        for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases
    )
    for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases:
        if not isinstance(rule, DescriptorRule) or rule.source_op is not scf.scf_select:
            continue
        result_type = next(
            guard.type_pattern
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE and guard.field == "result"
        )
        assert result_type.kind != "vector" or result_type.elements == ("i1",)


def test_reference_fallback_families_explicitly_reject_unmatched_native_cases() -> None:
    assert {
        rule.source_op
        for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases
        if isinstance(rule, UnsupportedRule)
    } == set(vector.VECTOR_CAST_OPS) | {
        arithmetic.scalar_minnumf,
        arithmetic.scalar_maxnumf,
        arithmetic.scalar_clampf,
    }


def test_float_extrema_recipes_cover_every_source_vector_carrier() -> None:
    actual = {
        (rule.source_op, guard.type_pattern)
        for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases
        if isinstance(rule, RecipeRule)
        and rule.source_op
        in {
            vector.vector_minimumf,
            vector.vector_maximumf,
            vector.vector_minnumf,
            vector.vector_maxnumf,
            vector.vector_clampf,
        }
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE and guard.field == "result"
    }
    narrow_types = {
        Vector(("f8E4M3", "f8E5M2"), minimum_lanes=1, maximum_lanes=16),
        Vector(("f16", "bf16"), minimum_lanes=1, maximum_lanes=8),
    }
    all_types = narrow_types | {
        Vector("f32", minimum_lanes=1, maximum_lanes=4),
        Vector("f64", minimum_lanes=1, maximum_lanes=2),
    }
    assert actual == {
        *(
            (source_op, value_type)
            for source_op in (vector.vector_minimumf, vector.vector_maximumf)
            for value_type in narrow_types
        ),
        *(
            (source_op, value_type)
            for source_op in (
                vector.vector_minnumf,
                vector.vector_maxnumf,
                vector.vector_clampf,
            )
            for value_type in all_types
        ),
    }


def test_narrow_scalar_selection_uses_integer_payloads() -> None:
    actual = {
        element: rule.descriptor.key
        for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases
        if isinstance(rule, DescriptorRule) and rule.source_op is scf.scf_select
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE
        and guard.field == "result"
        and guard.type_pattern.kind == "scalar"
        for element in guard.type_pattern.elements
        if element in ("i8", "f8E4M3", "f8E5M2", "i16", "f16", "bf16")
    }
    assert actual == {
        element: "wasm.i32.select"
        for element in ("i8", "f8E4M3", "f8E5M2", "i16", "f16", "bf16")
    }


def test_integer_bitwise_operations_cover_every_full_width_lane_shape() -> None:
    for source_op, instruction in (
        (vector.vector_andi, "and"),
        (vector.vector_ori, "or"),
        (vector.vector_xori, "xor"),
    ):
        actual = {
            shape: rule.descriptor.key
            for rule in WASM_CORE_SIMD128_CONTRACT_FRAGMENT.cases
            if isinstance(rule, DescriptorRule) and rule.source_op is source_op
            for shape in _numeric_vector_shapes(rule, "result")
        }
        assert actual == {
            (f"i{bits}", 128 // bits): f"wasm.v128.{instruction}"
            for bits in (8, 16, 32, 64)
        }
