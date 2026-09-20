# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source operation correspondence for spec-derived VM instructions."""

import struct

from iree.vm.bytecode.spec.isa.core.buffer import (
    BUFFER_ALLOCATE,
    BUFFER_ATOMIC_CARRIER_SELECTOR,
    BUFFER_ATOMIC_CMPXCHG,
    BUFFER_ATOMIC_KIND_SELECTOR,
    BUFFER_ATOMIC_ORDERING_SELECTOR,
    BUFFER_ATOMIC_REDUCE,
    BUFFER_ATOMIC_RMW,
    BUFFER_ATOMIC_SCOPE_SELECTOR,
    BUFFER_COMPARE,
    BUFFER_COPY,
    BUFFER_FILL,
    BUFFER_LENGTH,
    BUFFER_LOAD,
    BUFFER_STORE,
)
from iree.vm.bytecode.spec.isa.core.constant import CONSTANT_I64
from iree.vm.bytecode.spec.isa.core.float import (
    FLOAT_CLAMP_SELECTOR,
    FLOAT_COMPARE_SELECTOR,
    FloatBinaryOperation,
    FloatBinarySemantics,
    FloatClampSemantics,
    FloatClassifySemantics,
    FloatCompareSemantics,
    FloatFmaSemantics,
    FloatMathSemantics,
    FloatMinmaxSemantics,
    FloatUnaryOperation,
    FloatUnarySemantics,
)
from iree.vm.bytecode.spec.isa.core.integer import (
    INTEGER_COMPARE_SELECTOR,
    IntegerBinaryOperation,
    IntegerBinarySemantics,
    IntegerCompareSemantics,
    IntegerDivisionOperation,
    IntegerDivisionSemantics,
    IntegerUnaryOperation,
    IntegerUnarySemantics,
)
from iree.vm.bytecode.spec.isa.core.rules import RecordRuleKind
from iree.vm.bytecode.spec.isa.core.stack import MEMORY_FORMAT_SELECTOR
from iree.vm.bytecode.spec.isa.core.value import VALUE_COPY, VALUE_SELECT
from iree.vm.bytecode.spec.specification import SPECIFICATION

from loom.dialect import buffer, view
from loom.dialect.atomic import AtomicKind, AtomicOrdering, AtomicScope
from loom.dialect.globals import ALL_GLOBAL_OPS
from loom.dialect.globals.defs import global_load
from loom.dialect.index import ALL_INDEX_OPS, IndexPredicate
from loom.dialect.index import defs as index
from loom.dialect.scalar import (
    ALL_SCALAR_OPS,
    ClampFMode,
    arithmetic,
    bitwise,
    comparison,
    conversion,
    math,
)
from loom.dialect.scf import ALL_SCF_OPS, scf_select
from loom.ir import ScalarType, ScalarTypeKind
from loom.target.arch.vm.descriptors import VM_CORE_DESCRIPTOR_SET, scalar_result_type
from loom.target.contracts import (
    AttrProject,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    DirectDescriptorCase,
    EmitDescriptorOp,
    Guard,
    RecipeRule,
    Scalar,
    SelectDescriptorCase,
    SourceMemoryByteOffsetMaterializer,
    SourceMemoryConstraint,
    SourceMemoryIntegerConversion,
    SourceMemoryOperation,
    SourceMemoryProject,
    ValueAliasRule,
    ValueProject,
    ValueRef,
    binary_descriptor_rules,
    select_descriptor_rules,
    ternary_descriptor_rules,
    unary_descriptor_rules,
)
from loom.target.contracts.memory_spaces import MEMORY_SPACE_NAMES
from loom.target.low_descriptors import DescriptorOpKind, OperandRole

_BINARY_SOURCE_OPS = {
    IntegerBinaryOperation.ADD: arithmetic.scalar_addi,
    IntegerBinaryOperation.SUB: arithmetic.scalar_subi,
    IntegerBinaryOperation.MUL: arithmetic.scalar_muli,
    IntegerBinaryOperation.MIN_SIGNED: arithmetic.scalar_minsi,
    IntegerBinaryOperation.MIN_UNSIGNED: arithmetic.scalar_minui,
    IntegerBinaryOperation.MAX_SIGNED: arithmetic.scalar_maxsi,
    IntegerBinaryOperation.MAX_UNSIGNED: arithmetic.scalar_maxui,
    IntegerBinaryOperation.AND: bitwise.scalar_andi,
    IntegerBinaryOperation.OR: bitwise.scalar_ori,
    IntegerBinaryOperation.XOR: bitwise.scalar_xori,
    IntegerBinaryOperation.SHIFT_LEFT: bitwise.scalar_shli,
    IntegerBinaryOperation.SHIFT_RIGHT_SIGNED: bitwise.scalar_shrsi,
    IntegerBinaryOperation.SHIFT_RIGHT_UNSIGNED: bitwise.scalar_shrui,
    IntegerBinaryOperation.ROTATE_LEFT: bitwise.scalar_rotli,
    IntegerBinaryOperation.ROTATE_RIGHT: bitwise.scalar_rotri,
}

_UNARY_SOURCE_OPS = {
    IntegerUnaryOperation.NEGATE: arithmetic.scalar_negi,
    IntegerUnaryOperation.ABSOLUTE: arithmetic.scalar_absi,
    IntegerUnaryOperation.COUNT_LEADING_ZEROS: bitwise.scalar_ctlzi,
    IntegerUnaryOperation.COUNT_TRAILING_ZEROS: bitwise.scalar_cttzi,
    IntegerUnaryOperation.POPULATION_COUNT: bitwise.scalar_ctpopi,
}

_INDEX_SOURCE_OPS = {
    IntegerBinaryOperation.ADD: index.index_add,
    IntegerBinaryOperation.SUB: index.index_sub,
    IntegerBinaryOperation.MUL: index.index_mul,
    IntegerBinaryOperation.MIN_SIGNED: index.index_min,
    IntegerBinaryOperation.MAX_SIGNED: index.index_max,
    IntegerBinaryOperation.AND: index.index_andi,
    IntegerBinaryOperation.OR: index.index_ori,
    IntegerBinaryOperation.XOR: index.index_xori,
    IntegerBinaryOperation.SHIFT_LEFT: index.index_shli,
    IntegerBinaryOperation.SHIFT_RIGHT_SIGNED: index.index_shrsi,
    IntegerBinaryOperation.SHIFT_RIGHT_UNSIGNED: index.index_shrui,
    IntegerBinaryOperation.ROTATE_LEFT: index.index_rotli,
    IntegerBinaryOperation.ROTATE_RIGHT: index.index_rotri,
    IntegerUnaryOperation.COUNT_LEADING_ZEROS: index.index_ctlzi,
    IntegerUnaryOperation.COUNT_TRAILING_ZEROS: index.index_cttzi,
    IntegerUnaryOperation.POPULATION_COUNT: index.index_ctpopi,
    IntegerDivisionOperation.UNSIGNED_QUOTIENT: index.index_div,
    IntegerDivisionOperation.UNSIGNED_REMAINDER: index.index_rem,
}

_DIVISION_SOURCE_OPS = {
    IntegerDivisionOperation.SIGNED_QUOTIENT: arithmetic.scalar_divsi,
    IntegerDivisionOperation.UNSIGNED_QUOTIENT: arithmetic.scalar_divui,
    IntegerDivisionOperation.SIGNED_REMAINDER: arithmetic.scalar_remsi,
    IntegerDivisionOperation.UNSIGNED_REMAINDER: arithmetic.scalar_remui,
}

_FLOAT_BINARY_SOURCE_OPS = {
    FloatBinaryOperation.ADD: arithmetic.scalar_addf,
    FloatBinaryOperation.SUB: arithmetic.scalar_subf,
    FloatBinaryOperation.MUL: arithmetic.scalar_mulf,
    FloatBinaryOperation.DIV: arithmetic.scalar_divf,
    FloatBinaryOperation.REM: arithmetic.scalar_remf,
    FloatBinaryOperation.COPY_SIGN: arithmetic.scalar_copysignf,
}

_FLOAT_UNARY_SOURCE_OPS = {
    FloatUnaryOperation.NEGATE: arithmetic.scalar_negf,
    FloatUnaryOperation.ABSOLUTE: arithmetic.scalar_absf,
}

_SELECTED_SOURCE_OPS = {
    FloatMinmaxSemantics: {
        "minimum": arithmetic.scalar_minimumf,
        "maximum": arithmetic.scalar_maximumf,
        "minnum": arithmetic.scalar_minnumf,
        "maxnum": arithmetic.scalar_maxnumf,
    },
    FloatClassifySemantics: {
        "isnan": comparison.scalar_isnanf,
        "isinf": comparison.scalar_isinff,
        "isfinite": comparison.scalar_isfinitef,
    },
}
_MATH_SOURCE_OPS = {
    "ceil": math.scalar_ceilf,
    "floor": math.scalar_floorf,
    "round_even": math.scalar_roundevenf,
    "trunc": math.scalar_truncf,
    "sign": comparison.scalar_signf,
    # Despite the ISA spelling, these are frozen correctly rounded f32
    # mappings shared with source constant folding, including subnormals.
    "sin_turns.approx": math.scalar_sinturnsf,
    "cos_turns.approx": math.scalar_costurnsf,
}
_ATTRIBUTE_SOURCE_OPS = {
    IntegerCompareSemantics: (comparison.scalar_cmpi, "i", "predicate"),
    FloatCompareSemantics: (comparison.scalar_cmpf, "f", "predicate"),
    FloatClampSemantics: (arithmetic.scalar_clampf, "f", "mode"),
}

# Constants carry bits; the Low result retains the source interpretation.
_CONSTANT_SOURCES = {
    32: {
        "i1": ValueProject.exact_i64,
        "i8": ValueProject.i32_as_u32_bits,
        "i16": ValueProject.i32_as_u32_bits,
        "i32": ValueProject.i32_as_u32_bits,
        "f8E4M3": ValueProject.float_bits,
        "f8E5M2": ValueProject.float_bits,
        "f16": ValueProject.float_bits,
        "bf16": ValueProject.float_bits,
        "f32": ValueProject.float_bits,
    },
    64: {
        "i64": ValueProject.exact_i64,
        "f64": ValueProject.float_bits,
        "index": ValueProject.exact_i64,
        "offset": ValueProject.exact_i64,
    },
}
_SCALAR_TYPES = tuple(name for types in _CONSTANT_SOURCES.values() for name in types)
_SOURCE_SCALAR_NAMES = {name.lower(): name for name in _SCALAR_TYPES}

# Selectors carry their source/destination types in the canonical ISA spelling.
# Only the correspondence with Loom operations belongs in this projection.
_CONVERSION_SOURCE_OPS = {
    "integer.convert": {
        "s": conversion.scalar_extsi,
        "u": conversion.scalar_extui,
        "i": conversion.scalar_trunci,
    },
    "float.width": {"f32": conversion.scalar_extf, "f64": conversion.scalar_fptrunc},
    "float.extend": {"f": conversion.scalar_extf, "b": conversion.scalar_extf},
    "float.truncate": {"f": conversion.scalar_fptrunc},
    "integer.to.float": {"s": conversion.scalar_sitofp, "u": conversion.scalar_uitofp},
    "float.to.integer": {"s": conversion.scalar_fptosi, "u": conversion.scalar_fptoui},
}

_INSTRUCTIONS = {
    instruction.opcode: instruction for instruction in SPECIFICATION.instructions
}
_DESCRIPTORS = {
    descriptor.encoding_id: descriptor
    for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors
}

# A direct ordinal projection is valid only while both public enums agree.
for source_enum, selector in (
    (comparison.CmpIPredicate, INTEGER_COMPARE_SELECTOR),
    (IndexPredicate, INTEGER_COMPARE_SELECTOR),
    (comparison.CmpFPredicate, FLOAT_COMPARE_SELECTOR),
    (ClampFMode, FLOAT_CLAMP_SELECTOR),
    (AtomicOrdering, BUFFER_ATOMIC_ORDERING_SELECTOR),
    (AtomicScope, BUFFER_ATOMIC_SCOPE_SELECTOR),
):
    assert {case.keyword: case.value for case in source_enum.cases} == {
        value.name: value.value for value in selector.values
    }

# Source and machine spellings differ; only the correspondence lives here.
_ATOMIC_KIND_NAMES = {
    "xchgi": "exchange.integer",
    "xchgf": "exchange.float",
    "addi": "add.integer",
    "addf": "add.float",
    "subi": "subtract.integer",
    "andi": "and.integer",
    "ori": "or.integer",
    "xori": "xor.integer",
    "minsi": "minimum.signed",
    "maxsi": "maximum.signed",
    "minui": "minimum.unsigned",
    "maxui": "maximum.unsigned",
    "minimumf": "minimum.float",
    "maximumf": "maximum.float",
    "minnumf": "minnum.float",
    "maxnumf": "maxnum.float",
}
assert {_ATOMIC_KIND_NAMES[case.keyword]: case.value for case in AtomicKind.cases} == {
    value.name: value.value for value in BUFFER_ATOMIC_KIND_SELECTOR.values
}
_ATOMIC_CARRIERS = {
    int(value.name[1:]): value.value for value in BUFFER_ATOMIC_CARRIER_SELECTOR.values
}
_ATOMIC_ORDERING_PAIR = next(
    rule.data
    for rule in BUFFER_ATOMIC_CMPXCHG.rules
    if rule.kind is RecordRuleKind.PACKED_SELECTOR_PAIRS
)
# The shared pack recipe consumes consecutive source attributes.
_CMPXCHG_ATTR_NAMES = tuple(attr.name for attr in view.view_atomic_cmpxchg.attrs)
_CMPXCHG_ORDERING_START = _CMPXCHG_ATTR_NAMES.index(_ATOMIC_ORDERING_PAIR[0].name)
assert _CMPXCHG_ATTR_NAMES[
    _CMPXCHG_ORDERING_START : _CMPXCHG_ORDERING_START + len(_ATOMIC_ORDERING_PAIR)
] == tuple(part.name for part in _ATOMIC_ORDERING_PAIR)

VM_CORE_CONTRACT_DIALECT_OPS = {
    "buffer": buffer.ALL_BUFFER_OPS,
    "global": ALL_GLOBAL_OPS,
    "scalar": ALL_SCALAR_OPS,
    "index": ALL_INDEX_OPS,
    "scf": ALL_SCF_OPS,
    "view": view.ALL_VIEW_OPS,
}


def _direct_cases(semantics_type, source_ops, type_prefix="i"):
    for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors:
        semantics = _INSTRUCTIONS[descriptor.encoding_id].semantics
        if isinstance(semantics, semantics_type):
            source_type = Scalar(f"{type_prefix}{semantics.bit_width}")
            if semantics.bit_width == 32 and semantics.operation in (
                IntegerBinaryOperation.AND,
                IntegerBinaryOperation.OR,
                IntegerBinaryOperation.XOR,
            ):
                source_type = Scalar(("i1", "i32"))
            yield DirectDescriptorCase(
                source_ops[semantics.operation],
                descriptor,
                source_type,
            )


def _constant_cases():
    for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors:
        if descriptor.op_kind is not DescriptorOpKind.CONST:
            continue
        bit_width = descriptor.immediates[0].bit_width
        for source_type, projection in _CONSTANT_SOURCES[bit_width].items():
            yield DescriptorRule(
                source_op=(
                    index.index_constant
                    if source_type in ("index", "offset")
                    else conversion.scalar_constant
                ),
                descriptor=descriptor,
                guards=(Guard.value_type("result", Scalar(source_type)),),
                emit=(
                    EmitDescriptorOp(
                        descriptor=descriptor,
                        results={"destination_v8": ValueRef.result("result")},
                        immediates={"bits": projection("result")},
                        form=DescriptorEmitForm.CONST,
                    ),
                ),
            )


def _scalar_rule(
    descriptor, source_op, source_type, selector=None, *, result_type=None
):
    result_type = result_type or (
        Scalar("i1")
        if isinstance(
            _INSTRUCTIONS[descriptor.encoding_id].semantics,
            (IntegerCompareSemantics, FloatCompareSemantics, FloatClassifySemantics),
        )
        else source_type
    )
    operands = {
        target.field_name: ValueRef.operand(source.name)
        for target, source in zip(
            (
                value
                for value in descriptor.operands
                if value.role is OperandRole.OPERAND
            ),
            source_op.operands,
            strict=True,
        )
    }
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *(
                Guard.value_type(operand.name, source_type)
                for operand in source_op.operands
            ),
            Guard.value_type("result", result_type),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands=operands,
                results={"destination_v8": ValueRef.result("result")},
                immediates=(
                    {descriptor.immediates[0].field_name: selector}
                    if selector is not None
                    else {}
                ),
            ),
        ),
    )


def _selected_cases():
    for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors:
        semantics = _INSTRUCTIONS[descriptor.encoding_id].semantics
        if source := _ATTRIBUTE_SOURCE_OPS.get(type(semantics)):
            source_op, type_prefix, attribute = source
            yield _scalar_rule(
                descriptor,
                source_op,
                Scalar(f"{type_prefix}{semantics.bit_width}"),
                AttrProject.enum_ordinal(attribute),
            )
        elif source_ops := _SELECTED_SOURCE_OPS.get(type(semantics)):
            (selector,) = (
                field.rule.data
                for field in _INSTRUCTIONS[descriptor.encoding_id].fields
                if field.rule.data is not None
            )
            assert set(source_ops) == {value.name for value in selector.values}
            for value in selector.values:
                yield _scalar_rule(
                    descriptor,
                    source_ops[value.name],
                    Scalar(f"f{semantics.bit_width}"),
                    value.value,
                )


def _conversion_steps():
    for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors:
        if not descriptor.immediates:
            continue
        domain = descriptor.immediates[0].enum_domain
        if domain not in _CONVERSION_SOURCE_OPS:
            continue
        selector = _INSTRUCTIONS[descriptor.encoding_id].fields[-1].rule.data
        for value in selector.values:
            source, destination = value.name.split(".to.")
            source_type = "i" + source[1:] if source[0] in "su" else source
            result_type = (
                "i" + destination[1:] if destination[0] in "su" else destination
            )
            # Loom's FP8 names use capital E/M; ISA selectors use lowercase.
            source_type = _SOURCE_SCALAR_NAMES.get(source_type, source_type)
            result_type = _SOURCE_SCALAR_NAMES.get(result_type, result_type)
            if source_type not in _SCALAR_TYPES or result_type not in _SCALAR_TYPES:
                continue
            key = (
                source
                if domain == "float.width"
                else destination[0]
                if domain == "float.to.integer"
                else source[0]
            )
            yield (
                (_CONVERSION_SOURCE_OPS[domain][key], source_type, result_type),
                (descriptor, value.value),
            )


def _conversion_rule(source_op, path, steps):
    emits = []
    source = ValueRef.operand("input")
    for step_index, key in enumerate(path):
        descriptor, selector = steps[key]
        final = step_index == len(path) - 1
        result = (
            ValueRef.result("result")
            if final
            else ValueRef.temporary(f"conversion{step_index}")
        )
        if step_index:
            assert path[step_index - 1][2] == key[1]
        emits.append(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"source_v8": source},
                results={"destination_v8": result},
                result_types=None if final else {"destination_v8": Scalar(key[2])},
                immediates={descriptor.immediates[0].field_name: selector},
            )
        )
        source = result
    return DescriptorRule(
        source_op=source_op,
        descriptor=emits[-1].descriptor,
        guards=(
            Guard.value_type("input", Scalar(path[0][1])),
            Guard.value_type("result", Scalar(path[-1][2])),
        ),
        emit=tuple(emits),
    )


def _conversion_cases():
    steps = dict(_conversion_steps())
    paths = {key: (key,) for key in steps}
    bit_widths = {
        str(ScalarType(kind)): ScalarType(kind).bitwidth for kind in ScalarTypeKind
    }
    floats = tuple(name for name in _SCALAR_TYPES if name.startswith(("f", "bf")))
    integers = tuple(
        name for name in _SCALAR_TYPES if name.startswith("i") and name != "index"
    )

    for source in floats:
        if bit_widths[source] >= 32:
            continue
        extend = (conversion.scalar_extf, source, "f32")
        for result in floats:
            if bit_widths[source] == bit_widths[result]:
                continue
            source_op = (
                conversion.scalar_extf
                if bit_widths[source] < bit_widths[result]
                else conversion.scalar_fptrunc
            )
            key = (source_op, source, result)
            if key not in paths:
                final_op = (
                    conversion.scalar_extf
                    if result == "f64"
                    else conversion.scalar_fptrunc
                )
                paths[key] = (extend, (final_op, "f32", result))
        for source_op in (conversion.scalar_fptosi, conversion.scalar_fptoui):
            for result in integers:
                paths[(source_op, source, result)] = (
                    extend,
                    (source_op, "f32", result),
                )

    for source in integers:
        for source_op, extension in (
            (conversion.scalar_sitofp, conversion.scalar_extsi),
            (conversion.scalar_uitofp, conversion.scalar_extui),
        ):
            carrier = "i32" if bit_widths[source] < 32 else source
            prefix = ((extension, source, carrier),) if source != carrier else ()
            for result in floats:
                key = (source_op, source, result)
                if key in paths:
                    continue
                native = (source_op, carrier, result)
                if native in steps:
                    paths[key] = (*prefix, native)
                else:
                    # All integer values relevant to f16/FP8 finite rounding
                    # fit exactly in f32. Larger magnitudes overflow or saturate
                    # in the destination. Bfloat16 uses its direct selector:
                    # its wider exponent range makes intermediate rounding unsafe.
                    assert result in ("f16", "f8E4M3", "f8E5M2")
                    paths[key] = (
                        *prefix,
                        (source_op, carrier, "f32"),
                        (conversion.scalar_fptrunc, "f32", result),
                    )

    for (source_op, _, _), path in paths.items():
        yield _conversion_rule(source_op, path, steps)


def _math_cases():
    remaining = set(_MATH_SOURCE_OPS)
    for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors:
        instruction = _INSTRUCTIONS[descriptor.encoding_id]
        if not isinstance(instruction.semantics, FloatMathSemantics):
            continue
        selector = instruction.fields[-1].rule.data
        for value in selector.values:
            # Machine selectors with different denormal contracts require
            # separate source policies; a matching spelling is insufficient.
            if source_op := _MATH_SOURCE_OPS.get(value.name):
                remaining.discard(value.name)
                yield _scalar_rule(
                    descriptor,
                    source_op,
                    Scalar(f"f{instruction.semantics.bit_width}"),
                    value.value,
                )
    assert not remaining, f"math source mappings have no ISA selector: {remaining}"


def _f32_math_sequence(source_op, constants, steps, *, guards=()):
    # Rows name values, instructions and arguments in descriptor order. The
    # descriptors own field names, selector encodings and fixed result types;
    # bit-oriented constants/selects carry the source f32 interpretation.
    descriptors = {d.mnemonic: d for d in VM_CORE_DESCRIPTOR_SET.descriptors}
    domains = {
        domain.name: {value.token: value.value for value in domain.values}
        for domain in VM_CORE_DESCRIPTOR_SET.enum_domains
    }
    values = {
        operand.name: ValueRef.operand(operand.name) for operand in source_op.operands
    }
    values.update(
        {result.name: ValueRef.result(result.name) for result in source_op.results}
    )
    constants = tuple(
        (name, "constant.i32", struct.unpack("<I", struct.pack("<f", value))[0])
        for name, value in constants
    )
    emits = []
    for result, mnemonic, *arguments in (*constants, *steps):
        descriptor = descriptors[mnemonic]
        (form,) = descriptor.asm_forms
        (output,) = form.results
        arguments = iter(arguments)
        operands = {field: values[next(arguments)] for field in form.operands}
        result_type = scalar_result_type(_INSTRUCTIONS[descriptor.encoding_id])
        values.setdefault(result, ValueRef.temporary(result))
        emits.append(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands=operands,
                results={output: values[result]},
                result_types={
                    output: Scalar(
                        str(ScalarType(result_type))
                        if result_type is not None
                        else "f32"
                    )
                },
                immediates={
                    field.field_name: domains[field.enum_domain][value]
                    if field.enum_domain
                    else value
                    for field, value in zip(
                        descriptor.immediates, arguments, strict=True
                    )
                },
            )
        )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptors["float.math.unary.f32"],
        guards=(
            *(
                Guard.value_type(field.name, Scalar("f32"))
                for field in (*source_op.operands, *source_op.results)
            ),
            *guards,
        ),
        emit=tuple(emits),
    )


def _exp2_case():
    # The machine leaf flushes subnormal outputs. Shift those inputs into its
    # normal interval, then restore the exponent with preserving multiplication.
    # Adding 64 is exact throughout the f32 interval with a subnormal result.
    constants = (
        ("normal_limit", -126.0),
        ("shift", 64.0),
        ("zero", 0.0),
        ("scale", 2.0**-64),
        ("one", 1.0),
    )
    steps = (
        ("underflow", "float.compare.f32", "input", "normal_limit", "olt"),
        ("input_shift", "value.select", "underflow", "shift", "zero"),
        ("output_scale", "value.select", "underflow", "scale", "one"),
        ("shifted", "float.add.f32", "input", "input_shift"),
        ("normal", "float.math.unary.f32", "shifted", "exp2.approx"),
        ("result", "float.mul.f32", "normal", "output_scale"),
    )
    return _f32_math_sequence(
        math.scalar_exp2f,
        constants,
        steps,
        guards=(Guard.instance_flags_has_all("fastmath", "afn"),),
    )


def _root_cases():
    # Scaling subnormal inputs by an even power of two preserves the root's
    # significand. Both results are normal for every positive finite f32 input,
    # so restoring the exponent is exact, including each rsqrt rounding point.
    # Negative subnormals must normalize too, to reach the leaf's NaN case.
    for source_op, leaf, output_scale in (
        (math.scalar_sqrtf, "sqrt.approx", 2.0**-12),
        (math.scalar_rsqrtf, "rsqrt.approx", 2.0**12),
    ):
        constants = (
            ("normal_limit", 2.0**-126),
            ("input_scale", 2.0**24),
            ("output_scale", output_scale),
            ("one", 1.0),
        )
        steps = (
            ("magnitude", "float.abs.f32", "input"),
            ("subnormal", "float.compare.f32", "magnitude", "normal_limit", "olt"),
            ("scale_in", "value.select", "subnormal", "input_scale", "one"),
            ("scale_out", "value.select", "subnormal", "output_scale", "one"),
            ("scaled", "float.mul.f32", "input", "scale_in"),
            ("normal", "float.math.unary.f32", "scaled", leaf),
            ("result", "float.mul.f32", "normal", "scale_out"),
        )
        yield _f32_math_sequence(source_op, constants, steps)


def _log2_case():
    # Subnormal inputs normalize exactly. Correct the logarithm's exponent
    # afterward; negative subnormals must reach the leaf's NaN domain case.
    constants = (
        ("normal_limit", 2.0**-126),
        ("input_scale", 2.0**24),
        ("exponent", 24.0),
        ("one", 1.0),
        ("zero", 0.0),
    )
    steps = (
        ("magnitude", "float.abs.f32", "input"),
        ("subnormal", "float.compare.f32", "magnitude", "normal_limit", "olt"),
        ("scale", "value.select", "subnormal", "input_scale", "one"),
        ("correction", "value.select", "subnormal", "exponent", "zero"),
        ("scaled", "float.mul.f32", "input", "scale"),
        ("normal", "float.math.unary.f32", "scaled", "log2.approx"),
        ("result", "float.sub.f32", "normal", "correction"),
    )
    return _f32_math_sequence(
        math.scalar_log2f,
        constants,
        steps,
        guards=(Guard.instance_flags_has_all("fastmath", "afn"),),
    )


def _address_cases():
    # Address widths are fixed by vm.core. The shared verifier owns the index
    # domain restrictions; these rules consume that established source contract.
    integer_descriptors = {
        semantics.operation: descriptor
        for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors
        if isinstance(
            semantics := _INSTRUCTIONS[descriptor.encoding_id].semantics,
            (IntegerBinarySemantics, IntegerUnarySemantics, IntegerDivisionSemantics),
        )
        and semantics.bit_width == 64
    }
    for operation, source_op in _INDEX_SOURCE_OPS.items():
        yield _scalar_rule(
            integer_descriptors[operation],
            source_op,
            Scalar(("index", "offset"))
            if source_op in (index.index_add, index.index_sub)
            else Scalar("index"),
        )
    for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors:
        semantics = _INSTRUCTIONS[descriptor.encoding_id].semantics
        if isinstance(semantics, IntegerCompareSemantics) and semantics.bit_width == 64:
            yield _scalar_rule(
                descriptor,
                index.index_cmp,
                Scalar(("index", "offset")),
                AttrProject.enum_ordinal("predicate"),
            )
        if (
            not descriptor.immediates
            or descriptor.immediates[0].enum_domain != "integer.convert"
        ):
            continue
        selector = _INSTRUCTIONS[descriptor.encoding_id].fields[-1].rule.data
        for value in selector.values:
            source, destination = value.name.split(".to.")
            source_type = "i" + source[1:]
            if destination == "i64" and source != "s1":
                # Predicates enter both domains as zero/one. Other payloads are
                # signed coordinates or unsigned byte offsets, respectively.
                result_types = (
                    ("index", "offset")
                    if source == "u1"
                    else ("offset",)
                    if source[0] == "u"
                    else ("index",)
                )
                yield _scalar_rule(
                    descriptor,
                    index.index_cast,
                    Scalar(source_type),
                    value.value,
                    result_type=Scalar(result_types),
                )
            elif source == "i64":
                yield _scalar_rule(
                    descriptor,
                    index.index_cast,
                    Scalar(("index", "offset")),
                    value.value,
                    result_type=Scalar(destination),
                )
    yield ValueAliasRule(
        source_op=index.index_cast,
        source=ValueRef.operand("input"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("input", Scalar(("i64", "index", "offset"))),
            Guard.value_type("result", Scalar(("i64", "index", "offset"))),
        ),
    )
    multiply = integer_descriptors[IntegerBinaryOperation.MUL]
    add = integer_descriptors[IntegerBinaryOperation.ADD]
    yield DescriptorRule(
        source_op=index.index_scale,
        descriptor=multiply,
        guards=(
            Guard.value_type("index", Scalar("index")),
            Guard.value_type("stride", Scalar("offset")),
            Guard.value_type("result", Scalar("offset")),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=multiply,
                operands={
                    "left_v8": ValueRef.operand("index"),
                    "right_v8": ValueRef.operand("stride"),
                },
                results={"destination_v8": ValueRef.result("result")},
            ),
        ),
    )
    yield DescriptorRule(
        source_op=index.index_madd,
        descriptor=add,
        guards=tuple(
            Guard.value_type(name, Scalar("index"))
            for name in ("a", "b", "c", "result")
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=multiply,
                operands={
                    "left_v8": ValueRef.operand("a"),
                    "right_v8": ValueRef.operand("b"),
                },
                results={"destination_v8": ValueRef.temporary("product")},
                result_types={"destination_v8": Scalar("index")},
            ),
            EmitDescriptorOp(
                descriptor=add,
                operands={
                    "left_v8": ValueRef.temporary("product"),
                    "right_v8": ValueRef.operand("c"),
                },
                results={"destination_v8": ValueRef.result("result")},
            ),
        ),
    )


def _view_cases():
    # Views whose addresses are fully analyzed need no runtime object. Their
    # buffer identity and byte expression remain separate until each access.
    for source_op, operand in (
        (buffer.buffer_view, "buffer"),
        (view.view_subview, "source"),
        (view.view_refine, "source"),
    ):
        yield ValueAliasRule(
            source_op=source_op,
            source=ValueRef.operand(operand),
            result=ValueRef.result("result"),
        )
    integers = {
        semantics.operation: descriptor
        for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors
        if isinstance(
            semantics := _INSTRUCTIONS[descriptor.encoding_id].semantics,
            IntegerBinarySemantics,
        )
        and semantics.bit_width == 64
    }
    constant = _DESCRIPTORS[CONSTANT_I64.opcode]
    add = integers[IntegerBinaryOperation.ADD]
    materializer = SourceMemoryByteOffsetMaterializer(
        constant=constant,
        add=add,
        multiply=integers[IntegerBinaryOperation.MUL],
        shift_left=integers[IntegerBinaryOperation.SHIFT_LEFT],
        constant_immediate="bits",
        integer_conversions=tuple(
            SourceMemoryIntegerConversion(
                source_type, descriptor, (descriptor.immediates[0].field_name, selector)
            )
            for (source_op, source_type, result_type), (
                descriptor,
                selector,
            ) in _conversion_steps()
            if result_type == "i64"
            and (
                (source_type == "i1" and source_op is conversion.scalar_extui)
                or (
                    source_type in ("i8", "i16", "i32")
                    and source_op is conversion.scalar_extsi
                )
            )
        ),
    )
    for selector in MEMORY_FORMAT_SELECTOR.values:
        scalar, lanes = selector.name.split(".")
        if lanes != "x1":
            continue
        width = int(scalar[1:])
        types = tuple(
            str(ScalarType(kind))
            for kind in ScalarTypeKind
            if ScalarType(kind).bitwidth == width
            and str(ScalarType(kind)) in _SCALAR_TYPES
        )
        accesses = (
            (view.view_load, BUFFER_LOAD, SourceMemoryOperation.LOAD, "result"),
            (view.view_store, BUFFER_STORE, SourceMemoryOperation.STORE, "value"),
        )
        if width in _ATOMIC_CARRIERS:
            accesses += (
                (
                    view.view_atomic_reduce,
                    BUFFER_ATOMIC_REDUCE,
                    SourceMemoryOperation.ATOMIC_REDUCE,
                    "value",
                ),
                (
                    view.view_atomic_rmw,
                    BUFFER_ATOMIC_RMW,
                    SourceMemoryOperation.ATOMIC_RMW,
                    "result",
                ),
                (
                    view.view_atomic_cmpxchg,
                    BUFFER_ATOMIC_CMPXCHG,
                    SourceMemoryOperation.ATOMIC_CMPXCHG,
                    "old",
                ),
            )
        for source_op, instruction, operation, value_field in accesses:
            descriptor = _DESCRIPTORS[instruction.opcode]
            # Prefer the shorter zero-static recipe. The general form adds all
            # signed contributions before the VM checks the final unsigned range.
            for dynamic, zero_static in ((False, False), (True, True), (True, False)):
                memory = SourceMemoryConstraint(
                    operation=operation,
                    memory_spaces=tuple(sorted(MEMORY_SPACE_NAMES)),
                    element_byte_count=width // 8,
                    vector_lane_count=1,
                    vector_lane_byte_stride=width // 8,
                    static_byte_offset_minimum=0 if zero_static else -(2**63),
                    static_byte_offset_maximum=0 if zero_static else 2**63 - 1,
                    dynamic_term_count=None if dynamic else 0,
                    dynamic_term_count_minimum=1 if dynamic else 0,
                )
                emits = []
                coordinate = ValueRef.source_memory_dynamic_byte_offset()
                if not zero_static:
                    emits.append(
                        EmitDescriptorOp(
                            descriptor=constant,
                            form=DescriptorEmitForm.CONST,
                            results={"destination_v8": ValueRef.temporary("static")},
                            result_types={"destination_v8": Scalar("i64")},
                            immediates={
                                "bits": SourceMemoryProject.static_byte_offset()
                            },
                            source_memory=memory,
                        )
                    )
                    if dynamic:
                        emits.append(
                            EmitDescriptorOp(
                                descriptor=add,
                                operands={
                                    "left_v8": coordinate,
                                    "right_v8": ValueRef.temporary("static"),
                                },
                                results={
                                    "destination_v8": ValueRef.temporary("offset")
                                },
                                result_types={"destination_v8": Scalar("i64")},
                                source_memory=memory,
                                source_memory_byte_offset_materializer=materializer,
                            )
                        )
                    coordinate = ValueRef.temporary("offset" if dynamic else "static")
                operands = {"buffer_r8": ValueRef.operand("view")}
                results = {}
                if instruction in (BUFFER_LOAD, BUFFER_STORE):
                    operands.update(base_v8=coordinate, index_v8=coordinate)
                    immediates = {"scale_u8": 0, "format_u8": selector.value}
                    if operation == SourceMemoryOperation.STORE:
                        operands["source_v8"] = ValueRef.operand("value")
                    else:
                        results["destination_v8"] = ValueRef.result("result")
                else:
                    operands["offset_v8"] = coordinate
                    immediates = {
                        "carrier": _ATOMIC_CARRIERS[width],
                        "scope": AttrProject.enum_ordinal("scope"),
                    }
                    if instruction is BUFFER_ATOMIC_CMPXCHG:
                        immediates["orderings"] = AttrProject.attrs_pack_consecutive(
                            _ATOMIC_ORDERING_PAIR[0].name,
                            count=len(_ATOMIC_ORDERING_PAIR),
                            bit_width=_ATOMIC_ORDERING_PAIR[0].bit_length,
                        )
                        operands.update(
                            expected_v8=ValueRef.operand("expected"),
                            replacement_v8=ValueRef.operand("replacement"),
                        )
                    else:
                        operands["operand_v8"] = ValueRef.operand("value")
                        immediates.update(
                            kind=AttrProject.enum_ordinal("kind"),
                            ordering=AttrProject.enum_ordinal("ordering"),
                        )
                    if instruction is not BUFFER_ATOMIC_REDUCE:
                        results["old_v8"] = ValueRef.result(value_field)
                emits.append(
                    EmitDescriptorOp(
                        descriptor=descriptor,
                        operands=operands,
                        results=results,
                        immediates=immediates,
                        source_memory=memory,
                        source_memory_byte_offset_materializer=(
                            materializer if zero_static else None
                        ),
                    )
                )
                yield DescriptorRule(
                    source_op=source_op,
                    descriptor=descriptor,
                    guards=(Guard.value_type(value_field, Scalar(types)),),
                    emit=emits,
                    priority=1 if zero_static else 0,
                )


def _buffer_cases():
    # Source spelling is the only correspondence here: types and legal value
    # ranges come from the source op and wire fields respectively.
    byte_format = next(
        value.value for value in MEMORY_FORMAT_SELECTOR.values if value.name == "i8.x1"
    )
    # A zero scale uses the byte offset directly without a zero index register.
    byte_access = {"scale_u8": 0, "format_u8": byte_format}
    for source_op, instruction, operands, results, immediates in (
        (
            buffer.buffer_length,
            BUFFER_LENGTH,
            {"buffer_r8": "buffer"},
            {"destination_v8": "byte_length"},
            {},
        ),
        (
            buffer.buffer_copy,
            BUFFER_COPY,
            {
                "target_r8": "target",
                "target_offset_v8": "target_offset",
                "source_r8": "source",
                "source_offset_v8": "source_offset",
                "length_v8": "byte_length",
            },
            {},
            {},
        ),
        (
            buffer.buffer_compare,
            BUFFER_COMPARE,
            {
                "left_r8": "lhs",
                "left_offset_v8": "lhs_offset",
                "right_r8": "rhs",
                "right_offset_v8": "rhs_offset",
                "length_v8": "byte_length",
            },
            {"destination_v8": "order"},
            {},
        ),
        (
            buffer.buffer_load_i8_u,
            BUFFER_LOAD,
            {
                "buffer_r8": "source",
                "base_v8": "byte_offset",
                "index_v8": "byte_offset",
            },
            {"destination_v8": "result"},
            byte_access,
        ),
        (
            buffer.buffer_store_i8,
            BUFFER_STORE,
            {
                "buffer_r8": "target",
                "base_v8": "byte_offset",
                "index_v8": "byte_offset",
                "source_v8": "value",
            },
            {},
            byte_access,
        ),
    ):
        descriptor = _DESCRIPTORS[instruction.opcode]
        yield DescriptorRule(
            source_op=source_op,
            descriptor=descriptor,
            emit=(
                EmitDescriptorOp(
                    descriptor=descriptor,
                    form=DescriptorEmitForm.OP,
                    operands={
                        target: ValueRef.operand(source)
                        for target, source in operands.items()
                    },
                    results={
                        target: ValueRef.result(source)
                        for target, source in results.items()
                    },
                    immediates=immediates,
                ),
            ),
        )
    descriptor = _DESCRIPTORS[BUFFER_ALLOCATE.opcode]
    yield DescriptorRule(
        source_op=buffer.buffer_alloca,
        descriptor=descriptor,
        guards=(Guard.enum_attr_equals("memory_space", "private"),),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                form=DescriptorEmitForm.OP,
                operands={"length_v8": ValueRef.operand("byte_length")},
                results={"destination_r8": ValueRef.result("result")},
                immediates={
                    "minimum_alignment_log2_u8": AttrProject.i64_log2("base_alignment")
                },
            ),
        ),
    )
    descriptor = _DESCRIPTORS[BUFFER_FILL.opcode]
    pattern_field = next(
        field for field in BUFFER_FILL.fields if field.field.name == "pattern_width_u8"
    )
    for width in pattern_field.rule.values:
        types = tuple(
            str(ScalarType(kind))
            for kind in ScalarTypeKind
            if ScalarType(kind).bitwidth == width * 8
            and str(ScalarType(kind)) in _SCALAR_TYPES
        )
        yield DescriptorRule(
            source_op=buffer.buffer_fill,
            descriptor=descriptor,
            guards=(Guard.value_type("pattern", Scalar(types)),),
            emit=(
                EmitDescriptorOp(
                    descriptor=descriptor,
                    operands={
                        "buffer_r8": ValueRef.operand("target"),
                        "offset_v8": ValueRef.operand("target_offset"),
                        "length_v8": ValueRef.operand("byte_length"),
                        "pattern_v8": ValueRef.operand("pattern"),
                    },
                    results={},
                    immediates={"pattern_width_u8": width},
                ),
            ),
        )


VM_CORE_CONTRACT_FRAGMENT = ContractFragment(
    name="vm.core",
    descriptor_set=VM_CORE_DESCRIPTOR_SET,
    public_header="loom/target/arch/vm/contracts/core.h",
    cases=tuple(_constant_cases())
    + tuple(_selected_cases())
    + tuple(_conversion_cases())
    + tuple(_math_cases())
    + (_exp2_case(),)
    + tuple(_root_cases())
    + (_log2_case(),)
    + tuple(_address_cases())
    + tuple(_buffer_cases())
    + (RecipeRule(source_op=global_load),)
    + tuple(_view_cases())
    + select_descriptor_rules(
        (
            SelectDescriptorCase(
                scf_select,
                _DESCRIPTORS[VALUE_SELECT.opcode],
                Scalar("i1"),
                Scalar(_SCALAR_TYPES),
            ),
        ),
        descriptor_result="destination_v8",
        descriptor_condition="condition_v8",
        descriptor_true_value="true_v8",
        descriptor_false_value="false_v8",
    )
    + binary_descriptor_rules(
        tuple(_direct_cases(IntegerBinarySemantics, _BINARY_SOURCE_OPS))
        + tuple(_direct_cases(IntegerDivisionSemantics, _DIVISION_SOURCE_OPS))
        + tuple(_direct_cases(FloatBinarySemantics, _FLOAT_BINARY_SOURCE_OPS, "f")),
        form=DescriptorEmitForm.OP,
        descriptor_result="destination_v8",
        descriptor_lhs="left_v8",
        descriptor_rhs="right_v8",
    )
    + unary_descriptor_rules(
        tuple(_direct_cases(IntegerUnarySemantics, _UNARY_SOURCE_OPS))
        + tuple(_direct_cases(FloatUnarySemantics, _FLOAT_UNARY_SOURCE_OPS, "f"))
        + tuple(
            DirectDescriptorCase(
                conversion.scalar_bitcast,
                _DESCRIPTORS[VALUE_COPY.opcode],
                Scalar(types),
            )
            for types in (
                ("i8", "f8E4M3", "f8E5M2"),
                ("i16", "f16", "bf16"),
                ("i32", "f32"),
                ("i64", "f64"),
            )
        ),
        form=DescriptorEmitForm.OP,
        descriptor_result="destination_v8",
        descriptor_input="source_v8",
    )
    + ternary_descriptor_rules(
        tuple(
            DirectDescriptorCase(
                math.scalar_fmaf, descriptor, Scalar(f"f{semantics.bit_width}")
            )
            for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors
            if isinstance(
                semantics := _INSTRUCTIONS[descriptor.encoding_id].semantics,
                FloatFmaSemantics,
            )
        ),
        form=DescriptorEmitForm.OP,
        descriptor_result="destination_v8",
        descriptor_a="a_v8",
        descriptor_b="b_v8",
        descriptor_c="c_v8",
    ),
)
