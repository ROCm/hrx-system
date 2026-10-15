# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator for compact target-neutral math recipe programs."""

from __future__ import annotations

import argparse
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.func import ALL_FUNC_OPS
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scf import ALL_SCF_OPS
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dsl import Op
from loom.format.text.parser import Parser
from loom.gen.ops.c_names import c_enum_name
from loom.gen.ops.c_vector_scalarization import collect_vector_scalarization_rows
from loom.gen.support.files import write_text_file
from loom.gen.support.generated_file import line_comment_header
from loom.ir import F32, I1, I32, Module, Operation, ScalarType, ScalarTypeKind
from loom.verify import verify_module

_GENERATOR = "loom.gen.math.recipe_program"
_UNUSED_ARGUMENT = 0xFF
_MAX_ROW_INDEX = 0xFF
_MAX_INSTRUCTION_OFFSET = 0xFFFF

_PROGRAMS = (
    ("approximate_f32_exp2", "LOOM_MATH_RECIPE_PROGRAM_EXP2_F32"),
    ("approximate_f32_sin_turns", "LOOM_MATH_RECIPE_PROGRAM_SIN_TURNS_F32"),
    ("approximate_f32_cos_turns", "LOOM_MATH_RECIPE_PROGRAM_COS_TURNS_F32"),
)

_ATTR_NONE = "LOOM_MATH_RECIPE_ATTR_NONE"
_ATTR_CONSTANT_F32 = "LOOM_MATH_RECIPE_ATTR_CONSTANT_F32"
_ATTR_CONSTANT_I32 = "LOOM_MATH_RECIPE_ATTR_CONSTANT_I32"
_ATTR_ENUM = "LOOM_MATH_RECIPE_ATTR_ENUM"

_SCALAR_TYPE_C_NAMES = {
    ScalarTypeKind.I1: "LOOM_SCALAR_TYPE_I1",
    ScalarTypeKind.I32: "LOOM_SCALAR_TYPE_I32",
    ScalarTypeKind.F32: "LOOM_SCALAR_TYPE_F32",
}


@dataclass(frozen=True, slots=True)
class RecipeInstructionDescriptor:
    """Scalar/vector operation pair selected by one instruction index."""

    scalar_op: Op
    vector_op: Op
    operand_count: int
    result_element_type: ScalarTypeKind
    attribute_mode: str


@dataclass(frozen=True, slots=True)
class RecipeInstruction:
    """One packed recipe instruction."""

    descriptor_index: int
    arguments: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class RecipeProgram:
    """One function slice in the shared instruction array."""

    name: str
    c_name: str
    instruction_offset: int
    instruction_count: int
    value_count: int
    result_value: int


@dataclass(frozen=True, slots=True)
class RecipeProgramTable:
    """Validated compact representation of the authored recipe family."""

    descriptors: tuple[RecipeInstructionDescriptor, ...]
    instructions: tuple[RecipeInstruction, ...]
    constants: tuple[int, ...]
    programs: tuple[RecipeProgram, ...]

    @property
    def maximum_value_count(self) -> int:
        return max(program.value_count for program in self.programs)


def _parser() -> Parser:
    parser = Parser()
    parser.register_ops((*ALL_FUNC_OPS, *ALL_SCALAR_OPS, *ALL_SCF_OPS))
    parser.register_types(ALL_BUILTIN_TYPES)
    return parser


def _op_definitions() -> dict[str, Op]:
    return {op.name: op for op in (*ALL_FUNC_OPS, *ALL_SCALAR_OPS, *ALL_SCF_OPS)}


def _vector_counterparts() -> dict[str, Op]:
    counterparts = {row.scalar_op.name: row.vector_op for row in collect_vector_scalarization_rows()}
    scalar_ops = {op.name: op for op in ALL_SCALAR_OPS}
    vector_ops = {op.name: op for op in ALL_VECTOR_OPS}
    scf_ops = {op.name: op for op in ALL_SCF_OPS}
    counterparts.update(
        {
            scalar_ops["scalar.bitcast"].name: vector_ops["vector.bitcast"],
            scalar_ops["scalar.constant"].name: vector_ops["vector.constant"],
            scf_ops["scf.select"].name: vector_ops["vector.select"],
        }
    )
    return counterparts


def _scalar_result_type(module: Module, operation: Operation) -> ScalarType:
    if len(operation.results) != 1:
        raise ValueError(f"{operation.name}: recipe instructions require one result")
    result_type = module.values[operation.results[0]].type
    if not isinstance(result_type, ScalarType) or result_type not in (F32, I32, I1):
        raise ValueError(f"{operation.name}: recipe result must be scalar f32, i32, or i1; got {result_type}")
    return result_type


def _attribute_mode(operation: Operation, result_type: ScalarType) -> str:
    attribute_names = frozenset(operation.attributes)
    if operation.name == "scalar.constant":
        if attribute_names != {"value"}:
            raise ValueError(f"{operation.name}: expected only the value attribute, got {sorted(attribute_names)}")
        if result_type == F32:
            return _ATTR_CONSTANT_F32
        if result_type == I32:
            return _ATTR_CONSTANT_I32
        raise ValueError(f"{operation.name}: recipe constants must be f32 or i32")
    if operation.name in ("scalar.cmpf", "scalar.cmpi"):
        if attribute_names != {"predicate"}:
            raise ValueError(f"{operation.name}: expected only the predicate attribute, got {sorted(attribute_names)}")
        return _ATTR_ENUM
    if attribute_names:
        raise ValueError(f"{operation.name}: recipe operation has unsupported attributes {sorted(attribute_names)}")
    return _ATTR_NONE


def _enum_attribute_value(scalar_op: Op, vector_op: Op, keyword: str) -> int:
    scalar_attr = next((attr for attr in scalar_op.attrs if attr.name == "predicate"), None)
    vector_attr = next((attr for attr in vector_op.attrs if attr.name == "predicate"), None)
    if scalar_attr is None or vector_attr is None or scalar_attr.enum_def is None or vector_attr.enum_def is None:
        raise ValueError(f"{scalar_op.name}: comparison predicate schema is missing")
    scalar_value = scalar_attr.enum_def.case(keyword).value
    vector_value = vector_attr.enum_def.case(keyword).value
    if scalar_value != vector_value:
        raise ValueError(f"{scalar_op.name}/{vector_op.name}: predicate {keyword!r} uses different scalar/vector ordinals {scalar_value}/{vector_value}")
    return scalar_value


def _constant_bits(operation: Operation, result_type: ScalarType) -> int:
    value = operation.attributes["value"]
    if result_type == F32:
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if result_type == I32:
        integer = int(value)
        if integer != value or not -(1 << 31) <= integer < (1 << 31):
            raise ValueError(f"{operation.name}: i32 constant {value!r} is out of range")
        return integer & 0xFFFFFFFF
    raise ValueError(f"{operation.name}: unsupported constant result {result_type}")


def _compile_function(
    module: Module,
    function: Operation,
    c_name: str,
    op_definitions: Mapping[str, Op],
    vector_counterparts: Mapping[str, Op],
    descriptors: list[RecipeInstructionDescriptor],
    descriptor_indices: dict[tuple[object, ...], int],
    constants: list[int],
    constant_indices: dict[int, int],
    instructions: list[RecipeInstruction],
) -> RecipeProgram:
    if len(function.regions) != 1 or len(function.regions[0].blocks) != 1:
        raise ValueError(f"{function.attributes['callee']}: recipe must have one block")
    block = function.regions[0].blocks[0]
    if len(block.arg_ids) != 1 or module.values[block.arg_ids[0]].type != F32:
        raise ValueError(f"{function.attributes['callee']}: recipe must take one f32")

    local_values = {block.arg_ids[0]: 0}
    instruction_offset = len(instructions)
    result_value: int | None = None
    for operation in block.ops:
        if operation.name == "func.return":
            if result_value is not None or len(operation.operands) != 1:
                raise ValueError(f"{function.attributes['callee']}: expected one return")
            try:
                result_value = local_values[operation.operands[0]]
            except KeyError as error:
                raise ValueError(f"{function.attributes['callee']}: return uses an unknown value") from error
            continue
        if result_value is not None:
            raise ValueError(f"{function.attributes['callee']}: operation follows return")
        if operation.regions or operation.successors or operation.tied_results:
            raise ValueError(f"{operation.name}: recipe instructions must be regionless")
        if len(operation.operands) > 3:
            raise ValueError(f"{operation.name}: recipe instructions support at most 3 operands")

        scalar_op = op_definitions.get(operation.name)
        vector_op = vector_counterparts.get(operation.name)
        if scalar_op is None or vector_op is None:
            raise ValueError(f"{operation.name}: no validated scalar/vector recipe pair")
        result_type = _scalar_result_type(module, operation)
        attribute_mode = _attribute_mode(operation, result_type)
        descriptor_key = (
            scalar_op.name,
            vector_op.name,
            len(operation.operands),
            result_type.kind,
            attribute_mode,
        )
        descriptor_index = descriptor_indices.get(descriptor_key)
        if descriptor_index is None:
            descriptor_index = len(descriptors)
            if descriptor_index > _MAX_ROW_INDEX:
                raise ValueError("recipe descriptor count exceeds uint8_t")
            descriptor_indices[descriptor_key] = descriptor_index
            descriptors.append(
                RecipeInstructionDescriptor(
                    scalar_op=scalar_op,
                    vector_op=vector_op,
                    operand_count=len(operation.operands),
                    result_element_type=result_type.kind,
                    attribute_mode=attribute_mode,
                )
            )

        arguments = [_UNUSED_ARGUMENT] * 3
        for index, operand in enumerate(operation.operands):
            try:
                arguments[index] = local_values[operand]
            except KeyError as error:
                raise ValueError(f"{operation.name}: operand uses an unknown value") from error
        if attribute_mode in (_ATTR_CONSTANT_F32, _ATTR_CONSTANT_I32):
            bits = _constant_bits(operation, result_type)
            constant_index = constant_indices.get(bits)
            if constant_index is None:
                constant_index = len(constants)
                if constant_index > _MAX_ROW_INDEX:
                    raise ValueError("recipe constant count exceeds uint8_t")
                constant_indices[bits] = constant_index
                constants.append(bits)
            arguments[0] = constant_index
        elif attribute_mode == _ATTR_ENUM:
            arguments[2] = _enum_attribute_value(scalar_op, vector_op, str(operation.attributes["predicate"]))

        instructions.append(
            RecipeInstruction(
                descriptor_index,
                (arguments[0], arguments[1], arguments[2]),
            )
        )
        local_value = len(local_values)
        if local_value > _MAX_ROW_INDEX:
            raise ValueError(f"{function.attributes['callee']}: value count exceeds uint8_t")
        local_values[operation.results[0]] = local_value

    if result_value is None:
        raise ValueError(f"{function.attributes['callee']}: recipe has no return")
    instruction_count = len(instructions) - instruction_offset
    if instruction_offset > _MAX_INSTRUCTION_OFFSET:
        raise ValueError("recipe instruction offset exceeds uint16_t")
    if instruction_count > _MAX_ROW_INDEX:
        raise ValueError(f"{function.attributes['callee']}: instruction count exceeds uint8_t")
    return RecipeProgram(
        name=str(function.attributes["callee"]),
        c_name=c_name,
        instruction_offset=instruction_offset,
        instruction_count=instruction_count,
        value_count=len(local_values),
        result_value=result_value,
    )


def compile_recipe_program(source: str, *, filename: Path | None = None) -> RecipeProgramTable:
    """Parses and validates one complete authored recipe family."""

    ops = (*ALL_FUNC_OPS, *ALL_SCALAR_OPS, *ALL_SCF_OPS)
    module = _parser().parse(source, filename=str(filename or "<input>"))
    verify_module(module, ops=ops).raise_if_errors()
    functions: dict[str, Operation] = {}
    for symbol in module.symbols:
        if symbol.op is None:
            continue
        name = str(symbol.op.attributes.get("callee", ""))
        if not name or name in functions:
            raise ValueError(f"duplicate or unnamed recipe function {name!r}")
        functions[name] = symbol.op
    expected_names = {name for name, _ in _PROGRAMS}
    if set(functions) != expected_names:
        raise ValueError(f"recipe functions must be exactly {sorted(expected_names)}, got {sorted(functions)}")

    descriptors: list[RecipeInstructionDescriptor] = []
    descriptor_indices: dict[tuple[object, ...], int] = {}
    constants: list[int] = []
    constant_indices: dict[int, int] = {}
    instructions: list[RecipeInstruction] = []
    op_definitions = _op_definitions()
    vector_counterparts = _vector_counterparts()
    programs = tuple(
        _compile_function(
            module,
            functions[name],
            c_name,
            op_definitions,
            vector_counterparts,
            descriptors,
            descriptor_indices,
            constants,
            constant_indices,
            instructions,
        )
        for name, c_name in _PROGRAMS
    )
    return RecipeProgramTable(
        descriptors=tuple(descriptors),
        instructions=tuple(instructions),
        constants=tuple(constants),
        programs=programs,
    )


def _hex_byte(value: int) -> str:
    return f"UINT8_C(0x{value:02X})"


def generate_recipe_program_tables(table: RecipeProgramTable) -> str:
    """Emits private C initializers for one validated recipe table."""

    lines = [
        *line_comment_header("//", generator=_GENERATOR),
        "// clang-format off",
        "",
        "enum {",
        "  // Maximum local SSA values required by any generated recipe program.",
        f"  LOOM_MATH_RECIPE_PROGRAM_VALUE_CAPACITY = {table.maximum_value_count},",
        "};",
        "",
        "static const loom_math_recipe_instruction_descriptor_t",
        "    kMathRecipeInstructionDescriptors[] = {",
    ]
    for index, descriptor in enumerate(table.descriptors):
        lines.extend(
            [
                f"  // {index}: {descriptor.scalar_op.name} / {descriptor.vector_op.name}",
                "  {",
                f"    .scalar_op_kind = {c_enum_name(descriptor.scalar_op)},",
                f"    .vector_op_kind = {c_enum_name(descriptor.vector_op)},",
                f"    .operand_count = {descriptor.operand_count},",
                f"    .result_element_type = {_SCALAR_TYPE_C_NAMES[descriptor.result_element_type]},",
                f"    .attribute_mode = {descriptor.attribute_mode},",
                "  },",
            ]
        )
    lines.extend(
        [
            "};",
            "",
            "static const loom_math_recipe_instruction_t kMathRecipeInstructions[] = {",
        ]
    )
    for instruction in table.instructions:
        arguments = ", ".join(_hex_byte(argument) for argument in instruction.arguments)
        lines.append(f"  {{{instruction.descriptor_index}, {{{arguments}}}}},")
    lines.extend(
        [
            "};",
            "",
            "static const uint32_t kMathRecipeConstants[] = {",
        ]
    )
    lines.extend(f"  UINT32_C(0x{value:08X})," for value in table.constants)
    lines.extend(
        [
            "};",
            "",
            "static const loom_math_recipe_program_t",
            "    kMathRecipePrograms[LOOM_MATH_RECIPE_PROGRAM_COUNT_] = {",
        ]
    )
    for program in table.programs:
        lines.extend(
            [
                f"  [{program.c_name}] = {{",
                f"    .instruction_offset = {program.instruction_offset},",
                f"    .instruction_count = {program.instruction_count},",
                f"    .result_value = {program.result_value},",
                "  },",
            ]
        )
    lines.extend(
        [
            "};",
            "",
            "// clang-format on",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generates compact math recipe tables.")
    parser.add_argument("--input", required=True, type=Path, help="Authored Loom recipe source.")
    parser.add_argument("--output", type=Path, help="Generated C table include.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the recipe without writing an output file.",
    )
    args = parser.parse_args(argv)
    if args.check == (args.output is not None):
        parser.error("select exactly one of --check or --output")
    table = compile_recipe_program(args.input.read_text(), filename=args.input)
    if args.output is not None:
        write_text_file(args.output, generate_recipe_program_tables(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
