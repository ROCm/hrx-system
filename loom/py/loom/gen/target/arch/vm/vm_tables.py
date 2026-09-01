# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates the complete Loom-facing Core VM table family."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from itertools import groupby
from pathlib import Path

from model.isa import InstructionFieldRole
from model.schema import SCALAR_ENCODINGS

from loom.dsl import Op
from loom.gen.ops.c_names import COPYRIGHT, c_enum_name
from loom.gen.support.files import write_text_file
from loom.gen.support.generated_file import line_comment_header
from loom.gen.target.low.c_spelling import descriptor_ref_constant_name
from loom.gen.target.low.low_descriptors import write_descriptor_set_to_paths
from loom.ir import BufferType, ScalarType, Type
from loom.target.arch.vm.projection import (
    VM_CORE_DESCRIPTOR_SET,
    VM_INSTRUCTION_PROJECTIONS,
    VM_MODULE_RESOURCES,
    VM_PACKET_DESCRIPTORS,
)
from loom.target.arch.vm.source_lowering import (
    VM_ATOMIC_SOURCE_LOWERINGS,
    VM_SOURCE_CONVERSION_LOWERINGS,
    VM_SOURCE_CONVERSION_MAX_STEP_COUNT,
    VM_SOURCE_LOWERINGS,
)
from loom.target.arch.vm.verification import (
    VM_MEMORY_FORMAT_UNIT_COUNTS,
    VM_PACKED_IMMEDIATE_MASKS,
    VM_PACKET_CONSTRAINT_RANGES,
    VM_PACKET_CONSTRAINTS,
)


def _source_type_key(source_type: Type) -> str:
    """Returns one compact exact source-type key for generated C rows."""

    if isinstance(source_type, ScalarType):
        return f"LOOM_VM_SOURCE_TYPE_KEY(LOOM_TYPE_SCALAR, LOOM_SCALAR_TYPE_{source_type.kind.name})"
    if isinstance(source_type, BufferType):
        return "LOOM_VM_SOURCE_TYPE_KEY(LOOM_TYPE_BUFFER, LOOM_SCALAR_TYPE_NONE)"
    raise ValueError(f"unsupported concrete VM source lowering type {source_type!r}")


def _descriptor_ref_name(descriptor_key: str | None) -> str:
    if descriptor_key is None:
        return "UINT16_MAX"
    return descriptor_ref_constant_name(VM_CORE_DESCRIPTOR_SET, descriptor_key)


def _source_lowering_ranges() -> tuple[tuple[Op, int, int], ...]:
    """Returns contiguous row ranges grouped by exact source operation."""

    ranges: list[tuple[Op, int, int]] = []
    seen_op_names: set[str] = set()
    row_start = 0
    for source_op, rows_iter in groupby(
        VM_SOURCE_LOWERINGS,
        key=lambda row: row.source_op,
    ):
        if source_op.name in seen_op_names:
            raise ValueError(f"{source_op.name}: VM source lowering rows must be contiguous")
        seen_op_names.add(source_op.name)
        row_count = sum(1 for _ in rows_iter)
        if row_start > 0xFFFF or row_count > 0xFF:
            raise ValueError(f"{source_op.name}: VM source lowering range exceeds u16/u8")
        ranges.append((source_op, row_start, row_count))
        row_start += row_count
    if row_start != len(VM_SOURCE_LOWERINGS) or row_start > 0xFFFF:
        raise ValueError("VM source lowering row count exceeds u16")
    return tuple(ranges)


def _dialect_range_symbol(dialect_name: str) -> str:
    dialect_fragment = "".join(part.capitalize() for part in dialect_name.split("_"))
    return f"kVmSourceLowering{dialect_fragment}Ranges"


def _conversion_lowering_ranges() -> tuple[tuple[Op, int, int], ...]:
    """Returns contiguous conversion-lowering ranges grouped by source op."""

    ranges: list[tuple[Op, int, int]] = []
    seen_op_names: set[str] = set()
    lowering_start = 0
    for source_op, lowerings_iter in groupby(
        VM_SOURCE_CONVERSION_LOWERINGS,
        key=lambda lowering: lowering.source_op,
    ):
        if source_op.name in seen_op_names:
            raise ValueError(f"{source_op.name}: VM conversion lowerings must be contiguous")
        seen_op_names.add(source_op.name)
        lowering_count = sum(1 for _ in lowerings_iter)
        if lowering_start > 0xFFFF or lowering_count > 0xFF:
            raise ValueError(f"{source_op.name}: VM conversion range exceeds u16/u8")
        ranges.append((source_op, lowering_start, lowering_count))
        lowering_start += lowering_count
    if lowering_start != len(VM_SOURCE_CONVERSION_LOWERINGS) or lowering_start > 0xFFFF:
        raise ValueError("VM conversion lowering count exceeds u16")
    return tuple(ranges)


def generate_lowering_rows() -> str:
    """Returns the X-macro source-op signature projection."""

    maximum_operand_count = max((len(row.operand_types) for row in VM_SOURCE_LOWERINGS), default=0)
    maximum_result_count = max((len(row.result_types) for row in VM_SOURCE_LOWERINGS), default=0)
    lines = [
        COPYRIGHT.rstrip(),
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.vm.vm_tables"),
        "",
        "#if defined(LOOM_VM_MODULE_RESOURCE_ROW)",
    ]
    for resource in VM_MODULE_RESOURCES:
        arguments = (
            resource.kind.value,
            _descriptor_ref_name(resource.load_descriptor_key),
            _descriptor_ref_name(resource.store_descriptor_key),
            _descriptor_ref_name(resource.store_preserve_descriptor_key),
        )
        lines.append("LOOM_VM_MODULE_RESOURCE_ROW(" + ", ".join(arguments) + ")")
    lines.extend(
        [
            "#elif defined(LOOM_VM_ATOMIC_SOURCE_LOWERING_ROW)",
        ]
    )
    for lowering in VM_ATOMIC_SOURCE_LOWERINGS:
        component_arguments: list[str] = []
        for component in lowering.components:
            if component is None:
                component_arguments.extend(("UINT8_MAX", "0"))
            else:
                component_arguments.extend((str(component.selector_ordinal), str(component.bit_offset)))
        lines.append(
            "LOOM_VM_ATOMIC_SOURCE_LOWERING_ROW("
            + ", ".join(
                (
                    f"LOOM_LOW_SOURCE_MEMORY_OPERATION_{lowering.operation_kind}",
                    _descriptor_ref_name(lowering.descriptor_key),
                    str(lowering.selector_count),
                    *component_arguments,
                )
            )
            + ")"
        )
    lines.extend(
        [
            "#elif defined(LOOM_VM_SOURCE_LOWERING_ROW)",
            "LOOM_VM_SOURCE_LOWERING_LIMITS(",
            f"    {maximum_operand_count}, {maximum_result_count})",
        ]
    )
    for row in VM_SOURCE_LOWERINGS:
        operand_types = [_source_type_key(source_type) for source_type in row.operand_types]
        result_types = [_source_type_key(source_type) for source_type in row.result_types]
        unused_type = "LOOM_VM_SOURCE_TYPE_KEY(LOOM_TYPE_NONE, LOOM_SCALAR_TYPE_NONE)"
        operand_types.extend([unused_type] * (maximum_operand_count - len(operand_types)))
        result_types.extend([unused_type] * (maximum_result_count - len(result_types)))
        descriptor_name = descriptor_ref_constant_name(VM_CORE_DESCRIPTOR_SET, row.descriptor_key)
        arguments = [
            c_enum_name(row.source_op),
            descriptor_name,
            ("UINT8_MAX" if row.selector_immediate_ordinal is None else str(row.selector_immediate_ordinal)),
            ("UINT8_MAX" if row.selector_source_attr_ordinal is None else str(row.selector_source_attr_ordinal)),
            str(row.selector_value),
            *operand_types,
            *result_types,
        ]
        lines.append("LOOM_VM_SOURCE_LOWERING_ROW(")
        lines.append("    " + ", ".join(arguments) + ")")
    lines.append("#elif defined(LOOM_VM_SOURCE_LOWERING_DEFINE_RANGES)")
    ranges_by_dialect: dict[str, list[tuple[Op, int, int]]] = {}
    for source_op, row_start, row_count in _source_lowering_ranges():
        if source_op.group is None or source_op.group.dialect_id == 0:
            raise ValueError(f"{source_op.name}: VM source lowering requires a built-in dialect")
        ranges_by_dialect.setdefault(source_op.group.name, []).append((source_op, row_start, row_count))
    for dialect_name, ranges in ranges_by_dialect.items():
        symbol = _dialect_range_symbol(dialect_name)
        dialect_token = dialect_name.upper()
        lines.extend(
            [
                "static const loom_vm_source_lowering_range_t",
                f"    {symbol}[LOOM_OP_{dialect_token}_COUNT_] = {{",
            ]
        )
        lines.extend(f"        [{c_enum_name(source_op)} & 0xFF] = {{{row_start}, {row_count}}}," for source_op, row_start, row_count in ranges)
        lines.append("};")
    lines.extend(
        [
            "static const loom_vm_source_lowering_dialect_ranges_t",
            "    kVmSourceLoweringDialectRanges[LOOM_DIALECT_BUILTIN_COUNT_] = {",
        ]
    )
    for dialect_name in ranges_by_dialect:
        symbol = _dialect_range_symbol(dialect_name)
        lines.append(f"        [LOOM_DIALECT_{dialect_name.upper()}] = {{{symbol}, IREE_ARRAYSIZE({symbol})}},")
    lines.append("};")
    lines.append("#elif defined(LOOM_VM_CONVERSION_LOWERING_LIMITS)")
    lines.append(f"LOOM_VM_CONVERSION_LOWERING_LIMITS({VM_SOURCE_CONVERSION_MAX_STEP_COUNT})")
    lines.append("#elif defined(LOOM_VM_CONVERSION_LOWERING_STEP_ROW)")
    step_count = 0
    for lowering in VM_SOURCE_CONVERSION_LOWERINGS:
        for step in lowering.steps:
            if step_count > 0xFFFF:
                raise ValueError("VM conversion lowering step count exceeds u16")
            lines.append(
                "LOOM_VM_CONVERSION_LOWERING_STEP_ROW("
                + ", ".join(
                    (
                        descriptor_ref_constant_name(
                            VM_CORE_DESCRIPTOR_SET,
                            step.descriptor_key,
                        ),
                        str(step.selector_value),
                        f"LOOM_SCALAR_TYPE_{step.result_types[0].kind.name}",
                    )
                )
                + ")"
            )
            step_count += 1
    lines.append("#elif defined(LOOM_VM_CONVERSION_LOWERING_ROW)")
    step_start = 0
    for lowering in VM_SOURCE_CONVERSION_LOWERINGS:
        if len(lowering.steps) > 0xFF:
            raise ValueError(f"{lowering.source_op.name}: VM conversion lowering exceeds u8")
        lines.append(
            "LOOM_VM_CONVERSION_LOWERING_ROW("
            + ", ".join(
                (
                    f"LOOM_SCALAR_TYPE_{lowering.source_type.name}",
                    f"LOOM_SCALAR_TYPE_{lowering.result_type.name}",
                    str(step_start),
                    str(len(lowering.steps)),
                )
            )
            + ")"
        )
        step_start += len(lowering.steps)
    if step_start != step_count:
        raise ValueError("VM conversion lowering step projection drifted")
    lines.append("#elif defined(LOOM_VM_CONVERSION_LOWERING_DEFINE_RANGES)")
    lines.extend(
        (
            "static const loom_vm_conversion_lowering_range_t",
            "    kVmConversionLoweringRanges[LOOM_OP_SCALAR_COUNT_] = {",
        )
    )
    lines.extend(f"        [{c_enum_name(source_op)} & 0xFF] = {{{lowering_start}, {lowering_count}}}," for source_op, lowering_start, lowering_count in _conversion_lowering_ranges())
    lines.append("};")
    lines.append("#endif  // VM lowering projection")
    return "\n".join(lines) + "\n"


def _validate_instruction_encoding_projection() -> None:
    """Proves that the generic C encoder can populate every projected record."""

    encodings_by_id = {encoding.entity_id: encoding for encoding in SCALAR_ENCODINGS}
    if len(VM_PACKET_DESCRIPTORS) != len(VM_INSTRUCTION_PROJECTIONS):
        raise ValueError("VM packet descriptor projection is incomplete")
    for descriptor, projection in zip(VM_PACKET_DESCRIPTORS, VM_INSTRUCTION_PROJECTIONS, strict=True):
        instruction = projection.instruction
        if instruction.byte_length <= 0 or instruction.byte_length % 4:
            raise ValueError(f"{projection.key}: instruction length must be a positive multiple of four")
        if instruction.byte_length > 0xFF:
            raise ValueError(f"{projection.key}: instruction length exceeds uint8")
        if instruction.opcode < 0 or instruction.opcode > 0xFF:
            raise ValueError(f"{projection.key}: core opcode exceeds uint8")
        if descriptor.encoding_id != instruction.opcode:
            raise ValueError(f"{projection.key}: descriptor opcode drifted")

        for field in instruction.fields:
            encoding = encodings_by_id.get(field.encoding_id)
            if encoding is None:
                raise ValueError(f"{projection.key}: field {field.name} has a non-scalar encoding")
            field_byte_length = encoding.byte_length * field.array_length
            if field.offset + field_byte_length > instruction.byte_length:
                raise ValueError(f"{projection.key}: field {field.name} exceeds its record")
            if field.role in (
                InstructionFieldRole.RESULT,
                InstructionFieldRole.OPERAND,
            ):
                if encoding.byte_length != 1 or field.array_length != 1:
                    raise ValueError(f"{projection.key}: register field {field.name} must be one byte")
            elif field.role in (
                InstructionFieldRole.IMMEDIATE,
                InstructionFieldRole.CONSTRAINT_MEMBER,
            ):
                if encoding.byte_length not in (1, 2, 4, 8):
                    raise ValueError(f"{projection.key}: immediate field {field.name} is not a supported scalar")
            elif field.role is not InstructionFieldRole.PADDING:
                raise ValueError(f"{projection.key}: field {field.name} has unsupported role {field.role.value}")

        for immediate in descriptor.immediates:
            if immediate.encoding_slices:
                raise ValueError(f"{projection.key}: sliced immediates require a specialized encoder")
            if immediate.bit_width not in (8, 16, 32, 64):
                raise ValueError(f"{projection.key}: immediate width {immediate.bit_width} is unsupported")
            if immediate.encoding_field_id + immediate.bit_width // 8 > instruction.byte_length:
                raise ValueError(f"{projection.key}: immediate field exceeds its record")
        if descriptor.encoding_field_values:
            raise ValueError(f"{projection.key}: zero-initialized VM records need no fixed fields")


def generate_encoding_rows() -> str:
    """Returns the X-macro instruction-record size projection."""

    _validate_instruction_encoding_projection()
    maximum_record_byte_length = max(projection.instruction.byte_length for projection in VM_INSTRUCTION_PROJECTIONS)
    lines = [
        COPYRIGHT.rstrip(),
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.vm.vm_tables"),
        "",
        f"LOOM_VM_INSTRUCTION_ENCODING_LIMITS({maximum_record_byte_length})",
    ]
    lines.extend(f"LOOM_VM_INSTRUCTION_ENCODING_ROW({projection.instruction.byte_length})" for projection in VM_INSTRUCTION_PROJECTIONS)
    return "\n".join(lines) + "\n"


def generate_verification_rows() -> str:
    """Returns the compact target-low packet verifier tables."""

    if len(VM_PACKET_CONSTRAINT_RANGES) != len(VM_INSTRUCTION_PROJECTIONS):
        raise ValueError("VM packet constraint range count does not match packet descriptors")
    lines = [
        COPYRIGHT.rstrip(),
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.vm.vm_tables"),
        "",
        "#if defined(LOOM_VM_PACKET_CONSTRAINT_DEFINE_LIMITS)",
        "LOOM_VM_PACKET_CONSTRAINT_LIMITS(",
        f"    {len(VM_PACKET_CONSTRAINT_RANGES)}, {len(VM_PACKET_CONSTRAINTS)},",
        f"    {len(VM_PACKED_IMMEDIATE_MASKS)}, {len(VM_MEMORY_FORMAT_UNIT_COUNTS)})",
        "#elif defined(LOOM_VM_PACKET_CONSTRAINT_DEFINE_MEMORY_FORMAT_ROWS)",
    ]
    lines.extend(f"LOOM_VM_MEMORY_FORMAT_UNIT_COUNT_ROW({unit_count})" for unit_count in VM_MEMORY_FORMAT_UNIT_COUNTS)
    lines.append("#elif defined(LOOM_VM_PACKET_CONSTRAINT_DEFINE_PACKED_MASK_ROWS)")
    for words in VM_PACKED_IMMEDIATE_MASKS:
        word_arguments = ", ".join(f"UINT64_C(0x{word:016X})" for word in words)
        lines.append(f"LOOM_VM_PACKED_IMMEDIATE_MASK_ROW({word_arguments})")
    lines.append("#elif defined(LOOM_VM_PACKET_CONSTRAINT_DEFINE_RANGE_ROWS)")
    lines.extend(f"LOOM_VM_PACKET_CONSTRAINT_RANGE_ROW({constraint_start}, {constraint_count})" for constraint_start, constraint_count in VM_PACKET_CONSTRAINT_RANGES)
    lines.append("#elif defined(LOOM_VM_PACKET_CONSTRAINT_DEFINE_ROWS)")
    for constraint in VM_PACKET_CONSTRAINTS:
        arguments = [*constraint.arguments]
        arguments.extend([0] * (7 - len(arguments)))
        lines.append(
            "LOOM_VM_PACKET_CONSTRAINT_ROW("
            + ", ".join(
                (
                    constraint.kind.name,
                    str(constraint.parameter),
                    *(str(argument) for argument in arguments),
                )
            )
            + ")"
        )
    lines.append("#endif  // VM packet constraint projection")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Loom-facing Core VM C table family.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--lowering-rows", required=True, type=Path)
    parser.add_argument("--encoding-rows", required=True, type=Path)
    parser.add_argument("--verification-rows", required=True, type=Path)
    args = parser.parse_args(argv)

    write_descriptor_set_to_paths(
        VM_CORE_DESCRIPTOR_SET,
        header_path=args.header,
        source_path=args.source,
    )
    write_text_file(args.lowering_rows, generate_lowering_rows())
    write_text_file(args.encoding_rows, generate_encoding_rows())
    write_text_file(args.verification_rows, generate_verification_rows())
    return 0


if __name__ == "__main__":
    sys.exit(main())
