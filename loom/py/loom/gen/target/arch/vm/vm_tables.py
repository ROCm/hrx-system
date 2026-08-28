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
from pathlib import Path

from model.isa import InstructionFieldRole
from model.schema import SCALAR_ENCODINGS

from loom.gen.ops.c_names import COPYRIGHT, c_enum_name
from loom.gen.support.files import write_text_file
from loom.gen.support.generated_file import line_comment_header
from loom.gen.target.low.c_spelling import descriptor_ref_constant_name
from loom.gen.target.low.low_descriptors import write_descriptor_set_to_paths
from loom.ir import BufferType, ScalarType, Type
from loom.target.arch.vm.projection import (
    VM_CORE_DESCRIPTOR_SET,
    VM_INSTRUCTION_PROJECTIONS,
    VM_PACKET_DESCRIPTORS,
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


def generate_lowering_rows() -> str:
    """Returns the X-macro source-op signature projection."""

    maximum_operand_count = max((len(row.operand_types) for row in VM_SOURCE_LOWERINGS), default=0)
    maximum_result_count = max((len(row.result_types) for row in VM_SOURCE_LOWERINGS), default=0)
    lines = [
        COPYRIGHT.rstrip(),
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.vm.vm_tables"),
        "",
        "LOOM_VM_SOURCE_LOWERING_LIMITS(",
        f"    {maximum_operand_count}, {maximum_result_count})",
    ]
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
