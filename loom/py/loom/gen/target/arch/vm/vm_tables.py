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
from loom.target.arch.vm.projection import (
    VM_CORE_DESCRIPTOR_SET,
    VM_INSTRUCTION_PROJECTIONS,
    VM_SOURCE_LOWERINGS,
)


def _scalar_type_name(scalar_type) -> str:
    return f"LOOM_SCALAR_TYPE_{scalar_type.name}"


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
        operand_types = [_scalar_type_name(scalar_type) for scalar_type in row.operand_types]
        result_types = [_scalar_type_name(scalar_type) for scalar_type in row.result_types]
        operand_types.extend(["LOOM_SCALAR_TYPE_INDEX"] * (maximum_operand_count - len(operand_types)))
        result_types.extend(["LOOM_SCALAR_TYPE_INDEX"] * (maximum_result_count - len(result_types)))
        descriptor_name = descriptor_ref_constant_name(VM_CORE_DESCRIPTOR_SET, row.descriptor_key)
        arguments = [
            c_enum_name(row.source_op),
            descriptor_name,
            *operand_types,
            *result_types,
        ]
        lines.append("LOOM_VM_SOURCE_LOWERING_ROW(")
        lines.append("    " + ", ".join(arguments) + ")")
    return "\n".join(lines) + "\n"


def _validate_instruction_encoding_projection() -> None:
    """Proves that the generic C encoder can populate every projected record."""

    encodings_by_id = {encoding.entity_id: encoding for encoding in SCALAR_ENCODINGS}
    descriptors = VM_CORE_DESCRIPTOR_SET.descriptors
    if len(descriptors) != len(VM_INSTRUCTION_PROJECTIONS):
        raise ValueError("VM descriptor and instruction projection counts differ")
    for descriptor, projection in zip(descriptors, VM_INSTRUCTION_PROJECTIONS, strict=True):
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
            elif field.role is InstructionFieldRole.IMMEDIATE:
                if encoding.byte_length not in (1, 2, 4, 8) or field.array_length != 1:
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
        for field_value in descriptor.encoding_field_values:
            if field_value.value != 0:
                raise ValueError(f"{projection.key}: nonzero fixed fields require a specialized encoder")


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
    for ordinal, (descriptor, projection) in enumerate(
        zip(
            VM_CORE_DESCRIPTOR_SET.descriptors,
            VM_INSTRUCTION_PROJECTIONS,
            strict=True,
        )
    ):
        descriptor_name = descriptor_ref_constant_name(VM_CORE_DESCRIPTOR_SET, descriptor.key)
        lines.append(f"LOOM_VM_INSTRUCTION_ENCODING_ROW({ordinal}, {descriptor_name}, {projection.instruction.byte_length})")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Loom-facing Core VM C table family.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--lowering-rows", required=True, type=Path)
    parser.add_argument("--encoding-rows", required=True, type=Path)
    args = parser.parse_args(argv)

    write_descriptor_set_to_paths(
        VM_CORE_DESCRIPTOR_SET,
        header_path=args.header,
        source_path=args.source,
    )
    write_text_file(args.lowering_rows, generate_lowering_rows())
    write_text_file(args.encoding_rows, generate_encoding_rows())
    return 0


if __name__ == "__main__":
    sys.exit(main())
