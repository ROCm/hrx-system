# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Aggregation and version projection for the authoritative VM specification."""

import dataclasses

from iree.vm.bytecode.spec.isa import Instruction, InstructionFamily
from iree.vm.bytecode.spec.isa.core import FAMILIES, INSTRUCTIONS
from iree.vm.bytecode.spec.isa.validation import validate_instruction
from iree.vm.bytecode.spec.module import WireRecord
from iree.vm.bytecode.spec.module.records import RECORDS
from iree.vm.bytecode.spec.module.validation import validate_wire_record
from iree.vm.bytecode.spec.version import CORE_0, Version


@dataclasses.dataclass(frozen=True, slots=True)
class Specification:
    """One complete current specification and its versioned declarations."""

    name: str
    version: Version
    families: tuple[InstructionFamily, ...]
    instructions: tuple[Instruction, ...]
    records: tuple[WireRecord, ...]

    def __post_init__(self) -> None:
        if not self.version.is_valid():
            raise ValueError("invalid specification version")
        for item in (*self.families, *self.instructions, *self.records):
            if not item.since.is_valid() or not item.since.is_available_in(
                self.version
            ):
                raise ValueError("declaration is unavailable in the specification")

        if len({family.name for family in self.families}) != len(self.families):
            raise ValueError("duplicate instruction family name")
        for instruction in self.instructions:
            validate_instruction(instruction)
            if (
                instruction.family not in self.families
                or not instruction.family.since.is_available_in(instruction.since)
            ):
                raise ValueError(f"{instruction.mnemonic}: unknown family")
        opcodes = [instruction.opcode for instruction in self.instructions]
        mnemonics = [instruction.mnemonic for instruction in self.instructions]
        if len(set(opcodes)) != len(opcodes):
            raise ValueError("duplicate opcode")
        if len(set(mnemonics)) != len(mnemonics):
            raise ValueError("duplicate mnemonic")
        for record in self.records:
            validate_wire_record(record)
        if len({record.name for record in self.records}) != len(self.records):
            raise ValueError("duplicate wire record")

    def project(self, version: Version) -> "Specification":
        if not version.is_available_in(self.version):
            raise ValueError("requested version is unsupported")

        return Specification(
            self.name,
            version,
            version.select(self.families),
            version.select(self.instructions),
            version.select(self.records),
        )


SPECIFICATION = Specification(
    name="iree.vm.bytecode",
    version=CORE_0,
    families=FAMILIES,
    instructions=INSTRUCTIONS,
    records=RECORDS,
)
