# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Aggregation and version projection for the authoritative VM specification."""

import dataclasses

from iree.vm.bytecode.spec.isa import Instruction, InstructionFamily
from iree.vm.bytecode.spec.isa.core import FAMILIES, INSTRUCTIONS, SELECTORS
from iree.vm.bytecode.spec.isa.validation import validate_instruction
from iree.vm.bytecode.spec.module import ModuleFormat
from iree.vm.bytecode.spec.module import validation as module_validation
from iree.vm.bytecode.spec.module.container import MODULE_FORMAT
from iree.vm.bytecode.spec.schema import NumericTable, require, validate_numeric_table
from iree.vm.bytecode.spec.version import CORE_0, Version


@dataclasses.dataclass(frozen=True, slots=True)
class Specification:
    """One complete current specification and its versioned declarations."""

    name: str
    version: Version
    families: tuple[InstructionFamily, ...]
    selectors: tuple[NumericTable, ...]
    instructions: tuple[Instruction, ...]
    module_format: ModuleFormat

    def __post_init__(self) -> None:
        require(self.version.is_valid(), "invalid specification version")
        for item in (*self.families, *self.selectors, *self.instructions):
            if not item.since.is_available_in(self.version):
                raise ValueError("declaration is unavailable in the specification")

        if len({table.name for table in self.selectors}) != len(self.selectors):
            raise ValueError("duplicate selector name")
        for table in self.selectors:
            validate_numeric_table(table, self.version)
        if len({family.name for family in self.families}) != len(self.families):
            raise ValueError("duplicate instruction family name")
        for instruction in self.instructions:
            validate_instruction(instruction, self.selectors)
            if (
                instruction.family not in self.families
                or not instruction.family.since.is_available_in(instruction.since)
            ):
                raise ValueError(f"{instruction.mnemonic}: unknown family")
        if len({item.opcode for item in self.instructions}) != len(self.instructions):
            raise ValueError("duplicate opcode")
        if len({item.mnemonic for item in self.instructions}) != len(self.instructions):
            raise ValueError("duplicate mnemonic")
        module_validation.validate_module_format(self.module_format, self.version)

    def project(self, version: Version) -> "Specification":
        if not version.is_available_in(self.version):
            raise ValueError("requested version is unsupported")

        return Specification(
            self.name,
            version,
            version.select(self.families),
            tuple(
                table._replace(values=version.select(table.values))
                for table in version.select(self.selectors)
            ),
            version.select(self.instructions),
            module_validation.project_module_format(self.module_format, version),
        )


SPECIFICATION = Specification(
    name="iree.vm.bytecode",
    version=CORE_0,
    families=FAMILIES,
    selectors=SELECTORS,
    instructions=INSTRUCTIONS,
    module_format=MODULE_FORMAT,
)
