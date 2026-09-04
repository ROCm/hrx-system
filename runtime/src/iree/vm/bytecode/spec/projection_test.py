# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests the deterministic purpose-specific runtime projections."""

import unittest

from iree.vm.bytecode.spec.generate import generate_outputs
from iree.vm.bytecode.spec.specification import SPECIFICATION


def _bstring(value: str) -> str:
    return f'"\\x{len(value.encode("utf-8")):02x}" "{value}"'


class ProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.outputs = generate_outputs()

    def test_every_declaration_projects_to_each_required_surface(self) -> None:
        module_header = self.outputs["wire_module_header"]
        core_header = self.outputs["wire_core_header"]
        assertions = self.outputs["wire_assertions_source"]
        disassembler = self.outputs["disassembler_data"]
        documentation = self.outputs["documentation"]
        instruction_verifier = self.outputs["instruction_verifier_cases"]
        interpreter_data = self.outputs["interpreter_data"]
        module_verifier = self.outputs["module_verifier_cases"]
        self.assertNotIn("static_assert", module_header + core_header)
        self.assertNotIn("iree_string_view_t", disassembler)
        for record in SPECIFICATION.module_format.records:
            self.assertIn(f"typedef struct {record.c_type}", module_header)
            self.assertIn(
                f"sizeof({record.c_type}) == {record.byte_length}u", assertions
            )
            self.assertIn(_bstring(record.name), disassembler)
            self.assertIn(
                f"IREE_VM_BYTECODE_MODULE_RECORD_{record.name.upper()}",
                module_verifier,
            )
            self.assertEqual(documentation.count(f"#### `{record.name}`"), 1)
        for instruction in SPECIFICATION.instructions:
            c_type = f"iree_vm_bytecode_{instruction.mnemonic.replace('.', '_')}_t"
            self.assertIn(f"typedef struct {c_type}", core_header)
            self.assertIn(f"sizeof({c_type}) == {instruction.byte_length}u", assertions)
            self.assertIn(_bstring(instruction.mnemonic), disassembler)
            self.assertIn(instruction.mnemonic, instruction_verifier)
            self.assertIn(
                f"OP({instruction.mnemonic.replace('.', '_').upper()}, "
                f"{instruction.mnemonic.replace('.', '_')}, {c_type})",
                interpreter_data,
            )
            self.assertEqual(documentation.count(f"#### `{instruction.mnemonic}`"), 1)
        for table in (
            SPECIFICATION.module_format.numeric_tables + SPECIFICATION.selectors
        ):
            self.assertEqual(documentation.count(f"#### `{table.name}`"), 1)
        for section in SPECIFICATION.module_format.sections:
            self.assertEqual(documentation.count(f"### `{section.name}`"), 1)
        for family in SPECIFICATION.families:
            self.assertEqual(documentation.count(f"### {family.name}"), 1)
        self.assertIn("#### Structural verification obligations", documentation)
        self.assertIn("#### Reference pseudocode", documentation)

    def test_verification_projects_direct_runtime_checks(self) -> None:
        verifier_source = self.outputs["verifier_source"]
        instruction_cases = self.outputs["instruction_verifier_cases"]
        layout_data = self.outputs["layout_data"]
        module_cases = self.outputs["module_verifier_cases"]
        source = verifier_source + instruction_cases + module_cases + layout_data
        for fragment in (
            "iree_vm_bytecode_instruction_verification[256]",
            "iree_vm_bytecode_section_layouts[]",
            "iree_vm_bytecode_verify_control_target",
            "iree_vm_bytecode_verify_direct_call",
            "iree_vm_bytecode_verify_integer_bitstream_shape",
            "iree_vm_bytecode_verify_signature_descriptor",
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("switch (", verifier_source)
        self.assertNotIn("switch (", layout_data)
        self.assertNotIn("iree_status_t", verifier_source)
        self.assertNotIn("iree_status_t", instruction_cases)
        self.assertNotIn("VERIFICATION_RULE_", source)
        self.assertEqual(
            instruction_cases.count("case IREE_VM_BYTECODE_OPCODE_"),
            len(SPECIFICATION.instructions),
        )
        self.assertLess(
            instruction_cases.count("return true;"), len(SPECIFICATION.instructions)
        )

    def test_interpreter_data_covers_core_directly(self) -> None:
        data = self.outputs["interpreter_data"]
        opcode_list = data.partition(
            "#define IREE_VM_BYTECODE_INTERPRETER_INTEGER_TOTAL_LIST_COUNT"
        )[0]
        self.assertEqual(opcode_list.count("  OP("), len(SPECIFICATION.instructions))
        self.assertNotIn("verification", data.lower())
        for family_name, macro_name in (
            ("integer", "INTEGER"),
            ("float", "FLOAT"),
            ("conversion", "CONVERSION"),
        ):
            family_instructions = tuple(
                instruction
                for instruction in SPECIFICATION.instructions
                if instruction.family.name == family_name
            )
            if family_name == "float":
                self.assertIn(
                    f"#define IREE_VM_BYTECODE_INTERPRETER_{macro_name}_LIST_COUNT "
                    f"{len(family_instructions)}u",
                    data,
                )
            else:
                total_count = sum(
                    not instruction.failures for instruction in family_instructions
                )
                fallible_count = len(family_instructions) - total_count
                self.assertIn(
                    f"#define IREE_VM_BYTECODE_INTERPRETER_{macro_name}_TOTAL_LIST_COUNT "
                    f"{total_count}u",
                    data,
                )
                self.assertIn(
                    f"#define IREE_VM_BYTECODE_INTERPRETER_{macro_name}_FALLIBLE_LIST_COUNT "
                    f"{fallible_count}u",
                    data,
                )
        for table_name, macro_name in (
            ("integer.convert", "INTEGER_CONVERSION"),
            ("float.extend", "FLOAT_EXTEND"),
            ("float.truncate", "FLOAT_TRUNCATE"),
            ("float.width", "FLOAT_WIDTH"),
            ("integer.to.float", "INTEGER_TO_FLOAT"),
            ("float.to.integer", "FLOAT_TO_INTEGER"),
        ):
            table = next(
                table for table in SPECIFICATION.selectors if table.name == table_name
            )
            self.assertIn(
                f"#define IREE_VM_BYTECODE_{macro_name}_COUNT {len(table.values)}u",
                data,
            )
            for value in table.values:
                self.assertIn(value.name.replace(".", "_").upper(), data)


if __name__ == "__main__":
    unittest.main()
