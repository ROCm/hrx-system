# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests the deterministic purpose-specific runtime projections."""

import unittest

from iree.vm.bytecode.spec.generate import generate_outputs
from iree.vm.bytecode.spec.isa.core import INTEGER_INSTRUCTIONS
from iree.vm.bytecode.spec.render.c import _instruction_rules
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
        tooling = self.outputs["tooling_data"]
        documentation = self.outputs["documentation"]
        self.assertNotIn("static_assert", module_header + core_header)
        self.assertNotIn("iree_string_view_t", tooling)
        for record in SPECIFICATION.module_format.records:
            self.assertIn(f"typedef struct {record.c_type}", module_header)
            self.assertIn(
                f"sizeof({record.c_type}) == {record.byte_length}u", assertions
            )
            self.assertIn(_bstring(record.name), tooling)
            self.assertEqual(documentation.count(f"#### `{record.name}`"), 1)
        for instruction in SPECIFICATION.instructions:
            c_type = f"iree_vm_bytecode_{instruction.mnemonic.replace('.', '_')}_t"
            self.assertIn(f"typedef struct {c_type}", core_header)
            self.assertIn(f"sizeof({c_type}) == {instruction.byte_length}u", assertions)
            self.assertIn(_bstring(instruction.mnemonic), tooling)
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

    def test_verification_is_data_not_control_flow(self) -> None:
        source = self.outputs["verification_source"]
        for fragment in (
            "iree_vm_bytecode_instruction_verification[256]",
            "VERIFICATION_RULE_SELECTOR, 3u, 1u, 2u},  // float.math.unary.f64.selector_u8",
            "UINT32_C(0x000D0008), IREE_VM_BYTECODE_VERIFICATION_RULE_GLOBAL_ORDINAL",
            "VERIFICATION_RULE_LOCAL_BYTES_RANGE_MEMORY_FORMAT, 2u, 4u, 0u},  // stack.load.base_u16",
            "VERIFICATION_RULE_VALUE_REGISTER_FORMAT_RANGE, 1u, 4u, 0u},  // stack.load",
            "VERIFICATION_RULE_PACKED_SELECTORS, 6u, 64u, 3u},  // buffer.atomic.cmpxchg.selector0_u8",
            "VERIFICATION_RULE_PACKED_SELECTORS, 7u, 248u, 1u},  // buffer.atomic.cmpxchg.selector1_u8",
            "VERIFICATION_RULE_ATOMIC_CARRIER_SUPPORTED, 6u, 0u, 0u},  // buffer.atomic.cmpxchg",
            "VERIFICATION_RULE_PACKED_SELECTOR_PAIRS, 6u, 48u, 51u},  // buffer.atomic.cmpxchg",
            "UINT32_C(0x0033001F)",
            "UINT32_C(0x00170003)",
            "UINT32_C(0x00001A1F)",
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("switch (", source)
        self.assertNotIn("iree_status_t", source)
        parameters = []
        rule = _instruction_rules(INTEGER_INSTRUCTIONS[-3], parameters, {})[-1]
        self.assertEqual(
            (rule.kind, *rule[1:4], *parameters[rule.parameter :]),
            ("INTEGER_BITSTREAM_SHAPE", 3, 4, 5, 6, 7, 0, 64),
        )


if __name__ == "__main__":
    unittest.main()
