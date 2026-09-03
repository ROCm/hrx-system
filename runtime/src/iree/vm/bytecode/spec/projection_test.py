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


class ProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.outputs = generate_outputs()

    def assert_output_contains(self, output_name: str, *fragments: str) -> str:
        for fragment in fragments:
            self.assertIn(fragment, self.outputs[output_name])
        return self.outputs[output_name]

    def test_output_family_is_closed(self) -> None:
        self.assertEqual(
            set(self.outputs),
            {
                "wire_module_header",
                "wire_core_header",
                "wire_assertions_source",
                "verification_source",
                "tooling_data",
                "documentation",
            },
        )

    def test_wire_layout_has_one_assertion_translation_unit(self) -> None:
        module_header = self.assert_output_contains(
            "wire_module_header",
            "typedef struct iree_vm_bytecode_v0_image_header_t",
        )
        core_header = self.assert_output_contains(
            "wire_core_header",
            "typedef struct iree_vm_bytecode_stack_pack_i64_u32_x8_t",
        )
        self.assertNotIn("static_assert", module_header)
        self.assertNotIn("static_assert", core_header)
        self.assert_output_contains(
            "wire_assertions_source",
            "sizeof(iree_vm_bytecode_stack_pack_i64_u32_x8_t) == 36u",
        )

    def test_verification_is_data_not_control_flow(self) -> None:
        source = self.assert_output_contains(
            "verification_source",
            "iree_vm_bytecode_instruction_verification[256]",
            "VERIFICATION_RULE_SELECTOR, 3u, 1u, 2u},  // float.math.unary.f64.selector_u8",
            "UINT32_C(0x000D0008), IREE_VM_BYTECODE_VERIFICATION_RULE_GLOBAL_ORDINAL",
            "VERIFICATION_RULE_LOCAL_BYTES_RANGE_MEMORY_FORMAT, 2u, 4u, 0u},  // stack.load.base_u16",
            "VERIFICATION_RULE_VALUE_REGISTER_FORMAT_RANGE, 1u, 4u, 0u},  // stack.load",
        )
        self.assertNotIn("switch (", source)
        self.assertNotIn("iree_status_t", source)
        parameters = []
        rule = _instruction_rules(INTEGER_INSTRUCTIONS[-3], parameters, {})[-1]
        self.assertEqual(
            (rule.kind, *rule[1:4], *parameters[rule.parameter :]),
            ("INTEGER_BITSTREAM_SHAPE", 3, 4, 5, 6, 7, 0, 64),
        )

    def test_tooling_uses_one_byte_length_strings(self) -> None:
        data = self.assert_output_contains(
            "tooling_data",
            '"\\x1b" "conversion.float.to.integer"',
            "uint16_t iree_vm_bytecode_instruction_name_offsets[256]",
        )
        self.assertNotIn("iree_string_view_t", data)

    def test_documentation_carries_normative_surfaces(self) -> None:
        self.assert_output_contains(
            "documentation",
            "## Module container",
            "#### `metadata_value_type`",
            "#### Structural verification obligations",
            "#### `stack.load.indexed`",
            "#### `control.yield.s32`",
            "## Core selector domains",
            "#### `stack.copy.rodata`",
            "#### `stack.pack.i64.u32.x8`",
            "#### Preconditions",
            "#### Failures",
            "#### Ownership",
            "#### Reference pseudocode",
        )


if __name__ == "__main__":
    unittest.main()
