# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests the deterministic purpose-specific runtime projections."""

import unittest

from iree.vm.bytecode.spec.generate import generate_outputs


class ProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.outputs = generate_outputs()

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
        self.assertTrue(all(self.outputs.values()))

    def test_wire_layout_has_one_assertion_translation_unit(self) -> None:
        self.assertIn(
            "typedef struct iree_vm_bytecode_v0_image_header_t",
            self.outputs["wire_module_header"],
        )
        self.assertIn(
            "typedef struct iree_vm_bytecode_control_switch_t",
            self.outputs["wire_core_header"],
        )
        self.assertNotIn("static_assert", self.outputs["wire_module_header"])
        self.assertNotIn("static_assert", self.outputs["wire_core_header"])
        assertions = self.outputs["wire_assertions_source"]
        self.assertIn("sizeof(iree_vm_bytecode_v0_image_header_t) == 16u", assertions)
        self.assertIn(
            "offsetof(iree_vm_bytecode_control_switch_t, target_base_u32) == 4u",
            assertions,
        )

    def test_verification_is_data_not_control_flow(self) -> None:
        source = self.outputs["verification_source"]
        self.assertIn("iree_vm_bytecode_instruction_verification[256]", source)
        self.assertIn("IREE_VM_BYTECODE_VERIFICATION_RULE_CONTROL_TARGET_S16", source)
        self.assertIn("IREE_VM_BYTECODE_VERIFICATION_RULE_SIGNATURE_DESCRIPTOR", source)
        self.assertIn("IREE_VM_BYTECODE_CONTROL_FLOW_CONDITIONAL_BRANCH", source)
        self.assertNotIn("switch (", source)
        self.assertNotIn("iree_status_t", source)

    def test_tooling_uses_one_byte_length_strings(self) -> None:
        data = self.outputs["tooling_data"]
        self.assertIn('"\\x0d" "control.block"', data)
        self.assertIn("uint16_t iree_vm_bytecode_instruction_name_offsets[256]", data)
        self.assertNotIn("iree_string_view_t", data)

    def test_documentation_carries_normative_surfaces(self) -> None:
        markdown = self.outputs["documentation"]
        self.assertIn("## Module records", markdown)
        self.assertIn("#### `integer.add.i32`", markdown)
        self.assertIn("#### `control.yield.s32`", markdown)
        self.assertIn("#### Preconditions", markdown)
        self.assertIn("#### Failures", markdown)
        self.assertIn("#### Ownership", markdown)
        self.assertIn("#### Reference pseudocode", markdown)


if __name__ == "__main__":
    unittest.main()
