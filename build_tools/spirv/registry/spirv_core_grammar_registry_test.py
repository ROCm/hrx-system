# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import unittest
from collections import Counter
from pathlib import Path
from typing import cast

from build_tools.spirv.registry.spirv_core_grammar import (
    SpirvOperandCategory,
    SpirvVersion,
    load_spirv_core_grammar,
)

_GRAMMAR_PATH: Path | None = None


class SpirvCoreGrammarRegistryTest(unittest.TestCase):
    def test_imports_complete_pinned_core_grammar(self):
        self.assertIsNotNone(_GRAMMAR_PATH)
        grammar_path = cast(Path, _GRAMMAR_PATH)
        grammar = load_spirv_core_grammar(grammar_path)
        enumerants = tuple(
            enumerant for kind in grammar.operand_kinds for enumerant in kind.enumerants
        )

        self.assertEqual(grammar.magic_number, 0x07230203)
        self.assertEqual(grammar.version, SpirvVersion(1, 6))
        self.assertEqual(grammar.revision, 7)
        self.assertEqual(len(grammar.printing_classes), 28)
        self.assertEqual(len(grammar.printing_classes_by_tag), 28)
        self.assertEqual(len(grammar.instructions), 875)
        self.assertEqual(len(grammar.instructions_by_name), 930)
        self.assertEqual(len(grammar.instructions_by_opcode), 875)
        self.assertEqual(sum(len(item.operands) for item in grammar.instructions), 3704)
        self.assertEqual(
            sum(item.provisional for item in grammar.instructions),
            20,
        )
        self.assertEqual(
            sum(item.last_version is not None for item in grammar.instructions),
            2,
        )
        self.assertEqual(len(grammar.operand_kinds), 72)
        self.assertEqual(len(grammar.operand_kinds_by_name), 72)
        self.assertEqual(
            Counter(kind.category for kind in grammar.operand_kinds),
            {
                SpirvOperandCategory.BIT_ENUM: 16,
                SpirvOperandCategory.VALUE_ENUM: 42,
                SpirvOperandCategory.ID: 5,
                SpirvOperandCategory.LITERAL: 6,
                SpirvOperandCategory.COMPOSITE: 3,
            },
        )
        self.assertEqual(len(enumerants), 1107)
        self.assertEqual(
            sum(len(kind.enumerants_by_name) for kind in grammar.operand_kinds),
            1285,
        )
        self.assertEqual(
            sum(len(kind.enumerants_by_value) for kind in grammar.operand_kinds),
            1107,
        )
        self.assertEqual(sum(len(item.parameters) for item in enumerants), 187)
        self.assertEqual(sum(item.provisional for item in enumerants), 25)
        self.assertEqual(
            sum(item.last_version is not None for item in enumerants),
            1,
        )
        self.assertEqual(len(grammar.referenced_extensions), 194)
        self.assertEqual(len(grammar.capabilities_by_name), 353)
        self.assertEqual(len(grammar.capability_direct_implications), 353)
        self.assertEqual(len(grammar.capability_implication_closures), 353)

        demote = grammar.instructions_by_name["OpDemoteToHelperInvocation"]
        self.assertIs(
            grammar.instructions_by_name["OpDemoteToHelperInvocationEXT"],
            demote,
        )
        self.assertEqual(demote.opcode, 5380)

        storage_classes = grammar.operand_kinds_by_name["StorageClass"]
        physical_storage = storage_classes.enumerants_by_name["PhysicalStorageBuffer"]
        self.assertIs(
            storage_classes.enumerants_by_name["PhysicalStorageBufferEXT"],
            physical_storage,
        )
        self.assertEqual(physical_storage.value, 5349)

        matrix_operands = grammar.operand_kinds_by_name["CooperativeMatrixOperands"]
        self.assertEqual(matrix_operands.category, SpirvOperandCategory.BIT_ENUM)
        self.assertEqual(
            matrix_operands.enumerants_by_name["MatrixASignedComponentsKHR"].value,
            1,
        )

        bfloat16 = grammar.capabilities_by_name["BFloat16CooperativeMatrixKHR"]
        self.assertEqual(bfloat16.value, 5118)
        self.assertEqual(
            bfloat16.capabilities,
            ("BFloat16TypeKHR", "CooperativeMatrixKHR"),
        )
        self.assertEqual(
            grammar.capability_implication_closures[bfloat16.name],
            {"BFloat16TypeKHR", "CooperativeMatrixKHR"},
        )


def _grammar_path_from_argv(argv: list[str]) -> Path:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} <spirv.core.grammar.json>")
    grammar_path = Path(argv[1])
    if not grammar_path.is_file():
        raise SystemExit(f"SPIR-V core grammar does not exist: {grammar_path}")
    return grammar_path


if __name__ == "__main__":
    _GRAMMAR_PATH = _grammar_path_from_argv(sys.argv)
    unittest.main(argv=[sys.argv[0]])
