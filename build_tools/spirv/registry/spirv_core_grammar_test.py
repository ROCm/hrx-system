# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import json
import unittest

from build_tools.spirv.registry.registry_source import RegistrySourceError
from build_tools.spirv.registry.spirv_core_grammar import (
    SpirvOperandCategory,
    SpirvOperandQuantifier,
    SpirvVersion,
    parse_spirv_core_grammar,
)


def _minimal_document() -> dict[str, object]:
    return {
        "copyright": ["Test grammar", ""],
        "magic_number": "0x07230203",
        "major_version": 1,
        "minor_version": 6,
        "revision": 1,
        "instruction_printing_class": [
            {"tag": "Type-Declaration", "heading": "Type declarations"},
            {"tag": "@exclude"},
        ],
        "instructions": [
            {
                "opname": "OpTypeTest",
                "aliases": ["OpTypeTestAlias"],
                "class": "Type-Declaration",
                "opcode": 1,
                "version": "1.0",
                "capabilities": ["TestCapabilityAlias"],
                "extensions": ["SPV_TEST_extension"],
                "operands": [
                    {"kind": "IdResult"},
                    {"kind": "TestEnum", "name": "'Value'", "quantifier": "?"},
                ],
            }
        ],
        "operand_kinds": [
            {
                "category": "ValueEnum",
                "kind": "Capability",
                "enumerants": [
                    {
                        "enumerant": "Matrix",
                        "value": 0,
                        "version": "1.0",
                    },
                    {
                        "enumerant": "Shader",
                        "value": 1,
                        "version": "1.0",
                        "capabilities": ["Matrix"],
                    },
                    {
                        "enumerant": "TestCapability",
                        "aliases": ["TestCapabilityAlias"],
                        "value": 2,
                        "version": "None",
                        "capabilities": ["Shader"],
                        "extensions": ["SPV_TEST_extension"],
                    },
                ],
            },
            {
                "category": "ValueEnum",
                "kind": "TestEnum",
                "enumerants": [
                    {
                        "enumerant": "Value",
                        "value": 0,
                        "version": "1.0",
                    }
                ],
            },
            {
                "category": "BitEnum",
                "kind": "TestMask",
                "enumerants": [
                    {"enumerant": "None", "value": "0x0000"},
                    {
                        "enumerant": "Enabled",
                        "value": "0x0001",
                        "version": "None",
                    },
                ],
            },
            {"category": "Id", "kind": "IdResult", "doc": "Result ID"},
            {
                "category": "Literal",
                "kind": "LiteralInteger",
                "doc": "Integer literal",
            },
            {
                "category": "Composite",
                "kind": "PairIdResultIdResult",
                "bases": ["IdResult", "IdResult"],
            },
        ],
    }


def _parse_document(document: dict[str, object]):
    return parse_spirv_core_grammar(
        json.dumps(document), source="test.core.grammar.json"
    )


class SpirvCoreGrammarTest(unittest.TestCase):
    def test_builds_immutable_direct_indexes_and_capability_closure(self):
        grammar = _parse_document(_minimal_document())

        self.assertEqual(grammar.version, SpirvVersion(1, 6))
        self.assertEqual(grammar.magic_number, 0x07230203)
        self.assertEqual(grammar.copyright, ("Test grammar", ""))
        self.assertEqual(grammar.referenced_extensions, {"SPV_TEST_extension"})

        instruction = grammar.instructions_by_name["OpTypeTest"]
        self.assertIs(grammar.instructions_by_name["OpTypeTestAlias"], instruction)
        self.assertIs(grammar.instructions_by_opcode[1], instruction)
        self.assertEqual(instruction.capabilities, ("TestCapability",))
        self.assertEqual(
            instruction.operands[1].quantifier,
            SpirvOperandQuantifier.OPTIONAL,
        )

        mask_kind = grammar.operand_kinds_by_name["TestMask"]
        self.assertEqual(mask_kind.category, SpirvOperandCategory.BIT_ENUM)
        self.assertIsNone(mask_kind.enumerants_by_name["None"].version)
        self.assertEqual(mask_kind.enumerants_by_name["Enabled"].value, 1)
        self.assertIs(
            grammar.capabilities_by_name["TestCapabilityAlias"],
            grammar.capabilities_by_name["TestCapability"],
        )
        self.assertEqual(
            grammar.capability_direct_implications["TestCapabilityAlias"],
            ("Shader",),
        )
        self.assertEqual(
            grammar.capability_implication_closures["TestCapability"],
            {"Matrix", "Shader"},
        )

        with self.assertRaises(TypeError):
            grammar.instructions_by_name["OpFuture"] = instruction  # type: ignore[index]
        with self.assertRaises(TypeError):
            mask_kind.enumerants_by_value[2] = mask_kind.enumerants[0]  # type: ignore[index]

    def test_rejects_syntax_duplicate_keys_and_nonstandard_constants(self):
        with self.assertRaisesRegex(
            RegistrySourceError,
            r"broken\.json:line 1, column 2:.*property name",
        ):
            parse_spirv_core_grammar("{", source="broken.json")

        with self.assertRaisesRegex(
            RegistrySourceError, "duplicate JSON object field 'revision'"
        ):
            parse_spirv_core_grammar(
                '{"revision": 1, "revision": 2}', source="duplicate.json"
            )

        with self.assertRaisesRegex(
            RegistrySourceError, "non-standard JSON constant 'NaN'"
        ):
            parse_spirv_core_grammar("NaN", source="constant.json")

        with self.assertRaisesRegex(
            RegistrySourceError,
            r"encoded\.json:byte 0: grammar is not valid Unicode text",
        ):
            parse_spirv_core_grammar(b"\xff", source="encoded.json")

    def test_rejects_unknown_or_missing_fields_at_every_object_level(self):
        cases = []

        root_unknown = _minimal_document()
        root_unknown["future"] = True
        cases.append((root_unknown, r"\$: unknown fields: future"))

        instruction_unknown = _minimal_document()
        instruction_unknown["instructions"][0]["future"] = True  # type: ignore[index]
        cases.append(
            (instruction_unknown, r"instructions\[0\]: unknown fields: future")
        )

        operand_unknown = _minimal_document()
        operand_unknown["instructions"][0]["operands"][0]["future"] = True  # type: ignore[index]
        cases.append((operand_unknown, r"operands\[0\]: unknown fields: future"))

        kind_unknown = _minimal_document()
        kind_unknown["operand_kinds"][0]["future"] = True  # type: ignore[index]
        cases.append((kind_unknown, r"operand_kinds\[0\]: unknown fields: future"))

        enumerant_unknown = _minimal_document()
        enumerant_unknown["operand_kinds"][0]["enumerants"][0]["future"] = True  # type: ignore[index]
        cases.append((enumerant_unknown, r"enumerants\[0\]: unknown fields: future"))

        parameter_unknown = _minimal_document()
        parameter_unknown["operand_kinds"][1]["enumerants"][0]["parameters"] = [  # type: ignore[index]
            {"kind": "LiteralInteger", "future": True}
        ]
        cases.append((parameter_unknown, r"parameters\[0\]: unknown fields: future"))

        printing_class_missing = _minimal_document()
        del printing_class_missing["instruction_printing_class"][0]["tag"]  # type: ignore[index]
        cases.append((printing_class_missing, r"missing fields: tag"))

        for document, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RegistrySourceError, message):
                    _parse_document(document)

    def test_rejects_category_value_and_quantifier_shape_mismatches(self):
        cases = []

        unknown_category = _minimal_document()
        unknown_category["operand_kinds"][1]["category"] = "FutureEnum"  # type: ignore[index]
        cases.append((unknown_category, "unsupported operand category"))

        wrong_category_fields = _minimal_document()
        wrong_category_fields["operand_kinds"][3]["enumerants"] = []  # type: ignore[index]
        cases.append((wrong_category_fields, "requires only 'doc'"))

        numeric_bit_value = _minimal_document()
        numeric_bit_value["operand_kinds"][2]["enumerants"][1]["value"] = 1  # type: ignore[index]
        cases.append((numeric_bit_value, "expected a string"))

        string_value_enum = _minimal_document()
        string_value_enum["operand_kinds"][1]["enumerants"][0]["value"] = "0"  # type: ignore[index]
        cases.append((string_value_enum, "expected an integer"))

        bad_quantifier = _minimal_document()
        bad_quantifier["instructions"][0]["operands"][0]["quantifier"] = "+"  # type: ignore[index]
        cases.append((bad_quantifier, "unsupported operand quantifier"))

        for document, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RegistrySourceError, message):
                    _parse_document(document)

    def test_rejects_dangling_internal_references(self):
        cases = []

        unknown_class = _minimal_document()
        unknown_class["instructions"][0]["class"] = "Future"  # type: ignore[index]
        cases.append((unknown_class, "unknown instruction printing class"))

        unknown_operand = _minimal_document()
        unknown_operand["instructions"][0]["operands"][0]["kind"] = "Future"  # type: ignore[index]
        cases.append((unknown_operand, "unknown operand kind 'Future'"))

        unknown_base = _minimal_document()
        unknown_base["operand_kinds"][5]["bases"][0] = "Future"  # type: ignore[index]
        cases.append((unknown_base, "unknown operand kind 'Future'"))

        unknown_parameter = _minimal_document()
        unknown_parameter["operand_kinds"][1]["enumerants"][0]["parameters"] = [  # type: ignore[index]
            {"kind": "Future"}
        ]
        cases.append((unknown_parameter, "unknown operand kind 'Future'"))

        unknown_capability = _minimal_document()
        unknown_capability["instructions"][0]["capabilities"] = ["Future"]  # type: ignore[index]
        cases.append((unknown_capability, "unknown capability 'Future'"))

        repeated_capability_alias = _minimal_document()
        repeated_capability_alias["instructions"][0]["capabilities"] = [  # type: ignore[index]
            "TestCapability",
            "TestCapabilityAlias",
        ]
        cases.append((repeated_capability_alias, "repeated through aliases"))

        for document, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RegistrySourceError, message):
                    _parse_document(document)

    def test_rejects_duplicate_names_aliases_opcodes_and_values(self):
        cases = []

        instruction_name = _minimal_document()
        duplicate_instruction = copy.deepcopy(instruction_name["instructions"][0])  # type: ignore[index]
        duplicate_instruction["opname"] = "OpFuture"
        duplicate_instruction["aliases"] = ["OpTypeTest"]
        duplicate_instruction["opcode"] = 2
        instruction_name["instructions"].append(duplicate_instruction)  # type: ignore[union-attr]
        cases.append((instruction_name, "instruction name 'OpTypeTest' collides"))

        instruction_opcode = _minimal_document()
        duplicate_instruction = copy.deepcopy(instruction_opcode["instructions"][0])  # type: ignore[index]
        duplicate_instruction["opname"] = "OpFuture"
        duplicate_instruction["aliases"] = ["OpFutureAlias"]
        instruction_opcode["instructions"].append(duplicate_instruction)  # type: ignore[union-attr]
        cases.append((instruction_opcode, "opcode 1 collides"))

        kind_name = _minimal_document()
        duplicate_kind = copy.deepcopy(kind_name["operand_kinds"][3])  # type: ignore[index]
        kind_name["operand_kinds"].append(duplicate_kind)  # type: ignore[union-attr]
        cases.append((kind_name, "operand kind 'IdResult' collides"))

        enumerant_name = _minimal_document()
        enumerant_name["operand_kinds"][1]["enumerants"].append(  # type: ignore[index]
            {
                "enumerant": "Other",
                "aliases": ["Value"],
                "value": 1,
                "version": "1.0",
            }
        )
        cases.append((enumerant_name, "enumerant name 'Value' collides"))

        enumerant_value = _minimal_document()
        enumerant_value["operand_kinds"][1]["enumerants"].append(  # type: ignore[index]
            {"enumerant": "Other", "value": 0, "version": "1.0"}
        )
        cases.append((enumerant_value, "value 0 collides"))

        for document, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RegistrySourceError, message):
                    _parse_document(document)

    def test_rejects_noncanonical_instruction_and_enumerant_order(self):
        instruction_order = _minimal_document()
        instruction_order["instructions"][0]["opcode"] = 2  # type: ignore[index]
        earlier_instruction = copy.deepcopy(instruction_order["instructions"][0])  # type: ignore[index]
        earlier_instruction["opname"] = "OpEarlier"
        earlier_instruction["aliases"] = ["OpEarlierAlias"]
        earlier_instruction["opcode"] = 1
        instruction_order["instructions"].append(earlier_instruction)  # type: ignore[union-attr]
        with self.assertRaisesRegex(
            RegistrySourceError, "not in increasing opcode order"
        ):
            _parse_document(instruction_order)

        enumerant_order = _minimal_document()
        enumerant_order["operand_kinds"][1]["enumerants"][0]["value"] = 1  # type: ignore[index]
        enumerant_order["operand_kinds"][1]["enumerants"].append(  # type: ignore[index]
            {"enumerant": "Earlier", "value": 0, "version": "1.0"}
        )
        with self.assertRaisesRegex(
            RegistrySourceError, "not in increasing value order"
        ):
            _parse_document(enumerant_order)

    def test_rejects_invalid_versions_and_capability_cycles(self):
        cases = []

        future_version = _minimal_document()
        future_version["instructions"][0]["version"] = "1.7"  # type: ignore[index]
        cases.append((future_version, "exceeds grammar version 1.6"))

        invalid_version = _minimal_document()
        invalid_version["instructions"][0]["version"] = "0.9"  # type: ignore[index]
        cases.append((invalid_version, "invalid SPIR-V version '0.9'"))

        noncanonical_version = _minimal_document()
        noncanonical_version["instructions"][0]["version"] = "1.00"  # type: ignore[index]
        cases.append((noncanonical_version, "invalid SPIR-V version '1.00'"))

        reversed_last_version = _minimal_document()
        reversed_last_version["instructions"][0]["version"] = "1.1"  # type: ignore[index]
        reversed_last_version["instructions"][0]["lastVersion"] = "1.0"  # type: ignore[index]
        cases.append((reversed_last_version, "precedes version 1.1"))

        missing_enumerant_version = _minimal_document()
        del missing_enumerant_version["operand_kinds"][1]["enumerants"][0][  # type: ignore[index]
            "version"
        ]
        cases.append((missing_enumerant_version, "missing fields: version"))

        versioned_provisional = _minimal_document()
        versioned_provisional["instructions"][0]["provisional"] = True  # type: ignore[index]
        cases.append(
            (versioned_provisional, "provisional definitions must have version 'None'")
        )

        false_provisional = _minimal_document()
        false_provisional["instructions"][0]["provisional"] = False  # type: ignore[index]
        cases.append((false_provisional, "must be the boolean true"))

        unguarded_instruction = _minimal_document()
        del unguarded_instruction["instructions"][0]["capabilities"]  # type: ignore[index]
        del unguarded_instruction["instructions"][0]["extensions"]  # type: ignore[index]
        unguarded_instruction["instructions"][0]["version"] = "None"  # type: ignore[index]
        cases.append(
            (
                unguarded_instruction,
                "outside all core versions requires a capability or extension guard",
            )
        )

        unguarded_capability = _minimal_document()
        del unguarded_capability["operand_kinds"][0]["enumerants"][2][  # type: ignore[index]
            "extensions"
        ]
        cases.append(
            (
                unguarded_capability,
                "outside all core versions requires an extension guard",
            )
        )

        capability_cycle = _minimal_document()
        capability_cycle["operand_kinds"][0]["enumerants"][0]["capabilities"] = [  # type: ignore[index]
            "TestCapability"
        ]
        cases.append((capability_cycle, "capability dependency cycle"))

        for document, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RegistrySourceError, message):
                    _parse_document(document)


if __name__ == "__main__":
    unittest.main()
