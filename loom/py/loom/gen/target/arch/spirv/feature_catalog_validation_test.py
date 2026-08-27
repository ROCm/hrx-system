# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

from build_tools.spirv.registry.spirv_core_grammar import (
    SpirvCoreGrammar,
    load_spirv_core_grammar,
)
from build_tools.spirv.registry.vulkan_spirv_availability import (
    VulkanSpirvAvailabilityRegistry,
    load_vulkan_spirv_availability,
)

from loom.gen.target.arch.spirv.feature_catalog_validation import (
    validate_feature_catalog_sources,
)
from loom.gen.target.arch.spirv.spirv_features import generate_tables, main
from loom.target.arch.spirv.features import FEATURE_ATOMS, FeatureAtom


@contextmanager
def _raises_value_error(pattern: str) -> Iterator[None]:
    try:
        yield
    except ValueError as exc:
        if not re.search(pattern, str(exc)):
            raise AssertionError(f"{exc!s} did not match {pattern!r}") from exc
    else:
        raise AssertionError(f"expected ValueError matching {pattern!r}")


class FeatureCatalogValidationTest(unittest.TestCase):
    isa_path: ClassVar[Path]
    isa_header: ClassVar[str]
    grammar_path: ClassVar[Path]
    grammar: ClassVar[SpirvCoreGrammar]
    vulkan_registry_path: ClassVar[Path]
    vulkan_registry: ClassVar[VulkanSpirvAvailabilityRegistry]

    @classmethod
    def configure_sources(
        cls,
        isa_path: Path,
        grammar_path: Path,
        vulkan_registry_path: Path,
    ) -> None:
        cls.isa_path = isa_path
        cls.isa_header = isa_path.read_text(encoding="utf-8")
        cls.grammar_path = grammar_path
        cls.grammar = load_spirv_core_grammar(grammar_path)
        cls.vulkan_registry_path = vulkan_registry_path
        cls.vulkan_registry = load_vulkan_spirv_availability(vulkan_registry_path)

    def validate(
        self,
        *,
        atoms: tuple[FeatureAtom, ...] = FEATURE_ATOMS,
        isa_header: str | None = None,
        vulkan_registry: VulkanSpirvAvailabilityRegistry | None = None,
    ) -> None:
        validate_feature_catalog_sources(
            atoms=atoms,
            isa_header=self.isa_header if isa_header is None else isa_header,
            isa_source=str(self.isa_path),
            grammar=self.grammar,
            vulkan_registry=(self.vulkan_registry if vulkan_registry is None else vulkan_registry),
        )

    def replace_atom(self, key: str, **changes: object) -> tuple[FeatureAtom, ...]:
        return tuple(replace(atom, **changes) if atom.key == key else atom for atom in FEATURE_ATOMS)

    def test_current_catalog_satisfies_pinned_sources(self) -> None:
        self.validate()

    def test_generator_validation_preserves_emitted_tables(self) -> None:
        expected = generate_tables()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "features_tables.inl"
            result = main(
                [
                    f"--isa-header={self.isa_path}",
                    f"--spirv-grammar={self.grammar_path}",
                    f"--vulkan-registry={self.vulkan_registry_path}",
                    f"--tables={output_path}",
                ]
            )
            assert result == 0
            assert output_path.read_text(encoding="utf-8") == expected

    def test_alias_spelling_is_checked_against_its_canonical_opcode(self) -> None:
        bad_header = self.isa_header.replace(
            "LOOM_SPIRV_OP_TYPE_COOPERATIVE_VECTOR_NV = 5288,",
            "LOOM_SPIRV_OP_TYPE_COOPERATIVE_VECTOR_NV = 5289,",
        )
        with _raises_value_error(r"cooperative_vector_nv.*OpTypeVectorIdEXT.*encodes 5288"):
            self.validate(isa_header=bad_header)

    def test_each_operand_enumerant_category_checks_its_wire_value(self) -> None:
        cases = (
            ("LOOM_SPIRV_CAPABILITY_INT16", 22),
            ("LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER", 5349),
            ("LOOM_SPIRV_DECORATION_RESTRICT_POINTER", 5355),
            ("LOOM_SPIRV_ADDRESSING_MODEL_PHYSICAL_STORAGE_BUFFER64", 5348),
            ("LOOM_SPIRV_MEMORY_MODEL_VULKAN", 3),
        )
        for symbol, expected_value in cases:
            with self.subTest(symbol=symbol):
                bad_header = self.isa_header.replace(
                    f"{symbol} = {expected_value},",
                    f"{symbol} = {expected_value + 1},",
                )
                with _raises_value_error(rf"{symbol}.*encodes {expected_value}"):
                    self.validate(isa_header=bad_header)

    def test_unknown_isa_symbol_fails_at_the_grammar_boundary(self) -> None:
        atoms = self.replace_atom(
            "int16",
            capabilities=("LOOM_SPIRV_CAPABILITY_NOT_A_CAPABILITY",),
        )
        with _raises_value_error(r"int16.*NOT_A_CAPABILITY.*no primary or alias spelling.*grammar"):
            self.validate(atoms=atoms)

    def test_nonliteral_demanded_isa_assignment_is_rejected(self) -> None:
        bad_header = self.isa_header.replace(
            "LOOM_SPIRV_CAPABILITY_INT16 = 22,",
            "LOOM_SPIRV_CAPABILITY_INT16 = LOOM_SPIRV_CAPABILITY_INT8,",
        )
        with _raises_value_error(r"CAPABILITY_INT16.*literal integer enum assignment"):
            self.validate(isa_header=bad_header)

    def test_missing_demanded_isa_assignment_is_rejected(self) -> None:
        bad_header = self.isa_header.replace(
            "  LOOM_SPIRV_CAPABILITY_INT16 = 22,\n",
            "",
        )
        with _raises_value_error(r"int16.*CAPABILITY_INT16.*no literal assignment"):
            self.validate(isa_header=bad_header)

    def test_duplicate_demanded_isa_assignment_is_rejected(self) -> None:
        bad_header = self.isa_header + "\n  LOOM_SPIRV_CAPABILITY_INT16 = 22,\n"
        with _raises_value_error(r"CAPABILITY_INT16.*duplicates"):
            self.validate(isa_header=bad_header)

    def test_unreferenced_isa_assignments_stay_outside_the_demand_set(self) -> None:
        header_with_unparsed_unreferenced_row = self.isa_header.replace(
            "LOOM_SPIRV_OP_NOP = 0,",
            "LOOM_SPIRV_OP_NOP = LOOM_SPIRV_OP_MAX,",
        )
        self.validate(isa_header=header_with_unparsed_unreferenced_row)

    def test_minimum_version_cannot_exceed_the_pinned_grammar(self) -> None:
        atoms = self.replace_atom(
            "int16",
            minimum_spirv_version=0x00010700,
        )
        with _raises_value_error(r"int16.*version 1.7.*grammar.*version 1.6"):
            self.validate(atoms=atoms)

    def test_minimum_version_must_be_a_spirv_version_word(self) -> None:
        atoms = self.replace_atom(
            "int16",
            minimum_spirv_version=0x00010001,
        )
        with _raises_value_error(r"int16.*malformed minimum version"):
            self.validate(atoms=atoms)

    def test_capability_requires_a_version_or_extension_route(self) -> None:
        atoms = self.replace_atom(
            "storage_buffer_8bit_access",
            extensions=(),
        )
        with _raises_value_error(
            r"storage_buffer_8bit_access.*STORAGE_BUFFER8_BIT_ACCESS.*"
            r"SPV_KHR_8bit_storage",
        ):
            self.validate(atoms=atoms)

    def test_instruction_requires_its_capability_guard(self) -> None:
        atoms = self.replace_atom(
            "cooperative_matrix_khr",
            capabilities=(),
        )
        with _raises_value_error(
            r"cooperative_matrix_khr.*OP_TYPE_COOPERATIVE_MATRIX_KHR.*"
            r"CooperativeMatrixKHR",
        ):
            self.validate(atoms=atoms)

    def test_transitive_modeled_capability_implication_requires_provider(self) -> None:
        atoms = self.replace_atom(
            "bfloat16_cooperative_matrix_khr",
            required=("vulkan_shader", "bfloat16_type_khr"),
        )
        with _raises_value_error(
            r"bfloat16_cooperative_matrix_khr.*CooperativeMatrixKHR.*"
            r"cooperative_matrix_khr.*outside",
        ):
            self.validate(atoms=atoms)

    def test_catalog_extension_requires_a_vulkan_mapping(self) -> None:
        mappings = dict(self.vulkan_registry.spirv_extensions_by_name)
        del mappings["SPV_KHR_bfloat16"]
        registry = replace(
            self.vulkan_registry,
            spirv_extensions_by_name=MappingProxyType(mappings),
        )
        with _raises_value_error(r"bfloat16_type_khr.*SPV_KHR_bfloat16.*no Vulkan enable mapping"):
            self.validate(vulkan_registry=registry)

    def test_catalog_capability_requires_a_vulkan_mapping(self) -> None:
        mappings = dict(self.vulkan_registry.spirv_capabilities_by_name)
        del mappings["Int16"]
        registry = replace(
            self.vulkan_registry,
            spirv_capabilities_by_name=MappingProxyType(mappings),
        )
        with _raises_value_error(r"int16.*Int16.*no Vulkan enable mapping"):
            self.validate(vulkan_registry=registry)


def _paths_from_argv(argv: list[str]) -> tuple[Path, Path, Path]:
    if len(argv) != 4:
        raise SystemExit(f"usage: {argv[0]} <isa.h> <spirv.core.grammar.json> <vk.xml>")
    paths = tuple(Path(value) for value in argv[1:])
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"catalog validation input does not exist: {path}")
    return paths


def _main(argv: list[str]) -> None:
    FeatureCatalogValidationTest.configure_sources(*_paths_from_argv(argv))
    unittest.main(argv=[argv[0]])


if __name__ == "__main__":
    _main(sys.argv)
