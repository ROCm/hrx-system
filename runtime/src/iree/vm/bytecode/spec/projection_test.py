#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercises generated specification projections and their ownership boundary."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import re
import sys
import tempfile
import unittest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))

from execution import EXECUTABLE_INSTRUCTIONS  # noqa: E402
from generate import (  # noqa: E402
    generated_documentation_outputs,
    generated_table_outputs,
    generated_wire_outputs,
    write_or_check_documentation,
)
from model import (  # noqa: E402
    NumericTable,
    NumericValue,
)
from model.isa import Instruction, InstructionFamily  # noqa: E402
from model.isa.specification import ISA_SPECIFICATION  # noqa: E402
from model.module import (  # noqa: E402
    Section,
    ValidationObligation,
)
from model.module.v0 import (  # noqa: E402
    MODULE_SPECIFICATION,
)
from model.schema import WireRecord  # noqa: E402
from render import (  # noqa: E402
    render_projection_json,
)


def latest_projection(specification):
    entity_ids = tuple(entity.entity_id for entity in specification.entities)
    return specification.project(specification.derive_projection_versions(entity_ids))


class ProjectionTest(unittest.TestCase):
    def test_version_zero_surface_is_intentionally_complete(self) -> None:
        module_entities = MODULE_SPECIFICATION.entities
        isa_entities = ISA_SPECIFICATION.entities
        instructions = tuple(
            entity for entity in isa_entities if isinstance(entity, Instruction)
        )

        self.assertEqual(
            sum(isinstance(entity, WireRecord) for entity in module_entities),
            34,
        )
        self.assertEqual(
            sum(isinstance(entity, Section) for entity in module_entities),
            13,
        )
        self.assertEqual(
            sum(isinstance(entity, ValidationObligation) for entity in module_entities),
            56,
        )
        self.assertEqual(
            sum(isinstance(entity, NumericValue) for entity in module_entities),
            30,
        )
        self.assertEqual(len(instructions), 217)
        self.assertEqual(
            sum(instruction.since.domain == "core" for instruction in instructions),
            183,
        )
        self.assertEqual(
            sum(instruction.since.domain == "hal" for instruction in instructions),
            34,
        )
        self.assertEqual(
            sum(isinstance(entity, InstructionFamily) for entity in isa_entities),
            17,
        )

    def test_runtime_execution_covers_the_complete_core_page(self) -> None:
        core_instructions = {
            entity.mnemonic
            for entity in ISA_SPECIFICATION.entities
            if isinstance(entity, Instruction) and entity.since.domain == "core"
        }
        executable_instructions = tuple(
            instruction.mnemonic for instruction in EXECUTABLE_INSTRUCTIONS
        )

        self.assertEqual(
            len(executable_instructions), len(set(executable_instructions))
        )
        self.assertEqual(set(executable_instructions), core_instructions)

    def test_every_selector_has_normative_meaning_and_a_consumer(self) -> None:
        tables = tuple(
            entity
            for entity in ISA_SPECIFICATION.entities
            if isinstance(entity, NumericTable)
            and entity.table_kind.value == "selector"
        )
        values = tuple(
            entity
            for entity in ISA_SPECIFICATION.entities
            if isinstance(entity, NumericValue)
        )
        instructions = tuple(
            entity
            for entity in ISA_SPECIFICATION.entities
            if isinstance(entity, Instruction)
        )
        consumed_entity_ids = {
            entity_id
            for instruction in instructions
            for entity_id in instruction.referenced_entity_ids()
        }

        self.assertEqual(len(tables), 27)
        self.assertEqual(len(values), 215)
        for table in tables:
            self.assertNotIn("Closed selector domain", table.summary)
            self.assertIn(table.entity_id, consumed_entity_ids)
        for value in values:
            self.assertNotRegex(value.summary, rf"^Selects {re.escape(value.name)}")

    def test_reconciled_execution_contracts_are_explicit(self) -> None:
        isa_entities = ISA_SPECIFICATION.entity_map()
        module_entities = MODULE_SPECIFICATION.entity_map()

        machine_text = isa_entities["core.contract.machine"].normative_text
        self.assertIn("entry bits are unspecified", machine_text)
        self.assertIn(
            "capability-bearing function regions begin canonical null",
            machine_text,
        )
        self.assertIn("0xF0..0xFD select architectural extension pages", machine_text)
        self.assertIn("second-byte local opcode is in 0x01..0xFF", machine_text)
        self.assertIn("function pointer is non-null", machine_text)

        stack_text = isa_entities["core.family.stack"].normative_text
        self.assertIn("entry contents are unspecified", stack_text)
        self.assertNotIn("zero-initialized local byte", stack_text)

        float_text = isa_entities["core.family.float"].normative_text
        self.assertIn("every start or resume drive segment", float_text)
        self.assertIn("completion, suspension, and failure", float_text)

        buffer_text = isa_entities["core.family.buffer"].normative_text
        self.assertIn("immutable host-representable length", buffer_text)
        self.assertIn("current READ/WRITE access mask", buffer_text)
        self.assertNotIn(
            "immutable host-representable length and READ/WRITE", buffer_text
        )
        subspan = isa_entities["core.instruction.buffer.subspan"]
        self.assertNotIn(
            "permission_denied",
            {failure.status for failure in subspan.semantics.failures},
        )
        self.assertIn("check_buffer_open(source)", subspan.semantics.pseudocode)

        control_text = isa_entities["core.family.control"].normative_text
        self.assertIn("target module", control_text)
        self.assertIn("owns any frame it pushes", control_text)
        yield_instruction = isa_entities["core.instruction.control.yield.s32"]
        self.assertIn("_if_nonnull", yield_instruction.semantics.pseudocode)
        await_instruction = isa_entities["hal.instruction.semaphore.await"]
        self.assertIn(
            "current_record_persistent_state_or_null",
            await_instruction.semantics.pseudocode,
        )
        self.assertIn(
            "provider_resume_or_cancel(state)",
            await_instruction.semantics.pseudocode,
        )

        lifecycle_text = module_entities[
            "core.module.contract.verification_lifetime"
        ].normative_text
        self.assertIn("caller-supplied nonempty generic module name", lifecycle_text)
        self.assertIn("no public lifecycle", lifecycle_text)
        self.assertIn(
            "linked libraries have no implicit initialization", lifecycle_text
        )
        self.assertIn("returns no results", lifecycle_text)

        format_text = module_entities["core.module.format"].normative_text
        self.assertIn("0xF0..0xFD name architectural extension pages", format_text)
        self.assertIn("Every other nonzero authority is invalid", format_text)

        core_opcodes = generated_wire_outputs()["wire/core/opcodes.h"]
        self.assertIn(
            "IREE_VM_ISA_ARCHITECTURAL_EXTENSION_PAGE_MIN = 0xF0,",
            core_opcodes,
        )
        self.assertIn(
            "IREE_VM_ISA_ARCHITECTURAL_EXTENSION_PAGE_MAX = 0xFD,",
            core_opcodes,
        )

    def test_generated_file_set_is_deterministic_and_sharded(self) -> None:
        wire_outputs = generated_wire_outputs()
        documentation_outputs = generated_documentation_outputs()
        table_outputs = generated_table_outputs()

        self.assertEqual(wire_outputs, generated_wire_outputs())
        self.assertEqual(documentation_outputs, generated_documentation_outputs())
        self.assertEqual(table_outputs, generated_table_outputs())
        self.assertEqual(len(wire_outputs), 24)
        self.assertEqual(len(documentation_outputs), 21)
        self.assertEqual(
            set(table_outputs),
            {
                "execution_tables.inl",
                "isa_tables.c.inc",
                "module_tables.c.inc",
                "module_validation_obligations.inl",
            },
        )
        self.assertEqual(
            {path for path in documentation_outputs if path.endswith("/index.md")},
            {"isa/core/index.md", "isa/hal/index.md"},
        )
        self.assertEqual(
            {path for path in documentation_outputs if not path.startswith("isa/")},
            {"index.md", "module-format.md"},
        )
        self.assertEqual(
            {path for path in wire_outputs if path.endswith(".c")},
            {
                "wire/assertions.c",
                "wire/core/assertions.c",
                "wire/hal/assertions.c",
            },
        )
        self.assertTrue(all(path.endswith((".c", ".h")) for path in wire_outputs))
        self.assertTrue(all(path.endswith(".md") for path in documentation_outputs))
        self.assertIn(
            "Projection: `core=0.0`.",
            documentation_outputs["isa/core/index.md"],
        )
        self.assertNotIn("hal=", documentation_outputs["isa/core/index.md"])
        self.assertIn(
            "Projection: `core=0.0, hal=0.0`.",
            documentation_outputs["isa/hal/index.md"],
        )
        for path, contents in (
            wire_outputs | table_outputs | documentation_outputs
        ).items():
            self.assertNotIn(".notes/", contents, path)
            self.assertNotIn("legacy schema", contents.lower(), path)
            self.assertIn("GENERATED FILE: DO NOT EDIT.", contents, path)
            if path.endswith((".c", ".h", ".inc", ".inl")):
                self.assertEqual(contents.count("// clang-format off"), 1, path)
                self.assertEqual(contents.count("// clang-format on"), 1, path)

        execution_table = table_outputs["execution_tables.inl"]
        isa_table = table_outputs["isa_tables.c.inc"]
        module_table = table_outputs["module_tables.c.inc"]
        validation_obligations = table_outputs["module_validation_obligations.inl"]
        self.assertNotIn("IREE_SVL", isa_table + module_table)
        self.assertIn("iree_vm_bytecode_tooling_isa_string_table", isa_table)
        self.assertIn("iree_vm_bytecode_tooling_opcode_maps[][256]", isa_table)
        self.assertNotIn("iree_vm_bytecode_tooling_sections", isa_table)
        self.assertIn("iree_vm_bytecode_tooling_module_string_table", module_table)
        self.assertIn("iree_vm_bytecode_tooling_sections", module_table)
        self.assertNotIn("iree_vm_bytecode_tooling_opcode_maps", module_table)
        self.assertEqual(
            validation_obligations.count(
                "IREE_VM_BYTECODE_MODULE_VALIDATION_OBLIGATION("
            ),
            56,
        )
        self.assertEqual(
            execution_table.count("IREE_VM_BYTECODE_EXECUTION_INFO_ROW("), 256
        )
        self.assertIn("OP(INTEGER_XOR_I64, integer_xor_i64)", execution_table)
        self.assertIn("OP(FLOAT_MATH_UNARY_F32, float_math_unary_f32)", execution_table)
        self.assertIn(
            "OP(FLOAT_MATH_BINARY_F64, float_math_binary_f64)", execution_table
        )
        self.assertIn(
            "OP(FLOAT_MATH_TERNARY_F32, float_math_ternary_f32)", execution_table
        )

    def test_every_entity_has_exactly_one_markdown_anchor(self) -> None:
        module_projection = latest_projection(MODULE_SPECIFICATION)
        isa_projection = latest_projection(ISA_SPECIFICATION)
        outputs = generated_documentation_outputs()
        module_document = outputs["module-format.md"]
        isa_documents = {
            path: contents
            for path, contents in outputs.items()
            if path.startswith("isa/")
        }

        for entity in module_projection.entities:
            anchor = f'<a id="{entity.normative_anchor}"></a>'
            self.assertEqual(module_document.count(anchor), 1, entity.entity_id)
        for entity in isa_projection.entities:
            anchor = f'<a id="{entity.normative_anchor}"></a>'
            count = sum(document.count(anchor) for document in isa_documents.values())
            self.assertEqual(count, 1, entity.entity_id)

    def test_documentation_write_detects_nested_output_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = pathlib.Path(temporary_directory)
            documentation_directory = output_directory / "documentation"
            self.assertTrue(
                write_or_check_documentation(
                    documentation_directory,
                    check_only=False,
                )
            )
            self.assertTrue(
                write_or_check_documentation(
                    documentation_directory,
                    check_only=True,
                )
            )
            documentation_extra_path = documentation_directory / "unexpected.md"
            documentation_extra_path.write_text("unexpected\n", encoding="utf-8")
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                self.assertFalse(
                    write_or_check_documentation(
                        documentation_directory,
                        check_only=True,
                    )
                )
            self.assertIn("unexpected generated file", errors.getvalue())

    def test_generated_c_has_unique_guards_and_reviewable_lines(self) -> None:
        guards = {}
        for path, contents in generated_wire_outputs().items():
            if path.endswith(".h"):
                match = re.search(r"^#ifndef ([A-Z0-9_]+)$", contents, re.MULTILINE)
                self.assertIsNotNone(match, path)
                guard = match.group(1)
                self.assertNotIn(guard, guards, f"{path} and {guards.get(guard)}")
                guards[guard] = path
            self.assertLess(
                len(contents.splitlines()),
                2000,
                f"{path} needs a narrower generated ownership shard",
            )
            for line_number, line in enumerate(contents.splitlines(), 1):
                self.assertLessEqual(
                    len(line),
                    100,
                    f"{path}:{line_number} is not reviewable C",
                )

    def test_generated_headers_expose_versioned_feature_facts(self) -> None:
        outputs = generated_wire_outputs()
        module_header = outputs["wire/module_format.h"]
        isa_headers = "".join(
            contents
            for path, contents in outputs.items()
            if path.startswith("wire/")
            and path.endswith(".h")
            and path != "wire/module_format.h"
        )

        self.assertEqual(
            len(
                re.findall(
                    r"^  IREE_VM_BYTECODE_.*_SINCE_MINOR =",
                    module_header,
                    re.MULTILINE,
                )
            ),
            30 + 13,
        )
        self.assertIn(
            "IREE_VM_BYTECODE_SECTION_PRESENTATION_REQUIRED_FLAGS = 0x0001,",
            module_header,
        )
        self.assertIn(
            "#define IREE_VM_BYTECODE_IMAGE_HEADER_MAGIC_U8_BYTES "
            '"\\x49\\x52\\x45\\x45\\x56\\x4D\\x00\\x00"',
            module_header,
        )
        self.assertEqual(
            len(
                re.findall(
                    r"^  IREE_VM_ISA_.*_SINCE_MINOR =",
                    isa_headers,
                    re.MULTILINE,
                )
            ),
            215 + 217,
        )

    def test_generated_flag_types_follow_runtime_vocabulary(self) -> None:
        module_header = generated_wire_outputs()["wire/module_format.h"]
        self.assertIn(
            "typedef uint16_t iree_vm_bytecode_section_flags_t;",
            module_header,
        )
        self.assertIn(
            "enum iree_vm_bytecode_section_flag_bits_e {",
            module_header,
        )
        self.assertNotIn("iree_vm_bytecode_section_flag_t;", module_header)

    def test_wire_alignment_accepts_stricter_file_alignment_than_host_abi(self) -> None:
        outputs = generated_wire_outputs()
        module_assertions = outputs["wire/assertions.c"]
        self.assertIn(
            "8 % iree_alignof(iree_vm_bytecode_v0_section_directory_row_t) == 0",
            module_assertions,
        )
        self.assertNotIn(
            "iree_alignof(iree_vm_bytecode_v0_section_directory_row_t) == 8",
            module_assertions,
        )

    def test_layout_assertions_are_private_translation_units(self) -> None:
        outputs = generated_wire_outputs()
        for path, contents in outputs.items():
            if path.endswith(".h"):
                self.assertNotIn("static_assert", contents, path)
                self.assertNotIn("IREE_VM_BYTECODE_STATIC_ASSERT", contents, path)
                self.assertNotIn("IREE_VM_BYTECODE_ALIGNOF", contents, path)
            else:
                self.assertIn('#include "iree/base/alignment.h"', contents, path)
                self.assertIn("static_assert(", contents, path)
                self.assertIn("iree_alignof(", contents, path)
                self.assertNotIn("IREE_VM_BYTECODE_STATIC_ASSERT", contents, path)
                self.assertNotIn("IREE_VM_BYTECODE_ALIGNOF", contents, path)

    def test_json_remains_an_on_demand_versioned_projection(self) -> None:
        module_projection = json.loads(
            render_projection_json(latest_projection(MODULE_SPECIFICATION))
        )
        isa_projection = json.loads(
            render_projection_json(latest_projection(ISA_SPECIFICATION))
        )

        for projection in (module_projection, isa_projection):
            for entity in projection["entities"]:
                self.assertIn("since", entity)
                self.assertIn("minimum_consumer_version", entity)
        self.assertFalse(
            any(
                path.endswith(".json")
                for path in (
                    generated_wire_outputs() | generated_documentation_outputs()
                )
            )
        )

    def test_generated_c_is_declarative_only(self) -> None:
        forbidden_fragments = (
            "iree_status_t",
            "switch (",
            "goto ",
            "malloc(",
            "iree_hal_",
            "handler(",
        )
        for path, contents in generated_wire_outputs().items():
            for fragment in forbidden_fragments:
                self.assertNotIn(fragment, contents, path)


if __name__ == "__main__":
    unittest.main()
