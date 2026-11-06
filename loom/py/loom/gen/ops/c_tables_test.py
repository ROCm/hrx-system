# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Artifact ownership policy for checked-in and build-generated files."""

import pytest

from loom.assembly import Param
from loom.dsl import AttrDef, Dialect, EnumCase, EnumDef, TypeDef
from loom.gen.ops import model as c_table_model
from loom.gen.ops.c_dialect import generate_dialect_contents
from loom.gen.ops.c_tables import checked_in_file_set


def test_checked_in_file_set_separates_public_artifacts_from_build_outputs() -> None:
    dialect = Dialect("artifact_test", dialect_id=0x7D)
    build_generated_dialect = Dialect(
        "build_generated_test",
        dialect_id=0x7C,
        checked_in_headers=False,
    )
    model = c_table_model.GenerationModel(
        dialects=[
            c_table_model.DialectGeneration(
                dialect=dialect,
                ops=[],
                table_shards=None,
                types=[TypeDef("artifact_test.token", params=[AttrDef("payload", "type")], format=[Param("payload")])],
            ),
            c_table_model.DialectGeneration(
                dialect=build_generated_dialect,
                ops=[],
                table_shards=None,
                types=[TypeDef("build_generated_test.token", params=[AttrDef("payload", "type")], format=[Param("payload")])],
            ),
        ],
        types=[],
        generated_enums=(
            EnumDef(
                "GeneratedMode",
                [EnumCase("first", 0), EnumCase("second", 1)],
                c_type="loom_generated_mode_t",
                c_const_prefix="LOOM_GENERATED_MODE",
                c_include="loom/ops/generated_mode.h",
            ),
        ),
    )

    generated_file_set = checked_in_file_set(model)

    assert "loom/src/loom/ops/artifact_test/ops.h" in generated_file_set.output_paths
    assert "loom/src/loom/ops/build_generated_test/ops.h" in generated_file_set.obsolete_paths
    assert "loom/src/loom/ops/op_registry.h" in generated_file_set.output_paths
    assert "loom/src/loom/ops/artifact_test/type_builders.h" in generated_file_set.output_paths
    assert "loom/src/loom/ops/artifact_test/type_builders.c" in generated_file_set.obsolete_paths
    assert "loom/src/loom/ops/build_generated_test/type_builders.h" in generated_file_set.obsolete_paths
    assert "loom/src/loom/ops/type_builders.h" in generated_file_set.output_paths
    assert "loom/src/loom/ops/generated_mode.h" in generated_file_set.output_paths
    assert "loom/src/loom/ops/type_builders.c" in generated_file_set.obsolete_paths
    assert "loom/src/loom/ir/scalar_type_table.inc" in generated_file_set.output_paths
    assert "loom/src/loom/ops/artifact_test/ops.inc" in generated_file_set.obsolete_paths
    assert "loom/src/loom/ops/build_generated_test/ops.inc" in generated_file_set.obsolete_paths
    assert "loom/src/loom/ops/artifact_test/builders.c" in generated_file_set.obsolete_paths
    assert "loom/src/loom/ops/artifact_test/tables.c" in generated_file_set.obsolete_paths
    assert "loom/src/loom/ops/op_registry_tables.c" in generated_file_set.obsolete_paths
    assert "loom/src/loom/ops/op_registry_tables.h" in generated_file_set.obsolete_paths
    assert all(not path.endswith(".c") for path in generated_file_set.output_paths)
    assert set(generated_file_set.output_paths).isdisjoint(generated_file_set.obsolete_paths)

    generated_mode_header = next(generated_file.contents for generated_file in generated_file_set.files if generated_file.path == "loom/src/loom/ops/generated_mode.h")
    assert "typedef enum loom_generated_mode_e" in generated_mode_header
    assert "LOOM_GENERATED_MODE_FIRST = 0" in generated_mode_header
    assert "LOOM_GENERATED_MODE_SECOND = 1" in generated_mode_header
    assert "LOOM_GENERATED_MODE_COUNT_ = 2" in generated_mode_header


@pytest.mark.parametrize("parameterized", [False, True])
def test_dialect_construction_is_separate_from_type_metadata(parameterized: bool) -> None:
    type_def = TypeDef(
        "artifact_test.token",
        params=[AttrDef("payload", "type")] if parameterized else [],
        format=[Param("payload")] if parameterized else [],
    )
    outputs = generate_dialect_contents(
        c_table_model.DialectGeneration(
            dialect=Dialect("artifact_test", dialect_id=0x7D),
            ops=[],
            table_shards=None,
            types=[type_def],
        )
    )
    assert "loom_module_make_parameterized_type_by_kind" not in outputs["types.c"]
    if parameterized:
        assert '#include "loom/ops/artifact_test/type_builders.h"' in outputs["types.h"]
        assert "loom_module_make_parameterized_type_by_kind" in outputs["type_builders.c"]
        assert "LOOM_PARAMETERIZED_TYPE_KIND(LOOM_DIALECT_ARTIFACT_TEST, 0)" in outputs["type_builders.h"]
        assert "descriptor" not in outputs["type_builders.h"] + outputs["type_builders.c"]
    else:
        assert "type_builders.h" not in outputs
        assert "type_builders.c" not in outputs
