# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for compact target-neutral math recipe generation."""

from pathlib import Path

from loom.gen.math.recipe_program import compile_recipe_program


def test_complete_f32_recipe_family_stays_compact() -> None:
    source_path = Path(__file__).parents[4] / "src/loom/transforms/math/approximate_f32_recipes.loom"
    table = compile_recipe_program(source_path.read_text(), filename=source_path)

    assert tuple(program.name for program in table.programs) == (
        "approximate_f32_exp2",
        "approximate_f32_sin_turns",
        "approximate_f32_cos_turns",
    )
    assert tuple(program.instruction_count for program in table.programs) == (47, 82, 76)
    assert tuple(program.value_count for program in table.programs) == (48, 83, 77)
    assert len(table.instructions) == 205
    assert len(table.descriptors) == 20
    assert len(table.constants) == 35
    assert table.maximum_value_count == 83
