# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.ir import parse_scalar_type_kind
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.ordinary_vector import (
    ORDINARY_VECTOR_INSTRUCTIONS,
    OrdinaryVectorComponentKind,
    OrdinaryVectorComponentType,
    OrdinaryVectorType,
)
from loom.target.arch.spirv.ordinary_vector_integer import (
    ORDINARY_VECTOR_INTEGER_INSTRUCTIONS,
)


def test_ordinary_vector_asm_result_recipes_cover_the_source_table() -> None:
    descriptors_by_key = {
        descriptor.key: descriptor
        for descriptor in SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.descriptors
    }
    instruction_rows = (
        *ORDINARY_VECTOR_INSTRUCTIONS,
        *ORDINARY_VECTOR_INTEGER_INSTRUCTIONS,
    )
    recipe_count = 0
    no_recipe_count = 0
    for row in instruction_rows:
        descriptor = descriptors_by_key[row.key]
        assert len(descriptor.asm_forms) == 1, row.key
        asm_form = descriptor.asm_forms[0]
        component_type = (
            row.result_type.component_type
            if isinstance(row.result_type, OrdinaryVectorType)
            else row.result_type
        )
        uses_carrier_only_inference = not component_type.source_types or (
            isinstance(row.result_type, OrdinaryVectorComponentType)
            and component_type.kind == OrdinaryVectorComponentKind.OFFSET
        )
        if uses_carrier_only_inference:
            assert asm_form.result_value_types == (), row.key
            no_recipe_count += 1
            continue

        assert len(asm_form.result_value_types) == 1, row.key
        recipe = asm_form.result_value_types[0]
        assert recipe is not None, row.key
        expected_element_type = parse_scalar_type_kind(component_type.source_types[0])
        assert expected_element_type is not None, row.key
        assert recipe.element_type == expected_element_type, row.key
        expected_lane_count = (
            row.result_type.lane_count
            if isinstance(row.result_type, OrdinaryVectorType)
            else 0
        )
        assert recipe.vector_lane_count == expected_lane_count, row.key
        recipe_count += 1

    assert recipe_count > 0
    assert no_recipe_count > 0
