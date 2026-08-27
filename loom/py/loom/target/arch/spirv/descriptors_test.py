# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.ir import parse_scalar_type_kind
from loom.target.arch.spirv.builtins import (
    BUILTIN_DIMENSIONS,
    BUILTIN_INDEX_QUERIES,
)
from loom.target.arch.spirv.cooperative_matrix import (
    COOPERATIVE_MATRIX_CASES,
    cooperative_matrix_descriptor_key,
)
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.features import feature_bits_value
from loom.target.arch.spirv.ordinary_vector import (
    ORDINARY_VECTOR_INSTRUCTIONS,
    OrdinaryVectorComponentKind,
    OrdinaryVectorComponentType,
    OrdinaryVectorType,
)
from loom.target.arch.spirv.ordinary_vector_bit_layout import (
    ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTIONS,
)
from loom.target.arch.spirv.ordinary_vector_integer import (
    ORDINARY_VECTOR_INTEGER_INSTRUCTIONS,
)
from loom.target.arch.spirv.ordinary_vector_integer_conversion import (
    ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS,
)
from loom.target.arch.spirv.scalar_alu import (
    BOOLEAN_BINARY_OPERATIONS,
    BOOLEAN_CONSTANTS,
    FLOAT_BINARY_OPERATIONS,
    FLOAT_SCALAR_ALU_TYPES,
    INTEGER_BITWISE_BINARY_OPERATIONS,
    INTEGER_SCALAR_ALU_TYPE_PAIRS,
    OFFSET64_COMPARE_PREDICATES,
    SCALAR_ALU_TYPES,
    SIGNED_INTEGER_BINARY_OPERATIONS,
    SIGNED_INTEGER_COMPARE_PREDICATES,
    SIGNED_INTEGER_SCALAR_ALU_TYPES,
    UNSIGNED_INTEGER_BINARY_OPERATIONS,
    UNSIGNED_INTEGER_SCALAR_ALU_TYPES,
    UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES,
)
from loom.target.arch.spirv.scalar_constant import (
    BFLOAT16_CONSTANT_TYPE,
    FLOAT_CONSTANT_TYPES,
)
from loom.target.arch.spirv.scalar_conversion import (
    INTEGER_VALUE_VIEW_CONVERSIONS,
    LOW_SCALAR_CONVERSIONS,
)
from loom.target.arch.spirv.scalar_memory import (
    RAW_STORAGE_BUFFER_BYTE,
    STORAGE_BUFFER_SCALARS,
)
from loom.target.low_descriptors import AsmResultValueType


def _scalar_recipe(source_type: str) -> AsmResultValueType:
    element_type = parse_scalar_type_kind(source_type)
    assert element_type is not None
    return AsmResultValueType(element_type)


def test_result_asm_recipes_cover_every_spirv_descriptor_family() -> None:
    expected_recipes: dict[str, AsmResultValueType] = {}
    carrier_only_keys: set[str] = set()

    def add_recipe(key: str, recipe: AsmResultValueType) -> None:
        assert key not in expected_recipes
        assert key not in carrier_only_keys
        expected_recipes[key] = recipe

    def add_scalar_recipe(key: str, source_type: str) -> None:
        add_recipe(key, _scalar_recipe(source_type))

    def add_carrier_only(key: str) -> None:
        assert key not in expected_recipes
        assert key not in carrier_only_keys
        carrier_only_keys.add(key)

    for row in BOOLEAN_CONSTANTS:
        add_scalar_recipe(f"spirv.op_constant_{row.descriptor_suffix}.bool", "i1")
    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS:
        add_scalar_recipe(
            f"spirv.op_constant.{scalar_pair.signed.suffix}",
            scalar_pair.source_type,
        )
    for scalar in FLOAT_CONSTANT_TYPES:
        add_scalar_recipe(f"spirv.op_constant.{scalar.suffix}", scalar.source_type)
    add_carrier_only("spirv.op_constant.offset64")

    for scalar in SIGNED_INTEGER_SCALAR_ALU_TYPES:
        for operation in (
            *SIGNED_INTEGER_BINARY_OPERATIONS,
            *INTEGER_BITWISE_BINARY_OPERATIONS,
        ):
            add_scalar_recipe(
                f"spirv.op_{operation.descriptor_suffix}.{scalar.suffix}",
                scalar.source_type,
            )
    for scalar in FLOAT_SCALAR_ALU_TYPES:
        for operation in FLOAT_BINARY_OPERATIONS:
            add_scalar_recipe(
                f"spirv.op_{operation.descriptor_suffix}.{scalar.suffix}",
                scalar.source_type,
            )
    for operation in BOOLEAN_BINARY_OPERATIONS:
        add_scalar_recipe(f"spirv.op_{operation.descriptor_suffix}.bool", "i1")
    for scalar in UNSIGNED_INTEGER_SCALAR_ALU_TYPES:
        for operation in UNSIGNED_INTEGER_BINARY_OPERATIONS:
            add_carrier_only(f"spirv.op_{operation.descriptor_suffix}.{scalar.suffix}")

    unsigned_scalars = set(UNSIGNED_INTEGER_SCALAR_ALU_TYPES)
    for row in (*LOW_SCALAR_CONVERSIONS, *INTEGER_VALUE_VIEW_CONVERSIONS):
        if row.result_type in unsigned_scalars:
            add_carrier_only(row.key)
        else:
            add_scalar_recipe(row.key, row.result_type.source_type)

    for row in (
        *ORDINARY_VECTOR_INSTRUCTIONS,
        *ORDINARY_VECTOR_INTEGER_INSTRUCTIONS,
        *ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS,
        *ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTIONS,
    ):
        component_type = (
            row.result_type.component_type
            if isinstance(row.result_type, OrdinaryVectorType)
            else row.result_type
        )
        uses_carrier_only_type = not component_type.source_types or (
            isinstance(row.result_type, OrdinaryVectorComponentType)
            and component_type.kind == OrdinaryVectorComponentKind.OFFSET
        )
        if uses_carrier_only_type:
            add_carrier_only(row.key)
            continue
        lane_count = (
            row.result_type.lane_count
            if isinstance(row.result_type, OrdinaryVectorType)
            else 0
        )
        scalar_recipe = _scalar_recipe(component_type.source_types[0])
        add_recipe(
            row.key,
            AsmResultValueType(
                scalar_recipe.element_type,
                vector_lane_count=lane_count,
            ),
        )

    add_scalar_recipe("spirv.op_copy_object.i32", "i32")
    add_scalar_recipe("spirv.op_imul_add.i32", "i32")
    add_scalar_recipe("spirv.op_bit_count.i32", "i32")
    for opcode in ("iadd", "isub", "imul"):
        add_carrier_only(f"spirv.op_{opcode}.offset64")

    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS:
        scalar = (
            scalar_pair.signed if scalar_pair.bit_width == 64 else scalar_pair.unsigned
        )
        opcode = "bitcast" if scalar_pair.bit_width == 64 else "uconvert"
        add_carrier_only(f"spirv.op_{opcode}.{scalar.suffix}.offset64")
        from_offset_key = f"spirv.op_{opcode}.offset64.{scalar.suffix}"
        if scalar in unsigned_scalars:
            add_carrier_only(from_offset_key)
        else:
            add_scalar_recipe(from_offset_key, scalar.source_type)

    for query in BUILTIN_INDEX_QUERIES:
        for dimension in BUILTIN_DIMENSIONS:
            add_scalar_recipe(
                f"spirv.op_load_builtin.{query.descriptor_suffix}."
                f"{dimension.source_keyword}",
                "index",
            )

    compare_rows = (
        *(
            (predicate, scalar)
            for scalar in SIGNED_INTEGER_SCALAR_ALU_TYPES
            for predicate in SIGNED_INTEGER_COMPARE_PREDICATES
        ),
        *(
            (predicate, scalar)
            for scalar in UNSIGNED_INTEGER_SCALAR_ALU_TYPES
            for predicate in UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES
        ),
    )
    for predicate, scalar in compare_rows:
        add_scalar_recipe(
            f"spirv.op_{predicate.descriptor_suffix}.{scalar.suffix}", "i1"
        )
    for predicate in OFFSET64_COMPARE_PREDICATES:
        add_scalar_recipe(f"spirv.op_{predicate.descriptor_suffix}.offset64", "i1")

    for scalar in SCALAR_ALU_TYPES:
        add_scalar_recipe(f"spirv.op_select.{scalar.suffix}", scalar.source_type)
    add_scalar_recipe("spirv.op_select.bf16", BFLOAT16_CONSTANT_TYPE.source_type)
    add_scalar_recipe("spirv.op_select.bool", "i1")
    add_carrier_only("spirv.op_select.offset64")

    for scalar in STORAGE_BUFFER_SCALARS:
        add_carrier_only(
            f"spirv.op_ptr_access_chain.storage_buffer.{scalar.suffix}.byte_offset"
        )
        add_carrier_only(
            f"spirv.op_access_chain.workgroup.{scalar.suffix}.element_index"
        )
        for memory_space in ("storage_buffer", "workgroup"):
            load_key = f"spirv.op_load.{memory_space}.{scalar.suffix}"
            if scalar.source_rule_enabled:
                add_scalar_recipe(load_key, scalar.source_type)
            else:
                add_carrier_only(load_key)

    raw_byte_suffix = RAW_STORAGE_BUFFER_BYTE.suffix
    add_carrier_only(
        f"spirv.op_ptr_access_chain.storage_buffer.{raw_byte_suffix}.byte_offset"
    )
    add_carrier_only(f"spirv.op_load.storage_buffer.{raw_byte_suffix}")
    add_carrier_only(f"spirv.op_uconvert.{raw_byte_suffix}.u32")
    add_carrier_only(f"spirv.op_uconvert.u32.{raw_byte_suffix}")

    for case in COOPERATIVE_MATRIX_CASES:
        for role in ("lhs", "rhs", "init"):
            add_carrier_only(
                cooperative_matrix_descriptor_key(
                    "op_cooperative_matrix_load_khr",
                    role=role,
                    element=case.element,
                    m_size=case.m_size,
                    n_size=case.n_size,
                    k_size=case.k_size,
                    accumulator=case.accumulator,
                    scope="subgroup",
                    layout="row_major",
                )
            )
        add_carrier_only(
            cooperative_matrix_descriptor_key(
                "op_cooperative_matrix_mul_add_khr",
                element=case.element,
                m_size=case.m_size,
                n_size=case.n_size,
                k_size=case.k_size,
                accumulator=case.accumulator,
                scope="subgroup",
                operand_mode=case.operand_mode,
            )
        )
    add_carrier_only("spirv.op_variable.function.ptr")

    actual_forms = {
        descriptor.key: descriptor.asm_forms[0]
        for descriptor in SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.descriptors
        if descriptor.asm_forms and descriptor.asm_forms[0].results
    }
    assert set(actual_forms) == set(expected_recipes) | carrier_only_keys
    for key, expected_recipe in expected_recipes.items():
        assert actual_forms[key].result_value_types == (expected_recipe,), key
    for key in carrier_only_keys:
        assert actual_forms[key].result_value_types == (), key


def test_raw_storage_byte_descriptors_do_not_require_int8() -> None:
    suffix = RAW_STORAGE_BUFFER_BYTE.suffix
    keys = (
        f"spirv.op_ptr_access_chain.storage_buffer.{suffix}.byte_offset",
        f"spirv.op_load.storage_buffer.{suffix}",
        f"spirv.op_store.storage_buffer.{suffix}",
        f"spirv.op_uconvert.{suffix}.u32",
        f"spirv.op_uconvert.u32.{suffix}",
    )
    descriptors = {
        descriptor.key: descriptor
        for descriptor in SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.descriptors
    }
    storage_feature_mask = feature_bits_value(("storage_buffer_8bit_access",))
    for key in keys:
        assert descriptors[key].feature_mask_words == (storage_feature_mask,)
