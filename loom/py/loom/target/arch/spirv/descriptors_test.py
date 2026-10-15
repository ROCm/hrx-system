# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.ir import parse_scalar_type_kind
from loom.target.arch.spirv.atomic import (
    ATOMIC_FLOAT_OPERATIONS,
    ATOMIC_FLOAT_SCALARS,
    ATOMIC_INTEGER_OPERATIONS,
    ATOMIC_INTEGER_SCALARS,
    ATOMIC_SCOPES,
    ATOMIC_STORAGE_CLASSES,
    atomic_descriptor_key,
    float_atomic_cas_strategies,
    float_atomic_descriptor_key,
)
from loom.target.arch.spirv.barrier import CONTROL_BARRIER_CASES
from loom.target.arch.spirv.builtins import (
    BUILTIN_DIMENSIONS,
    BUILTIN_INDEX_QUERIES,
    BUILTIN_SCALAR_INDEX_QUERIES,
)
from loom.target.arch.spirv.cooperative_matrix import (
    COOPERATIVE_MATRIX_CASES,
    COOPERATIVE_MATRIX_TYPES,
    cooperative_matrix_descriptor_key,
    cooperative_matrix_memory_variants,
)
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.extended_math import EXTENDED_MATH_INSTRUCTIONS
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
from loom.target.arch.spirv.ordinary_vector_conversion import (
    ORDINARY_VECTOR_CONVERSION_INSTRUCTIONS,
)
from loom.target.arch.spirv.ordinary_vector_float import (
    ORDINARY_VECTOR_FLOAT_BINARY_INSTRUCTIONS,
    ORDINARY_VECTOR_FLOAT_COMPARE_INSTRUCTIONS,
    ORDINARY_VECTOR_FLOAT_SCALE_INSTRUCTIONS,
)
from loom.target.arch.spirv.ordinary_vector_integer import (
    INTEGER_BIT_COUNT_INSTRUCTIONS,
    ORDINARY_VECTOR_INTEGER_INSTRUCTIONS,
)
from loom.target.arch.spirv.scalar_alu import (
    BOOLEAN_BINARY_OPERATIONS,
    BOOLEAN_CONSTANTS,
    FLOAT_BINARY_OPERATIONS,
    FLOAT_COMPARE_PREDICATES,
    FLOAT_COMPARE_TYPES,
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
    DIRECT_SCALAR_CONVERSIONS,
    INTEGER_VALUE_VIEW_CONVERSIONS,
    LOW_SCALAR_CONVERSIONS,
)
from loom.target.arch.spirv.scalar_memory import (
    RAW_STORAGE_BUFFER_BYTE,
    STORAGE_BUFFER_SCALARS,
)
from loom.target.arch.spirv.subgroup import (
    SPIRV_SUBGROUP_BALLOT_PACKING_INSTRUCTIONS,
    SPIRV_SUBGROUP_INSTRUCTIONS,
)
from loom.target.low_descriptors import (
    AsmResultValueType,
    EffectFlag,
    EffectKind,
    InstructionClass,
    MemorySpace,
)


def test_bfloat16_float32_conversions_are_bidirectional() -> None:
    conversions = {
        (row.source_type.source_type, row.result_type.source_type): row
        for row in DIRECT_SCALAR_CONVERSIONS
    }
    widening = conversions[("bf16", "f32")]
    narrowing = conversions[("f32", "bf16")]
    assert widening.source_op_key == "extf"
    assert widening.feature_bits == feature_bits_value(
        ("bfloat16_type_khr", "float32_denorm_preserve")
    )
    assert narrowing.source_op_key == "fptrunc"
    assert narrowing.feature_bits == feature_bits_value(("bfloat16_type_khr",))


def test_native_float_narrowing_requires_exact_execution_controls() -> None:
    conversions = {
        (row.source_type.source_type, row.result_type.source_type): row
        for row in DIRECT_SCALAR_CONVERSIONS
        if row.source_op_key == "fptrunc"
    }
    expected_features = {
        ("f64", "f32"): (
            "float32_denorm_preserve",
            "float32_rounding_rte",
            "float32_signed_zero_inf_nan_preserve",
            "float64_signed_zero_inf_nan_preserve",
        ),
        ("f64", "f16"): (
            "float16_64_denorm_preserve",
            "float16_64_rounding_rte",
            "float16_signed_zero_inf_nan_preserve",
            "float64_signed_zero_inf_nan_preserve",
            "float64_to_float16_conversion",
        ),
        ("f32", "f16"): (
            "float16_64_denorm_preserve",
            "float16_64_rounding_rte",
            "float16_signed_zero_inf_nan_preserve",
            "float32_signed_zero_inf_nan_preserve",
        ),
    }

    for pair, feature_atoms in expected_features.items():
        assert conversions[pair].feature_atoms == feature_atoms


def test_control_barrier_descriptors_cover_the_complete_source_family() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.descriptors
        if descriptor.key.startswith("spirv.op_control_barrier.")
    }
    assert set(descriptors) == {case.descriptor_key for case in CONTROL_BARRIER_CASES}
    for case in CONTROL_BARRIER_CASES:
        descriptor = descriptors[case.descriptor_key]
        expected_memory_space = {
            "global": MemorySpace.GLOBAL,
            "workgroup": MemorySpace.WORKGROUP,
        }[case.memory_space.source_keyword]
        assert descriptor.effects[0].kind is EffectKind.BARRIER
        assert descriptor.effects[0].memory_space is expected_memory_space
        assert descriptor.effects[0].flags == (
            EffectFlag.ORDERED,
            EffectFlag.DEPENDENCY,
        )
        assert descriptor.effects[1].kind is EffectKind.CONVERGENT
        assert InstructionClass.EXECUTION_BARRIER in descriptor.instruction_classes


def test_builtin_queries_cover_dispatch_and_subgroup_inputs() -> None:
    assert {query.source_op_key for query in BUILTIN_INDEX_QUERIES} == {
        "workgroup_id",
        "workitem_id",
        "workitem_dispatch_id",
        "workgroup_count",
    }
    assert {query.source_op_key for query in BUILTIN_SCALAR_INDEX_QUERIES} == {
        "subgroup_lane_id",
        "subgroup_size",
        "subgroup_id",
        "subgroup_count",
    }
    assert all(query.component_count == 3 for query in BUILTIN_INDEX_QUERIES)
    assert all(query.component_count == 1 for query in BUILTIN_SCALAR_INDEX_QUERIES)
    queries = (*BUILTIN_INDEX_QUERIES, *BUILTIN_SCALAR_INDEX_QUERIES)
    assert len({query.variable_enum for query in queries}) == len(queries)
    assert len({query.builtin_enum for query in queries}) == len(queries)


def test_subgroup_builtins_require_group_non_uniform() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.descriptors
    }
    for query in BUILTIN_SCALAR_INDEX_QUERIES:
        descriptor = descriptors[f"spirv.op_load_builtin.{query.descriptor_suffix}"]
        assert descriptor.feature_mask_words == (
            feature_bits_value(("group_non_uniform",)),
        )


def test_subgroup_instructions_are_convergent_and_capability_guarded() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.descriptors
    }
    for row in SPIRV_SUBGROUP_INSTRUCTIONS:
        descriptor = descriptors[row.key]
        assert descriptor.feature_mask_words == (row.feature_bits,)
        assert len(descriptor.effects) == 1
        assert descriptor.effects[0].kind is EffectKind.CONVERGENT
        assert descriptor.effects[0].flags == (EffectFlag.ORDERED,)


def _scalar_recipe(source_type: str) -> AsmResultValueType:
    element_type = parse_scalar_type_kind(source_type)
    assert element_type is not None
    return AsmResultValueType(element_type)


def _atomic_result_recipes() -> dict[str, AsmResultValueType]:
    recipes: dict[str, AsmResultValueType] = {}

    def add(key: str, source_type: str) -> None:
        assert key not in recipes
        recipes[key] = _scalar_recipe(source_type)

    for scalar in ATOMIC_INTEGER_SCALARS:
        for storage_class in ATOMIC_STORAGE_CLASSES:
            for scope in ATOMIC_SCOPES:
                for operation in ATOMIC_INTEGER_OPERATIONS:
                    add(
                        atomic_descriptor_key(
                            "rmw",
                            scalar,
                            storage_class,
                            scope,
                            operation=operation,
                        ),
                        scalar.source_type,
                    )
                for success_ordering in scope.orderings:
                    add(
                        atomic_descriptor_key(
                            "cmpxchg",
                            scalar,
                            storage_class,
                            scope,
                            success_ordering=success_ordering,
                        ),
                        scalar.source_type,
                    )

    for scalar in ATOMIC_FLOAT_SCALARS:
        for storage_class in ATOMIC_STORAGE_CLASSES:
            for scope in ATOMIC_SCOPES:
                for operation in ATOMIC_FLOAT_OPERATIONS:
                    if operation.native_opcode is not None:
                        add(
                            float_atomic_descriptor_key(
                                "rmw",
                                "native",
                                scalar,
                                storage_class,
                                scope,
                                operation=operation,
                            ),
                            scalar.source_type,
                        )
                    if scalar.integer_scalar_enum is None:
                        continue
                    strategies = (
                        ("bitcast",)
                        if operation.source_kind == "xchgf"
                        else float_atomic_cas_strategies(scalar, operation)
                    )
                    for strategy in strategies:
                        add(
                            float_atomic_descriptor_key(
                                "rmw",
                                strategy,
                                scalar,
                                storage_class,
                                scope,
                                operation=operation,
                            ),
                            scalar.source_type,
                        )
                if scalar.integer_scalar_enum is None:
                    continue
                for success_ordering in scope.orderings:
                    add(
                        float_atomic_descriptor_key(
                            "cmpxchg",
                            "bitcast",
                            scalar,
                            storage_class,
                            scope,
                            success_ordering=success_ordering,
                        ),
                        scalar.source_type,
                    )

    return recipes


def _extended_math_result_recipes() -> dict[str, AsmResultValueType]:
    recipes = {}
    for row in EXTENDED_MATH_INSTRUCTIONS:
        component_type = row.value_type
        lane_count = 0
        if isinstance(row.value_type, OrdinaryVectorType):
            component_type = row.value_type.component_type
            lane_count = row.value_type.lane_count
        recipes[row.descriptor_key] = AsmResultValueType(
            _scalar_recipe(component_type.source_types[0]).element_type,
            vector_lane_count=lane_count,
        )
    return recipes


def _cooperative_matrix_result_keys() -> list[str]:
    keys = [matrix_type.constructor_key for matrix_type in COOPERATIVE_MATRIX_TYPES]
    for case in COOPERATIVE_MATRIX_CASES:
        keys.extend(
            case.descriptor_key(
                "op_cooperative_matrix_load_khr",
                role=role,
                memory_space=memory.descriptor_suffix,
            )
            for role, scalar in (
                ("lhs", case.lhs_scalar),
                ("rhs", case.rhs_scalar),
                ("init", case.accumulator_scalar),
            )
            for memory in cooperative_matrix_memory_variants(scalar)
        )
        keys.append(
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
    return keys


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
        *ORDINARY_VECTOR_FLOAT_BINARY_INSTRUCTIONS,
        *ORDINARY_VECTOR_FLOAT_SCALE_INSTRUCTIONS,
        *ORDINARY_VECTOR_FLOAT_COMPARE_INSTRUCTIONS,
        *ORDINARY_VECTOR_INTEGER_INSTRUCTIONS,
        *INTEGER_BIT_COUNT_INSTRUCTIONS,
        *ORDINARY_VECTOR_CONVERSION_INSTRUCTIONS,
        *ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTIONS,
        *SPIRV_SUBGROUP_BALLOT_PACKING_INSTRUCTIONS,
        *SPIRV_SUBGROUP_INSTRUCTIONS,
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

    extended_math_recipes = _extended_math_result_recipes()
    assert expected_recipes.keys().isdisjoint(extended_math_recipes)
    assert carrier_only_keys.isdisjoint(extended_math_recipes)
    expected_recipes.update(extended_math_recipes)

    add_scalar_recipe("spirv.op_copy_object.i32", "i32")
    add_scalar_recipe("spirv.op_imul_add.i32", "i32")
    for opcode in ("iadd", "isub", "imul"):
        add_carrier_only(f"spirv.op_{opcode}.offset64")

    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS:
        scalar = (
            scalar_pair.signed if scalar_pair.bit_width == 64 else scalar_pair.unsigned
        )
        opcode = "bitcast" if scalar_pair.bit_width == 64 else "uconvert"
        add_carrier_only(f"spirv.op_{opcode}.{scalar.suffix}.offset64")
        if scalar_pair.bit_width < 64:
            add_carrier_only(f"spirv.op_s_convert.{scalar_pair.signed.suffix}.offset64")
        from_offset_key = f"spirv.op_{opcode}.offset64.{scalar.suffix}"
        if scalar in unsigned_scalars:
            add_carrier_only(from_offset_key)
        else:
            add_scalar_recipe(from_offset_key, scalar.source_type)

    builtin_descriptor_keys = tuple(
        f"spirv.op_load_builtin.{query.descriptor_suffix}.{dimension.source_keyword}"
        for query in BUILTIN_INDEX_QUERIES
        for dimension in BUILTIN_DIMENSIONS
    ) + tuple(
        f"spirv.op_load_builtin.{query.descriptor_suffix}"
        for query in BUILTIN_SCALAR_INDEX_QUERIES
    )
    for descriptor_key in builtin_descriptor_keys:
        add_scalar_recipe(descriptor_key, "index")

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
        *(
            (predicate, compare_type.scalar)
            for compare_type in FLOAT_COMPARE_TYPES
            for predicate in FLOAT_COMPARE_PREDICATES
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
    add_carrier_only("spirv.op_select.storage_buffer")

    for scalar in STORAGE_BUFFER_SCALARS:
        add_carrier_only(
            f"spirv.op_ptr_access_chain.storage_buffer.{scalar.suffix}.byte_offset"
        )
        add_carrier_only(
            f"spirv.op_access_chain.workgroup.{scalar.suffix}.element_index"
        )
        add_carrier_only(f"spirv.op_access_chain.workgroup.{scalar.suffix}.byte_offset")
        for memory_space in ("storage_buffer", "workgroup"):
            load_key = f"spirv.op_load.{memory_space}.{scalar.suffix}"
            if scalar.source_rule_enabled:
                add_scalar_recipe(load_key, scalar.source_type)
            else:
                add_carrier_only(load_key)

    for key, recipe in _atomic_result_recipes().items():
        add_recipe(key, recipe)

    raw_byte_suffix = RAW_STORAGE_BUFFER_BYTE.suffix
    add_carrier_only(
        f"spirv.op_ptr_access_chain.storage_buffer.{raw_byte_suffix}.byte_offset"
    )
    add_carrier_only(f"spirv.op_load.storage_buffer.{raw_byte_suffix}")
    add_carrier_only(f"spirv.op_uconvert.{raw_byte_suffix}.u32")
    add_carrier_only(f"spirv.op_uconvert.u32.{raw_byte_suffix}")

    for key in _cooperative_matrix_result_keys():
        add_carrier_only(key)
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


def test_workgroup_matrix_memory_needs_no_storage_buffer_features() -> None:
    descriptors = {
        row.key: row for row in SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.descriptors
    }
    storage_bits = feature_bits_value(
        ("storage_buffer_8bit_access", "storage_buffer_16bit_access")
    )
    for case in COOPERATIVE_MATRIX_CASES:
        for role, scalar in (
            ("lhs", case.lhs_scalar),
            ("rhs", case.rhs_scalar),
            ("init", case.accumulator_scalar),
            ("result", case.result_scalar),
        ):
            for memory in cooperative_matrix_memory_variants(scalar):
                if memory.space != "workgroup":
                    continue
                descriptor = descriptors[
                    case.descriptor_key(
                        "op_cooperative_matrix_store_khr"
                        if role == "result"
                        else "op_cooperative_matrix_load_khr",
                        role=role,
                        memory_space=memory.descriptor_suffix,
                    )
                ]
                features = descriptor.feature_mask_words[0]
                assert not features & storage_bits
                assert (
                    features & scalar.register_feature_bits
                    == scalar.register_feature_bits
                )
                assert (
                    features & memory.carrier.register_feature_bits
                    == memory.carrier.register_feature_bits
                )
                assert descriptor.immediates[0].unsigned_max == 1
                pointer = next(
                    operand
                    for operand in descriptor.operands
                    if operand.field_name == "ptr"
                )
                assert tuple(
                    alternative.reg_class for alternative in pointer.reg_alts
                ) == (f"spirv.ptr.workgroup.{memory.carrier.suffix}",)
                memory_effect = next(
                    effect for effect in descriptor.effects if effect.is_memory_access
                )
                assert memory_effect.memory_space == MemorySpace.WORKGROUP
                assert memory_effect.width_bits == 0
