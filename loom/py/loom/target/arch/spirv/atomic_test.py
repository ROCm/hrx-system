# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.dialect.atomic import AtomicKind, AtomicOrdering, AtomicScope
from loom.target.arch.spirv.atomic import (
    ATOMIC_FLOAT_OPERATIONS,
    ATOMIC_FLOAT_SCALARS,
    ATOMIC_INTEGER_OPERATIONS,
    ATOMIC_INTEGER_SCALARS,
    ATOMIC_ORDERINGS,
    ATOMIC_SCOPES,
    ATOMIC_STORAGE_CLASSES,
    atomic_descriptor_key,
    atomic_feature_bits,
    cmpxchg_failure_orderings,
    float_atomic_cas_feature_bits,
    float_atomic_descriptor_key,
    float_atomic_native_feature_bits,
)
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.features import atom_by_key


def _atomic_descriptors():
    return {
        descriptor.key: descriptor
        for descriptor in SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.descriptors
        if descriptor.key.startswith("spirv.atomic.")
    }


def test_source_atomic_operation_vocabulary_is_completely_classified() -> None:
    source_kinds = {case.keyword for case in AtomicKind.cases}
    integer_kinds = {operation.source_kind for operation in ATOMIC_INTEGER_OPERATIONS}
    float_kinds = {operation.source_kind for operation in ATOMIC_FLOAT_OPERATIONS}

    assert integer_kinds.isdisjoint(float_kinds)
    assert integer_kinds | float_kinds == source_kinds
    assert {scalar.source_type for scalar in ATOMIC_INTEGER_SCALARS} == {
        "i32",
        "i64",
    }
    assert {scalar.source_type for scalar in ATOMIC_FLOAT_SCALARS} == {
        "f16",
        "f32",
        "f64",
    }


def test_vulkan_scalar_type_boundary_is_explicit() -> None:
    source_integer_types = {"i1", "i8", "i16", "i32", "i64"}
    source_float_types = {
        "f8E4M3",
        "f8E5M2",
        "f16",
        "bf16",
        "f32",
        "f64",
    }
    source_cmpxchg_types = {
        "index",
        *(source_integer_types - {"i1"}),
        *source_float_types,
    }

    integer_types = {scalar.source_type for scalar in ATOMIC_INTEGER_SCALARS}
    float_types = {scalar.source_type for scalar in ATOMIC_FLOAT_SCALARS}
    float_cmpxchg_types = {
        scalar.source_type
        for scalar in ATOMIC_FLOAT_SCALARS
        if scalar.integer_scalar_enum is not None
    }

    assert integer_types == {"i32", "i64"}
    assert source_integer_types - integer_types == {"i1", "i8", "i16"}
    assert float_types == {"f16", "f32", "f64"}
    assert source_float_types - float_types == {"f8E4M3", "f8E5M2", "bf16"}
    assert integer_types | float_cmpxchg_types == {"i32", "i64", "f32", "f64"}
    assert source_cmpxchg_types - integer_types - float_cmpxchg_types == {
        "index",
        "i8",
        "i16",
        "f8E4M3",
        "f8E5M2",
        "f16",
        "bf16",
    }


def test_atomic_features_require_their_scalar_capabilities() -> None:
    features = atom_by_key()
    f16 = next(scalar for scalar in ATOMIC_FLOAT_SCALARS if scalar.source_type == "f16")
    f64 = next(scalar for scalar in ATOMIC_FLOAT_SCALARS if scalar.source_type == "f64")
    required_capabilities = {
        "storage_buffer_int64_atomics": "int64",
        "workgroup_int64_atomics": "int64",
        f16.storage_buffer_basic_feature_atom: "float16",
        f16.workgroup_basic_feature_atom: "float16",
        f16.storage_buffer_add_feature_atom: "float16",
        f16.workgroup_add_feature_atom: "float16",
        f64.storage_buffer_basic_feature_atom: "float64",
        f64.workgroup_basic_feature_atom: "float64",
        f64.storage_buffer_add_feature_atom: "float64",
        f64.workgroup_add_feature_atom: "float64",
    }

    for feature_key, required_capability in required_capabilities.items():
        assert required_capability in features[feature_key].required


def test_vulkan_scope_and_ordering_boundary_is_explicit() -> None:
    assert tuple(ordering.source_keyword for ordering in ATOMIC_ORDERINGS) == tuple(
        case.keyword for case in AtomicOrdering.cases if case.keyword != "seq_cst"
    )
    assert tuple(scope.source_keyword for scope in ATOMIC_SCOPES) == tuple(
        case.keyword for case in AtomicScope.cases if case.keyword != "system"
    )
    assert ATOMIC_SCOPES[0].source_keyword == "thread"
    assert tuple(
        ordering.source_keyword for ordering in ATOMIC_SCOPES[0].orderings
    ) == ("relaxed",)

    expected_failure_orderings = {
        "relaxed": ("relaxed",),
        "acquire": ("relaxed", "acquire"),
        "release": ("relaxed",),
        "acq_rel": ("relaxed", "acquire"),
    }
    for success_ordering in ATOMIC_ORDERINGS:
        assert (
            tuple(
                ordering.source_keyword
                for ordering in cmpxchg_failure_orderings(success_ordering)
            )
            == expected_failure_orderings[success_ordering.source_keyword]
        )


def test_integer_atomic_descriptor_matrix_is_complete() -> None:
    descriptors = _atomic_descriptors()
    expected_keys = set[str]()

    for scalar in ATOMIC_INTEGER_SCALARS:
        for storage_class in ATOMIC_STORAGE_CLASSES:
            for scope in ATOMIC_SCOPES:
                expected_feature_mask = atomic_feature_bits(
                    scalar, storage_class, scope
                )
                for operation in ATOMIC_INTEGER_OPERATIONS:
                    forms = ("rmw", "reduce") if operation.supports_reduce else ("rmw",)
                    for form in forms:
                        key = atomic_descriptor_key(
                            form,
                            scalar,
                            storage_class,
                            scope,
                            operation=operation,
                        )
                        expected_keys.add(key)
                        descriptor = descriptors[key]
                        assert descriptor.feature_mask_words == (
                            (expected_feature_mask,) if expected_feature_mask else ()
                        )
                for success_ordering in scope.orderings:
                    key = atomic_descriptor_key(
                        "cmpxchg",
                        scalar,
                        storage_class,
                        scope,
                        success_ordering=success_ordering,
                    )
                    expected_keys.add(key)
                    descriptor = descriptors[key]
                    assert descriptor.feature_mask_words == (
                        (expected_feature_mask,) if expected_feature_mask else ()
                    )

    actual_keys = {
        key
        for key in descriptors
        if ".native." not in key and ".bitcast." not in key and ".cas." not in key
    }
    assert len(expected_keys) == 356
    assert actual_keys == expected_keys


def test_float_atomic_descriptor_matrix_is_complete() -> None:
    descriptors = _atomic_descriptors()
    expected_keys = set[str]()

    for scalar in ATOMIC_FLOAT_SCALARS:
        for storage_class in ATOMIC_STORAGE_CLASSES:
            for scope in ATOMIC_SCOPES:
                for operation in ATOMIC_FLOAT_OPERATIONS:
                    if operation.native_opcode is not None:
                        forms = (
                            ("rmw", "reduce") if operation.supports_reduce else ("rmw",)
                        )
                        for form in forms:
                            key = float_atomic_descriptor_key(
                                form,
                                "native",
                                scalar,
                                storage_class,
                                scope,
                                operation=operation,
                            )
                            expected_keys.add(key)
                            feature_mask = float_atomic_native_feature_bits(
                                scalar, operation, storage_class, scope
                            )
                            assert descriptors[key].feature_mask_words == (
                                (feature_mask,) if feature_mask else ()
                            )
                    if scalar.integer_scalar_enum is None:
                        continue
                    strategy = "bitcast" if operation.source_kind == "xchgf" else "cas"
                    forms = ("rmw", "reduce") if operation.supports_reduce else ("rmw",)
                    for form in forms:
                        key = float_atomic_descriptor_key(
                            form,
                            strategy,
                            scalar,
                            storage_class,
                            scope,
                            operation=operation,
                        )
                        expected_keys.add(key)
                        feature_mask = float_atomic_cas_feature_bits(
                            scalar, storage_class, scope
                        )
                        assert descriptors[key].feature_mask_words == (
                            (feature_mask,) if feature_mask else ()
                        )
                if scalar.integer_scalar_enum is None:
                    continue
                for success_ordering in scope.orderings:
                    key = float_atomic_descriptor_key(
                        "cmpxchg",
                        "bitcast",
                        scalar,
                        storage_class,
                        scope,
                        success_ordering=success_ordering,
                    )
                    expected_keys.add(key)
                    feature_mask = float_atomic_cas_feature_bits(
                        scalar, storage_class, scope
                    )
                    assert descriptors[key].feature_mask_words == (
                        (feature_mask,) if feature_mask else ()
                    )

    actual_keys = {
        key
        for key in descriptors
        if ".native." in key or ".bitcast." in key or ".cas." in key
    }
    assert len(expected_keys) == 300
    assert actual_keys == expected_keys
    assert len(descriptors) == 656


def test_workgroup_float_fallback_descriptors_use_integer_pointer_classes() -> None:
    descriptors = _atomic_descriptors()
    workgroup = next(
        storage_class
        for storage_class in ATOMIC_STORAGE_CLASSES
        if storage_class.suffix == "workgroup"
    )
    for scalar in ATOMIC_FLOAT_SCALARS:
        if scalar.integer_suffix is None:
            continue
        for scope in ATOMIC_SCOPES:
            for operation in ATOMIC_FLOAT_OPERATIONS:
                strategy = "bitcast" if operation.source_kind == "xchgf" else "cas"
                key = float_atomic_descriptor_key(
                    "rmw",
                    strategy,
                    scalar,
                    workgroup,
                    scope,
                    operation=operation,
                )
                pointer = next(
                    operand
                    for operand in descriptors[key].operands
                    if operand.field_name == "ptr"
                )
                assert tuple(
                    alternative.reg_class for alternative in pointer.reg_alts
                ) == (f"spirv.ptr.workgroup.{scalar.integer_suffix}",)
