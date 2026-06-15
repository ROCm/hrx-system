# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager

from loom.target.arch.x86.packed_dot_data import (
    FEATURE_AVX10_2,
    FEATURE_AVX512_BF16,
    FEATURE_AVX512_VL,
    FEATURE_AVX512_VNNI,
    FEATURE_AVX_VNNI,
    FEATURE_AVX_VNNI_INT8,
    FEATURE_AVX_VNNI_INT16,
)
from loom.target.arch.x86.target_info import (
    X86_DESCRIPTOR_SET_INFOS,
    X86_ISA_TIER_SCALAR,
    X86_NATIVE_TARGET_SELECTOR_PROFILE_KEYS,
    X86_REG_CLASS_GPR32,
    X86_REG_CLASS_GPR64,
    X86_REG_CLASS_K,
    X86_REG_CLASS_XMM,
    X86_REG_CLASS_YMM,
    X86_REG_CLASS_ZMM,
    X86_TARGET_PROFILE_INFOS,
    X86DescriptorSetInfo,
    X86TargetProfileInfo,
    sorted_descriptor_set_infos,
    sorted_target_profile_infos,
    validate_x86_target_info_tables,
    x86_descriptor_set_info_by_generator_target,
    x86_descriptor_set_ordinal,
    x86_descriptor_set_storage_info_by_generator_target,
    x86_descriptor_set_view_infos_by_storage_generator_target,
    x86_target_profile_info_by_key,
)


@contextmanager
def _raises_value_error(match: str) -> Iterator[None]:
    try:
        yield
    except ValueError as exc:
        if re.search(match, str(exc)) is None:
            raise AssertionError(
                f"ValueError message {exc!s} did not match {match}"
            ) from exc
    else:
        raise AssertionError("expected ValueError")


def test_descriptor_set_lookup_and_ordinals_are_key_sorted() -> None:
    infos = sorted_descriptor_set_infos()
    keys = [info.key for info in infos]
    assert keys == sorted(keys)

    for ordinal, key in enumerate(keys):
        assert x86_descriptor_set_ordinal(key) == ordinal

    with _raises_value_error("unknown x86 descriptor set"):
        x86_descriptor_set_ordinal("x86.nope.core")


def test_descriptor_storage_target_lookup_classifies_current_views() -> None:
    assert (
        x86_descriptor_set_storage_info_by_generator_target("avx512").generator_target
        == "avx512_packed_dot"
    )
    assert (
        x86_descriptor_set_storage_info_by_generator_target(
            "avx512_packed_dot"
        ).generator_target
        == "avx512_packed_dot"
    )

    view_infos = x86_descriptor_set_view_infos_by_storage_generator_target(
        "avx512_packed_dot"
    )
    assert [info.generator_target for info in view_infos] == [
        "avx10_2",
        "avx2",
        "avx512",
        "avx512_bf16",
        "avx512_vnni",
        "avx_vnni",
        "avx_vnni_int16",
        "avx_vnni_int8",
        "packed_dot",
        "scalar",
        "simd128",
    ]


def test_packed_dot_feature_rows_record_feature_and_width_requirements() -> None:
    rows_by_target = {
        info.generator_target: info
        for info in x86_descriptor_set_view_infos_by_storage_generator_target(
            "avx512_packed_dot"
        )
    }

    assert rows_by_target["avx_vnni"].required_feature_bits == FEATURE_AVX_VNNI
    assert rows_by_target["avx_vnni"].register_classes == (
        X86_REG_CLASS_XMM,
        X86_REG_CLASS_YMM,
    )
    assert rows_by_target["avx512_vnni"].required_feature_bits == (
        FEATURE_AVX512_VNNI | FEATURE_AVX512_VL
    )
    assert rows_by_target["avx512_bf16"].required_feature_bits == (
        FEATURE_AVX512_BF16 | FEATURE_AVX512_VL
    )
    assert rows_by_target["avx_vnni_int8"].required_feature_bits == (
        FEATURE_AVX_VNNI_INT8
    )
    assert rows_by_target["avx_vnni_int16"].required_feature_bits == (
        FEATURE_AVX_VNNI_INT16
    )
    assert rows_by_target["avx10_2"].required_feature_bits == FEATURE_AVX10_2


def test_descriptor_view_lookup_rejects_view_storage_target() -> None:
    with _raises_value_error("is a view target, not a storage target"):
        x86_descriptor_set_view_infos_by_storage_generator_target("avx512")


def test_target_profile_lookup_records_current_feature_projection() -> None:
    profile = x86_target_profile_info_by_key("x86.avx512_packed_dot")
    assert profile.descriptor_set_key == "x86.avx512_packed_dot.core"
    assert profile.native_bundle_key == "x86-avx512-packed-dot"
    assert profile.register_classes == (
        X86_REG_CLASS_GPR32,
        X86_REG_CLASS_GPR64,
        X86_REG_CLASS_XMM,
        X86_REG_CLASS_YMM,
        X86_REG_CLASS_ZMM,
        X86_REG_CLASS_K,
    )


def test_target_profile_lookup_records_planned_scalar_and_packed_dot_rows() -> None:
    scalar = x86_target_profile_info_by_key("x86.scalar")
    assert scalar.descriptor_set_key == "x86.scalar.core"
    assert scalar.register_classes == (X86_REG_CLASS_GPR32, X86_REG_CLASS_GPR64)
    assert scalar.contract_feature_bits == 0
    assert scalar.native_bundle_key == "x86-scalar"

    simd128 = x86_target_profile_info_by_key("x86.simd128")
    assert simd128.descriptor_set_key == "x86.simd128.core"
    assert simd128.register_classes == (
        X86_REG_CLASS_GPR32,
        X86_REG_CLASS_GPR64,
        X86_REG_CLASS_XMM,
    )
    assert simd128.native_bundle_key == "x86-simd128"

    avx2 = x86_target_profile_info_by_key("x86.avx2")
    assert avx2.descriptor_set_key == "x86.avx2.core"
    assert avx2.register_classes == (
        X86_REG_CLASS_GPR32,
        X86_REG_CLASS_GPR64,
        X86_REG_CLASS_XMM,
        X86_REG_CLASS_YMM,
    )
    assert avx2.native_bundle_key == "x86-avx2"

    packed_dot = x86_target_profile_info_by_key("x86.packed_dot")
    assert packed_dot.contract_feature_bits == (
        FEATURE_AVX512_BF16
        | FEATURE_AVX512_VL
        | FEATURE_AVX_VNNI
        | FEATURE_AVX_VNNI_INT8
    )
    assert packed_dot.register_classes == (
        X86_REG_CLASS_XMM,
        X86_REG_CLASS_YMM,
        X86_REG_CLASS_ZMM,
    )


def test_target_profiles_are_profile_key_sorted() -> None:
    profile_keys = [info.profile_key for info in sorted_target_profile_infos()]
    assert profile_keys == sorted(profile_keys)


def test_native_selector_profiles_are_backed_by_native_bundles() -> None:
    bundle_keys = [
        x86_target_profile_info_by_key(profile_key).native_bundle_key
        for profile_key in X86_NATIVE_TARGET_SELECTOR_PROFILE_KEYS
        if profile_key is not None
    ]
    assert bundle_keys == [
        "x86-avx512",
        "x86-packed-dot",
        "x86-avx512-packed-dot",
        "x86-scalar",
        "x86-simd128",
        "x86-avx2",
    ]


def test_descriptor_generator_target_lookup_rejects_unknown_target() -> None:
    with _raises_value_error("unknown x86 descriptor generator target"):
        x86_descriptor_set_info_by_generator_target("bad_target")


def test_table_validation_rejects_duplicate_descriptor_keys() -> None:
    duplicate_sets = (
        *X86_DESCRIPTOR_SET_INFOS,
        X86DescriptorSetInfo(
            generator_target="duplicate_scalar",
            key="x86.scalar.core",
            isa_tier=X86_ISA_TIER_SCALAR,
            register_classes=(X86_REG_CLASS_GPR32,),
        ),
    )

    with _raises_value_error("duplicate x86 descriptor set key"):
        validate_x86_target_info_tables(
            descriptor_sets=duplicate_sets,
            target_profiles=X86_TARGET_PROFILE_INFOS,
        )


def test_table_validation_rejects_view_as_storage_target() -> None:
    descriptor_sets = (
        X86DescriptorSetInfo(
            generator_target="storage",
            key="x86.storage.core",
            isa_tier=X86_ISA_TIER_SCALAR,
            register_classes=(X86_REG_CLASS_GPR32,),
        ),
        X86DescriptorSetInfo(
            generator_target="view",
            key="x86.view.core",
            isa_tier=X86_ISA_TIER_SCALAR,
            register_classes=(X86_REG_CLASS_GPR32,),
            storage_generator_target="storage",
        ),
        X86DescriptorSetInfo(
            generator_target="bad_view",
            key="x86.bad_view.core",
            isa_tier=X86_ISA_TIER_SCALAR,
            register_classes=(X86_REG_CLASS_GPR32,),
            storage_generator_target="view",
        ),
    )

    with _raises_value_error("uses view-only target 'view' as storage"):
        validate_x86_target_info_tables(descriptor_sets=descriptor_sets)


def test_table_validation_rejects_profile_descriptor_mismatch() -> None:
    bad_profile = X86TargetProfileInfo(
        profile_key="x86.bad",
        descriptor_generator_target="scalar",
        descriptor_set_key="x86.other.core",
        register_classes=(X86_REG_CLASS_GPR32,),
    )

    with _raises_value_error("expects descriptor set 'x86.other.core'"):
        validate_x86_target_info_tables(
            target_profiles=(*X86_TARGET_PROFILE_INFOS, bad_profile)
        )
