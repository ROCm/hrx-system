# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

from build_tools.amdgpu.target_map_data import (
    AMDGPU_EXACT_TARGET_INFOS,
    AMDGPU_GENERIC_CODE_OBJECT_INFOS,
    AMDGPU_PHYSICAL_TARGET_INFOS,
    AMDGPU_TARGET_OVERLAY_INFOS,
)

from loom.target.arch.amdgpu.lds_bank_service import (
    AMDGPU_LDS_BANK_SERVICE_MODELS_WAVE32_B128_QUAD_PHASES,
)
from loom.target.arch.amdgpu.target_info import (
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_OCP_FP8_NONCANONICAL_NAN,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_ARITHMETIC,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_COMPARE,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_CONVERSION,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_DUAL_MOV_SRC2_CACHE,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_NUMERIC_MINMAX_MNEMONICS,
    AMDGPU_DESCRIPTOR_SET_INFOS,
    AMDGPU_GENERIC_MATRIX_FEATURE_EXCLUSIONS,
    AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT,
    AMDGPU_INSTRUCTION_CONSTRAINT_KNOWN_BITS,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12_5_GENERIC,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250,
    AMDGPU_MATRIX_FEATURES_BY_PROFILE,
    AMDGPU_PROCESSOR_INFOS,
    AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU,
    AMDGPU_TARGET_ID_FEATURE_SUPPORT_NONE,
    AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC,
    AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK,
    AMDGPU_TARGET_INFOS,
    AmdgpuOccupancyDomainInfo,
    AmdgpuOccupancyModelInfo,
    _occupancy_capacity,
    _occupancy_capacity_change_points,
    _validate_portable_occupancy_model,
    amdgpu_descriptor_set_info_by_generator_target,
    amdgpu_descriptor_set_storage_info_by_generator_target,
    amdgpu_descriptor_set_supported_target_contract_keys,
    amdgpu_descriptor_set_view_infos_by_storage_generator_target,
    amdgpu_generic_code_object_compatibility_info,
    amdgpu_target_descriptor_set_key,
    amdgpu_target_info_by_name,
    amdgpu_target_instruction_constraints,
    validate_amdgpu_code_object_processor_rows,
    validate_amdgpu_descriptor_set_isa_xml,
    validate_amdgpu_generic_contracts,
    validate_amdgpu_target_id_processor_rows,
    validate_amdgpu_target_rows,
)


@dataclass(frozen=True, slots=True)
class _IsaArchitecture:
    source_name: str
    architecture_name: str
    architecture_id: int


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


def test_descriptor_set_isa_xml_validation_accepts_matching_architecture() -> None:
    spec = _IsaArchitecture(
        source_name="rdna4.xml",
        architecture_name="AMD RDNA 4",
        architecture_id=10,
    )

    validate_amdgpu_descriptor_set_isa_xml(
        amdgpu_descriptor_set_info_by_generator_target("rdna4"), spec
    )
    validate_amdgpu_descriptor_set_isa_xml(
        amdgpu_descriptor_set_info_by_generator_target("rdna4_gfx125x"), spec
    )


def test_noncanonical_native_fp8_nan_is_scoped_to_gfx12_descriptor_sets() -> None:
    flagged_generator_targets = {
        info.generator_target
        for info in AMDGPU_DESCRIPTOR_SET_INFOS
        if info.flags & AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_OCP_FP8_NONCANONICAL_NAN
    }
    assert flagged_generator_targets == {"rdna4", "gfx12_generic"}


def test_native_scalar_float_arithmetic_is_scoped_to_rdna35_and_newer() -> None:
    flagged_generator_targets = {
        info.generator_target
        for info in AMDGPU_DESCRIPTOR_SET_INFOS
        if info.flags & AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_ARITHMETIC
    }
    assert flagged_generator_targets == {
        "rdna3_5",
        "rdna4m",
        "rdna4",
        "rdna4_gfx1250_a0",
        "rdna4_gfx1251",
        "rdna4_gfx125x",
        "gfx12_generic",
        "gfx12_5_generic",
    }

    conversion_generator_targets = {
        info.generator_target
        for info in AMDGPU_DESCRIPTOR_SET_INFOS
        if info.flags & AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_CONVERSION
    }
    assert conversion_generator_targets == flagged_generator_targets

    compare_generator_targets = {
        info.generator_target
        for info in AMDGPU_DESCRIPTOR_SET_INFOS
        if info.flags & AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_COMPARE
    }
    assert compare_generator_targets == flagged_generator_targets


def test_numeric_minmax_mnemonics_are_scoped_to_rdna4_and_newer() -> None:
    flagged_generator_targets = {
        info.generator_target
        for info in AMDGPU_DESCRIPTOR_SET_INFOS
        if info.flags & AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_NUMERIC_MINMAX_MNEMONICS
    }
    assert flagged_generator_targets == {
        "rdna4m",
        "rdna4",
        "rdna4_gfx1250_a0",
        "rdna4_gfx1251",
        "rdna4_gfx125x",
        "gfx12_generic",
        "gfx12_5_generic",
    }


def test_vopd_dual_move_src2_cache_is_scoped_to_gfx117x_and_newer() -> None:
    flagged_generator_targets = {
        info.generator_target
        for info in AMDGPU_DESCRIPTOR_SET_INFOS
        if info.flags & AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_DUAL_MOV_SRC2_CACHE
    }
    assert flagged_generator_targets == {
        "rdna4m",
        "rdna4",
        "rdna4_gfx1250_a0",
        "rdna4_gfx1251",
        "rdna4_gfx125x",
        "gfx12_generic",
        "gfx12_5_generic",
    }


def test_descriptor_set_isa_xml_validation_rejects_mismatched_architecture() -> None:
    spec = _IsaArchitecture(
        source_name="rdna4.xml",
        architecture_name="AMD RDNA 4",
        architecture_id=10,
    )

    with _raises_value_error(
        "amdgpu.rdna3.core expects AMD RDNA 3 architecture id 8, "
        "found AMD RDNA 4 architecture id 10"
    ):
        validate_amdgpu_descriptor_set_isa_xml(
            amdgpu_descriptor_set_info_by_generator_target("rdna3"), spec
        )


def test_rdna4m_processors_publish_gfx12_matrix_contracts() -> None:
    processors = {
        info.processor: info
        for info in AMDGPU_PROCESSOR_INFOS
        if info.processor in ("gfx1170", "gfx1171", "gfx1172")
    }

    assert set(processors) == {"gfx1170", "gfx1171", "gfx1172"}
    assert all(
        info.features.matrix == AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12
        for info in processors.values()
    )


def test_rdna4m_processors_publish_delay_alu_scheduling() -> None:
    processors = {
        info.processor: info
        for info in AMDGPU_PROCESSOR_INFOS
        if info.processor in ("gfx1170", "gfx1171", "gfx1172")
    }

    assert set(processors) == {"gfx1170", "gfx1171", "gfx1172"}
    assert all(
        info.features.scheduling == AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU
        for info in processors.values()
    )


def test_descriptor_set_generator_target_lookup_rejects_unknown_target() -> None:
    with _raises_value_error("unknown AMDGPU descriptor generator target"):
        amdgpu_descriptor_set_info_by_generator_target("gfx999")


def test_descriptor_set_storage_target_lookup_classifies_storage_targets() -> None:
    storage_infos = tuple(
        info
        for info in AMDGPU_DESCRIPTOR_SET_INFOS
        if info.storage_generator_target is None
    )
    view_infos = tuple(
        info
        for info in AMDGPU_DESCRIPTOR_SET_INFOS
        if info.storage_generator_target is not None
    )
    assert storage_infos
    assert view_infos

    for storage_info in storage_infos:
        assert (
            amdgpu_descriptor_set_storage_info_by_generator_target(
                storage_info.generator_target
            )
            == storage_info
        )
        assert amdgpu_descriptor_set_view_infos_by_storage_generator_target(
            storage_info.generator_target
        ) == tuple(
            sorted(
                (
                    info
                    for info in view_infos
                    if info.storage_generator_target == storage_info.generator_target
                ),
                key=lambda info: info.key,
            )
        )

    for view_info in view_infos:
        assert (
            amdgpu_descriptor_set_storage_info_by_generator_target(
                view_info.generator_target
            ).generator_target
            == view_info.storage_generator_target
        )


def test_generic_descriptor_sets_have_independent_contracts() -> None:
    descriptor_sets_by_key = {info.key: info for info in AMDGPU_DESCRIPTOR_SET_INFOS}
    processors_by_name = {info.processor: info for info in AMDGPU_PROCESSOR_INFOS}
    generic_processors = tuple(
        info for info in AMDGPU_PROCESSOR_INFOS if info.processor.endswith("-generic")
    )
    compiler_generic_processors = tuple(
        info for info in generic_processors if info.descriptor_set.key
    )
    generic_descriptor_sets = tuple(
        info for info in AMDGPU_DESCRIPTOR_SET_INFOS if info.member_generator_targets
    )
    assert {info.descriptor_set.key for info in compiler_generic_processors} == {
        info.key for info in generic_descriptor_sets
    }

    for generic_processor in compiler_generic_processors:
        descriptor_set = descriptor_sets_by_key[generic_processor.descriptor_set.key]
        exact_members = tuple(
            processors_by_name[compatibility.exact_processor]
            for compatibility in AMDGPU_EXACT_TARGET_INFOS
            if compatibility.code_object_processor == generic_processor.processor
            and compatibility.generic_introduction_version
            <= generic_processor.elf.generic_version
        )
        assert exact_members
        assert all(
            member.descriptor_set.key != descriptor_set.key for member in exact_members
        )
        assert {
            descriptor_sets_by_key[member.descriptor_set.key].generator_target
            for member in exact_members
        } == set(descriptor_set.member_generator_targets)


def test_generic_descriptor_sets_derive_supported_target_contracts() -> None:
    expected_contract_keys_by_generator_target = {
        "gfx9_4_generic": (
            "amdgpu.cdna3.core",
            "amdgpu.cdna4.core",
        ),
        "gfx11_generic": (
            "amdgpu.rdna3.core",
            "amdgpu.rdna3_5.core",
        ),
        "gfx12_generic": ("amdgpu.rdna4.core",),
        "gfx12_5_generic": (
            "amdgpu.rdna4.gfx1250_a0.core",
            "amdgpu.rdna4.gfx1251.core",
            "amdgpu.rdna4.gfx125x.core",
        ),
    }

    for (
        generator_target,
        expected_contract_keys,
    ) in expected_contract_keys_by_generator_target.items():
        descriptor_set = amdgpu_descriptor_set_info_by_generator_target(
            generator_target
        )
        assert (
            amdgpu_descriptor_set_supported_target_contract_keys(descriptor_set)
            == expected_contract_keys
        )

    exact_descriptor_set = amdgpu_descriptor_set_info_by_generator_target("rdna3_5")
    assert (
        amdgpu_descriptor_set_supported_target_contract_keys(exact_descriptor_set) == ()
    )


def test_matrix_feature_profiles_model_replacement_instruction_shapes() -> None:
    assert (
        "mfma_gfx940_xf32"
        in AMDGPU_MATRIX_FEATURES_BY_PROFILE[AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940]
    )
    assert (
        "mfma_gfx940_xf32"
        not in AMDGPU_MATRIX_FEATURES_BY_PROFILE[
            AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950
        ]
    )
    assert (
        "wmma_gfx11"
        not in AMDGPU_MATRIX_FEATURES_BY_PROFILE[
            AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12
        ]
    )
    generic_features = set(
        AMDGPU_MATRIX_FEATURES_BY_PROFILE[
            AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC
        ]
    )
    assert "mfma_gfx940_i8" in generic_features
    assert "mfma_gfx940_fp8" not in generic_features
    assert "smfmac_gfx940_fp8" not in generic_features
    member_intersection = set(
        AMDGPU_MATRIX_FEATURES_BY_PROFILE[AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940]
    ) & set(
        AMDGPU_MATRIX_FEATURES_BY_PROFILE[AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950]
    )
    assert generic_features == (
        member_intersection
        - set(AMDGPU_GENERIC_MATRIX_FEATURE_EXCLUSIONS["gfx9-4-generic"])
    )
    gfx12_5_generic_features = set(
        AMDGPU_MATRIX_FEATURES_BY_PROFILE[
            AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12_5_GENERIC
        ]
    )
    assert gfx12_5_generic_features == (
        set(
            AMDGPU_MATRIX_FEATURES_BY_PROFILE[
                AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250
            ]
        )
        - set(AMDGPU_GENERIC_MATRIX_FEATURE_EXCLUSIONS["gfx12-5-generic"])
    )


def test_generic_contracts_are_portable_member_intersections() -> None:
    validate_amdgpu_generic_contracts(
        AMDGPU_PROCESSOR_INFOS, AMDGPU_DESCRIPTOR_SET_INFOS
    )


def test_occupancy_capacity_change_points_cover_every_positive_demand() -> None:
    maximum_units = 257
    granularities = (3, 8, 24)
    change_points = _occupancy_capacity_change_points(maximum_units, granularities)

    for units in range(1, maximum_units + 1):
        representative = max(point for point in change_points if point <= units)
        for granularity in granularities:
            assert (units + granularity - 1) // granularity == (
                representative + granularity - 1
            ) // granularity
            for pool_units in (17, 64, maximum_units):
                assert _occupancy_capacity(
                    pool_units, granularity, units
                ) == _occupancy_capacity(pool_units, granularity, representative)


def test_occupancy_validation_checks_member_capacity_change_points() -> None:
    generic_model = AmdgpuOccupancyModelInfo(
        max_waves_per_simd=16,
        domain=AmdgpuOccupancyDomainInfo(
            simd_count=4,
            local_memory_bytes=64,
            local_memory_allocation_granularity=16,
            max_barrier_workgroup_count=16,
        ),
        register_classes=(),
    )
    member_model = replace(
        generic_model,
        domain=replace(
            generic_model.domain,
            local_memory_bytes=48,
            local_memory_allocation_granularity=8,
        ),
    )

    with _raises_value_error("overstates local-memory capacity at 9 bytes"):
        _validate_portable_occupancy_model(
            "test-generic", 32, generic_model, (member_model,)
        )


def test_instruction_constraints_are_attached_to_canonical_targets() -> None:
    processors = {info.processor: info for info in AMDGPU_PROCESSOR_INFOS}
    gfx1250 = processors["gfx1250"]
    gfx12_5_generic = processors["gfx12-5-generic"]
    gfx1250_target = amdgpu_target_info_by_name("gfx1250")
    gfx1250_a0_target = amdgpu_target_info_by_name("gfx1250-a0")
    gfx12_5_generic_target = amdgpu_target_info_by_name("gfx12-5-generic")
    assert gfx1250_target is not None
    assert gfx1250_a0_target is not None
    assert gfx12_5_generic_target is not None

    a0_constraints = gfx1250_a0_target.semantics.instruction_constraints
    assert gfx1250.instructions.base_constraints == 0
    assert amdgpu_target_instruction_constraints(gfx1250_target, gfx1250) == 0
    assert (
        amdgpu_target_instruction_constraints(gfx1250_a0_target, gfx1250)
        == a0_constraints
    )
    assert a0_constraints != 0
    assert a0_constraints & ~AMDGPU_INSTRUCTION_CONSTRAINT_KNOWN_BITS == 0
    assert (
        amdgpu_target_instruction_constraints(gfx12_5_generic_target, gfx12_5_generic)
        == a0_constraints
    )


def test_lds_bank_service_models_are_structural_target_data() -> None:
    processors = {info.processor: info for info in AMDGPU_PROCESSOR_INFOS}
    gfx1250 = processors["gfx1250"]
    gfx1250_a0 = amdgpu_target_info_by_name("gfx1250-a0")
    assert gfx1250_a0 is not None

    assert (
        gfx1250.features.lds_bank_service_models
        == AMDGPU_LDS_BANK_SERVICE_MODELS_WAVE32_B128_QUAD_PHASES
    )
    assert gfx1250_a0.semantics.lds_bank_service_models is None
    assert processors["gfx1251"].features.lds_bank_service_models == ()
    assert processors["gfx12-5-generic"].features.lds_bank_service_models == ()


def test_physical_targets_resolve_to_canonical_target_rows() -> None:
    targets = {target.target: target for target in AMDGPU_TARGET_INFOS}
    mappings = {
        (physical.processor, physical.asic_revision): physical.target
        for physical in AMDGPU_PHYSICAL_TARGET_INFOS
    }

    assert mappings == {
        ("gfx1250", 0): "gfx1250-a0",
        ("gfx1250", 1): "gfx1250",
    }
    for physical in AMDGPU_PHYSICAL_TARGET_INFOS:
        assert targets[physical.target].processor == physical.processor


def test_target_semantics_are_keyed_by_canonical_target() -> None:
    processors = {info.processor: info for info in AMDGPU_PROCESSOR_INFOS}
    gfx1250 = amdgpu_target_info_by_name("gfx1250")
    gfx1250_a0 = amdgpu_target_info_by_name("gfx1250-a0")
    assert gfx1250 is not None
    assert gfx1250_a0 is not None

    assert gfx1250.semantics.kernel_metadata_extensions == (
        (".gfx1250_revision", "B0"),
    )
    assert gfx1250_a0.semantics.kernel_metadata_extensions == (
        (".gfx1250_revision", "A0"),
    )
    assert (
        amdgpu_target_descriptor_set_key(gfx1250, processors["gfx1250"])
        == "amdgpu.rdna4.gfx125x.core"
    )
    assert (
        amdgpu_target_descriptor_set_key(gfx1250_a0, processors["gfx1250"])
        == "amdgpu.rdna4.gfx1250_a0.core"
    )


def test_target_rows_reject_noncanonical_overlay_identity() -> None:
    targets = list(AMDGPU_TARGET_INFOS)
    overlay_index = next(
        index for index, target in enumerate(targets) if target.target == "gfx1250-a0"
    )
    targets[overlay_index] = replace(
        targets[overlay_index],
        target="gfx1250-experimental",
    )

    with _raises_value_error("omit canonical overlays"):
        validate_amdgpu_target_rows(
            AMDGPU_PROCESSOR_INFOS,
            targets,
        )


def test_target_rows_reject_unrelated_descriptor_override() -> None:
    targets = list(AMDGPU_TARGET_INFOS)
    overlay_index = next(
        index for index, target in enumerate(targets) if target.target == "gfx1250-a0"
    )
    overlay = targets[overlay_index]
    targets[overlay_index] = replace(
        overlay,
        semantics=replace(
            overlay.semantics,
            descriptor_set_key="amdgpu.rdna4.core",
        ),
    )

    with _raises_value_error("does not view its processor descriptor contract"):
        validate_amdgpu_target_rows(
            AMDGPU_PROCESSOR_INFOS,
            targets,
        )


def test_generic_contracts_reject_duplicated_member_constraints() -> None:
    targets = list(AMDGPU_TARGET_INFOS)
    generic_index = next(
        index for index, info in enumerate(targets) if info.target == "gfx12-5-generic"
    )
    generic = targets[generic_index]
    targets[generic_index] = replace(
        generic,
        semantics=replace(
            generic.semantics,
            instruction_constraints=(
                AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT
            ),
        ),
    )

    with _raises_value_error("duplicates derived member instruction constraints"):
        validate_amdgpu_generic_contracts(
            AMDGPU_PROCESSOR_INFOS,
            AMDGPU_DESCRIPTOR_SET_INFOS,
            targets,
        )


def test_physical_target_models_reject_nonportable_processor_base() -> None:
    targets = list(AMDGPU_TARGET_INFOS)
    target_index = next(
        index for index, info in enumerate(targets) if info.target == "gfx1250-a0"
    )
    target = targets[target_index]
    targets[target_index] = replace(
        target,
        semantics=replace(target.semantics, lds_bank_service_models=()),
    )

    with _raises_value_error(
        "gfx1250 LDS bank-service models do not match its physical target intersection"
    ):
        validate_amdgpu_target_rows(AMDGPU_PROCESSOR_INFOS, targets)


def test_generic_models_reject_nonportable_member_model() -> None:
    processors = list(AMDGPU_PROCESSOR_INFOS)
    generic_index = next(
        index
        for index, info in enumerate(processors)
        if info.processor == "gfx12-5-generic"
    )
    generic = processors[generic_index]
    processors[generic_index] = replace(
        generic,
        features=replace(
            generic.features,
            lds_bank_service_models=(
                AMDGPU_LDS_BANK_SERVICE_MODELS_WAVE32_B128_QUAD_PHASES
            ),
        ),
    )

    with _raises_value_error(
        "gfx12-5-generic LDS bank-service models do not match the member intersection"
    ):
        validate_amdgpu_generic_contracts(processors, AMDGPU_DESCRIPTOR_SET_INFOS)


def test_rdna3_5_processors_use_gfx11_matrix_shapes() -> None:
    descriptor_set = amdgpu_descriptor_set_info_by_generator_target("rdna3_5")
    processors = tuple(
        info
        for info in AMDGPU_PROCESSOR_INFOS
        if info.descriptor_set.key == descriptor_set.key
    )
    assert {processor.processor for processor in processors} == {
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1153",
    }
    for processor in processors:
        assert processor.features.matrix == AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11


def test_rdna4m_processors_use_distinct_descriptor_contract() -> None:
    descriptor_set = amdgpu_descriptor_set_info_by_generator_target("rdna4m")
    processors = {
        info.processor
        for info in AMDGPU_PROCESSOR_INFOS
        if info.descriptor_set.key == descriptor_set.key
    }
    assert processors == {"gfx1170", "gfx1171", "gfx1172"}


def test_processor_rows_cover_canonical_code_object_relation() -> None:
    processor_infos = {info.processor: info for info in AMDGPU_PROCESSOR_INFOS}
    validate_amdgpu_code_object_processor_rows(AMDGPU_PROCESSOR_INFOS)

    generic_processors = {info.processor for info in AMDGPU_GENERIC_CODE_OBJECT_INFOS}
    assert {
        processor for processor in processor_infos if processor.endswith("-generic")
    } == generic_processors
    for generic_info in AMDGPU_GENERIC_CODE_OBJECT_INFOS:
        assert (
            processor_infos[generic_info.processor].elf.generic_version
            == generic_info.current_version
        )

    for compatibility in AMDGPU_EXACT_TARGET_INFOS:
        assert compatibility.exact_processor in processor_infos
        assert compatibility.code_object_processor in processor_infos
        derived_compatibility = amdgpu_generic_code_object_compatibility_info(
            compatibility.exact_processor
        )
        if compatibility.generic_introduction_version != 0:
            assert derived_compatibility == compatibility
            generic_processor = processor_infos[compatibility.code_object_processor]
            assert (
                compatibility.generic_introduction_version
                <= generic_processor.elf.generic_version
            )
        else:
            assert derived_compatibility is None


def test_processor_rows_cover_canonical_target_id_qualification() -> None:
    processors = {info.processor: info for info in AMDGPU_PROCESSOR_INFOS}
    validate_amdgpu_target_id_processor_rows(AMDGPU_PROCESSOR_INFOS)

    assert processors["gfx942"].target_id.supported_features == (
        AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC
        | AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK
    )
    assert processors["gfx9-4-generic"].target_id.supported_features == (
        AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC
        | AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK
    )
    assert (
        processors["gfx1151"].target_id.supported_features
        == AMDGPU_TARGET_ID_FEATURE_SUPPORT_NONE
    )
    assert (
        processors["gfx1250"].target_id.supported_features
        == AMDGPU_TARGET_ID_FEATURE_SUPPORT_NONE
    )
    assert (
        processors["gfx1251"].target_id.supported_features
        == AMDGPU_TARGET_ID_FEATURE_SUPPORT_NONE
    )


def test_target_rows_cover_canonical_target_map() -> None:
    validate_amdgpu_target_rows(AMDGPU_PROCESSOR_INFOS, AMDGPU_TARGET_INFOS)
    processor_names = {processor.processor for processor in AMDGPU_PROCESSOR_INFOS}
    target_names = {target.target for target in AMDGPU_TARGET_INFOS}
    assert all(target.processor in processor_names for target in AMDGPU_TARGET_INFOS)
    assert {
        info.target
        for info in AMDGPU_TARGET_OVERLAY_INFOS
        if info.processor in target_names
    }.issubset(target_names)


def test_target_rows_reject_non_dense_enum_values() -> None:
    targets = list(AMDGPU_TARGET_INFOS)
    target_index = next(
        index for index, info in enumerate(targets) if info.target == "gfx1250-a0"
    )
    targets[target_index] = replace(
        targets[target_index],
        enum_value=100,
    )

    with _raises_value_error("dense and one-based"):
        validate_amdgpu_target_rows(AMDGPU_PROCESSOR_INFOS, targets)


def test_code_object_relation_rejects_missing_canonical_processor() -> None:
    processors = tuple(
        info for info in AMDGPU_PROCESSOR_INFOS if info.processor != "gfx1151"
    )
    with _raises_value_error("missing canonical.*gfx1151"):
        validate_amdgpu_code_object_processor_rows(processors)


def test_target_id_qualification_rejects_missing_canonical_processor() -> None:
    processors = tuple(
        info for info in AMDGPU_PROCESSOR_INFOS if info.processor != "gfx1151"
    )
    with _raises_value_error("missing canonical target-ID.*gfx1151"):
        validate_amdgpu_target_id_processor_rows(processors)


def test_target_kernel_metadata_extensions_reject_invalid_rows() -> None:
    targets = list(AMDGPU_TARGET_INFOS)
    target_index = next(
        index for index, info in enumerate(targets) if info.target == "gfx1250-a0"
    )
    target = targets[target_index]
    targets[target_index] = replace(
        target,
        semantics=replace(
            target.semantics,
            kernel_metadata_extensions=(
                (".z_target", "A0"),
                (".a_target", "A0"),
            ),
        ),
    )

    with _raises_value_error("metadata keys are not unique and sorted"):
        validate_amdgpu_target_rows(AMDGPU_PROCESSOR_INFOS, targets)


def test_target_kernel_metadata_extensions_reject_standard_field_collision() -> None:
    targets = list(AMDGPU_TARGET_INFOS)
    target_index = next(
        index for index, info in enumerate(targets) if info.target == "gfx1250-a0"
    )
    target = targets[target_index]
    targets[target_index] = replace(
        target,
        semantics=replace(
            target.semantics,
            kernel_metadata_extensions=((".wavefront_size", "32"),),
        ),
    )

    with _raises_value_error("replaces a standard kernel metadata field"):
        validate_amdgpu_target_rows(AMDGPU_PROCESSOR_INFOS, targets)
