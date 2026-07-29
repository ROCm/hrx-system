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
)

from loom.target.arch.amdgpu.lds_bank_service import (
    AMDGPU_LDS_BANK_SERVICE_MODELS_WAVE32_B128_QUAD_PHASES,
)
from loom.target.arch.amdgpu.target_info import (
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
    AMDGPU_TARGET_ID_FEATURE_SUPPORT_NONE,
    AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC,
    AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK,
    AmdgpuProcessorAsicRevisionSemantics,
    amdgpu_descriptor_set_info_by_generator_target,
    amdgpu_descriptor_set_storage_info_by_generator_target,
    amdgpu_descriptor_set_view_infos_by_storage_generator_target,
    amdgpu_generic_code_object_compatibility_info,
    amdgpu_processor_default_instruction_constraints,
    processor_info,
    validate_amdgpu_code_object_processor_rows,
    validate_amdgpu_descriptor_set_isa_xml,
    validate_amdgpu_generic_contracts,
    validate_amdgpu_processor_revision_rows,
    validate_amdgpu_target_id_processor_rows,
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


def test_instruction_constraints_separate_default_and_revision_behavior() -> None:
    processors = {info.processor: info for info in AMDGPU_PROCESSOR_INFOS}
    gfx1250 = processors["gfx1250"]
    revisions = {revision.name: revision for revision in gfx1250.asic_revisions}

    assert gfx1250.instructions.base_constraints == 0
    assert revisions["a0"].instruction_constraints != 0
    assert (
        revisions["a0"].instruction_constraints
        & ~AMDGPU_INSTRUCTION_CONSTRAINT_KNOWN_BITS
        == 0
    )
    assert revisions["b0"].instruction_constraints == 0
    assert gfx1250.default_asic_revision == revisions["b0"].value
    assert amdgpu_processor_default_instruction_constraints(gfx1250) == 0
    assert (
        amdgpu_processor_default_instruction_constraints(processors["gfx12-5-generic"])
        == 0
    )


def test_lds_bank_service_models_are_structural_target_data() -> None:
    processors = {info.processor: info for info in AMDGPU_PROCESSOR_INFOS}
    gfx1250 = processors["gfx1250"]

    assert (
        gfx1250.features.lds_bank_service_models
        == AMDGPU_LDS_BANK_SERVICE_MODELS_WAVE32_B128_QUAD_PHASES
    )
    assert all(
        revision.lds_bank_service_models == gfx1250.features.lds_bank_service_models
        for revision in gfx1250.asic_revisions
    )
    assert processors["gfx1251"].features.lds_bank_service_models == ()
    assert processors["gfx12-5-generic"].features.lds_bank_service_models == ()


def test_revision_semantics_join_canonical_revision_identity() -> None:
    processor = processor_info(
        "gfx1250",
        0x049,
        asic_revision_semantics=(
            AmdgpuProcessorAsicRevisionSemantics(
                value=0,
                instruction_constraints=(
                    AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT
                ),
                kernel_metadata_extensions=((".test_revision", "A0"),),
            ),
        ),
    )

    assert tuple(
        (revision.value, revision.name) for revision in processor.asic_revisions
    ) == ((0, "a0"), (1, "b0"))
    assert processor.asic_revisions[0].instruction_constraints == (
        AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT
    )
    assert processor.asic_revisions[0].kernel_metadata_extensions == (
        (".test_revision", "A0"),
    )
    assert processor.asic_revisions[1].instruction_constraints == 0
    assert processor.asic_revisions[1].kernel_metadata_extensions == ()


def test_revision_semantics_reject_unknown_canonical_revision() -> None:
    with _raises_value_error("semantic overlays for unknown ASIC revisions: 7"):
        processor_info(
            "gfx1250",
            0x049,
            asic_revision_semantics=(AmdgpuProcessorAsicRevisionSemantics(value=7),),
        )


def test_revision_semantics_reject_noncanonical_order() -> None:
    with _raises_value_error("semantic overlays are not in ascending value order"):
        processor_info(
            "gfx1250",
            0x049,
            asic_revision_semantics=(
                AmdgpuProcessorAsicRevisionSemantics(value=1),
                AmdgpuProcessorAsicRevisionSemantics(value=0),
            ),
        )


def test_generic_contracts_reject_nonportable_instruction_constraints() -> None:
    processors = list(AMDGPU_PROCESSOR_INFOS)
    generic_index = next(
        index
        for index, info in enumerate(processors)
        if info.processor == "gfx12-5-generic"
    )
    generic = processors[generic_index]
    processors[generic_index] = replace(
        generic,
        instructions=replace(
            generic.instructions,
            base_constraints=(
                AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT
            ),
        ),
    )

    with _raises_value_error(
        "gfx12-5-generic instruction constraints do not match the union"
    ):
        validate_amdgpu_generic_contracts(processors, AMDGPU_DESCRIPTOR_SET_INFOS)


def test_revision_models_reject_nonportable_processor_base() -> None:
    processors = list(AMDGPU_PROCESSOR_INFOS)
    processor_index = next(
        index for index, info in enumerate(processors) if info.processor == "gfx1250"
    )
    processor = processors[processor_index]
    revisions = list(processor.asic_revisions)
    revisions[0] = replace(revisions[0], lds_bank_service_models=())
    processors[processor_index] = replace(
        processor,
        asic_revisions=tuple(revisions),
    )

    with _raises_value_error(
        "gfx1250 LDS bank-service models do not match the ASIC-revision intersection"
    ):
        validate_amdgpu_processor_revision_rows(processors)


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
    assert processors
    for processor in processors:
        assert processor.features.matrix == AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11


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


def test_processor_rows_cover_canonical_physical_revisions() -> None:
    validate_amdgpu_processor_revision_rows(AMDGPU_PROCESSOR_INFOS)


def test_processor_revisions_reject_noncanonical_default() -> None:
    processors = list(AMDGPU_PROCESSOR_INFOS)
    processor_index = next(
        index for index, info in enumerate(processors) if info.processor == "gfx1250"
    )
    processors[processor_index] = replace(
        processors[processor_index],
        default_asic_revision=0,
    )

    with _raises_value_error("default ASIC revision disagrees"):
        validate_amdgpu_processor_revision_rows(processors)


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


def test_revision_kernel_metadata_extensions_reject_invalid_rows() -> None:
    processors = list(AMDGPU_PROCESSOR_INFOS)
    processor_index = next(
        index for index, info in enumerate(processors) if info.processor == "gfx1250"
    )
    processor = processors[processor_index]
    revisions = list(processor.asic_revisions)
    revisions[0] = replace(
        revisions[0],
        kernel_metadata_extensions=(
            (".z_revision", "A0"),
            (".a_revision", "A0"),
        ),
    )
    processors[processor_index] = replace(
        processor,
        asic_revisions=tuple(revisions),
    )

    with _raises_value_error("metadata keys are not unique and sorted"):
        validate_amdgpu_processor_revision_rows(processors)


def test_revision_kernel_metadata_extensions_reject_standard_field_collision() -> None:
    processors = list(AMDGPU_PROCESSOR_INFOS)
    processor_index = next(
        index for index, info in enumerate(processors) if info.processor == "gfx1250"
    )
    processor = processors[processor_index]
    revisions = list(processor.asic_revisions)
    revisions[0] = replace(
        revisions[0],
        kernel_metadata_extensions=((".wavefront_size", "32"),),
    )
    processors[processor_index] = replace(
        processor,
        asic_revisions=tuple(revisions),
    )

    with _raises_value_error("replaces a standard kernel metadata field"):
        validate_amdgpu_processor_revision_rows(processors)
