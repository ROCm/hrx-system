# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from build_tools.amdgpu.target_map_data import AMDGPU_GENERIC_CODE_OBJECT_INFOS

from loom.target.arch.amdgpu.target_info import (
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION,
    AMDGPU_DESCRIPTOR_SET_INFOS,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12,
    AMDGPU_MATRIX_FEATURES_BY_PROFILE,
    AMDGPU_PROCESSOR_INFOS,
    amdgpu_descriptor_set_info_by_generator_target,
    amdgpu_descriptor_set_storage_info_by_generator_target,
    amdgpu_descriptor_set_view_infos_by_storage_generator_target,
    amdgpu_processor_info_by_name,
    validate_amdgpu_descriptor_set_isa_xml,
    validate_amdgpu_generic_contracts,
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
    assert (
        amdgpu_descriptor_set_storage_info_by_generator_target(
            "rdna4_gfx125x"
        ).generator_target
        == "rdna4_gfx125x"
    )
    assert (
        amdgpu_descriptor_set_storage_info_by_generator_target("cdna3").generator_target
        == "cdna3"
    )

    for storage_target, view_target in (
        ("cdna3", "gfx9_4_generic"),
        ("rdna3", "gfx11_generic"),
        ("rdna4", "gfx12_generic"),
        ("rdna4_gfx125x", "gfx12_5_generic"),
    ):
        view_info = amdgpu_descriptor_set_info_by_generator_target(view_target)
        assert (
            amdgpu_descriptor_set_storage_info_by_generator_target(
                view_target
            ).generator_target
            == storage_target
        )
        assert amdgpu_descriptor_set_view_infos_by_storage_generator_target(
            storage_target
        ) == (view_info,)


def test_generic_descriptor_sets_have_independent_contracts() -> None:
    cases = (
        (
            "gfx9_4_generic",
            "amdgpu.gfx9_4.generic.core",
            ("cdna3", "cdna4"),
            ("cdna3", "cdna4"),
            "gfx9-4-generic",
            ("gfx940", "gfx950"),
            False,
        ),
        (
            "gfx11_generic",
            "amdgpu.gfx11.generic.core",
            ("rdna3", "rdna3_5"),
            ("rdna3", "rdna3_5"),
            "gfx11-generic",
            ("gfx1100", "gfx1151"),
            False,
        ),
        (
            "gfx12_generic",
            "amdgpu.gfx12.generic.core",
            ("rdna4",),
            ("rdna4",),
            "gfx12-generic",
            ("gfx1200", "gfx1201"),
            True,
        ),
        (
            "gfx12_5_generic",
            "amdgpu.gfx12_5.generic.core",
            ("rdna4_gfx125x",),
            ("rdna4",),
            "gfx12-5-generic",
            ("gfx1250", "gfx1251"),
            True,
        ),
    )
    for (
        generator_target,
        descriptor_set_key,
        member_generator_targets,
        isa_xml_keys,
        generic_processor_name,
        exact_processor_names,
        supports_vopd,
    ) in cases:
        info = amdgpu_descriptor_set_info_by_generator_target(generator_target)
        assert info.key == descriptor_set_key
        assert info.member_generator_targets == member_generator_targets
        assert (
            tuple(isa_info.isa_xml_key for isa_info in info.isa_infos) == isa_xml_keys
        )
        assert (
            bool(info.flags & AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION)
            == supports_vopd
        )

        generic_processor = amdgpu_processor_info_by_name(generic_processor_name)
        assert generic_processor is not None
        assert generic_processor.descriptor_set.key == info.key
        for exact_processor_name in exact_processor_names:
            exact_processor = amdgpu_processor_info_by_name(exact_processor_name)
            assert exact_processor is not None
            assert exact_processor.descriptor_set.key != info.key


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
    assert set(
        AMDGPU_MATRIX_FEATURES_BY_PROFILE[
            AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC
        ]
    ) == (
        set(
            AMDGPU_MATRIX_FEATURES_BY_PROFILE[AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940]
        )
        & set(
            AMDGPU_MATRIX_FEATURES_BY_PROFILE[AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950]
        )
    )


def test_generic_contracts_are_portable_member_intersections() -> None:
    validate_amdgpu_generic_contracts(
        AMDGPU_PROCESSOR_INFOS, AMDGPU_DESCRIPTOR_SET_INFOS
    )


def test_rdna3_5_processors_use_gfx11_matrix_shapes() -> None:
    for processor_name in ("gfx1150", "gfx1151", "gfx1152", "gfx1153", "gfx1170"):
        processor_info = amdgpu_processor_info_by_name(processor_name)
        assert processor_info is not None
        assert (
            processor_info.features.matrix == AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11
        )


def test_generic_processor_elf_flags_use_canonical_code_object_versions() -> None:
    processor_infos = {info.processor: info for info in AMDGPU_PROCESSOR_INFOS}
    generic_processors = {info.processor for info in AMDGPU_GENERIC_CODE_OBJECT_INFOS}
    assert {
        processor for processor in processor_infos if processor.endswith("-generic")
    } == generic_processors
    for generic_info in AMDGPU_GENERIC_CODE_OBJECT_INFOS:
        assert (
            processor_infos[generic_info.processor].elf.generic_version
            == generic_info.current_version
        )
