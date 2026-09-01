# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.target.arch.amdgpu.target_catalog import (
    AMDGPU_GENERIC_CODE_OBJECT_INFOS,
    TARGET_ID_FEATURE_SRAMECC,
    TARGET_ID_FEATURE_XNACK,
    AmdgpuExactTargetInfo,
    AmdgpuGenericCodeObjectInfo,
    AmdgpuPhysicalTargetInfo,
    AmdgpuTargetOverlayInfo,
    generic_code_object_current_version,
    physical_target_info,
    processor_has_physical_target_infos,
    target_id_features_for_processor,
    target_processor,
    validate_amdgpu_target_catalog,
)


def test_canonical_catalog_is_closed_and_versioned() -> None:
    validate_amdgpu_target_catalog()
    assert {
        info.processor: generic_code_object_current_version(info.processor)
        for info in AMDGPU_GENERIC_CODE_OBJECT_INFOS
    } == {
        info.processor: info.current_version
        for info in AMDGPU_GENERIC_CODE_OBJECT_INFOS
    }


def test_target_id_features_cover_exact_and_generic_processors() -> None:
    assert target_id_features_for_processor("gfx942") == (
        TARGET_ID_FEATURE_SRAMECC,
        TARGET_ID_FEATURE_XNACK,
    )
    assert target_id_features_for_processor("gfx9-4-generic") == (
        TARGET_ID_FEATURE_SRAMECC,
        TARGET_ID_FEATURE_XNACK,
    )
    assert target_id_features_for_processor("gfx1151") == ()


def test_target_overlay_selects_one_backend_processor() -> None:
    assert target_processor("gfx1250-a0") == "gfx1250"
    assert target_processor("gfx1250") == "gfx1250"
    assert target_processor("gfx12-5-generic") == "gfx12-5-generic"
    assert target_processor("gfx-future") is None


def test_physical_revision_resolves_to_canonical_target() -> None:
    assert processor_has_physical_target_infos("gfx1250")
    assert not processor_has_physical_target_infos("gfx1100")
    assert physical_target_info("gfx1250", 0) == AmdgpuPhysicalTargetInfo(
        "gfx1250", 0, "gfx1250-a0"
    )
    assert physical_target_info("gfx1250", 1) == AmdgpuPhysicalTargetInfo(
        "gfx1250", 1, "gfx1250"
    )
    assert physical_target_info("gfx1250", 2) is None


def test_rejects_member_newer_than_generic_code_object() -> None:
    with pytest.raises(ValueError, match=r"outside .* supported range"):
        validate_amdgpu_target_catalog(
            (AmdgpuGenericCodeObjectInfo("gfx-test-generic", 1),),
            (AmdgpuExactTargetInfo("gfx-test", "gfx-test-generic", 2),),
            (),
            (),
        )


def test_rejects_overlay_for_unknown_processor() -> None:
    with pytest.raises(ValueError, match="unknown exact processor"):
        validate_amdgpu_target_catalog(
            (),
            (AmdgpuExactTargetInfo("gfx-test", "gfx-test", 0),),
            (AmdgpuTargetOverlayInfo("gfx-test-a0", "gfx-future"),),
            (),
        )


def test_rejects_physical_target_from_another_processor() -> None:
    with pytest.raises(ValueError, match="from another processor"):
        validate_amdgpu_target_catalog(
            (),
            (
                AmdgpuExactTargetInfo("gfx-test-a", "gfx-test-a", 0),
                AmdgpuExactTargetInfo("gfx-test-b", "gfx-test-b", 0),
            ),
            (AmdgpuTargetOverlayInfo("gfx-test-a-a0", "gfx-test-a"),),
            (
                AmdgpuPhysicalTargetInfo("gfx-test-b", 0, "gfx-test-a-a0"),
                AmdgpuPhysicalTargetInfo("gfx-test-b", 1, "gfx-test-b"),
            ),
        )
