# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest
from build_tools.amdgpu.target_map_data import (
    AMDGPU_TARGET_ID_FEATURE_ORDER,
    target_id_features_for_processor,
)

from loom.target.arch.amdgpu.target_identity import (
    AmdgpuArtifactTargetKeyError,
    AmdgpuTargetFeatureState,
    parse_amdgpu_artifact_target_key,
)
from loom.target.arch.amdgpu.target_info import AMDGPU_TARGET_INFOS


def test_resolves_target_overlay_without_losing_processor_identity() -> None:
    identity = parse_amdgpu_artifact_target_key("gfx1250-a0")

    assert identity is not None
    assert identity.target.target == "gfx1250-a0"
    assert identity.processor.processor == "gfx1250"
    assert identity.feature_state("sramecc") is AmdgpuTargetFeatureState.UNSUPPORTED
    assert identity.feature_state("xnack") is AmdgpuTargetFeatureState.UNSUPPORTED


def test_resolves_qualified_target_into_structured_feature_states() -> None:
    identity = parse_amdgpu_artifact_target_key("gfx942:xnack-:sramecc+")

    assert identity is not None
    assert identity.target.target == "gfx942"
    assert identity.processor.processor == "gfx942"
    assert identity.feature_state("sramecc") is AmdgpuTargetFeatureState.ON
    assert identity.feature_state("xnack") is AmdgpuTargetFeatureState.OFF
    assert [feature.name for feature in identity.amdhsa_features] == list(
        AMDGPU_TARGET_ID_FEATURE_ORDER
    )


def test_bare_artifact_keys_cover_every_canonical_target() -> None:
    for target in AMDGPU_TARGET_INFOS:
        identity = parse_amdgpu_artifact_target_key(target.target)

        assert identity is not None
        assert identity.target is target
        assert identity.processor.processor == target.processor
        supported_feature_names = set(
            target_id_features_for_processor(target.processor)
        )
        for feature in identity.amdhsa_features:
            expected_state = (
                AmdgpuTargetFeatureState.ANY
                if feature.name in supported_feature_names
                else AmdgpuTargetFeatureState.UNSUPPORTED
            )
            assert feature.state is expected_state


def test_feature_coordinates_cover_every_canonical_target() -> None:
    for target in AMDGPU_TARGET_INFOS:
        supported_feature_names = set(
            target_id_features_for_processor(target.processor)
        )
        for feature_name in AMDGPU_TARGET_ID_FEATURE_ORDER:
            for suffix, expected_state in (
                ("-", AmdgpuTargetFeatureState.OFF),
                ("+", AmdgpuTargetFeatureState.ON),
            ):
                target_key = f"{target.target}:{feature_name}{suffix}"
                if feature_name not in supported_feature_names:
                    with pytest.raises(AmdgpuArtifactTargetKeyError):
                        parse_amdgpu_artifact_target_key(target_key)
                    continue
                identity = parse_amdgpu_artifact_target_key(target_key)

                assert identity is not None
                assert identity.target is target
                assert identity.feature_state(feature_name) is expected_state


@pytest.mark.parametrize(
    "value",
    [
        "",
        "gfx942:",
        "gfx942::xnack+",
        "gfx942:xnack",
        "gfx942:xnack=",
        "gfx942:wavefrontsize64+",
        "gfx942:xnack+:xnack-",
        "gfx1151:xnack+",
    ],
)
def test_rejects_malformed_or_unsupported_coordinates(value: str) -> None:
    with pytest.raises(AmdgpuArtifactTargetKeyError):
        parse_amdgpu_artifact_target_key(value)


def test_unknown_target_is_not_inferred() -> None:
    assert parse_amdgpu_artifact_target_key("gfx9999") is None
