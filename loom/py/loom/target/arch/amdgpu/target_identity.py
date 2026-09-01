# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structured AMDGPU artifact target identities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from loom.target.arch.amdgpu.target_catalog import (
    AMDGPU_TARGET_ID_FEATURE_ORDER,
    target_id_features_for_processor,
)
from loom.target.arch.amdgpu.target_info import (
    AmdgpuProcessorInfo,
    AmdgpuTargetInfo,
    amdgpu_processor_info_by_name,
    amdgpu_target_info_by_name,
)


class AmdgpuArtifactTargetKeyError(ValueError):
    """Raised when an AMDGPU artifact key is malformed or unsupported."""


class AmdgpuTargetFeatureState(Enum):
    """One normalized AMDHSA target-ID feature state."""

    ANY = "any"
    UNSUPPORTED = "unsupported"
    OFF = "off"
    ON = "on"


@dataclass(frozen=True, slots=True)
class AmdgpuTargetFeature:
    """One named AMDHSA target-ID feature coordinate."""

    name: str
    state: AmdgpuTargetFeatureState


@dataclass(frozen=True, slots=True)
class AmdgpuTargetIdentity:
    """Canonical target, backend processor, and AMDHSA feature coordinates."""

    target: AmdgpuTargetInfo
    processor: AmdgpuProcessorInfo
    amdhsa_features: tuple[AmdgpuTargetFeature, ...]

    def feature_state(self, name: str) -> AmdgpuTargetFeatureState:
        """Returns the normalized state of a known target-ID feature."""
        for feature in self.amdhsa_features:
            if feature.name == name:
                return feature.state
        raise KeyError(name)


def parse_amdgpu_artifact_target_key(value: str) -> AmdgpuTargetIdentity | None:
    """Parses a canonical AMDGPU artifact key through the shared target tables.

    Returns ``None`` when the target selector is unknown. Malformed coordinates
    and coordinates unsupported by a known target raise
    ``AmdgpuArtifactTargetKeyError``.
    """
    if not value:
        raise AmdgpuArtifactTargetKeyError("AMDGPU artifact target key is empty")
    components = value.split(":")
    if any(not component for component in components):
        raise AmdgpuArtifactTargetKeyError(
            "AMDGPU artifact target key has an empty coordinate"
        )

    target = amdgpu_target_info_by_name(components[0])
    if target is None:
        return None
    processor = amdgpu_processor_info_by_name(target.processor)
    assert processor is not None, "validated AMDGPU target has no processor row"

    supported_feature_names = frozenset(
        target_id_features_for_processor(processor.processor)
    )
    features = [
        AmdgpuTargetFeature(
            name,
            (
                AmdgpuTargetFeatureState.ANY
                if name in supported_feature_names
                else AmdgpuTargetFeatureState.UNSUPPORTED
            ),
        )
        for name in AMDGPU_TARGET_ID_FEATURE_ORDER
    ]
    feature_ordinals = {
        feature.name: ordinal for ordinal, feature in enumerate(features)
    }
    seen_feature_names: set[str] = set()
    for coordinate in components[1:]:
        if len(coordinate) < 2 or coordinate[-1] not in ("-", "+"):
            raise AmdgpuArtifactTargetKeyError(
                f"AMDGPU target coordinate {coordinate!r} requires '+' or '-'"
            )
        feature_name = coordinate[:-1]
        feature_ordinal = feature_ordinals.get(feature_name)
        if feature_ordinal is None:
            raise AmdgpuArtifactTargetKeyError(
                f"unknown AMDGPU target-ID feature {feature_name!r}"
            )
        if feature_name in seen_feature_names:
            raise AmdgpuArtifactTargetKeyError(
                f"AMDGPU target-ID feature {feature_name!r} is repeated"
            )
        if feature_name not in supported_feature_names:
            raise AmdgpuArtifactTargetKeyError(
                f"AMDGPU target {target.target!r} does not support target-ID "
                f"feature {feature_name!r}"
            )
        seen_feature_names.add(feature_name)
        features[feature_ordinal] = AmdgpuTargetFeature(
            feature_name,
            (
                AmdgpuTargetFeatureState.ON
                if coordinate[-1] == "+"
                else AmdgpuTargetFeatureState.OFF
            ),
        )

    return AmdgpuTargetIdentity(
        target=target,
        processor=processor,
        amdhsa_features=tuple(features),
    )
