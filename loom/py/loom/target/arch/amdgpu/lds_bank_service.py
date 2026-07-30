# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Immutable AMDGPU LDS bank-service model data.

Processor and canonical target rows select tuples of model keys. The C table
generator interns those tuples into dense model-set ordinals and emits the
structural models consumed by lowering. Adding another target that shares an
existing model therefore changes target data only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

AMDGPU_LDS_BANK_SERVICE_EVIDENCE_PUBLIC_VENDOR_DOCUMENTATION = (
    "public-vendor-documentation"
)
AMDGPU_LDS_BANK_SERVICE_EVIDENCE_VENDOR_SOFTWARE_MODEL_UNVALIDATED = (
    "vendor-software-model-unvalidated"
)
AMDGPU_LDS_BANK_SERVICE_EVIDENCE_SILICON_CALIBRATED_VENDOR_MODEL = (
    "silicon-calibrated-vendor-model"
)
AMDGPU_LDS_BANK_SERVICE_EVIDENCE_CLASSES = (
    AMDGPU_LDS_BANK_SERVICE_EVIDENCE_PUBLIC_VENDOR_DOCUMENTATION,
    AMDGPU_LDS_BANK_SERVICE_EVIDENCE_VENDOR_SOFTWARE_MODEL_UNVALIDATED,
    AMDGPU_LDS_BANK_SERVICE_EVIDENCE_SILICON_CALIBRATED_VENDOR_MODEL,
)

AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ = "read"
AMDGPU_LDS_BANK_SERVICE_DIRECTION_WRITE = "write"
AMDGPU_LDS_BANK_SERVICE_DIRECTIONS = (
    AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ,
    AMDGPU_LDS_BANK_SERVICE_DIRECTION_WRITE,
)

AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COUNT_EACH = "count-each"
AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COALESCE_IDENTICAL_READS = (
    "coalesce-identical-reads"
)
AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICIES = (
    AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COUNT_EACH,
    AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COALESCE_IDENTICAL_READS,
)

AMDGPU_LDS_BANK_SERVICE_MAX_WAVE_SIZE = 64
AMDGPU_LDS_BANK_SERVICE_MAX_BANK_COUNT = 64
AMDGPU_LDS_BANK_SERVICE_MAX_PHASE_COUNT = 8
AMDGPU_LDS_BANK_SERVICE_MAX_PACKET_WORD_COUNT = 4


@dataclass(frozen=True, slots=True)
class AmdgpuLdsBankServiceModelInfo:
    key: str
    revision: str
    descriptor_key: str
    evidence_class: str
    direction: str
    request_policy: str
    wave_size: int
    bank_count: int
    bank_word_byte_count: int
    packet_word_count: int
    phase_lane_masks: tuple[int, ...]


AMDGPU_LDS_BANK_SERVICE_MODEL_INFOS: tuple[AmdgpuLdsBankServiceModelInfo, ...] = (
    AmdgpuLdsBankServiceModelInfo(
        key="amdgpu.lds.wave32.b128.quad-phases.read.count-each",
        revision="ROCm/rocm-libraries@a7e3879c8847:LDSModel.cpp",
        descriptor_key="amdgpu.ds_read_b128",
        evidence_class=(
            AMDGPU_LDS_BANK_SERVICE_EVIDENCE_VENDOR_SOFTWARE_MODEL_UNVALIDATED
        ),
        direction=AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ,
        request_policy=AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COUNT_EACH,
        wave_size=32,
        bank_count=32,
        bank_word_byte_count=4,
        packet_word_count=4,
        phase_lane_masks=(
            0x0000000F,
            0x000000F0,
            0x00000F00,
            0x0000F000,
            0x000F0000,
            0x00F00000,
            0x0F000000,
            0xF0000000,
        ),
    ),
    AmdgpuLdsBankServiceModelInfo(
        key="amdgpu.lds.wave32.b128.quad-phases.write.count-each",
        revision="ROCm/rocm-libraries@a7e3879c8847:LDSModel.cpp",
        descriptor_key="amdgpu.ds_write_b128",
        evidence_class=(
            AMDGPU_LDS_BANK_SERVICE_EVIDENCE_VENDOR_SOFTWARE_MODEL_UNVALIDATED
        ),
        direction=AMDGPU_LDS_BANK_SERVICE_DIRECTION_WRITE,
        request_policy=AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COUNT_EACH,
        wave_size=32,
        bank_count=32,
        bank_word_byte_count=4,
        packet_word_count=4,
        phase_lane_masks=(
            0x0000000F,
            0x000000F0,
            0x00000F00,
            0x0000F000,
            0x000F0000,
            0x00F00000,
            0x0F000000,
            0xF0000000,
        ),
    ),
)

# Structural service model shared by every target selecting these rows.
AMDGPU_LDS_BANK_SERVICE_MODELS_WAVE32_B128_QUAD_PHASES = (
    "amdgpu.lds.wave32.b128.quad-phases.read.count-each",
    "amdgpu.lds.wave32.b128.quad-phases.write.count-each",
)


def amdgpu_lds_bank_service_model_info_by_key(
    model_infos: Sequence[
        AmdgpuLdsBankServiceModelInfo
    ] = AMDGPU_LDS_BANK_SERVICE_MODEL_INFOS,
) -> dict[str, AmdgpuLdsBankServiceModelInfo]:
    return {info.key: info for info in model_infos}


def validate_amdgpu_lds_bank_service_model_infos(
    descriptor_ref_keys: Iterable[str],
    model_infos: Sequence[
        AmdgpuLdsBankServiceModelInfo
    ] = AMDGPU_LDS_BANK_SERVICE_MODEL_INFOS,
) -> None:
    descriptor_ref_key_set = frozenset(descriptor_ref_keys)
    keys = tuple(info.key for info in model_infos)
    if keys != tuple(sorted(keys)):
        raise ValueError("AMDGPU LDS bank-service model keys must be sorted")
    if len(keys) != len(set(keys)):
        raise ValueError("AMDGPU LDS bank-service model keys must be unique")

    for info in model_infos:
        owner = f"AMDGPU LDS bank-service model '{info.key}'"
        if not info.key:
            raise ValueError("AMDGPU LDS bank-service model key is required")
        if not info.revision:
            raise ValueError(f"{owner} source revision is required")
        if info.descriptor_key not in descriptor_ref_key_set:
            raise ValueError(
                f"{owner} references unknown descriptor '{info.descriptor_key}'"
            )
        if info.evidence_class not in AMDGPU_LDS_BANK_SERVICE_EVIDENCE_CLASSES:
            raise ValueError(
                f"{owner} has unknown evidence class '{info.evidence_class}'"
            )
        if info.direction not in AMDGPU_LDS_BANK_SERVICE_DIRECTIONS:
            raise ValueError(f"{owner} has unknown direction '{info.direction}'")
        if info.request_policy not in AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICIES:
            raise ValueError(
                f"{owner} has unknown request policy '{info.request_policy}'"
            )
        if (
            info.request_policy
            == AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COALESCE_IDENTICAL_READS
            and info.direction != AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ
        ):
            raise ValueError(f"{owner} coalesces identical requests on a write model")
        if (
            info.wave_size <= 0
            or info.wave_size > AMDGPU_LDS_BANK_SERVICE_MAX_WAVE_SIZE
        ):
            raise ValueError(f"{owner} wave size must be in 1..64")
        if (
            info.bank_count <= 0
            or info.bank_count > AMDGPU_LDS_BANK_SERVICE_MAX_BANK_COUNT
        ):
            raise ValueError(f"{owner} bank count must be in 1..64")
        if info.bank_word_byte_count <= 0:
            raise ValueError(f"{owner} bank-word byte count must be positive")
        if (
            info.packet_word_count <= 0
            or info.packet_word_count > AMDGPU_LDS_BANK_SERVICE_MAX_PACKET_WORD_COUNT
        ):
            raise ValueError(f"{owner} packet word count must be in 1..4")
        if (
            not info.phase_lane_masks
            or len(info.phase_lane_masks) > AMDGPU_LDS_BANK_SERVICE_MAX_PHASE_COUNT
        ):
            raise ValueError(f"{owner} phase count must be in 1..8")

        valid_lane_mask = (1 << info.wave_size) - 1
        covered_lane_mask = 0
        for phase_lane_mask in info.phase_lane_masks:
            if phase_lane_mask <= 0 or phase_lane_mask & ~valid_lane_mask:
                raise ValueError(f"{owner} has an invalid phase lane mask")
            if covered_lane_mask & phase_lane_mask:
                raise ValueError(f"{owner} phase lane masks overlap")
            covered_lane_mask |= phase_lane_mask
        if covered_lane_mask != valid_lane_mask:
            raise ValueError(f"{owner} phase lane masks do not cover the wave")


def validate_amdgpu_lds_bank_service_model_selection(
    owner: str,
    model_keys: Sequence[str],
    model_infos_by_key: Mapping[str, AmdgpuLdsBankServiceModelInfo] | None = None,
) -> None:
    if model_infos_by_key is None:
        model_infos_by_key = amdgpu_lds_bank_service_model_info_by_key()
    if len(model_keys) != len(set(model_keys)):
        raise ValueError(f"{owner} repeats an LDS bank-service model")
    unknown_keys = tuple(key for key in model_keys if key not in model_infos_by_key)
    if unknown_keys:
        raise ValueError(
            f"{owner} references unknown LDS bank-service models: "
            + ", ".join(unknown_keys)
        )
    descriptor_keys = tuple(
        model_infos_by_key[key].descriptor_key for key in model_keys
    )
    if len(descriptor_keys) != len(set(descriptor_keys)):
        raise ValueError(f"{owner} selects multiple LDS models for one descriptor")
    if descriptor_keys != tuple(sorted(descriptor_keys)):
        raise ValueError(
            f"{owner} LDS bank-service models must be sorted by descriptor key"
        )


def validate_amdgpu_lds_bank_service_model_coverage(
    model_sets: Sequence[Sequence[str]],
    model_infos: Sequence[
        AmdgpuLdsBankServiceModelInfo
    ] = AMDGPU_LDS_BANK_SERVICE_MODEL_INFOS,
) -> None:
    known_keys = {info.key for info in model_infos}
    selected_keys = {model_key for model_keys in model_sets for model_key in model_keys}
    unselected_keys = tuple(sorted(known_keys - selected_keys))
    if unselected_keys:
        raise ValueError(
            "AMDGPU LDS bank-service models are not selected by a target row: "
            + ", ".join(unselected_keys)
        )
