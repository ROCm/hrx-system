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
from dataclasses import dataclass, replace

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
AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COMBINE_DISJOINT_WRITES = (
    "combine-disjoint-writes"
)
AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICIES = (
    AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COUNT_EACH,
    AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COALESCE_IDENTICAL_READS,
    AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COMBINE_DISJOINT_WRITES,
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
    packet_byte_count: int
    phase_lane_masks: tuple[int, ...]


def _b128_octet_model(
    family: str,
    wave_size: int,
    direction: str,
    evidence_class: str,
) -> AmdgpuLdsBankServiceModelInfo:
    # AMD CK documents distinct read/write octets:
    # https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html
    # Read broadcast semantics: AMD ROCm Programming Guide 7.2.3, section 6.3.3.
    # Native gfx1100/gfx1151 controls in both wave modes and gfx942 wave64
    # distinguish the phase maps with read-only and write-only conflicts.
    # Broadcast reads match contiguous cost. These controls qualify service
    # structure, not a cycle prediction.
    read = direction == AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ
    request_policy = (
        AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COALESCE_IDENTICAL_READS
        if read
        else AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COUNT_EACH
    )
    half_wave_masks = (
        (0x00F0000F, 0x000F00F0, 0xF0000F00, 0x0F00F000)
        if read
        else (0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000)
    )
    return AmdgpuLdsBankServiceModelInfo(
        key=f"amdgpu.lds.{family}.wave{wave_size}.b128.{direction}.{request_policy}",
        revision="AMD:CK-LDS-2025-07-25;ROCm-guide-7.2.3:6.3.3",
        descriptor_key=f"amdgpu.ds_{direction}_b128",
        evidence_class=evidence_class,
        direction=direction,
        request_policy=request_policy,
        wave_size=wave_size,
        bank_count=32,
        bank_word_byte_count=4,
        packet_byte_count=16,
        phase_lane_masks=tuple(
            mask << half_wave_start
            for half_wave_start in range(0, wave_size, 32)
            for mask in half_wave_masks
        ),
    )


_B128_OCTET_MODELS = tuple(
    _b128_octet_model(family, wave_size, direction, evidence)
    for family, wave_sizes, evidence in (
        ("cdna3", (64,), AMDGPU_LDS_BANK_SERVICE_EVIDENCE_PUBLIC_VENDOR_DOCUMENTATION),
        (
            "gfx942",
            (64,),
            AMDGPU_LDS_BANK_SERVICE_EVIDENCE_SILICON_CALIBRATED_VENDOR_MODEL,
        ),
        (
            "gfx1100",
            (32, 64),
            AMDGPU_LDS_BANK_SERVICE_EVIDENCE_SILICON_CALIBRATED_VENDOR_MODEL,
        ),
        (
            "gfx1151",
            (32, 64),
            AMDGPU_LDS_BANK_SERVICE_EVIDENCE_SILICON_CALIBRATED_VENDOR_MODEL,
        ),
    )
    for direction in AMDGPU_LDS_BANK_SERVICE_DIRECTIONS
    for wave_size in wave_sizes
)


def _contiguous_model(
    processor: str, wave_size: int, byte_count: int, direction: str
) -> AmdgpuLdsBankServiceModelInfo:
    # Native gfx1100/gfx1151 wave32/wave64 and gfx942 wave64 controls qualify
    # contiguous 32-lane phases for b16/b32 and 16-lane phases for b64. Amplified
    # conflicts and bank permutations distinguish these from crossed phases
    # and a whole-wave organization. Same-word reads and disjoint halfword
    # writes combine; a two-byte base shift can change conflicts. These are
    # service rules, not cycle costs.
    read = direction == AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ
    policy = (
        AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COALESCE_IDENTICAL_READS
        if read
        else AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COMBINE_DISJOINT_WRITES
        if byte_count == 2
        else AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COUNT_EACH
    )
    packet = "u16" if read and byte_count == 2 else f"b{byte_count * 8}"
    phase_lane_count = 16 if byte_count == 8 else 32
    return AmdgpuLdsBankServiceModelInfo(
        key=(
            f"amdgpu.lds.{processor}.wave{wave_size}.b{byte_count * 8}."
            f"{direction}.{policy}"
        ),
        revision=(
            "AMD:ROCm-guide-7.2.3:6.3.3;native-b64-2026-09-25"
            if byte_count == 8
            else "AMD:ROCm-guide-7.2.3:6.3.3;native-narrow-2026-09-24"
        ),
        descriptor_key=f"amdgpu.ds_{direction}_{packet}",
        evidence_class=AMDGPU_LDS_BANK_SERVICE_EVIDENCE_SILICON_CALIBRATED_VENDOR_MODEL,
        direction=direction,
        request_policy=policy,
        wave_size=wave_size,
        bank_count=32,
        bank_word_byte_count=4,
        packet_byte_count=byte_count,
        phase_lane_masks=tuple(
            ((1 << phase_lane_count) - 1) << start
            for start in range(0, wave_size, phase_lane_count)
        ),
    )


_NATIVE_CONTIGUOUS_MODELS = tuple(
    _contiguous_model(processor, wave_size, byte_count, direction)
    for processor, wave_sizes in (
        ("gfx1100", (32, 64)),
        ("gfx1151", (32, 64)),
        ("gfx942", (64,)),
    )
    for byte_count in (2, 4, 8)
    for direction in AMDGPU_LDS_BANK_SERVICE_DIRECTIONS
    for wave_size in wave_sizes
)


def _d16_read_model(
    processor: str, wave_size: int, half: str
) -> AmdgpuLdsBankServiceModelInfo:
    # Native low/high-half controls independently qualify the same 32-lane
    # phases and same-word read combining as u16. Destination preservation was
    # checked separately; it is not evidence for the bank-service organization.
    model = _contiguous_model(
        processor, wave_size, 2, AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ
    )
    return replace(
        model,
        key=(
            f"amdgpu.lds.{processor}.wave{wave_size}.b16.d16-{half}."
            f"read.{model.request_policy}"
        ),
        revision="AMD:ROCm-guide-7.2.3:6.3.3;native-d16-2026-09-25",
        descriptor_key=(
            "amdgpu.ds_load_u16_d16_hi" if half == "high" else "amdgpu.ds_load_u16_d16"
        ),
    )


_NATIVE_D16_READ_MODELS = tuple(
    _d16_read_model(processor, wave_size, half)
    for processor in ("gfx1100", "gfx1151")
    for wave_size in (32, 64)
    for half in ("low", "high")
)


AMDGPU_LDS_BANK_SERVICE_MODEL_INFOS: tuple[AmdgpuLdsBankServiceModelInfo, ...] = tuple(
    sorted(
        (
            *_B128_OCTET_MODELS,
            *_NATIVE_CONTIGUOUS_MODELS,
            *_NATIVE_D16_READ_MODELS,
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
                packet_byte_count=16,
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
                packet_byte_count=16,
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
        ),
        key=lambda info: info.key,
    )
)


def _family_model_keys(family: str) -> tuple[str, ...]:
    return tuple(
        info.key
        for info in sorted(
            (
                info
                for info in AMDGPU_LDS_BANK_SERVICE_MODEL_INFOS
                if f".{family}." in info.key
            ),
            key=lambda info: (info.descriptor_key, info.wave_size),
        )
    )


AMDGPU_LDS_BANK_SERVICE_MODELS_CDNA3 = _family_model_keys("cdna3")
AMDGPU_LDS_BANK_SERVICE_MODELS_GFX942 = _family_model_keys("gfx942")
AMDGPU_LDS_BANK_SERVICE_MODELS_GFX1100 = _family_model_keys("gfx1100")
AMDGPU_LDS_BANK_SERVICE_MODELS_GFX1151 = _family_model_keys("gfx1151")

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
            info.request_policy
            == AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COMBINE_DISJOINT_WRITES
            and info.direction != AMDGPU_LDS_BANK_SERVICE_DIRECTION_WRITE
        ):
            raise ValueError(f"{owner} combines disjoint writes on a read model")
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
        if info.bank_word_byte_count not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError(
                f"{owner} bank-word byte count must be a power of two in 1..64"
            )
        if (
            info.packet_byte_count <= 0
            or info.packet_byte_count
            > info.bank_word_byte_count * AMDGPU_LDS_BANK_SERVICE_MAX_PACKET_WORD_COUNT
            or info.packet_byte_count & (info.packet_byte_count - 1)
            or info.packet_byte_count > 16
        ):
            raise ValueError(
                f"{owner} packet byte count must be a power of two in 1..16 "
                "spanning at most four bank words"
            )
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
    binding_keys = tuple(
        (model_infos_by_key[key].descriptor_key, model_infos_by_key[key].wave_size)
        for key in model_keys
    )
    if len(binding_keys) != len(set(binding_keys)):
        raise ValueError(
            f"{owner} selects multiple LDS models for one descriptor and wave size"
        )
    if binding_keys != tuple(sorted(binding_keys)):
        raise ValueError(
            f"{owner} LDS bank-service models must be sorted by "
            "descriptor and wave size"
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
