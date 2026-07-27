# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU source-to-low memory descriptor candidates."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[7]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.gen.target.arch.amdgpu.lower.candidates.validation import (  # noqa: E402
    dense_candidate_ranges,
    descriptor_ref_constant_name,
    require_descriptor_refs,
)
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    AmdgpuMemoryAddressForm,
    AmdgpuMemoryDescriptorCandidate,
    AmdgpuMemoryDescriptorDomain,
    AmdgpuMemoryOperationKind,
    AmdgpuMemoryPayloadFormat,
    AmdgpuMemoryPayloadRegisterClass,
    amdgpu_memory_descriptor_candidates,
)

_MEMORY_DESCRIPTOR_DOMAIN_INDEX = {
    AmdgpuMemoryDescriptorDomain.BUFFER_RESOURCE: 0,
    AmdgpuMemoryDescriptorDomain.GLOBAL_SADDR: 1,
    AmdgpuMemoryDescriptorDomain.LDS: 2,
    AmdgpuMemoryDescriptorDomain.GLOBAL_FLAT: 3,
    AmdgpuMemoryDescriptorDomain.GLOBAL_SMEM: 4,
    AmdgpuMemoryDescriptorDomain.SCRATCH: 5,
}

_MEMORY_ADDRESS_FORM_INDEX = {
    AmdgpuMemoryAddressForm.DEFAULT: 0,
    AmdgpuMemoryAddressForm.BUFFER_OFF_ZERO: 1,
    AmdgpuMemoryAddressForm.DS_2ADDR: 2,
    AmdgpuMemoryAddressForm.GLOBAL_SADDR: 3,
    AmdgpuMemoryAddressForm.DS_ADDTID: 4,
    AmdgpuMemoryAddressForm.FLAT: 5,
    AmdgpuMemoryAddressForm.GLOBAL_SMEM: 6,
    AmdgpuMemoryAddressForm.SCRATCH_VADDR: 7,
}

_MEMORY_OPERATION_KIND_INDEX = {
    AmdgpuMemoryOperationKind.LOAD: 0,
    AmdgpuMemoryOperationKind.STORE: 1,
}

_MEMORY_PAYLOAD_REGISTER_CLASS_INDEX = {
    AmdgpuMemoryPayloadRegisterClass.VGPR: 0,
    AmdgpuMemoryPayloadRegisterClass.SGPR: 1,
}

_MEMORY_PAYLOAD_FORMAT_INDEX = {
    AmdgpuMemoryPayloadFormat.GENERIC: 0,
    AmdgpuMemoryPayloadFormat.LOW_16BIT_FLOAT: 1,
    AmdgpuMemoryPayloadFormat.SIGNED_16BIT_INTEGER: 2,
}

_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_MAX = 31
_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_MAX = 7


def _candidate_range_key(
    candidate: AmdgpuMemoryDescriptorCandidate,
) -> tuple[
    AmdgpuMemoryDescriptorDomain,
    AmdgpuMemoryAddressForm,
    AmdgpuMemoryOperationKind,
    int,
    AmdgpuMemoryPayloadRegisterClass,
    AmdgpuMemoryPayloadFormat,
    int,
]:
    return (
        candidate.domain,
        candidate.address_form,
        candidate.operation_kind,
        candidate.packet_byte_count,
        candidate.payload_register_class,
        candidate.payload_format,
        candidate.payload_register_count,
    )


def _range_key_packed_value(
    key: tuple[
        AmdgpuMemoryDescriptorDomain,
        AmdgpuMemoryAddressForm,
        AmdgpuMemoryOperationKind,
        int,
        AmdgpuMemoryPayloadRegisterClass,
        AmdgpuMemoryPayloadFormat,
        int,
    ],
) -> int:
    (
        domain,
        address_form,
        operation_kind,
        packet_byte_count,
        payload_register_class,
        payload_format,
        payload_register_count,
    ) = key
    return (
        _MEMORY_DESCRIPTOR_DOMAIN_INDEX[domain]
        | (_MEMORY_ADDRESS_FORM_INDEX[address_form] << 3)
        | (_MEMORY_OPERATION_KIND_INDEX[operation_kind] << 6)
        | (packet_byte_count << 7)
        | (_MEMORY_PAYLOAD_REGISTER_CLASS_INDEX[payload_register_class] << 12)
        | (_MEMORY_PAYLOAD_FORMAT_INDEX[payload_format] << 13)
        | (payload_register_count << 15)
    )


def _validate_uint_field(
    owner: str,
    field_name: str,
    value: int,
    maximum: int,
) -> None:
    if value < 0 or value > maximum:
        raise ValueError(f"{owner} has {field_name} {value}, expected 0..{maximum}")


def _validate_candidate(candidate: AmdgpuMemoryDescriptorCandidate) -> None:
    owner = f"AMDGPU memory descriptor candidate {candidate.descriptor_key}"
    _validate_uint_field(
        owner,
        "packet_byte_count",
        candidate.packet_byte_count,
        _MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_MAX,
    )
    _validate_uint_field(
        owner,
        "payload_register_count",
        candidate.payload_register_count,
        _MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_MAX,
    )


def _ordered_candidates(
    candidates: Sequence[AmdgpuMemoryDescriptorCandidate],
) -> tuple[AmdgpuMemoryDescriptorCandidate, ...]:
    require_descriptor_refs(
        "AMDGPU memory descriptor candidate table",
        (candidate.descriptor_key for candidate in candidates),
    )
    for candidate in candidates:
        _validate_candidate(candidate)
    return tuple(sorted(candidates, key=lambda candidate: _range_key_packed_value(_candidate_range_key(candidate))))


def _candidate_ranges(
    candidates: Sequence[AmdgpuMemoryDescriptorCandidate],
) -> tuple[
    tuple[
        tuple[
            AmdgpuMemoryDescriptorDomain,
            AmdgpuMemoryAddressForm,
            AmdgpuMemoryOperationKind,
            int,
            AmdgpuMemoryPayloadRegisterClass,
            AmdgpuMemoryPayloadFormat,
            int,
        ],
        int,
        int,
    ],
    ...,
]:
    return dense_candidate_ranges(
        candidates,
        _candidate_range_key,
        owner="AMDGPU memory descriptor candidate table",
        range_sort_key=_range_key_packed_value,
    )


def _candidate_initializer(candidate: AmdgpuMemoryDescriptorCandidate) -> str:
    descriptor_ref = descriptor_ref_constant_name(candidate.descriptor_key)
    return "\n".join(
        [
            "LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE(",
            f"    {candidate.domain.c_name}, {candidate.address_form.c_name},",
            f"    {candidate.operation_kind.c_name}, {candidate.packet_byte_count},",
            f"    {candidate.payload_register_class.c_name},",
            f"    {candidate.payload_format.c_name}, {candidate.payload_register_count},",
            f"    {descriptor_ref})",
        ]
    )


def _range_initializer(
    key: tuple[
        AmdgpuMemoryDescriptorDomain,
        AmdgpuMemoryAddressForm,
        AmdgpuMemoryOperationKind,
        int,
        AmdgpuMemoryPayloadRegisterClass,
        AmdgpuMemoryPayloadFormat,
        int,
    ],
    first_candidate: int,
    candidate_count: int,
) -> str:
    (
        domain,
        address_form,
        operation_kind,
        packet_byte_count,
        payload_register_class,
        payload_format,
        payload_register_count,
    ) = key
    return "\n".join(
        [
            "LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_RANGE(",
            f"    {domain.c_name}, {address_form.c_name},",
            f"    {operation_kind.c_name}, {packet_byte_count},",
            f"    {payload_register_class.c_name},",
            f"    {payload_format.c_name}, {payload_register_count},",
            f"    {first_candidate}, {candidate_count})",
        ]
    )


def _generated_header() -> list[str]:
    return [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header(
            "//",
            generator="loom.gen.target.arch.amdgpu.lower.candidates.amdgpu_memory_candidates",
        ),
        "",
    ]


def _emit_candidate_rows() -> str:
    candidates = _ordered_candidates(amdgpu_memory_descriptor_candidates())
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_candidate_initializer(candidate) for candidate in candidates),
            ]
        )
        + "\n"
    )


def _emit_candidate_ranges() -> str:
    candidates = _ordered_candidates(amdgpu_memory_descriptor_candidates())
    ranges = _candidate_ranges(candidates)
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_range_initializer(key, first_candidate, candidate_count) for key, first_candidate, candidate_count in ranges),
            ]
        )
        + "\n"
    )


def _write_output(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU source-to-low memory descriptor candidates.")
    parser.add_argument(
        "--candidate-rows",
        type=Path,
        help="Generated memory descriptor candidate row fragment path.",
    )
    parser.add_argument(
        "--candidate-ranges",
        type=Path,
        help="Generated memory descriptor candidate range fragment path.",
    )
    args = parser.parse_args(argv)

    if args.candidate_rows is None and args.candidate_ranges is None:
        parser.error("at least one output path is required")
    if args.candidate_rows is not None:
        _write_output(args.candidate_rows, _emit_candidate_rows())
    if args.candidate_ranges is not None:
        _write_output(args.candidate_ranges, _emit_candidate_ranges())
    return 0


if __name__ == "__main__":
    sys.exit(main())
