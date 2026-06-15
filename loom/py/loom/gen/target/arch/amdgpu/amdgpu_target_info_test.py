# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager

from loom.gen.target.arch.amdgpu import amdgpu_target_info
from loom.target.arch.amdgpu.target_info import (
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH,
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID,
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11,
    AmdgpuDescriptorSetInfo,
    AmdgpuKernelDescriptorVgprGranules,
    AmdgpuProcessorKernelDescriptorInfo,
    processor_info,
)


@contextmanager
def _raises_value_error(match: str) -> Iterator[None]:
    try:
        yield
    except ValueError as exc:
        if re.search(match, str(exc)) is None:
            raise AssertionError(f"ValueError message {exc!s} did not match {match}") from exc
    else:
        raise AssertionError("expected ValueError")


def _descriptor_set_info() -> AmdgpuDescriptorSetInfo:
    return AmdgpuDescriptorSetInfo(
        generator_target="test",
        key="amdgpu.test.core",
        isa_xml_key="test",
        isa_architecture_name="AMDGPU Test",
        isa_architecture_id=1,
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
    )


def test_profileless_kernel_descriptor_accepts_packed_workitem_id_fact() -> None:
    processor = processor_info(
        "gfx-test",
        0x001,
        descriptor_set_key="amdgpu.test.core",
        kernel_descriptor=AmdgpuProcessorKernelDescriptorInfo(
            flags=AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID,
        ),
    )

    amdgpu_target_info._validate_processors((processor,), (_descriptor_set_info(),))


def test_profileless_kernel_descriptor_rejects_vgpr_granules() -> None:
    processor = processor_info(
        "gfx-test",
        0x001,
        descriptor_set_key="amdgpu.test.core",
        kernel_descriptor=AmdgpuProcessorKernelDescriptorInfo(
            vgpr_granules=AmdgpuKernelDescriptorVgprGranules(wave32=8, wave64=4),
        ),
    )

    with _raises_value_error("no kernel descriptor profile but has VGPR"):
        amdgpu_target_info._validate_processors((processor,), (_descriptor_set_info(),))


def test_profileless_kernel_descriptor_rejects_profile_owned_flags() -> None:
    processor = processor_info(
        "gfx-test",
        0x001,
        descriptor_set_key="amdgpu.test.core",
        kernel_descriptor=AmdgpuProcessorKernelDescriptorInfo(
            flags=AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH,
        ),
    )

    with _raises_value_error("profile-owned ABI flags"):
        amdgpu_target_info._validate_processors((processor,), (_descriptor_set_info(),))


def test_profiled_kernel_descriptor_requires_vgpr_granules() -> None:
    processor = processor_info(
        "gfx-test",
        0x001,
        descriptor_set_key="amdgpu.test.core",
        kernel_descriptor=AmdgpuProcessorKernelDescriptorInfo(
            profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11,
        ),
    )

    with _raises_value_error("descriptor profile but no VGPR encoding granules"):
        amdgpu_target_info._validate_processors((processor,), (_descriptor_set_info(),))
