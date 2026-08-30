# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

from loom.gen.target.arch.amdgpu.descriptors import amdgpu_lower_capabilities
from loom.target.low_descriptors import Descriptor, DescriptorSet


def _descriptor(semantic_tag: str | None) -> Descriptor:
    return Descriptor(
        key="amdgpu.test",
        mnemonic="test",
        semantic_tag=semantic_tag,
        operands=(),
        schedule_class="test",
    )


def _descriptor_set(key: str, *descriptors: Descriptor) -> DescriptorSet:
    return DescriptorSet(
        key=key,
        target_key="amdgpu",
        feature_key=None,
        c_header_path=Path("test.h"),
        c_source_path=Path("test.c"),
        header_guard="TEST_H_",
        public_header="test.h",
        function_name="test_descriptor_set",
        c_table_prefix="test",
        c_enum_prefix="TEST",
        generator_version=0,
        reg_classes=(),
        resources=(),
        schedule_classes=(),
        descriptors=descriptors,
    )


def test_lower_capabilities_follow_selected_descriptor_semantics() -> None:
    base = _descriptor_set("amdgpu.base", _descriptor("arithmetic.add.i32"))
    cluster = _descriptor_set("amdgpu.cluster", _descriptor("memory.cluster.load.to_lds.u128"))
    tensor = _descriptor_set("amdgpu.tensor", _descriptor("memory.tensor.load.to_lds"))

    emit = amdgpu_lower_capabilities._emit_header
    base_header = emit((base,))
    cluster_header = emit((base, cluster))
    combined_header = emit((base, cluster, tensor))

    assert "LOOM_AMDGPU_LOWER_CAPABILITY_ASYNC_CLUSTER_GATHER 0" in base_header
    assert "LOOM_AMDGPU_LOWER_CAPABILITY_ASYNC_TENSOR_LOAD_TO_LDS 0" in base_header
    assert "LOOM_AMDGPU_LOWER_CAPABILITY_ASYNC_CLUSTER_GATHER 1" in cluster_header
    assert "LOOM_AMDGPU_LOWER_CAPABILITY_ASYNC_TENSOR_LOAD_TO_LDS 0" in cluster_header
    assert "LOOM_AMDGPU_LOWER_CAPABILITY_ASYNC_CLUSTER_GATHER 1" in combined_header
    assert "LOOM_AMDGPU_LOWER_CAPABILITY_ASYNC_TENSOR_LOAD_TO_LDS 1" in combined_header


def test_lower_capability_header_has_stable_guard() -> None:
    header = amdgpu_lower_capabilities._emit_header(())
    assert "#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CAPABILITIES_H_" in header
    assert "#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CAPABILITIES_H_" in header
    assert header.endswith("#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CAPABILITIES_H_\n")
