# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.build_tools import target_presets


def test_default_is_the_non_wasm_subset_of_all() -> None:
    default = target_presets.resolve_presets(("default",))
    all_targets = target_presets.resolve_presets(("all",))

    assert "wasm" not in default.architectures
    assert "wasm" not in default.artifacts
    assert "wasm_host" not in default.execution
    assert set(all_targets.architectures).difference(default.architectures) == {"wasm"}
    assert set(all_targets.artifacts).difference(default.artifacts) == {"wasm"}
    assert set(all_targets.execution).difference(default.execution) == {"wasm_host"}


def test_amd_is_the_union_of_gpu_and_npu() -> None:
    amd = target_presets.resolve_presets(("amd",))
    separate = target_presets.resolve_presets(("amd_gpu", "amd_npu"))

    assert amd == separate


def test_composition_is_order_independent_and_canonical() -> None:
    cpu_amd = target_presets.resolve_presets(("cpu", "amd"))
    amd_cpu = target_presets.resolve_presets(("amd", "cpu"))

    assert cpu_amd == amd_cpu
    assert cpu_amd.architectures == tuple(
        architecture
        for architecture in target_presets.ARCHITECTURES
        if architecture in cpu_amd.architectures
    )
