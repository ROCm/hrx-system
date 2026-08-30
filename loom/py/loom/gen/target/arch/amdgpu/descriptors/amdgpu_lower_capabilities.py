# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates AMDGPU lowering capabilities from selected descriptors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from loom.gen.support.files import write_text_file
from loom.gen.support.generated_file import line_comment_header
from loom.target.low_descriptors import DescriptorSet


@dataclass(frozen=True, slots=True)
class _LowerCapability:
    """One lowerer whose reachability is proven by descriptor semantics."""

    name: str
    semantic_tag_prefix: str


_LOWER_CAPABILITIES = (
    _LowerCapability(
        name="ASYNC_CLUSTER_GATHER",
        semantic_tag_prefix="memory.cluster.load.to_lds.",
    ),
    _LowerCapability(
        name="ASYNC_TENSOR_LOAD_TO_LDS",
        semantic_tag_prefix="memory.tensor.load.to_lds",
    ),
)


def _selected_capability_names(
    descriptor_sets: Sequence[DescriptorSet],
) -> frozenset[str]:
    semantic_tags: list[str] = []
    for descriptor_set in descriptor_sets:
        semantic_tags.extend(descriptor.semantic_tag for descriptor in descriptor_set.descriptors if descriptor.semantic_tag is not None)

    selected_names = set[str]()
    for capability in _LOWER_CAPABILITIES:
        for semantic_tag in semantic_tags:
            if semantic_tag.startswith(capability.semantic_tag_prefix):
                selected_names.add(capability.name)
                break
    return frozenset(selected_names)


def _emit_header(descriptor_sets: Sequence[DescriptorSet]) -> str:
    selected_capability_names = _selected_capability_names(descriptor_sets)
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header(
            "//",
            generator=__name__,
        ),
        "",
        "#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CAPABILITIES_H_",
        "#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CAPABILITIES_H_",
        "",
    ]
    for capability in _LOWER_CAPABILITIES:
        value = int(capability.name in selected_capability_names)
        lines.append(f"#define LOOM_AMDGPU_LOWER_CAPABILITY_{capability.name} {value}")
    lines.extend(
        [
            "",
            "#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CAPABILITIES_H_",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_lower_capability_header(descriptor_sets: Sequence[DescriptorSet], header_path: Path) -> None:
    """Writes build-time lowerer capability facts for selected descriptors."""

    write_text_file(header_path, _emit_header(descriptor_sets))


__all__ = ("generate_lower_capability_header",)
