# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared emission for one AMDGPU runtime target-table family."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from loom.gen.support.files import write_text_file
from loom.gen.target.arch.amdgpu.amdgpu_target_table_family import (
    AmdgpuTargetTableFamily,
)
from loom.gen.target.arch.amdgpu.descriptors.amdgpu_descriptors import (
    generate_amdgpu_descriptor_table_family,
)
from loom.gen.target.arch.amdgpu.encoding.amdgpu_encoding_tables import (
    generate_amdgpu_encoding_table_source,
)
from loom.target.arch.amdgpu.isa_xml import AmdgpuIsaFactSource
from loom.target.low_descriptors import DescriptorSet

_DESCRIPTOR_SOURCE_HEADER = "loom/codegen/low/descriptors.h"
_ENCODING_SOURCE_HEADER = "loom/target/arch/amdgpu/encoding/encoding.h"


def generate_amdgpu_target_table_outputs(
    family: AmdgpuTargetTableFamily,
    isa_specs: Mapping[str, AmdgpuIsaFactSource],
    descriptor_sets_by_generator_target: Mapping[str, DescriptorSet],
    *,
    descriptor_source_path: Path,
    encoding_source_path: Path,
) -> None:
    """Emits the descriptor and encoding sources for one storage family."""

    descriptor_family = generate_amdgpu_descriptor_table_family(
        family,
        descriptor_sets_by_generator_target,
        source_public_header=_DESCRIPTOR_SOURCE_HEADER,
    )
    encoding_source = generate_amdgpu_encoding_table_source(
        family,
        isa_specs,
        descriptor_sets_by_generator_target,
        public_header=_ENCODING_SOURCE_HEADER,
    )
    write_text_file(descriptor_source_path, descriptor_family.source)
    write_text_file(encoding_source_path, encoding_source)
