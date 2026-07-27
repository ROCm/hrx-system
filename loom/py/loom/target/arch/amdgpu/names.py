# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU target naming contracts shared by build and C generators."""

from __future__ import annotations

import re
from typing import Protocol


class AmdgpuDescriptorSetNameInfo(Protocol):
    @property
    def generator_target(self) -> str: ...


def amdgpu_label_fragment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def amdgpu_c_identifier_fragment(value: str) -> str:
    identifier = amdgpu_label_fragment(value).strip("_")
    if not identifier:
        return "EMPTY"
    if identifier[0].isdigit():
        identifier = "_" + identifier
    return identifier.upper()


def amdgpu_descriptor_set_capability(key: str) -> str:
    prefix = "amdgpu."
    if not key.startswith(prefix):
        raise ValueError(
            f"AMDGPU descriptor-set key '{key}' must start with '{prefix}'"
        )
    return "descriptor_set_" + amdgpu_label_fragment(key.removeprefix(prefix))


def amdgpu_descriptor_set_define(key: str) -> str:
    return "LOOM_AMDGPU_" + amdgpu_descriptor_set_capability(key).upper()


def amdgpu_descriptor_set_ordinal_suffix(key: str) -> str:
    prefix = "amdgpu."
    suffix = ".core"
    if not key.startswith(prefix) or not key.endswith(suffix):
        raise ValueError(f"AMDGPU descriptor-set key '{key}' must be a core key")
    return key.removeprefix(prefix).removesuffix(suffix).replace(".", "_").upper()


def amdgpu_descriptor_set_ordinal_constant_name(key: str) -> str:
    suffix = amdgpu_descriptor_set_ordinal_suffix(key)
    return f"LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_{suffix}"


def amdgpu_low_descriptor_header(info: AmdgpuDescriptorSetNameInfo) -> str:
    return f"loom/target/arch/amdgpu/descriptors/{info.generator_target}_descriptors.h"


def amdgpu_low_descriptor_provider_symbol(key: str) -> str:
    return "loom_" + amdgpu_label_fragment(key) + "_descriptor_set"


def amdgpu_encoding_table_header(info: AmdgpuDescriptorSetNameInfo) -> str:
    return f"loom/target/arch/amdgpu/encoding/{info.generator_target}_encoding_tables.h"


def amdgpu_encoding_table_symbol(info: AmdgpuDescriptorSetNameInfo) -> str:
    return f"loom_amdgpu_{info.generator_target}_encoding_table"
