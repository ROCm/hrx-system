# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from loom.gen.target.arch.amdgpu.descriptors import amdgpu_descriptor_tables


def test_descriptor_identity_source_leaves_unselected_domain_slots_empty() -> None:
    descriptor = SimpleNamespace(
        key="amdgpu.selected",
        semantic_tag=None,
    )
    descriptor_set = SimpleNamespace(descriptors=(descriptor,))
    descriptor_strings = amdgpu_descriptor_tables.CStringBank.build(
        frozenset((descriptor.key,)),
        symbol="test_descriptor_strings",
        header="test/descriptor_strings.h",
    )
    shared_strings = amdgpu_descriptor_tables.CStringBank.build(
        frozenset(),
        symbol="test_shared_strings",
        header="test/shared_strings.h",
    )
    identity_domain = SimpleNamespace(
        keys=("amdgpu.unselected", descriptor.key),
    )

    with mock.patch.object(
        amdgpu_descriptor_tables,
        "amdgpu_descriptor_identity_domain",
        return_value=identity_domain,
    ):
        source = amdgpu_descriptor_tables._descriptor_identity_source(
            (descriptor_set,),
            descriptor_strings,
            shared_strings,
        )

    assert "[LOOM_AMDGPU_DESCRIPTOR_IDENTITY_COUNT] = {" in source
    assert "    [0] = {" not in source
    assert "    [1] = {" in source
    assert ".key_string_ref =" in source
