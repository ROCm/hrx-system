# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.target.arch.amdgpu.descriptors import (
    _gfx11_core_overlays,
    _gfx12_core_overlays,
    _gfx117x_core_overlays,
    _gfx940_core_overlays,
    _gfx950_core_overlays,
    _gfx1250_core_overlays,
)
from loom.target.arch.amdgpu.low_aliases import (
    sorted_amdgpu_blocked_low_aliases,
    validate_amdgpu_blocked_low_aliases,
)


def _amdgpu_core_overlay_sets():
    return (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx117x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx1250_core_overlays(),
    )


def test_blocked_low_aliases_are_sorted_and_unique() -> None:
    aliases = sorted_amdgpu_blocked_low_aliases()
    validate_amdgpu_blocked_low_aliases(aliases)
    descriptor_keys = [alias.descriptor_key for alias in aliases]
    assert descriptor_keys == sorted(descriptor_keys)
    assert len(descriptor_keys) == len(set(descriptor_keys))

    lookup_names = [
        lookup_name for alias in aliases for lookup_name in alias.lookup_names
    ]
    assert len(lookup_names) == len(set(lookup_names))


def test_blocked_low_aliases_are_not_authorable_descriptors() -> None:
    blocked_descriptor_keys = {
        alias.descriptor_key for alias in sorted_amdgpu_blocked_low_aliases()
    }

    for descriptor_set in _amdgpu_core_overlay_sets():
        descriptor_keys = {descriptor.descriptor_key for descriptor in descriptor_set}
        assert not blocked_descriptor_keys & descriptor_keys


def test_blocked_low_alias_replacements_are_authorable_descriptors() -> None:
    aliases = sorted_amdgpu_blocked_low_aliases()

    for descriptor_set in _amdgpu_core_overlay_sets():
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        for alias in aliases:
            descriptor = descriptors[alias.replacement_descriptor_key]
            assert descriptor.mnemonic == alias.replacement_mnemonic
