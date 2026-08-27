# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the top-level Loom Python package import contract."""

import sys

import pytest

import loom

_AUTHORING_MODULES_LOADED_WITH_ROOT = {
    module_name
    for module_name in ("loom.builders", "loom.diagnostics", "loom.verify")
    if module_name in sys.modules
}


def test_root_package_defers_authoring_surface() -> None:
    assert not _AUTHORING_MODULES_LOADED_WITH_ROOT


def test_root_package_resolves_and_caches_public_exports() -> None:
    builder_type = loom.LoomBuilder

    assert builder_type.__module__ == "loom.builders"
    assert loom.LoomBuilder is builder_type
    assert set(loom.__all__).issubset(dir(loom))
    assert all(getattr(loom, name) is getattr(loom, name) for name in loom.__all__)


def test_root_package_rejects_unknown_exports() -> None:
    with pytest.raises(AttributeError, match="not_a_public_export"):
        _ = loom.not_a_public_export
