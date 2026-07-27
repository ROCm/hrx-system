# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.importers.core import NameAllocator, sanitize_identifier, source_name


class _NamedSource:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name


def test_sanitize_identifier_preserves_meaningful_names() -> None:
    assert sanitize_identifier("%i_tried_to_name_this") == "i_tried_to_name_this"
    assert sanitize_identifier("%123") == "_123"
    assert sanitize_identifier("a.b-c") == "a_b_c"


def test_name_allocator_deduplicates_stably() -> None:
    names = NameAllocator(["reserved"])

    assert names.fresh("value") == "value"
    assert names.fresh("value") == "value_2"
    assert names.reserve_or_fresh("reserved") == "reserved_2"


def test_source_name_prefers_foreign_get_name() -> None:
    assert source_name(_NamedSource("%named")) == "%named"
