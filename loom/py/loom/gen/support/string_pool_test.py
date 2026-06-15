# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.gen.support.string_pool import CStringPool


def test_intern_records_unique_entries_with_byte_offsets() -> None:
    pool = CStringPool("TEST")

    assert pool.intern("first.value", "alpha") == "first_value"
    assert pool.intern("second-value", "beta") == "second_value"

    assert [(entry.label, entry.value, entry.offset) for entry in pool.entries] == [
        ("first_value", "alpha", 0),
        ("second_value", "beta", 6),
    ]
    assert pool.next_offset == 11


def test_intern_aliases_duplicate_values_to_primary_label() -> None:
    pool = CStringPool("TEST")

    assert pool.intern("first", "same") == "first"
    assert pool.intern("alias", "same") == "first"

    assert len(pool.entries) == 1
    assert pool.ref("alias") == "TEST_STRING_first"


def test_intern_allows_reusing_label_for_same_value() -> None:
    pool = CStringPool("TEST")

    assert pool.intern("label", "same") == "label"
    assert pool.intern("label", "same") == "label"


def test_intern_rejects_reusing_label_for_different_values() -> None:
    pool = CStringPool("TEST")
    pool.intern("label", "first")

    with pytest.raises(ValueError, match="reused for different values"):
        pool.intern("label", "second")


def test_intern_rejects_bstring_payload_overflow() -> None:
    pool = CStringPool("TEST")

    with pytest.raises(ValueError, match="exceeds 255 bytes"):
        pool.intern("label", "x" * 256)


def test_canonical_label_matches_c_identifier_policy() -> None:
    assert CStringPool.canonical_label("...") == "empty"
    assert CStringPool.canonical_label("9-lives") == "_9_lives"
