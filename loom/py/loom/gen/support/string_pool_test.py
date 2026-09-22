# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.gen.support.string_pool import CStringPool, _slice_reference, emit_c_string_pool, layout_c_string_pool


def test_intern_records_unique_entries_with_stable_labels() -> None:
    pool = CStringPool("TEST")

    assert pool.intern("first.value", "alpha") == "first_value"
    assert pool.intern("second-value", "beta") == "second_value"

    assert [(entry.label, entry.value) for entry in pool.entries] == [
        ("first_value", "alpha"),
        ("second_value", "beta"),
    ]


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


def test_intern_rejects_slice_payload_overflow() -> None:
    pool = CStringPool("TEST")

    with pytest.raises(ValueError, match="exceeds 255 bytes"):
        pool.intern("label", "x" * 256)


def test_intern_can_collect_unbounded_c_strings() -> None:
    pool = CStringPool("TEST", max_payload_length=None)

    pool.intern("long", "x" * 1024)

    assert pool.entries[0].value == "x" * 1024


def test_canonical_label_matches_c_identifier_policy() -> None:
    assert CStringPool.canonical_label("...") == "empty"
    assert CStringPool.canonical_label("9-lives") == "_9_lives"


def test_layout_shares_qualified_names_and_mnemonics() -> None:
    pool = CStringPool("TEST")
    pool.intern("qualified", "amdgpu.flat_atomic_and_b64")
    pool.intern("mnemonic", "flat_atomic_and_b64")
    pool.intern("infix", "atomic_and")
    pool.intern("empty", "")
    layout = layout_c_string_pool(pool)
    assert layout.data == b"amdgpu.flat_atomic_and_b64"
    assert layout.references["empty"] == 0
    for entry in pool.entries:
        reference = layout.references[entry.label]
        offset, length = reference & 0xFFFFFF, reference >> 24
        assert layout.data[offset : offset + length] == entry.value.encode()
    assert layout.references["mnemonic"] & 0xFFFFFF == 7


def test_layout_does_not_depend_on_intern_order() -> None:
    values = ["zero", "zero_suffix", "middle", ""]
    layouts = []
    for ordered in [values, list(reversed(values))]:
        pool = CStringPool("TEST")
        for index, value in enumerate(ordered):
            pool.intern(f"value_{index}", value)
        layout = layout_c_string_pool(pool)
        layouts.append((layout.data, {entry.value: layout.references[entry.label] for entry in pool.entries}))
    assert layouts[0] == layouts[1]


def test_layout_counts_utf8_bytes_and_accepts_maximum_length() -> None:
    pool = CStringPool("TEST")
    pool.intern("utf8", "é" * 127 + "x")
    pool.intern("suffix", "éx")
    layout = layout_c_string_pool(pool)
    assert len(layout.data) == 255
    assert layout.references["utf8"] >> 24 == 255
    assert layout.references["suffix"] >> 24 == 3
    with pytest.raises(ValueError, match="exceeds 255 bytes"):
        pool.intern("overflow", "é" * 128)


def test_slice_layout_rejects_unbounded_payloads() -> None:
    pool = CStringPool("TEST", max_payload_length=None)
    pool.intern("long", "x" * 256)
    with pytest.raises(ValueError, match="exceeds 255 bytes"):
        layout_c_string_pool(pool)


def test_emit_preserves_c_escapes_and_uses_byte_references() -> None:
    pool = CStringPool("TEST")
    pool.intern("escaped", 'é\0"\\\n')
    lines = emit_c_string_pool(pool, "kTestStringData")
    assert "static const char kTestStringData[] =" in lines
    assert "  TEST_STRING_escaped = 0x06000000u," in lines
    assert "\\000" in "\n".join(lines)
    assert "string pool byte length must match generated slices" in "\n".join(lines)


def test_empty_pool_and_empty_string() -> None:
    pool = CStringPool("TEST")
    assert emit_c_string_pool(pool, "kData") == []
    pool.intern("empty", "")
    layout = layout_c_string_pool(pool)
    assert layout.data == b""
    assert layout.references == {"empty": 0}
    assert '    "";' in emit_c_string_pool(pool, "kData")


def test_slice_reference_pool_bound_and_none_sentinel() -> None:
    assert _slice_reference(0xFFFFFF, 1) == 0x01FFFFFF
    assert _slice_reference(0xFFFF01, 255) == 0xFFFFFF01
    with pytest.raises(ValueError, match="24-bit"):
        _slice_reference(0xFFFFFF, 2)
    with pytest.raises(ValueError, match="24-bit"):
        _slice_reference(0x1000000, 0)
    with pytest.raises(ValueError, match="24-bit"):
        _slice_reference(0xFFFFFF, 255)
