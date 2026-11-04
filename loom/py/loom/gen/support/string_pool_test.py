# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.gen.support.string_pool import CStringBank, CStringPool, _slice_reference, emit_c_string_pool, layout_c_string_pool


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
    for bank in (0, 1):
        assert _slice_reference(0x7FFFFF, 1, bank) == 0x017FFFFF | bank << 23
        assert _slice_reference(0x7FFF01, 255, bank) == 0xFF7FFF01 | bank << 23
        with pytest.raises(ValueError, match="23-bit"):
            _slice_reference(0x7FFFFF, 2, bank)
        with pytest.raises(ValueError, match="23-bit"):
            _slice_reference(0x800000, 0, bank)
        with pytest.raises(ValueError, match="23-bit"):
            _slice_reference(0x7FFFFF, 255, bank)
    with pytest.raises(ValueError, match="primary or secondary"):
        _slice_reference(0, 0, 2)


def test_shared_bank_preserves_bytes_and_independent_layout(tmp_path) -> None:
    bank = CStringBank.build(frozenset({"amdgpu.s_add_f32", "é\0suffix"}), symbol="test_strings", header="test/strings.h")
    path = tmp_path / "strings.json"
    path.write_text(bank.manifest())
    assert CStringBank.read(path) == bank
    pool = CStringPool("TEST")
    for label, value in (("name", "amdgpu.s_add_f32"), ("suffix", "s_add_f32"), ("private", "private_field"), ("bytes", "é\0suffix"), ("empty", "")):
        pool.intern(label, value)
    layout = layout_c_string_pool(pool, bank)
    assert layout.shared is bank
    assert layout.data == b"private_field"
    assert layout.references["empty"] == 0
    for entry in pool.entries:
        reference = layout.references[entry.label]
        data = (layout.data, bank.data)[(reference >> 23) & 1]
        start = reference & 0x7FFFFF
        assert data[start : start + (reference >> 24)] == entry.value.encode()
    assert "test_strings" in layout.initializer("private_data")
    assert "extern const char test_strings[" in bank.emit_header()
    assert '#include "test/strings.h"' in bank.emit_source()


def test_external_primary_bank_composes_with_shared_bank() -> None:
    shared = CStringBank.build(
        frozenset({"amdgpu.s_add_f32"}),
        symbol="test_instruction_strings",
        header="test/instruction_strings.h",
    )
    values = frozenset(
        {
            "",
            "amdgpu.s_add_f32",
            "descriptor-only metadata",
        }
    )
    primary = CStringBank.build(
        values,
        symbol="test_descriptor_strings",
        header="test/descriptor_strings.h",
        shared=shared,
    )
    assert primary.values == frozenset({"", "descriptor-only metadata"})

    pool = CStringPool("TEST")
    for index, value in enumerate(sorted(values)):
        pool.intern(f"value_{index}", value)
    layout = layout_c_string_pool(pool, shared, primary=primary)

    assert layout.data == b""
    assert layout.primary is primary
    assert layout.shared is shared
    assert "test_descriptor_strings" in layout.initializer("unused_local_data")
    assert "test_instruction_strings" in layout.initializer("unused_local_data")
    emitted = emit_c_string_pool(pool, "unused_local_data", layout=layout)
    assert not any("const char unused_local_data" in line for line in emitted)
    assert any("sizeof(test_descriptor_strings)" in line for line in emitted)

    incomplete_primary = CStringBank.build(
        frozenset({""}),
        symbol="test_incomplete_strings",
        header="test/incomplete_strings.h",
    )
    with pytest.raises(ValueError, match="does not contain 'descriptor-only metadata'"):
        layout_c_string_pool(pool, shared, primary=incomplete_primary)


def test_incidental_substrings_do_not_acquire_shared_bank() -> None:
    bank = CStringBank.build(frozenset({"amdgpu.s_add_f32"}), symbol="test_strings", header="test/strings.h")
    pool = CStringPool("TEST")
    pool.intern("field", "add")
    pool.intern("empty", "")
    layout = layout_c_string_pool(pool, bank)
    assert layout.shared is None
    assert layout.data == b"add"
