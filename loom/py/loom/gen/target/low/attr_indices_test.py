# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace

import pytest

from loom.gen.target.low import attr_indices
from loom.gen.target.low.low_descriptors import generate_descriptor_set, generate_descriptor_set_family
from loom.target.low_descriptors import AsmImmediate, ImmediateFlag
from loom.target.test.descriptors import TEST_LOW_CONST_I32_DESCRIPTOR, TEST_LOW_CORE_DESCRIPTOR_SET


def _descriptor(*field_names: str):
    template = TEST_LOW_CONST_I32_DESCRIPTOR
    return replace(
        template,
        immediates=tuple(replace(template.immediates[0], field_name=name) for name in field_names),
        asm_forms=tuple(replace(form, immediates=tuple(AsmImmediate(name) for name in field_names)) for form in template.asm_forms),
    )


def _generate(descriptor):
    return generate_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,)))


def _reader(name: str, index: int, *, optional: bool = False):
    function = "loom_low_optional_immediate_attr" if optional else "loom_low_immediate_attr"
    return f"#define {name}(attributes) \\\n  {function}((attributes), {index})"


def test_attr_readers_follow_dictionary_order_after_schema_edits():
    descriptor = _descriptor("zeta", "alpha")
    original = _generate(descriptor)
    assert _reader("loom_test_low_core_test_const_i32_alpha", 0) in original.header
    assert _reader("loom_test_low_core_test_const_i32_zeta", 1) in original.header

    reordered = _generate(replace(descriptor, immediates=tuple(reversed(descriptor.immediates))))
    assert original.header == reordered.header

    inserted = _generate(_descriptor("zeta", "middle", "alpha"))
    assert _reader("loom_test_low_core_test_const_i32_alpha", 0) in inserted.header
    assert _reader("loom_test_low_core_test_const_i32_middle", 1) in inserted.header
    assert _reader("loom_test_low_core_test_const_i32_zeta", 2) in inserted.header

    renamed = _generate(_descriptor("omega", "alpha"))
    assert "loom_test_low_core_test_const_i32_zeta" not in renamed.header
    assert _reader("loom_test_low_core_test_const_i32_omega", 1) in renamed.header

    removed = _generate(_descriptor("alpha"))
    assert "loom_test_low_core_test_const_i32_zeta" not in removed.header
    assert _reader("loom_test_low_core_test_const_i32_alpha", 0) in removed.header


def test_non_trailing_optional_field_invalidates_its_and_later_positions():
    descriptor = _descriptor("zeta", "middle", "alpha")
    descriptor = replace(
        descriptor,
        immediates=tuple(replace(immediate, flags=(ImmediateFlag.DEFAULT_VALUE,)) if immediate.field_name == "middle" else immediate for immediate in descriptor.immediates),
    )
    generated = _generate(descriptor)
    assert _reader("loom_test_low_core_test_const_i32_alpha", 0) in generated.header
    assert "loom_test_low_core_test_const_i32_middle" not in generated.header
    assert "loom_test_low_core_test_const_i32_zeta" not in generated.header


def test_trailing_optional_reader_tracks_presence_after_schema_edits():
    def generate(*field_names):
        descriptor = _descriptor(*field_names)
        descriptor = replace(
            descriptor,
            immediates=tuple(replace(immediate, flags=(ImmediateFlag.DEFAULT_VALUE,)) if immediate.field_name == "offset" else immediate for immediate in descriptor.immediates),
        )
        return _generate(descriptor).header

    original = generate("offset")
    assert _reader("loom_test_low_core_test_const_i32_offset", 0, optional=True) in original

    inserted_before = generate("offset", "alignment")
    assert _reader("loom_test_low_core_test_const_i32_alignment", 0) in inserted_before
    assert _reader("loom_test_low_core_test_const_i32_offset", 1, optional=True) in inserted_before

    inserted_after = generate("alignment", "offset", "zeta")
    assert _reader("loom_test_low_core_test_const_i32_alignment", 0) in inserted_after
    assert "loom_test_low_core_test_const_i32_offset" not in inserted_after
    assert "loom_test_low_core_test_const_i32_zeta" not in inserted_after


def test_descriptor_view_readers_use_the_view_namespace():
    storage = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(_descriptor("zeta", "alpha"),))
    view = replace(storage, key="test.low.view", c_enum_prefix="TEST_LOW_VIEW", function_name="loom_test_low_view_descriptor_set")
    generated = generate_descriptor_set_family(storage, (view,))
    assert _reader("loom_test_low_view_test_const_i32_alpha", 0) in generated.view_headers[0]
    assert _reader("loom_test_low_view_test_const_i32_zeta", 1) in generated.view_headers[0]
    assert "loom_test_low_core_test_const_i32_alpha" not in generated.view_headers[0]


def test_common_readers_require_agreement_across_descriptor_variants():
    variants = (_descriptor("alpha", "zeta"), _descriptor("alpha", "middle", "zeta"))
    lines = attr_indices.emit_attr_accessors("TEST", ((descriptor.key, descriptor.immediates) for descriptor in variants), target_key="test")
    header = "\n".join(lines)
    assert _reader("loom_test_const_i32_alpha", 0) in header
    assert "loom_test_const_i32_middle" not in header
    assert "loom_test_const_i32_zeta" not in header

    missing = attr_indices.emit_attr_accessors("TEST", ((descriptor.key, descriptor.immediates) for descriptor in (*variants, _descriptor())), target_key="test")
    assert missing == []

    required = _descriptor("alpha")
    optional = replace(required, immediates=(replace(required.immediates[0], flags=(ImmediateFlag.DEFAULT_VALUE,)),))
    assert attr_indices.emit_attr_accessors("TEST", ((descriptor.key, descriptor.immediates) for descriptor in (required, optional)), target_key="test") == []


def test_attr_names_reject_c_identifier_collisions():
    descriptor = _descriptor("same.name", "same_name")
    with pytest.raises(ValueError, match="collides between"):
        attr_indices.emit_attr_accessors("TEST", ((descriptor.key, descriptor.immediates),), target_key="test")


def test_attr_reader_macros_preserve_optional_presence():
    descriptor = _descriptor("alpha", "offset")
    descriptor = replace(
        descriptor,
        immediates=tuple(replace(immediate, flags=(ImmediateFlag.DEFAULT_VALUE,)) if immediate.field_name == "offset" else immediate for immediate in descriptor.immediates),
    )
    header = _generate(descriptor).header
    assert "#define loom_test_low_core_test_const_i32_alpha(attributes)" in header
    assert "loom_low_immediate_attr((attributes), 0)" in header
    assert "#define loom_test_low_core_test_const_i32_offset(attributes)" in header
    assert "loom_low_optional_immediate_attr((attributes), 1)" in header
    assert "static inline" not in header
