# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.gen.target.arch.amdgpu.encoding.amdgpu_encoding_tables import (
    _bit_range,
    _descriptor_encoding_field_names,
    _DescriptorEncodingContract,
    _EncodingContract,
    _field,
    _identifier_seed_without_fields,
    _project_encoding_contract,
    _replace_encoding_fields,
    _with_vop3_unused_source_defaults,
)
from loom.target.arch.amdgpu.isa_xml import AmdgpuIsaEncoding
from loom.target.low_descriptors import Descriptor


def _encoding_contract(*descriptor_keys: str) -> _EncodingContract:
    return _EncodingContract(
        descriptors=tuple(
            _DescriptorEncodingContract(
                descriptor_key=descriptor_key,
                format_id=1,
                bit_count=32,
                word_count=1,
                identifier_seed=0,
                fields=(),
            )
            for descriptor_key in descriptor_keys
        ),
        source_literal=255,
        scalar_source_literal=255,
        scalar_inline_u32=(128, 65),
        inline_f32_sources=(),
        vector_source_vgprs=(256, 256),
        s_mov_b32_opcode=0,
        v_mov_b32_opcode=1,
    )


def test_encoding_contract_projection_preserves_view_order() -> None:
    contract = _encoding_contract("test.first", "test.second", "test.third")

    projected = _project_encoding_contract(
        contract,
        ("test.third", "test.first"),
    )

    assert tuple(descriptor.descriptor_key for descriptor in projected.descriptors) == ("test.third", "test.first")
    assert projected.source_literal == contract.source_literal
    assert projected.vector_source_vgprs == contract.vector_source_vgprs


def test_encoding_contract_projection_rejects_missing_descriptor() -> None:
    contract = _encoding_contract("test.present")

    with pytest.raises(
        ValueError,
        match=r"storage contract is missing view descriptor 'test\.missing'",
    ):
        _project_encoding_contract(contract, ("test.missing",))


def test_descriptor_encoding_contract_includes_opcode_field() -> None:
    descriptor = Descriptor(
        key="amdgpu.test",
        mnemonic="test",
        semantic_tag=None,
        operands=(),
        schedule_class="test",
        encoding_format_id=1,
        encoding_id=1,
    )

    assert _descriptor_encoding_field_names(descriptor) == {"OP"}


def test_identifier_seed_ignores_fields_written_during_packing() -> None:
    opcode_field = _field("OP", _bit_range(16, 7))

    first_seed = _identifier_seed_without_fields(0xD3BE0000, (opcode_field,))
    second_seed = _identifier_seed_without_fields(0xD3AD0000, (opcode_field,))

    assert first_seed == 0xD3800000
    assert second_seed == first_seed


def test_identifier_seed_retains_unwritten_format_bits() -> None:
    opcode_field = _field("OP", _bit_range(16, 7))

    lhs_seed = _identifier_seed_without_fields(0xD3BE0000, (opcode_field,))
    rhs_seed = _identifier_seed_without_fields(0xD2BE0000, (opcode_field,))

    assert lhs_seed != rhs_seed


def test_vop3_encoding_seeds_default_unwritten_sources_to_inline_zero() -> None:
    source_fields = (
        _field("SRC0", _bit_range(32, 9)),
        _field("SRC1", _bit_range(41, 9)),
        _field("SRC2", _bit_range(50, 9)),
    )
    encoding = AmdgpuIsaEncoding(
        name="ENC_VOP3",
        order=0,
        bit_count=64,
        identifier_mask=0,
        identifier_values=((0x1FF << 32) | (0x1FF << 41) | (0x1FF << 50),),
        fields=source_fields,
    )

    (defaulted_encoding,) = _with_vop3_unused_source_defaults((encoding,), 0x80)

    assert defaulted_encoding.identifier_values == ((0x80 << 32) | (0x80 << 41) | (0x80 << 50),)


def test_vop3_source_defaults_leave_other_encoding_families_unchanged() -> None:
    encoding = AmdgpuIsaEncoding(
        name="ENC_VINTERP",
        order=0,
        bit_count=64,
        identifier_mask=0,
        identifier_values=(0,),
        fields=(
            _field("SRC0", _bit_range(32, 9)),
            _field("SRC1", _bit_range(41, 9)),
            _field("SRC2", _bit_range(50, 9)),
        ),
    )

    assert _with_vop3_unused_source_defaults((encoding,), 0x80) == (encoding,)


def test_encoding_field_replacement_preserves_field_order() -> None:
    encoding = AmdgpuIsaEncoding(
        name="ENC_VOP3P",
        order=0,
        bit_count=64,
        identifier_mask=0,
        identifier_values=(0,),
        fields=(
            _field("OP", _bit_range(16, 7)),
            _field("SRC0", _bit_range(32, 9)),
        ),
    )

    replaced = _replace_encoding_fields(encoding, (_field("OP", _bit_range(16, 8)),))

    assert tuple(field.name for field in replaced.fields) == ("OP", "SRC0")
    assert replaced.fields[0].ranges == (_bit_range(16, 8),)


def test_encoding_field_replacement_rejects_unknown_field() -> None:
    encoding = AmdgpuIsaEncoding(
        name="ENC_VOP3P",
        order=0,
        bit_count=64,
        identifier_mask=0,
        identifier_values=(0,),
        fields=(_field("OP", _bit_range(16, 7)),),
    )

    with pytest.raises(ValueError, match="cannot replace missing fields: UNKNOWN"):
        _replace_encoding_fields(encoding, (_field("UNKNOWN", _bit_range(16, 8)),))
