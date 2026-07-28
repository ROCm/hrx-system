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
    _field,
    _identifier_seed_without_fields,
    _replace_encoding_fields,
)
from loom.target.arch.amdgpu.isa_xml import AmdgpuIsaEncoding
from loom.target.low_descriptors import Descriptor


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
