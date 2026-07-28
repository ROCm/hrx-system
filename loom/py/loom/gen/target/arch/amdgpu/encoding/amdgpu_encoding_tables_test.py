# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.gen.target.arch.amdgpu.encoding.amdgpu_encoding_tables import (
    _bit_range,
    _descriptor_encoding_field_names,
    _field,
    _identifier_seed_without_fields,
)
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

    cdna3_seed = _identifier_seed_without_fields(0xD3BE0000, (opcode_field,))
    cdna4_seed = _identifier_seed_without_fields(0xD3AD0000, (opcode_field,))

    assert cdna3_seed == 0xD3800000
    assert cdna4_seed == cdna3_seed


def test_identifier_seed_retains_unwritten_format_bits() -> None:
    opcode_field = _field("OP", _bit_range(16, 7))

    lhs_seed = _identifier_seed_without_fields(0xD3BE0000, (opcode_field,))
    rhs_seed = _identifier_seed_without_fields(0xD2BE0000, (opcode_field,))

    assert lhs_seed != rhs_seed
