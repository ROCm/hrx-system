# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.gen.support.c import CIdentifierCase, c_identifier
from loom.gen.target.arch.amdgpu import amdgpu_config_tables
from loom.target.arch.amdgpu.encoding import (
    AMDGPU_ENCODING_FIELD_IDS,
    AMDGPU_ENCODING_FIELD_NAMES,
)


def _encoding_field_row(name: str) -> str:
    suffix = c_identifier(name, case=CIdentifierCase.UPPER, empty="EMPTY")
    return f"LOOM_AMDGPU_ENCODING_FIELD(LOOM_AMDGPU_ENCODING_FIELD_{suffix}, {AMDGPU_ENCODING_FIELD_IDS[name]})"


def test_encoding_field_ids_fragment_is_row_data_only() -> None:
    source = amdgpu_config_tables._emit_encoding_field_ids()

    assert "typedef " not in source
    assert "enum " not in source
    assert "#ifndef " not in source
    assert "#define " not in source
    assert "#include " not in source
    assert "\nif " not in source
    assert "\nreturn " not in source

    lines = source.splitlines()
    assert len(lines) == len(AMDGPU_ENCODING_FIELD_NAMES)
    assert lines[0] == _encoding_field_row(AMDGPU_ENCODING_FIELD_NAMES[0])
    assert lines[-1] == _encoding_field_row(AMDGPU_ENCODING_FIELD_NAMES[-1])
