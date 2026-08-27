# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from loom.gen.target.arch.x86 import x86_packed_dot_contract


def test_checked_in_file_set_owns_only_source_maintenance_header() -> None:
    generated_file_set = x86_packed_dot_contract.checked_in_file_set()

    assert generated_file_set.output_paths == ("loom/src/loom/target/arch/x86/packed_dot_contract_data.h",)
    assert generated_file_set.obsolete_paths == ()
    assert "loom_x86_packed_dot_builtin_descriptors" in generated_file_set.files[0].contents


def test_main_requires_an_explicit_mode_or_build_output() -> None:
    with pytest.raises(SystemExit, match="2"):
        x86_packed_dot_contract.main([])


def test_main_selects_checked_in_maintenance_modes() -> None:
    with mock.patch.object(
        x86_packed_dot_contract,
        "maintain_checked_in_files",
        return_value=SimpleNamespace(ok=True),
    ) as maintain_checked_in_files:
        assert x86_packed_dot_contract.main(["--check"]) == 0
        assert x86_packed_dot_contract.main(["--in-place"]) == 0

    assert maintain_checked_in_files.call_args_list == [
        mock.call("check"),
        mock.call("update"),
    ]


def test_main_rejects_mixed_maintenance_and_build_outputs(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        x86_packed_dot_contract.main(["--check", "--source", str(tmp_path / "packed_dot_contract_data.c")])


def test_main_writes_explicit_build_outputs(tmp_path: Path) -> None:
    header_path = tmp_path / "include/packed_dot_contract_data.h"
    source_path = tmp_path / "src/packed_dot_contract_data.c"

    assert (
        x86_packed_dot_contract.main(
            [
                "--header",
                str(header_path),
                "--source",
                str(source_path),
            ]
        )
        == 0
    )

    assert "loom_x86_packed_dot_builtin_descriptors" in header_path.read_text(encoding="utf-8")
    source = source_path.read_text(encoding="utf-8")
    assert '#include "loom/target/arch/x86/packed_dot_contract_data.h"' in source
    assert "loom_x86_packed_dot_builtin_descriptor_count" in source
