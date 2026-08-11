# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from loom.gen import checked_in_artifacts
from loom.gen.support.generated_file import (
    GeneratedFileMaintenanceResult,
    GeneratedFileSet,
)


def test_checked_in_artifact_families_register_expected_families() -> None:
    empty_file_set = GeneratedFileSet.from_mapping({})
    amdgpu_target_config = SimpleNamespace(
        DESCRIPTION="AMDGPU target configuration",
        REGENERATE_COMMAND="python target_config.py --in-place",
        checked_in_file_set=mock.Mock(return_value=empty_file_set),
    )
    with (
        mock.patch.object(
            checked_in_artifacts,
            "_load_amdgpu_target_config",
            return_value=amdgpu_target_config,
        ),
        mock.patch.object(
            checked_in_artifacts.package_inits,
            "checked_in_file_set",
            return_value=empty_file_set,
        ),
        mock.patch.object(
            checked_in_artifacts.builders_pyi,
            "checked_in_file_set",
            return_value=empty_file_set,
        ),
        mock.patch.object(
            checked_in_artifacts.c_tables,
            "checked_in_file_set",
            return_value=empty_file_set,
        ),
        mock.patch.object(
            checked_in_artifacts.textmate,
            "checked_in_file_set",
            return_value=empty_file_set,
        ),
        mock.patch.object(
            checked_in_artifacts.x86_packed_dot_contract,
            "checked_in_file_set",
            return_value=empty_file_set,
        ),
        mock.patch.object(
            checked_in_artifacts.numeric_conversion_matrix,
            "checked_in_file_set",
            return_value=empty_file_set,
        ),
    ):
        families = checked_in_artifacts.checked_in_artifact_families()

    assert tuple(family.description for family in families) == (
        "Python package initializers",
        "Python builder stubs",
        "C op table artifacts",
        "AMDGPU target configuration",
        "TextMate grammars",
        "x86 packed-dot contract header",
        "FP8 numeric conversion witnesses",
    )
    amdgpu_target_config.checked_in_file_set.assert_called_once_with()


def test_registered_artifact_ownership_is_disjoint() -> None:
    owners: dict[str, str] = {}

    for family in checked_in_artifacts.checked_in_artifact_families():
        for path in (*family.file_set.output_paths, *family.file_set.obsolete_paths):
            assert path not in owners, f"{path} is owned by both {owners[path]} and {family.description}"
            owners[path] = family.description

    assert owners


def test_main_selects_check_and_update_modes() -> None:
    with mock.patch.object(
        checked_in_artifacts,
        "maintain_checked_in_artifacts",
        return_value=GeneratedFileMaintenanceResult(True),
    ) as maintain_checked_in_artifacts:
        assert checked_in_artifacts.main(["--check"]) == 0
        assert checked_in_artifacts.main(["--in-place"]) == 0

    assert maintain_checked_in_artifacts.call_args_list == [
        mock.call("check"),
        mock.call("update"),
    ]
