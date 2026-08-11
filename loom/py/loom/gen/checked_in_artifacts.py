# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Authoritative checked-in Loom generated-artifact registry."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Sequence
from types import ModuleType

from loom.gen import bootstrap as _bootstrap
from loom.gen.editor import textmate
from loom.gen.ops import c_tables
from loom.gen.python import builders_pyi, package_inits
from loom.gen.support.generated_file import (
    GeneratedFileFamily,
    GeneratedFileMaintenanceMode,
    GeneratedFileMaintenanceResult,
    maintain_generated_file_families,
)
from loom.gen.target.arch.x86 import x86_packed_dot_contract
from loom.gen.test import numeric_conversion_matrix

_AMDGPU_TARGET_CONFIG_MODULE_NAME = "loom_build_tools_amdgpu_target_config"


def _load_amdgpu_target_config() -> ModuleType:
    existing_module = sys.modules.get(_AMDGPU_TARGET_CONFIG_MODULE_NAME)
    if existing_module is not None:
        return existing_module
    module_path = _bootstrap.REPO_ROOT / "loom/build_tools/amdgpu/target_config.py"
    spec = importlib.util.spec_from_file_location(
        _AMDGPU_TARGET_CONFIG_MODULE_NAME,
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def checked_in_artifact_families() -> tuple[GeneratedFileFamily, ...]:
    """Returns every checked-in Loom generated artifact family."""
    amdgpu_target_config = _load_amdgpu_target_config()
    return (
        GeneratedFileFamily(
            description=package_inits.DESCRIPTION,
            regenerate_command=package_inits.REGENERATE_COMMAND,
            file_set=package_inits.checked_in_file_set(),
        ),
        GeneratedFileFamily(
            description=builders_pyi.DESCRIPTION,
            regenerate_command=builders_pyi.REGENERATE_COMMAND,
            file_set=builders_pyi.checked_in_file_set(),
        ),
        GeneratedFileFamily(
            description=c_tables.DESCRIPTION,
            regenerate_command=c_tables.REGENERATE_COMMAND,
            file_set=c_tables.checked_in_file_set(),
        ),
        GeneratedFileFamily(
            description=amdgpu_target_config.DESCRIPTION,
            regenerate_command=amdgpu_target_config.REGENERATE_COMMAND,
            file_set=amdgpu_target_config.checked_in_file_set(),
        ),
        GeneratedFileFamily(
            description=textmate.DESCRIPTION,
            regenerate_command=textmate.REGENERATE_COMMAND,
            file_set=textmate.checked_in_file_set(),
        ),
        GeneratedFileFamily(
            description=x86_packed_dot_contract.DESCRIPTION,
            regenerate_command=x86_packed_dot_contract.REGENERATE_COMMAND,
            file_set=x86_packed_dot_contract.checked_in_file_set(),
        ),
        GeneratedFileFamily(
            description=numeric_conversion_matrix.DESCRIPTION,
            regenerate_command=numeric_conversion_matrix.REGENERATE_COMMAND,
            file_set=numeric_conversion_matrix.checked_in_file_set(),
        ),
    )


def maintain_checked_in_artifacts(
    mode: GeneratedFileMaintenanceMode,
) -> GeneratedFileMaintenanceResult:
    """Checks or updates every registered checked-in artifact family."""
    return maintain_generated_file_families(
        _bootstrap.REPO_ROOT,
        checked_in_artifact_families(),
        mode=mode,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Maintain all checked-in Loom generated artifacts.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--in-place", action="store_true")
    args = parser.parse_args(argv)

    result = maintain_checked_in_artifacts("update" if args.in_place else "check")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
