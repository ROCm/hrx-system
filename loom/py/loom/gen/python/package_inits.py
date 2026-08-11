# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: package-sentinel ``__init__.py`` files."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from loom.gen import bootstrap as _bootstrap
from loom.gen.support.generated_file import (
    GeneratedFileMaintenanceMode,
    GeneratedFileMaintenanceResult,
    GeneratedFileSet,
    line_comment_header,
    maintain_generated_file_set,
)

DESCRIPTION = "Python package initializers"
REGENERATE_COMMAND = "python3 loom/py/loom/gen/run.py package_inits --in-place"


@dataclass(frozen=True, slots=True)
class PackageInit:
    path: str
    docstring: str


PACKAGE_INITS = (
    PackageInit(
        "loom/py/loom/target/__init__.py",
        "Target descriptor inputs for Loom code generation.",
    ),
    PackageInit(
        "loom/py/loom/target/arch/__init__.py",
        "Target-family descriptor inputs.",
    ),
    PackageInit(
        "loom/py/loom/target/arch/amdgpu/__init__.py",
        "AMDGPU target-family descriptor and dialect inputs.",
    ),
    PackageInit(
        "loom/py/loom/target/arch/amdgpu/contracts/__init__.py",
        "Sharded AMDGPU source-to-low contract fragments.",
    ),
    PackageInit(
        "loom/py/loom/target/arch/ireevm/__init__.py",
        "IREE VM target-family descriptor and dialect inputs.",
    ),
    PackageInit(
        "loom/py/loom/target/arch/spirv/__init__.py",
        "SPIR-V target-family descriptor and dialect inputs.",
    ),
    PackageInit(
        "loom/py/loom/target/arch/wasm/__init__.py",
        "Wasm target descriptor inputs.",
    ),
    PackageInit(
        "loom/py/loom/target/arch/x86/__init__.py",
        "x86 target-family descriptor inputs.",
    ),
    PackageInit(
        "loom/py/loom/target/arch/x86/contracts/__init__.py",
        "x86 source-to-low contract fragments.",
    ),
    PackageInit(
        "loom/py/loom/target/arch/ireevm/contracts/__init__.py",
        "IREE VM source-to-low contract fragments.",
    ),
    PackageInit(
        "loom/py/loom/target/emit/__init__.py",
        "Emission-target descriptor inputs.",
    ),
    PackageInit(
        "loom/py/loom/target/emit/llvmir/__init__.py",
        "LLVMIR debug projection metadata.",
    ),
    PackageInit(
        "loom/py/loom/target/emit/llvmir/x86/__init__.py",
        "x86 LLVMIR debug projection metadata.",
    ),
    PackageInit(
        "loom/py/loom/target/emit/wasm/__init__.py",
        "Wasm emission target contract data.",
    ),
    PackageInit(
        "loom/py/loom/target/test/__init__.py",
        "Backend-independent test target descriptor inputs.",
    ),
)


def generate_package_init_files(
    package_inits: Sequence[PackageInit] = PACKAGE_INITS,
) -> dict[str, str]:
    return {package_init.path: _generate_package_init(package_init) for package_init in package_inits}


def _generate_package_init(package_init: PackageInit) -> str:
    lines = [
        "# Copyright 2026 The IREE Authors",
        "#",
        "# Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "# See https://llvm.org/LICENSE.txt for license information.",
        "# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
    ]
    lines.extend(
        line_comment_header(
            "#",
            generator="loom.gen.python.package_inits",
            regenerate=REGENERATE_COMMAND,
        )
    )
    lines.extend(
        [
            "",
            f'"""{package_init.docstring}"""',
        ]
    )
    return "\n".join(lines) + "\n"


def checked_in_file_set() -> GeneratedFileSet:
    """Returns the complete checked-in package-initializer ownership set."""
    return GeneratedFileSet.from_mapping(generate_package_init_files())


def maintain_checked_in_files(
    mode: GeneratedFileMaintenanceMode,
) -> GeneratedFileMaintenanceResult:
    """Checks or updates all checked-in package initializers."""
    return maintain_generated_file_set(
        _bootstrap.REPO_ROOT,
        checked_in_file_set(),
        mode=mode,
        description=DESCRIPTION,
        regenerate_command=REGENERATE_COMMAND,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--in-place", action="store_true")
    args = parser.parse_args(argv)

    result = maintain_checked_in_files("update" if args.in_place else "check")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
