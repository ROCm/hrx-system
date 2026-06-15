# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: package-sentinel ``__init__.py`` files."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from loom.gen import bootstrap as _bootstrap
from loom.gen.support.generated_file import line_comment_header


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
            regenerate="python3 loom/py/loom/gen/run.py package_inits --in-place",
        )
    )
    lines.extend(
        [
            "",
            f'"""{package_init.docstring}"""',
        ]
    )
    return "\n".join(lines) + "\n"


def _write_files(files: Mapping[str, str]) -> None:
    for rel_path, content in sorted(files.items()):
        path = _bootstrap.REPO_ROOT / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"  {rel_path}")


def _check_files(files: Mapping[str, str]) -> int:
    stale_or_changed: list[str] = []
    for rel_path, expected in sorted(files.items()):
        path = _bootstrap.REPO_ROOT / rel_path
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            stale_or_changed.append(rel_path)
    if stale_or_changed:
        print(
            "error: generated package init files are stale; regenerate with python3 loom/py/loom/gen/run.py package_inits --in-place",
            file=sys.stderr,
        )
        for rel_path in stale_or_changed:
            print(f"  {rel_path}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--in-place", action="store_true")
    args = parser.parse_args(argv)

    files = generate_package_init_files()
    if args.in_place:
        _write_files(files)
        return 0
    return _check_files(files)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
