# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.gen.python.package_inits import PackageInit, generate_package_init_files
from loom.gen.support.generated_file import GENERATED_FILE_MARKER


def test_generates_canonical_package_sentinel() -> None:
    files = generate_package_init_files(
        (
            PackageInit(
                "loom/py/loom/example/__init__.py",
                "Example package.",
            ),
        )
    )

    assert files == {
        "loom/py/loom/example/__init__.py": (
            "# Copyright 2026 The IREE Authors\n"
            "#\n"
            "# Licensed under the Apache License v2.0 with LLVM Exceptions.\n"
            "# See https://llvm.org/LICENSE.txt for license information.\n"
            "# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n"
            "\n"
            f"# {GENERATED_FILE_MARKER}\n"
            "# Generator: loom.gen.python.package_inits.\n"
            "# Regenerate: python3 loom/py/loom/gen/run.py package_inits --in-place\n"
            "\n"
            '"""Example package."""\n'
        )
    }
