# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Audit and tag a Linux wheel without changing its installed payload."""

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

from auditwheel.main import main as auditwheel_main


def check_payload(source: Path, audited: Path) -> None:
    """Only wheel tags and their RECORD digests may change during the audit."""
    with ZipFile(source) as before, ZipFile(audited) as after:
        old_files = {
            entry.filename for entry in before.infolist() if not entry.is_dir()
        }
        new_files = {entry.filename for entry in after.infolist() if not entry.is_dir()}
        if old_files != new_files:
            raise ValueError(
                "ABI audit changed the SDK file set: "
                f"added={sorted(new_files - old_files)}, "
                f"removed={sorted(old_files - new_files)}"
            )
        for name in sorted(old_files):
            if name.endswith((".dist-info/WHEEL", ".dist-info/RECORD")):
                continue
            with before.open(name) as old, after.open(name) as new:
                if (
                    hashlib.file_digest(old, "sha256").digest()
                    != hashlib.file_digest(new, "sha256").digest()
                ):
                    raise ValueError(f"ABI audit changed SDK payload: {name}")


def main(source: str, output: str, platform: str) -> None:
    source_path = Path(source).absolute()
    output_path = Path(output).absolute()
    with tempfile.TemporaryDirectory() as temporary:
        # The wheel already owns its complete native closure and loader paths.
        # Refuse any repair that needs to graft or rewrite a system library.
        sys.argv = [
            "auditwheel",
            "repair",
            "--plat",
            platform,
            "--only-plat",
            "--patcher=none",
            "--zip-compression-level=9",
            "--wheel-dir",
            temporary,
            str(source_path),
        ]
        result = auditwheel_main()
        if result:
            raise RuntimeError(f"Wheel ABI audit failed: {result}")
        audited = Path(temporary) / output_path.name
        check_payload(source_path, audited)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(audited, output_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
