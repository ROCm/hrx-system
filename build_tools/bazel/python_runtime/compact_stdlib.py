# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Builds the deterministic standard-library archive used by Windows Python."""

from __future__ import annotations

import argparse
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePath

_ARCHIVE_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_REQUIRED_MEMBERS = (
    "encodings/__init__.py",
    "importlib/__init__.py",
    "site.py",
)
# Preserve direct filesystem access through ssl.__file__.
_UNARCHIVED_MEMBERS = frozenset(("ssl.py",))
_RUNTIME_MODULE_MEMBER = "_hrx_windows_runtime.py"
_SITE_CUSTOMIZATION_CALL = b"    execsitecustomize()\n"
_RUNTIME_INITIALIZATION = (
    b"    import _hrx_windows_runtime as _hrx_runtime\n"
    b"    _hrx_runtime._configure_runtime(\n"
    b"        _hrx_runtime.__file__, sys, os.environ)\n" + _SITE_CUSTOMIZATION_CALL
)


def is_archive_member(relative_path: PurePath) -> bool:
    """Returns whether a Lib-relative path belongs in the runtime archive."""

    if "__pycache__" in relative_path.parts:
        return False
    if relative_path.parts and relative_path.parts[0] == "site-packages":
        return False
    if relative_path.as_posix() in _UNARCHIVED_MEMBERS:
        return False
    return relative_path.suffix not in (".pyc", ".pyo")


def archive_members(library_directory: Path) -> Iterator[tuple[Path, PurePath]]:
    """Yields stable source and archive paths for the standard library."""

    for source_path in sorted(library_directory.rglob("*")):
        relative_path = source_path.relative_to(library_directory)
        if source_path.is_file() and is_archive_member(relative_path):
            yield source_path, relative_path


def write_standard_library_archive(
    library_directory: Path, runtime_module_path: Path, output_path: Path
) -> None:
    """Writes a byte-stable, uncompressed CPython standard-library archive."""

    if not library_directory.is_dir():
        raise ValueError(f"Python standard library does not exist: {library_directory}")
    if not runtime_module_path.is_file():
        raise ValueError(
            f"Python runtime initializer does not exist: {runtime_module_path}"
        )
    members = list(archive_members(library_directory))
    member_names = {relative_path.as_posix() for _, relative_path in members}
    missing_members = set(_REQUIRED_MEMBERS) - member_names
    if missing_members:
        raise ValueError(
            f"Python standard library is missing required files: {sorted(missing_members)}"
        )
    if _RUNTIME_MODULE_MEMBER in member_names:
        raise ValueError(
            f"Python standard library already contains {_RUNTIME_MODULE_MEMBER}"
        )
    members.append((runtime_module_path, PurePath(_RUNTIME_MODULE_MEMBER)))
    members.sort(key=lambda member: member[1].as_posix())

    # Storing source bytes avoids host-zlib differences in repository content.
    with zipfile.ZipFile(
        output_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for source_path, relative_path in members:
            contents = source_path.read_bytes()
            if relative_path.as_posix() == "site.py":
                if contents.count(_SITE_CUSTOMIZATION_CALL) != 1:
                    raise ValueError(
                        "Python site bootstrap does not contain exactly one "
                        "application customization call"
                    )
                contents = contents.replace(
                    _SITE_CUSTOMIZATION_CALL, _RUNTIME_INITIALIZATION
                )
            archive_info = zipfile.ZipInfo(
                relative_path.as_posix(), date_time=_ARCHIVE_TIMESTAMP
            )
            archive_info.compress_type = zipfile.ZIP_STORED
            archive_info.create_system = 3
            archive_info.external_attr = 0o100644 << 16
            archive.writestr(archive_info, contents)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("library_directory", type=Path)
    parser.add_argument("runtime_module_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    write_standard_library_archive(
        args.library_directory, args.runtime_module_path, args.output_path
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
