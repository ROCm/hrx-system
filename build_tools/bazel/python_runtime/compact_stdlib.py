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
_SITE_INITIALIZATION_MARKER = b"\n_fixup_stdlib_paths()\n"
_COMPACT_RUNTIME_INITIALIZATION = _SITE_INITIALIZATION_MARKER + (
    b"\n"
    b"def _configure_compact_windows_runtime():\n"
    b"    runtime_root = os.path.dirname(\n"
    b"        os.path.join(_RUNFILES_ROOT, _INTERPRETER_ACTUAL_PATH)\n"
    b"    )\n"
    b'    os.environ["TCL_LIBRARY"] = os.path.join(\n'
    b'        runtime_root, "tcl", "tcl8.6"\n'
    b"    )\n"
    b'    os.environ["TK_LIBRARY"] = os.path.join(\n'
    b'        runtime_root, "tcl", "tk8.6"\n'
    b"    )\n"
    b"\n"
    b"\n"
    b"_configure_compact_windows_runtime()\n"
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


def write_standard_library_archive(library_directory: Path, output_path: Path) -> None:
    """Writes a byte-stable, uncompressed CPython standard-library archive."""

    if not library_directory.is_dir():
        raise ValueError(f"Python standard library does not exist: {library_directory}")
    members = list(archive_members(library_directory))
    member_names = {relative_path.as_posix() for _, relative_path in members}
    missing_members = set(_REQUIRED_MEMBERS) - member_names
    if missing_members:
        raise ValueError(
            f"Python standard library is missing required files: {sorted(missing_members)}"
        )
    # Storing source bytes avoids host-zlib differences in repository content.
    with zipfile.ZipFile(
        output_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for source_path, relative_path in members:
            contents = source_path.read_bytes()
            archive_info = zipfile.ZipInfo(
                relative_path.as_posix(), date_time=_ARCHIVE_TIMESTAMP
            )
            archive_info.compress_type = zipfile.ZIP_STORED
            archive_info.create_system = 3
            archive_info.external_attr = 0o100644 << 16
            archive.writestr(archive_info, contents)


def write_site_initialization_template(
    upstream_template_path: Path, output_path: Path
) -> None:
    """Adds compact-runtime environment setup to rules_python startup."""

    contents = upstream_template_path.read_bytes()
    if contents.count(_SITE_INITIALIZATION_MARKER) != 1:
        raise ValueError(
            "rules_python site initialization template does not contain exactly "
            "one standard-library path fixup"
        )
    output_path.write_bytes(
        contents.replace(_SITE_INITIALIZATION_MARKER, _COMPACT_RUNTIME_INITIALIZATION)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("library_directory", type=Path)
    parser.add_argument("upstream_site_template", type=Path)
    parser.add_argument("output_archive", type=Path)
    parser.add_argument("output_site_template", type=Path)
    args = parser.parse_args()
    write_standard_library_archive(args.library_directory, args.output_archive)
    write_site_initialization_template(
        args.upstream_site_template, args.output_site_template
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
