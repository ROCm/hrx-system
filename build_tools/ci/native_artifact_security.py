#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualifies security properties of final native release artifacts."""

from __future__ import annotations

import argparse
import enum
import json
import os
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class NativeArtifactSecurityError(ValueError):
    """Raised when a native artifact violates the release security contract."""


class ElfRunpathPolicy(enum.Enum):
    """Expected dynamic-library lookup policy for an ELF release artifact."""

    BUNDLED_ORIGIN = "bundled-origin"
    ABSENT = "absent"


@dataclass(frozen=True)
class NativeArtifactSecurity:
    """Security properties established for one final native artifact."""

    format: str
    properties: frozenset[str]


def _property_names(record: dict[str, Any], key: str) -> set[str]:
    try:
        return {item["Name"] for item in record[key]["Flags"]}
    except (KeyError, TypeError) as error:
        raise NativeArtifactSecurityError(
            f"llvm-readobj returned invalid {key} flags"
        ) from error


def _parse_elf(
    document: Any,
    *,
    executable: bool,
    runpath_policy: ElfRunpathPolicy,
) -> NativeArtifactSecurity:
    """Parses the JSON form emitted by llvm-readobj for one ELF file."""
    if not isinstance(document, list) or len(document) != 1:
        raise NativeArtifactSecurityError(
            "llvm-readobj returned an unexpected ELF file list"
        )
    file_record = document[0]
    try:
        header = file_record["ElfHeader"]
        program_headers = [
            item["ProgramHeader"] for item in file_record["ProgramHeaders"]
        ]
        dynamic = file_record["DynamicSection"]
    except (KeyError, TypeError) as error:
        raise NativeArtifactSecurityError(
            "llvm-readobj returned an invalid ELF document"
        ) from error

    properties = set()
    if str(header.get("Type", "")).startswith("SharedObject"):
        properties.add("position-independent")

    stacks = []
    has_relro = False
    has_writable_executable_load = False
    for program_header in program_headers:
        try:
            type_name = program_header["Type"]["Name"]
            flags = _property_names(program_header, "Flags")
        except (KeyError, TypeError) as error:
            raise NativeArtifactSecurityError(
                "llvm-readobj returned an invalid ELF program header"
            ) from error
        if type_name == "PT_GNU_STACK":
            stacks.append(flags)
        elif type_name == "PT_GNU_RELRO":
            has_relro = True
        elif type_name == "PT_LOAD" and {"PF_W", "PF_X"} <= flags:
            has_writable_executable_load = True
    if len(stacks) == 1 and "PF_X" not in stacks[0]:
        properties.add("non-executable-stack")
    if has_relro:
        properties.add("relro")
    if not has_writable_executable_load:
        properties.add("writable-xor-executable")

    bind_now = False
    pie = False
    runpaths = []
    has_rpath = False
    for entry in dynamic:
        if not isinstance(entry, dict):
            raise NativeArtifactSecurityError(
                "llvm-readobj returned an invalid ELF dynamic entry"
            )
        entry_type = entry.get("Type")
        flags = entry.get("Flags", [])
        if entry_type in {"FLAGS", "FLAGS_1"}:
            if not isinstance(flags, list):
                raise NativeArtifactSecurityError(
                    "llvm-readobj returned invalid ELF dynamic flags"
                )
            bind_now |= "BIND_NOW" in flags or "NOW" in flags
            pie |= entry_type == "FLAGS_1" and "PIE" in flags
        elif entry_type == "RUNPATH":
            paths = entry.get("Path")
            if not isinstance(paths, list) or not all(
                isinstance(path, str) for path in paths
            ):
                raise NativeArtifactSecurityError(
                    "llvm-readobj returned an invalid ELF runpath"
                )
            runpaths.append(paths)
        elif entry_type == "RPATH":
            has_rpath = True
    if bind_now:
        properties.add("eager-binding")
    if (
        runpath_policy is ElfRunpathPolicy.BUNDLED_ORIGIN
        and not has_rpath
        and runpaths == [["$ORIGIN"]]
    ):
        properties.add("origin-runpath")
    elif runpath_policy is ElfRunpathPolicy.ABSENT and not has_rpath and not runpaths:
        properties.add("no-runpath")
    if executable and pie:
        properties.add("pie")

    required = {
        "position-independent",
        "non-executable-stack",
        "relro",
        "eager-binding",
        "writable-xor-executable",
    }
    required.add(
        "origin-runpath"
        if runpath_policy is ElfRunpathPolicy.BUNDLED_ORIGIN
        else "no-runpath"
    )
    if executable:
        required.add("pie")
    missing = sorted(required - properties)
    if missing:
        raise NativeArtifactSecurityError("ELF artifact lacks " + ", ".join(missing))
    return NativeArtifactSecurity("ELF", frozenset(properties))


def _parse_integer_field(output: str, name: str) -> int:
    match = re.search(
        rf"^\s*{re.escape(name)}:\s*(0x[0-9a-fA-F]+|[0-9]+)\s*$",
        output,
        flags=re.MULTILINE,
    )
    if match is None:
        raise NativeArtifactSecurityError(f"llvm-readobj returned no {name} field")
    return int(match.group(1), 0)


def _parse_pe(output: str) -> NativeArtifactSecurity:
    """Parses the stable text form emitted by llvm-readobj for one PE file."""
    if "Format: COFF-x86-64" not in output or "ImageOptionalHeader {" not in output:
        raise NativeArtifactSecurityError(
            "llvm-readobj returned an invalid x86-64 PE document"
        )
    properties = set()
    characteristics = {
        "IMAGE_DLL_CHARACTERISTICS_DYNAMIC_BASE": "aslr",
        "IMAGE_DLL_CHARACTERISTICS_HIGH_ENTROPY_VA": "high-entropy-aslr",
        "IMAGE_DLL_CHARACTERISTICS_NX_COMPAT": "data-execution-prevention",
    }
    for token, property_name in characteristics.items():
        if token in output:
            properties.add(property_name)
    if _parse_integer_field(output, "SecurityCookie") != 0:
        properties.add("stack-cookie")

    sections = re.findall(
        r"^  Section \{\n(.*?)^  \}", output, flags=re.DOTALL | re.MULTILINE
    )
    if not sections:
        raise NativeArtifactSecurityError("llvm-readobj returned no PE sections")
    writable_executable = any(
        "IMAGE_SCN_MEM_WRITE" in section and "IMAGE_SCN_MEM_EXECUTE" in section
        for section in sections
    )
    if not writable_executable:
        properties.add("writable-xor-executable")

    required = {
        "aslr",
        "high-entropy-aslr",
        "data-execution-prevention",
        "stack-cookie",
        "writable-xor-executable",
    }
    missing = sorted(required - properties)
    if missing:
        raise NativeArtifactSecurityError("PE artifact lacks " + ", ".join(missing))
    return NativeArtifactSecurity("PE", frozenset(properties))


def inspect_file(
    llvm_readobj: Path,
    path: Path,
    *,
    executable: bool,
    runpath_policy: ElfRunpathPolicy,
) -> NativeArtifactSecurity:
    """Inspects one final ELF or PE artifact with llvm-readobj."""
    with path.open("rb") as stream:
        signature = stream.read(4)
    if signature == b"\x7fELF":
        result = subprocess.run(
            [
                os.fspath(llvm_readobj),
                "--elf-output-style=JSON",
                "--file-headers",
                "--program-headers",
                "--dynamic-table",
                os.fspath(path),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        try:
            document = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise NativeArtifactSecurityError(
                "llvm-readobj returned invalid ELF JSON"
            ) from error
        return _parse_elf(
            document,
            executable=executable,
            runpath_policy=runpath_policy,
        )
    if signature[:2] == b"MZ":
        result = subprocess.run(
            [
                os.fspath(llvm_readobj),
                "--file-headers",
                "--coff-load-config",
                "--sections",
                os.fspath(path),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        return _parse_pe(result.stdout)
    raise NativeArtifactSecurityError(f"Unsupported native artifact: {path}")


def qualify_wheel(
    llvm_readobj: Path, wheel: Path
) -> list[tuple[str, NativeArtifactSecurity]]:
    """Qualifies every native executable and shared library in one wheel."""
    reports = []
    with zipfile.ZipFile(wheel) as archive, tempfile.TemporaryDirectory() as temporary:
        for ordinal, member in enumerate(
            sorted(archive.infolist(), key=lambda item: item.filename)
        ):
            if member.is_dir():
                continue
            name = Path(member.filename).name
            lower_name = name.lower()
            contents = archive.read(member)
            if contents.startswith(b"\x7fELF") or contents.startswith(b"MZ"):
                path = Path(temporary) / f"{ordinal}.artifact"
                path.write_bytes(contents)
                library = lower_name.endswith((".so", ".dll")) or ".so." in lower_name
                try:
                    report = inspect_file(
                        llvm_readobj,
                        path,
                        executable=not library,
                        runpath_policy=ElfRunpathPolicy.BUNDLED_ORIGIN,
                    )
                except (
                    NativeArtifactSecurityError,
                    OSError,
                    subprocess.CalledProcessError,
                ) as error:
                    raise NativeArtifactSecurityError(
                        f"{wheel.name}:{member.filename}: {error}"
                    ) from error
                reports.append((member.filename, report))
            elif contents.startswith(
                b"Microsoft C/C++ MSF 7.00"
            ) or contents.startswith(b"#!"):
                continue
            elif ".data/scripts/" in member.filename:
                raise NativeArtifactSecurityError(
                    f"{wheel.name}:{member.filename}: unsupported native payload"
                )
    if not reports:
        raise NativeArtifactSecurityError(f"SDK wheel has no native artifacts: {wheel}")
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm-readobj", required=True, type=Path)
    parser.add_argument("--wheel", action="append", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        for wheel in args.wheel:
            for name, report in qualify_wheel(args.llvm_readobj, wheel):
                properties = ",".join(sorted(report.properties))
                print(f"security[{wheel.name}:{name}]={report.format}:{properties}")
    except (
        NativeArtifactSecurityError,
        OSError,
        subprocess.CalledProcessError,
        zipfile.BadZipFile,
    ) as error:
        parser.exit(1, f"Native artifact security qualification failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
