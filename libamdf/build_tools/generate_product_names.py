#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiles libdrm's amdgpu.ids into immutable PCI product-name lookup data."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_database(text: str) -> tuple[dict[int, str], dict[int, str]]:
    devices: dict[int, str] = {}
    subsystems: dict[int, str] = {}
    version_seen = False
    parent = None
    first_parent = False
    for number, line in enumerate(text.splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        if not version_seen:
            if line != "1.0.0":
                raise ValueError(f"line {number}: unsupported database version")
            version_seen = True
            continue
        child = line.startswith((" ", "\t"))
        if child and (not line.strip() or line.lstrip().startswith("#")):
            continue
        fields = line.split(",", 2)
        if len(fields) != 3:
            raise ValueError(f"line {number}: expected two IDs and a name")
        if any(not re.fullmatch(r"[ \t]*[0-9a-fA-F]+", x) for x in fields[:2]):
            raise ValueError(f"line {number}: invalid hexadecimal identifier")
        first, second = (int(field, 16) for field in fields[:2])
        if first > 0xFFFF or second > (0xFFFF if child else 0xFF):
            raise ValueError(f"line {number}: identifier exceeds its PCI width")
        name = fields[2].lstrip(" \t")
        if not name or any(
            ord(character) < 32 or ord(character) == 127 for character in name
        ):
            raise ValueError(f"line {number}: empty name or control character")
        if child:
            if parent is None:
                raise ValueError(f"line {number}: subsystem without a device")
            if first_parent:
                subsystems.setdefault((parent << 32) | (first << 16) | second, name)
        else:
            parent = (first << 8) | second
            first_parent = parent not in devices
            devices.setdefault(parent, name)
    if not devices:
        raise ValueError("database contains no device names")
    return devices, subsystems


def c_string(value: bytes) -> str:
    # Fixed-width octal escapes preserve UTF-8 and cannot consume a following
    # digit. Ordinary printable ASCII keeps the generated input readable.
    return (
        '"'
        + "".join(
            chr(byte)
            if 32 <= byte < 127 and byte not in (34, 63, 92)
            else f"\\{byte:03o}"
            for byte in value
        )
        + '"'
    )


def render_tables(devices: dict[int, str], subsystems: dict[int, str]) -> str:
    names = sorted(set(devices.values()) | set(subsystems.values()))
    offsets = {}
    strings = []
    size = 0
    for name in names:
        offsets[name] = size
        encoded = name.encode("utf-8") + b"\0"
        strings.append(encoded)
        size += len(encoded)
    offset_type = "uint16_t" if max(offsets.values()) <= 0xFFFF else "uint32_t"

    lines = [
        f"#define AMDF_PCI_PRODUCT_DEVICE_COUNT {len(devices)}",
        f"#define AMDF_PCI_PRODUCT_SUBSYSTEM_COUNT {len(subsystems)}",
        f"#define AMDF_PCI_PRODUCT_MAX_NAME_LENGTH {max(len(x) - 1 for x in strings)}",
        "",
        f"static const char amdf_pci_product_names[{size}] =",
        *("    " + c_string(value) for value in strings),
        "    ;",
        "",
    ]
    for suffix, records, key_type, key_format in (
        ("device", devices, "uint32_t", "0x{key:06X}u"),
        ("subsystem", subsystems, "uint64_t", "UINT64_C(0x{key:014X})"),
    ):
        if not records:
            continue
        keys = sorted(records)
        for label, c_type, values in (
            ("keys", key_type, [key_format.format(key=key) for key in keys]),
            ("offsets", offset_type, [str(offsets[records[key]]) for key in keys]),
        ):
            lines.append(
                f"static const {c_type} amdf_pci_product_{suffix}_{label}[] = {{"
            )
            for start in range(0, len(values), 6):
                lines.append("    " + ", ".join(values[start : start + 6]) + ",")
            lines.extend(["};", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        devices, subsystems = parse_database(args.input.read_text(encoding="utf-8"))
        tables = render_tables(devices, subsystems)
    except ValueError as error:
        parser.error(f"{args.input}: {error}")
    template = args.template.read_text(encoding="utf-8")
    assert template.count("@TABLES@") == 1
    license_text = args.license.read_text(encoding="utf-8")
    notice = "// Product-name data derived from libdrm (MIT license):\n"
    notice += "\n".join("// " + line for line in license_text.splitlines()) + "\n\n"
    output = "// Generated from amdgpu.ids; edit the generator or its template.\n"
    output += notice + template.replace("@TABLES@", tables)
    if not args.output.exists() or args.output.read_text(encoding="utf-8") != output:
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
