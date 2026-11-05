#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Embeds physical variants of a self-contained AMDHSA V6 kernel.

The input descriptor and AMDGPU metadata own all resource and argument facts.
The generated image preserves read-only data and text addresses relative to an
aligned upload base. Queue admission, argument values and final-use ownership
belong to the typed caller; this cold tool neither compiles nor executes the
kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path


class ImageError(ValueError):
    """The input is outside the self-contained kernel image contract."""


def require(condition, message):
    if not condition:
        raise ImageError(message)


def extent(data, offset, size, description):
    require(
        0 <= offset <= len(data) and 0 <= size <= len(data) - offset,
        f"out-of-bounds {description}",
    )
    return data[offset : offset + size]


def string_at(data, offset, description):
    require(0 <= offset < len(data), f"out-of-bounds {description}")
    end = data.find(b"\0", offset)
    require(end >= 0, f"unterminated {description}")
    try:
        return data[offset:end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ImageError(f"invalid UTF-8 {description}") from error


def unsigned(value, description, maximum=0xFFFFFFFF):
    require(
        type(value) is int and 0 <= value <= maximum,
        f"invalid unsigned {description}",
    )
    return value


def power_of_two(value):
    return value > 0 and value & (value - 1) == 0


class MessagePack:
    """Decodes the integer/string/container subset used by AMDGPU metadata."""

    def __init__(self, data):
        self.data = data
        self.position = 0

    def take(self, size):
        result = extent(self.data, self.position, size, "MessagePack value")
        self.position += size
        return result

    def integer(self, size, signed=False):
        return int.from_bytes(self.take(size), "big", signed=signed)

    def text(self, size):
        try:
            return self.take(size).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ImageError("invalid UTF-8 metadata string") from error

    def array(self, count, depth):
        require(count <= len(self.data) - self.position, "truncated metadata array")
        return [self.value(depth + 1) for _ in range(count)]

    def mapping(self, count, depth):
        require(
            count <= (len(self.data) - self.position) // 2,
            "truncated metadata map",
        )
        result = {}
        for _ in range(count):
            key = self.value(depth + 1)
            require(type(key) is str, "metadata map key is not a string")
            require(key not in result, f"duplicate metadata key: {key}")
            result[key] = self.value(depth + 1)
        return result

    def value(self, depth=0):
        require(depth <= 32, "metadata nesting exceeds the supported format")
        tag = self.integer(1)
        if tag < 0x80:
            return tag
        if tag >= 0xE0:
            return tag - 256
        if 0xA0 <= tag <= 0xBF:
            return self.text(tag & 0x1F)
        if 0x90 <= tag <= 0x9F:
            return self.array(tag & 0xF, depth)
        if 0x80 <= tag <= 0x8F:
            return self.mapping(tag & 0xF, depth)
        if tag == 0xC0:
            return None
        if tag in (0xC2, 0xC3):
            return tag == 0xC3
        if 0xCC <= tag <= 0xCF:
            return self.integer(1 << (tag - 0xCC))
        if 0xD0 <= tag <= 0xD3:
            return self.integer(1 << (tag - 0xD0), signed=True)
        if 0xD9 <= tag <= 0xDB:
            return self.text(self.integer(1 << (tag - 0xD9)))
        if tag in (0xDC, 0xDD):
            return self.array(self.integer(2 if tag == 0xDC else 4), depth)
        if tag in (0xDE, 0xDF):
            return self.mapping(self.integer(2 if tag == 0xDE else 4), depth)
        raise ImageError(f"unsupported metadata MessagePack tag 0x{tag:02x}")

    def decode(self):
        result = self.value()
        require(self.position == len(self.data), "trailing metadata bytes")
        return result


@dataclass(frozen=True)
class Section:
    # Name from the ELF section-name string table.
    name: str
    # ELF SHT_* section kind.
    type: int
    # ELF SHF_* section attributes.
    flags: int
    # Linked virtual byte address.
    address: int
    # Byte offset in the complete HSACO file.
    offset: int
    # Complete section byte extent.
    size: int
    # Associated section index, used for symbol string tables.
    link: int
    # Required byte alignment, or zero if unspecified.
    alignment: int
    # Fixed record byte size, or zero for unstructured sections.
    entry_size: int


class Elf:
    """Reads the bounded ELF64 layout emitted for the kernel product."""

    def __init__(self, data):
        self.data = data
        header = struct.unpack("<16sHHIQQQIHHHHHH", extent(data, 0, 64, "ELF header"))
        identity = header[0]
        require(identity[:7] == b"\x7fELF\x02\x01\x01", "expected ELF64 little-endian")
        require(identity[7:9] == bytes((64, 4)), "expected AMDHSA code object V6")
        require(header[1:4] == (3, 224, 1), "expected AMDGPU ET_DYN ELF version 1")
        require(header[4] == 0 and header[8] == 64, "unsupported ELF entry/header")
        require(header[9] == 56 and header[11] == 64, "unsupported ELF table layout")
        require(header[12] > 0, "extended or absent ELF section table")
        self.flags = header[7]
        program_data = extent(data, header[5], header[9] * header[10], "program table")
        self.programs = list(struct.iter_unpack("<IIQQQQQQ", program_data))
        section_data = extent(data, header[6], header[11] * header[12], "section table")
        raw_sections = list(struct.iter_unpack("<IIQQQQIIQQ", section_data))
        require(0 < header[13] < len(raw_sections), "invalid section name table")
        names_section = raw_sections[header[13]]
        require(names_section[1] == 3, "invalid section name string table")
        names = extent(data, names_section[4], names_section[5], "section names")
        self.sections = []
        self.by_name = {}
        for index, raw in enumerate(raw_sections):
            name = string_at(names, raw[0], "section name")
            section = Section(name, *raw[1:7], raw[8], raw[9])
            if index == 0:
                require(raw == (0,) * 10, "unsupported ELF null section")
            else:
                require(
                    name and name not in self.by_name, "duplicate/empty ELF section"
                )
                require(
                    section.alignment in (0, 1) or power_of_two(section.alignment),
                    f"invalid section alignment: {name}",
                )
                if section.type != 8:
                    self.contents(section)
                require(
                    section.type not in (4, 9, 19) or section.size == 0,
                    f"unsupported relocation section: {name}",
                )
                self.by_name[name] = section
            self.sections.append(section)
        for program in self.programs:
            kind, _, offset, address, _, file_size, memory_size, alignment = program
            if kind != 1:
                continue
            extent(data, offset, file_size, "load segment")
            require(file_size <= memory_size, "load file extent exceeds memory extent")
            require(address + memory_size < 1 << 64, "overflowing load address")
            require(
                alignment in (0, 1)
                or (
                    power_of_two(alignment)
                    and offset % alignment == address % alignment
                ),
                "invalid load alignment",
            )

    def contents(self, section):
        return extent(self.data, section.offset, section.size, section.name)

    def section(self, name, kind):
        require(name in self.by_name, f"missing section: {name}")
        result = self.by_name[name]
        require(result.type == kind, f"unexpected section type: {name}")
        return result

    def require_loaded(self, section, flags):
        require(section.flags == flags, f"unexpected section flags: {section.name}")
        permissions = 5 if flags & 4 else 4
        for program in self.programs:
            kind, mode, offset, address, _, file_size, _, _ = program
            delta = section.address - address
            if (
                kind == 1
                and mode == permissions
                and 0 <= delta <= file_size
                and section.size <= file_size - delta
                and section.offset == offset + delta
            ):
                return
        raise ImageError(
            f"section is not in its read/execute load segment: {section.name}"
        )

    def symbols(self, section):
        require(
            section.entry_size == 24 and section.size % 24 == 0, "invalid symbol table"
        )
        require(
            0 < section.link < len(self.sections), "invalid symbol string table link"
        )
        strings = self.sections[section.link]
        require(strings.type == 3, "invalid symbol string table")
        names = self.contents(strings)
        result = {}
        for symbol in struct.iter_unpack("<IBBHQQ", self.contents(section)):
            name = string_at(names, symbol[0], "symbol name")
            if name:
                require(symbol[3] != 0, f"undefined symbol: {name}")
                require(name not in result, f"duplicate symbol: {name}")
                result[name] = symbol
        return result


def read_metadata(elf):
    note = elf.section(".note", 7)
    data = elf.contents(note)
    position = 0
    metadata = None
    while position < len(data):
        name_size, value_size, kind = struct.unpack(
            "<III", extent(data, position, 12, "note header")
        )
        position += 12
        name = extent(data, position, name_size, "note name")
        position += (name_size + 3) & ~3
        payload = extent(data, position, value_size, "note payload")
        position += (value_size + 3) & ~3
        require(position <= len(data), "truncated note padding")
        if name == b"AMDGPU\0" and kind == 32:
            require(metadata is None, "multiple AMDGPU metadata notes")
            metadata = MessagePack(payload).decode()
    require(type(metadata) is dict, "missing AMDGPU metadata map")
    return metadata


def kernel_metadata(metadata, symbol, descriptor):
    require(
        metadata.get("amdhsa.version") == [1, 2], "unsupported AMDGPU metadata version"
    )
    target = metadata.get("amdhsa.target")
    require(
        type(target) is str and target.startswith("amdgcn-amd-amdhsa--gfx"),
        "missing AMDHSA target",
    )
    kernels = metadata.get("amdhsa.kernels")
    require(type(kernels) is list and len(kernels) == 1, "expected one metadata kernel")
    kernel = kernels[0]
    require(type(kernel) is dict, "invalid metadata kernel")
    require(
        kernel.get(".name") == symbol and kernel.get(".symbol") == symbol + ".kd",
        "kernel metadata/symbol mismatch",
    )
    for offset, name in (
        (0, ".group_segment_fixed_size"),
        (4, ".private_segment_fixed_size"),
        (8, ".kernarg_segment_size"),
    ):
        value = unsigned(kernel.get(name), name)
        require(
            value == struct.unpack_from("<I", descriptor, offset)[0],
            f"descriptor/metadata {name} mismatch",
        )
    alignment = unsigned(kernel.get(".kernarg_segment_align"), "kernarg alignment")
    require(power_of_two(alignment), "invalid kernarg alignment")
    maximum = unsigned(kernel.get(".max_flat_workgroup_size"), "maximum workgroup size")
    require(maximum > 0, "zero maximum workgroup size")
    if ".reqd_workgroup_size" in kernel:
        required = kernel[".reqd_workgroup_size"]
        require(
            type(required) is list and len(required) == 3,
            "invalid required XYZ geometry",
        )
        for value in required:
            require(
                unsigned(value, "workgroup dimension", 0xFFFF) > 0,
                "zero required workgroup dimension",
            )
        require(math.prod(required) <= maximum, "required workgroup exceeds maximum")
    wave = unsigned(kernel.get(".wavefront_size"), "wavefront size")
    properties = struct.unpack_from("<H", descriptor, 56)[0]
    require(wave in (32, 64), "unsupported wavefront size")
    require(
        bool(properties & (1 << 10)) == (wave == 32),
        "descriptor/metadata wavefront mismatch",
    )
    require(
        not properties & (1 << 11),
        "dynamic private stack requires a separate caller contract",
    )
    require(
        not kernel.get(".uses_dynamic_stack", False), "dynamic private stack metadata"
    )
    require(
        ".workgroup_cluster_size" not in kernel,
        "workgroup clusters require a separate caller contract",
    )
    for name in (".sgpr_count", ".vgpr_count"):
        unsigned(kernel.get(name), name)
    arguments = kernel.get(".args")
    require(type(arguments) is list, "missing kernel argument array")
    occupied = []
    for argument in arguments:
        require(type(argument) is dict, "invalid kernel argument")
        offset = unsigned(argument.get(".offset"), "argument offset")
        size = unsigned(argument.get(".size"), "argument size")
        align = unsigned(argument.get(".align"), "argument alignment")
        require(size > 0 and power_of_two(align), "invalid argument size/alignment")
        require(
            align <= alignment and offset % align == 0,
            "argument alignment exceeds kernarg layout",
        )
        require(
            offset + size <= kernel[".kernarg_segment_size"],
            "argument exceeds kernarg extent",
        )
        require(
            all(offset + size <= start or offset >= end for start, end in occupied),
            "overlapping kernel arguments",
        )
        occupied.append((offset, offset + size))
        kind = argument.get(".value_kind")
        require(
            kind in ("global_buffer", "by_value", "dynamic_shared_pointer"),
            f"unsupported argument kind: {kind}",
        )
        if kind == "global_buffer":
            require(
                size == 8 and argument.get(".address_space") == "global",
                "unsupported global buffer argument",
            )
        require(type(argument.get(".name", "")) is str, "invalid argument name")
    return target, kernel


@dataclass(frozen=True)
class KernelImage:
    # Complete read-only data and text with their linked placement intact.
    image: bytes
    # Exact compiler-emitted 64-byte descriptor.
    descriptor: bytes
    # Descriptor byte position within the aligned image.
    descriptor_offset: int
    # Complete .rodata storage extent, beginning at descriptor_offset.
    rodata_size: int
    # Kernel entry byte position within the aligned image.
    entry_offset: int
    # Function-symbol byte extent, independent of complete text storage.
    entry_size: int
    # Complete .text storage extent in bytes.
    text_size: int
    # Native ELF target flags copied from the file header.
    elf_flags: int
    # Digest of the complete input HSACO, including loader metadata.
    hsaco_sha256: str
    # Native AMDHSA target name from the metadata note.
    target: str
    # Validated compiler-owned argument and launch resource facts.
    metadata: dict


def extract_image(data, symbol):
    elf = Elf(data)
    allowed_allocated = {
        ".note",
        ".dynsym",
        ".hash",
        ".dynstr",
        ".rodata",
        ".text",
        ".dynamic",
    }
    for section in elf.sections:
        require(
            not (
                section.flags & 2
                and section.size
                and section.name not in allowed_allocated
            ),
            f"unsupported allocated section: {section.name}",
        )
        if section.type in (2, 11):
            elf.symbols(section)
    dynamic = elf.section(".dynamic", 6)
    require(dynamic.size % 16 == 0, "invalid dynamic table")
    terminated = False
    for tag, value in struct.iter_unpack("<qQ", elf.contents(dynamic)):
        require(
            tag in (0, 4, 5, 6, 10, 11), f"unsupported dynamic dependency/tag: {tag}"
        )
        require(
            not terminated or (tag == 0 and value == 0),
            "dynamic entries after terminator",
        )
        terminated = terminated or tag == 0
    require(terminated, "unterminated dynamic table")
    symbols = elf.symbols(elf.section(".dynsym", 11))
    require(symbol in symbols, "missing kernel symbol")
    require(symbol + ".kd" in symbols, "missing kernel descriptor symbol")
    text = elf.section(".text", 1)
    rodata = elf.section(".rodata", 1)
    elf.require_loaded(rodata, 2)
    elf.require_loaded(text, 6)
    rodata_index = elf.sections.index(rodata)
    entry = symbols[symbol]
    descriptor_symbol = symbols[symbol + ".kd"]
    require(
        descriptor_symbol[1] == 0x11
        and descriptor_symbol[3] == rodata_index
        and descriptor_symbol[4] == rodata.address
        and descriptor_symbol[5] == 64
        and rodata.size >= 64,
        "rodata must begin with the complete kernel descriptor",
    )
    occupied_rodata = [(0, 64, symbol + ".kd")]
    for name, object_symbol in symbols.items():
        if name in (symbol, symbol + ".kd"):
            continue
        require(
            object_symbol[1] == 0x11,
            f"extra symbol is not a global object: {name}",
        )
        require(
            object_symbol[3] == rodata_index,
            f"extra object is outside rodata: {name}",
        )
        object_offset = object_symbol[4] - rodata.address
        object_size = object_symbol[5]
        require(
            0 <= object_offset <= rodata.size
            and object_size > 0
            and object_size <= rodata.size - object_offset,
            f"extra object exceeds rodata: {name}",
        )
        require(
            all(
                object_offset + object_size <= start or object_offset >= end
                for start, end, _ in occupied_rodata
            ),
            f"overlapping rodata object: {name}",
        )
        occupied_rodata.append((object_offset, object_offset + object_size, name))
    require(
        entry[1] == 0x12
        and entry[3] == elf.sections.index(text)
        and entry[4] == text.address
        and 0 < entry[5] <= text.size
        and entry[5] % 4 == 0
        and text.size % 4 == 0,
        "text must begin with the complete kernel entry",
    )
    require(
        rodata.address % 64 == 0 and text.address % 256 == 0,
        "descriptor/entry alignment",
    )
    rodata_contents = elf.contents(rodata)
    descriptor = rodata_contents[:64]
    require(
        descriptor[12:16] == bytes(4)
        and descriptor[24:44] == bytes(20)
        and descriptor[60:64] == bytes(4),
        "unsupported descriptor reserved fields",
    )
    delta = struct.unpack_from("<q", descriptor, 16)[0]
    require(
        delta >= rodata.size and rodata.address + delta == text.address,
        "descriptor entry offset mismatch",
    )
    target, metadata = kernel_metadata(read_metadata(elf), symbol, descriptor)
    phase = rodata.address % 256
    image_size = phase + delta + text.size
    require(image_size <= len(data), "unsupported sparse rodata/text image")
    require(image_size <= 0xFFFFFFFF, "image exceeds the caller's 32-bit extent")
    image = bytearray(image_size)
    image[phase : phase + rodata.size] = rodata_contents
    image[phase + delta :] = elf.contents(text)
    return KernelImage(
        image=bytes(image),
        descriptor=descriptor,
        descriptor_offset=phase,
        rodata_size=rodata.size,
        entry_offset=phase + delta,
        entry_size=entry[5],
        text_size=text.size,
        elf_flags=elf.flags,
        hsaco_sha256=hashlib.sha256(data).hexdigest(),
        target=target,
        metadata=metadata,
    )


def render_variant(kernel, namespace):
    """Stores one image and its metadata in a private implementation namespace."""
    metadata = kernel.metadata
    required = metadata.get(".reqd_workgroup_size", [0, 0, 0])
    arguments = metadata[".args"]
    lines = [
        f"namespace {namespace} {{",
        "",
        f'inline constexpr char kHsacoSha256[] = "{kernel.hsaco_sha256}";',
        f'inline constexpr char kImageSha256[] = "{hashlib.sha256(kernel.image).hexdigest()}";',
    ]
    constants = {
        "kDescriptorByteOffset": kernel.descriptor_offset,
        "kRodataByteLength": kernel.rodata_size,
        "kEntryByteOffset": kernel.entry_offset,
        "kEntryByteLength": kernel.entry_size,
        "kTextByteLength": kernel.text_size,
        "kKernargByteLength": metadata[".kernarg_segment_size"],
        "kKernargAlignment": metadata[".kernarg_segment_align"],
        "kGroupSegmentByteLength": metadata[".group_segment_fixed_size"],
        "kPrivateSegmentByteLength": metadata[".private_segment_fixed_size"],
        "kMaxFlatWorkgroupSize": metadata[".max_flat_workgroup_size"],
        "kWavefrontSize": metadata[".wavefront_size"],
        "kSgprCount": metadata[".sgpr_count"],
        "kVgprCount": metadata[".vgpr_count"],
        "kComputePgmRsrc1": struct.unpack_from("<I", kernel.descriptor, 48)[0],
        "kComputePgmRsrc2": struct.unpack_from("<I", kernel.descriptor, 52)[0],
        "kComputePgmRsrc3": struct.unpack_from("<I", kernel.descriptor, 44)[0],
        "kKernelCodeProperties": struct.unpack_from("<H", kernel.descriptor, 56)[0],
        "kKernargPreload": struct.unpack_from("<H", kernel.descriptor, 58)[0],
    }
    for name, value in constants.items():
        lines.append(f"inline constexpr uint32_t {name} = {value}u;")

    def array(name, ctype, values):
        lines.append(f"inline constexpr std::array<{ctype}, {len(values)}> {name} = {{")
        lines.extend(
            "    " + ", ".join(values[i : i + 8]) + ","
            for i in range(0, len(values), 8)
        )
        lines.append("};")

    array("kRequiredWorkgroupSize", "uint32_t", [str(v) + "u" for v in required])
    for name, key in (
        ("kArgumentByteOffsets", ".offset"),
        ("kArgumentByteLengths", ".size"),
        ("kArgumentAlignments", ".align"),
    ):
        array(name, "uint32_t", [str(argument[key]) + "u" for argument in arguments])
    for name, key in (
        ("kArgumentNames", ".name"),
        ("kArgumentValueKinds", ".value_kind"),
    ):
        array(
            name,
            "std::string_view",
            [json.dumps(argument.get(key, "")) for argument in arguments],
        )
    words = struct.unpack(f"<{len(kernel.image) // 4}I", kernel.image)
    lines.extend(
        [
            "",
            f"alignas(256) inline constexpr std::array<uint32_t, {len(words)}> kImage = {{",
        ]
    )
    for index in range(0, len(words), 5):
        lines.append(
            "    "
            + ", ".join(f"0x{word:08x}u" for word in words[index : index + 5])
            + ","
        )
    lines.extend(
        [
            "};",
            "",
            "inline constexpr ::kernels::Image kExecutable = {",
            "    kImage.data(),",
            "    sizeof(kImage),",
            "    kDescriptorByteOffset,",
            "    kRodataByteLength,",
            "    kImageSha256,",
            "};",
            "",
            f"}}  // namespace {namespace}",
            "",
        ]
    )
    return "\n".join(lines)


def render_set(variants, namespace, header_name):
    require(
        re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*(?:::[A-Za-z_][A-Za-z_0-9]*)*", namespace),
        "invalid C++ namespace",
    )
    guard = "AMDF_CTS_GPU_KERNELS_" + namespace.replace("::", "_").upper() + "_SET_H_"
    preamble = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        "// Generated from Loom compiler products by embed.py.",
        "",
    ]
    header = preamble + [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        '#include "libamdf/cts/gpu/kernels/kernel.h"',
        "",
        f"namespace {namespace} {{",
        "extern const ::kernels::KernelSet kKernels;",
        f"}}  // namespace {namespace}",
        "",
        f"#endif  // {guard}",
        "",
    ]
    source = preamble + [
        f'#include "{header_name}"',
        "",
        "#include <array>",
        "#include <cstdint>",
        "#include <string_view>",
        "",
        "namespace {",
        "",
    ]
    for selector, kernel in variants:
        require(
            re.fullmatch(r"gfx[0-9a-f]+(?:-strict)?", selector),
            "invalid target selector",
        )
        require(
            kernel.target.split(":", 1)[0] == f"amdgcn-amd-amdhsa--{selector}",
            f"compiler target {kernel.target!r} does not match {selector!r}",
        )
        source += [
            render_variant(kernel, selector.replace("-", "_")),
            "",
        ]
    source += [f"constexpr ::kernels::Kernel kVariants[{len(variants)}] = {{"]
    for selector, _ in variants:
        prefix = selector.replace("-", "_") + "::"

        def value(name):
            return prefix + name

        source += [
            "    {",
            f"        {json.dumps(selector)},",
            f"        {value('kHsacoSha256')},",
            f"        {value('kExecutable')},",
            f"        {value('kEntryByteOffset')},",
            f"        {value('kEntryByteLength')},",
            f"        {value('kTextByteLength')},",
            "        {"
            + ", ".join(
                value(name)
                for name in (
                    "kKernargByteLength",
                    "kKernargAlignment",
                    "kArgumentByteOffsets",
                    "kArgumentByteLengths",
                    "kArgumentAlignments",
                    "kArgumentNames",
                    "kArgumentValueKinds",
                )
            )
            + "},",
            f"        {value('kRequiredWorkgroupSize')},",
            f"        {value('kMaxFlatWorkgroupSize')},",
            f"        {value('kWavefrontSize')},",
            f"        {value('kGroupSegmentByteLength')},",
            f"        {value('kPrivateSegmentByteLength')},",
            "        {"
            + ", ".join(
                value(name)
                for name in (
                    "kComputePgmRsrc1",
                    "kComputePgmRsrc2",
                    "kComputePgmRsrc3",
                    "kSgprCount",
                    "kVgprCount",
                    "kKernelCodeProperties",
                    "kKernargPreload",
                )
            )
            + "},",
            "    },",
        ]
    source += [
        "};",
        "",
        "}  // namespace",
        "",
        f"namespace {namespace} {{",
        "const ::kernels::KernelSet kKernels = {kVariants};",
        f"}}  // namespace {namespace}",
        "",
    ]
    return "\n".join(header), "\n".join(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", action="append", required=True, metavar="TARGET=PATH"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()
    try:
        variants = []
        for item in args.variant:
            selector, separator, path = item.partition("=")
            require(separator and path, "variant must be TARGET=PATH")
            require(selector not in dict(variants), "duplicate target selector")
            variants.append(
                (selector, extract_image(Path(path).read_bytes(), args.symbol))
            )
        header, source = render_set(variants, args.namespace, args.output.name)
        args.implementation.write_text(source, encoding="utf-8")
        args.output.write_text(header, encoding="utf-8")
    except (ImageError, OSError) as error:
        parser.exit(1, f"embed.py: {error}\n")


if __name__ == "__main__":
    main()
