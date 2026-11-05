#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks extraction against freshly compiled GPU kernel products."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import struct
import sys
import unittest
from dataclasses import replace
from pathlib import Path


def load_embed():
    spec = importlib.util.spec_from_file_location(
        "amdf_kernel_embed", Path(__file__).with_name("embed.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


embed = load_embed()


def sections(data):
    """Returns raw ELF fields for independent byte comparisons/mutations."""
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    table = [
        struct.unpack_from("<IIQQQQIIQQ", data, header[6] + i * 64)
        for i in range(header[12])
    ]
    names_section = table[header[13]]
    names = data[names_section[4] : names_section[4] + names_section[5]]
    return {
        names[section[0] :].split(b"\0", 1)[0].decode(): (header[6] + i * 64, section)
        for i, section in enumerate(table)
    }


def section_index(data, name):
    """Returns one ELF section index without using the extractor under test."""
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    table = [
        struct.unpack_from("<IIQQQQIIQQ", data, header[6] + i * 64)
        for i in range(header[12])
    ]
    names_section = table[header[13]]
    names = data[names_section[4] : names_section[4] + names_section[5]]
    for index, section in enumerate(table):
        section_name = names[section[0] :].split(b"\0", 1)[0].decode()
        if section_name == name:
            return index
    raise AssertionError(f"missing section: {name}")


def symbols(data, section_name=".dynsym"):
    """Returns symbol record positions and fields for independent mutations."""
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    table = [
        struct.unpack_from("<IIQQQQIIQQ", data, header[6] + i * 64)
        for i in range(header[12])
    ]
    symbol_section = table[section_index(data, section_name)]
    string_section = table[symbol_section[6]]
    names = data[string_section[4] : string_section[4] + string_section[5]]
    result = {}
    for position in range(symbol_section[4], symbol_section[4] + symbol_section[5], 24):
        record = struct.unpack_from("<IBBHQQ", data, position)
        name = names[record[0] :].split(b"\0", 1)[0].decode()
        if name:
            result[name] = (position, record)
    return result


class EmbedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.products = {target: path.read_bytes() for target, path in cls.variants}
        cls.product_data = next(iter(cls.products.values()))

    def test_real_products_keep_transform_abi_and_compiler_resources(self):
        for target, data in self.products.items():
            with self.subTest(target=target):
                kernel = embed.extract_image(data, self.symbol)
                processor = target
                self.assertEqual(kernel.target, f"amdgcn-amd-amdhsa--{processor}")
                self.assertEqual(kernel.metadata[".kernarg_segment_size"], 24)
                self.assertEqual(kernel.metadata[".reqd_workgroup_size"], [64, 1, 1])
                self.assertEqual(kernel.metadata[".max_flat_workgroup_size"], 64)
                self.assertEqual(kernel.metadata[".group_segment_fixed_size"], 0)
                self.assertEqual(kernel.metadata[".private_segment_fixed_size"], 0)
                self.assertEqual(
                    kernel.metadata[".wavefront_size"],
                    64 if processor.startswith("gfx9") else 32,
                )
                self.assertEqual(
                    [
                        (a[".offset"], a[".size"], a[".value_kind"])
                        for a in kernel.metadata[".args"]
                    ],
                    [
                        (0, 8, "global_buffer"),
                        (8, 8, "global_buffer"),
                        (16, 4, "by_value"),
                        (20, 4, "by_value"),
                    ],
                )
                header, implementation = embed.render_set(
                    [(target, kernel)], "kernels::transform", "transform_kernels.h"
                )
                for index, offset in ((1, 48), (2, 52), (3, 44)):
                    actual = struct.unpack_from("<I", kernel.descriptor, offset)[0]
                    self.assertIn(
                        f"kComputePgmRsrc{index} = {actual}u;", implementation
                    )
                self.assertEqual(kernel.hsaco_sha256, hashlib.sha256(data).hexdigest())
                self.assertIn(hashlib.sha256(kernel.image).hexdigest(), implementation)
                self.assertIn(f'"{target}"', implementation)
                self.assertIn("extern const ::kernels::KernelSet kKernels;", header)

    def test_complete_text_and_descriptor_preserve_linked_layout(self):
        for target, data in self.products.items():
            with self.subTest(target=target):
                table = sections(data)
                rodata = table[".rodata"][1]
                text = table[".text"][1]
                kernel = embed.extract_image(data, self.symbol)
                phase = rodata[3] % 256
                entry = phase + text[3] - rodata[3]
                self.assertEqual(kernel.descriptor_offset, phase)
                self.assertEqual(kernel.rodata_size, 64)
                self.assertEqual(kernel.entry_offset, entry)
                self.assertEqual(kernel.image[:phase], bytes(phase))
                self.assertEqual(
                    kernel.image[phase : phase + 64], data[rodata[4] : rodata[4] + 64]
                )
                self.assertEqual(
                    kernel.image[phase + 64 : entry], bytes(entry - phase - 64)
                )
                self.assertEqual(
                    kernel.image[entry:], data[text[4] : text[4] + text[5]]
                )
                self.assertEqual(len(kernel.image), entry + text[5])

    def test_nonzero_descriptor_phase_is_retained(self):
        data = bytearray(self.product_data)
        table = sections(data)
        section_position, rodata = table[".rodata"]
        descriptor = bytes(data[rodata[4] : rodata[4] + 64])
        data[rodata[4] + 64 : rodata[4] + 128] = descriptor
        struct.pack_into(
            "<QQ", data, section_position + 16, rodata[3] + 64, rodata[4] + 64
        )
        delta = struct.unpack_from("<q", descriptor, 16)[0]
        struct.pack_into("<q", data, rodata[4] + 80, delta - 64)
        # The read segment must continue to cover the moved descriptor.
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
        for i in range(header[10]):
            position = header[5] + i * 56
            program = struct.unpack_from("<IIQQQQQQ", data, position)
            if program[0:2] == (1, 4):
                struct.pack_into(
                    "<QQ", data, position + 32, program[5] + 64, program[6] + 64
                )
        for name in (".dynsym", ".symtab"):
            symbols = table[name][1]
            for offset in range(symbols[4], symbols[4] + symbols[5], 24):
                if struct.unpack_from("<Q", data, offset + 8)[0] == rodata[3]:
                    struct.pack_into("<Q", data, offset + 8, rodata[3] + 64)
        kernel = embed.extract_image(bytes(data), self.symbol)
        self.assertEqual(kernel.descriptor_offset, (rodata[3] + 64) % 256)
        self.assertEqual(kernel.entry_offset % 256, 0)
        self.assertEqual(
            kernel.descriptor_offset
            + struct.unpack_from("<q", kernel.descriptor, 16)[0],
            kernel.entry_offset,
        )

    def test_rejects_truncated_or_wrong_format_products(self):
        original = self.product_data
        for data in (original[:32], original[:-64]):
            with self.subTest(size=len(data)), self.assertRaises(embed.ImageError):
                embed.extract_image(data, self.symbol)
        for offset, value in ((4, 1), (5, 2), (7, 0), (8, 3)):
            data = bytearray(original)
            data[offset] = value
            with self.subTest(offset=offset), self.assertRaises(embed.ImageError):
                embed.extract_image(bytes(data), self.symbol)

    def test_rejects_relocations_extra_backing_and_external_dependencies(self):
        original = self.product_data
        table = sections(original)
        mutations = [
            (table[".symtab"][0] + 4, "<I", 4, "relocation"),
            (table[".strtab"][0] + 8, "<Q", 2, "allocated section"),
            (table[".dynsym"][1][4] + 24 + 6, "<H", 0, "undefined symbol"),
            (table[".dynamic"][1][4], "<q", 1, "dynamic dependency"),
            (table[".text"][0] + 24, "<Q", len(original), "out-of-bounds"),
        ]
        for offset, format_string, value, message in mutations:
            data = bytearray(original)
            struct.pack_into(format_string, data, offset, value)
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(embed.ImageError, message),
            ):
                embed.extract_image(bytes(data), self.symbol)

    def test_rejects_descriptor_disagreement(self):
        for target, original in self.products.items():
            with self.subTest(target=target):
                descriptor = sections(original)[".rodata"][1][4]
                properties = struct.unpack_from("<H", original, descriptor + 56)[0]
                mutations = [
                    (0, "<I", 512, "group_segment"),
                    (4, "<I", 16, "private_segment"),
                    (8, "<I", 32, "kernarg_segment"),
                    (16, "<q", 64, "entry offset"),
                    (24, "<I", 1, "reserved"),
                    (56, "<H", properties ^ 0x400, "wavefront"),
                    (56, "<H", properties | 0x800, "dynamic private stack"),
                ]
                for offset, format_string, value, message in mutations:
                    data = bytearray(original)
                    struct.pack_into(format_string, data, descriptor + offset, value)
                    with (
                        self.subTest(message=message),
                        self.assertRaisesRegex(embed.ImageError, message),
                    ):
                        embed.extract_image(bytes(data), self.symbol)

    def metadata(self):
        data = self.product_data
        return embed.read_metadata(embed.Elf(data)), embed.extract_image(
            data, self.symbol
        )

    def test_runtime_geometry_is_not_invented(self):
        metadata, kernel = self.metadata()
        del metadata["amdhsa.kernels"][0][".reqd_workgroup_size"]
        _, parsed = embed.kernel_metadata(metadata, self.symbol, kernel.descriptor)
        source = embed.render_variant(
            replace(kernel, metadata=parsed), "kernels::geometry"
        )
        self.assertIn("kMaxFlatWorkgroupSize = 64u;", source)
        self.assertIn("    0u, 0u, 0u,", source)

    def test_rounded_kernarg_extent_is_not_the_last_argument_end(self):
        metadata, kernel = self.metadata()
        metadata["amdhsa.kernels"][0][".kernarg_segment_size"] = 32
        descriptor = bytearray(kernel.descriptor)
        struct.pack_into("<I", descriptor, 8, 32)
        _, parsed = embed.kernel_metadata(metadata, self.symbol, bytes(descriptor))
        source = embed.render_variant(
            replace(kernel, descriptor=bytes(descriptor), metadata=parsed),
            "kernels::rounded_arguments",
        )
        self.assertIn("kKernargByteLength = 32u;", source)
        self.assertEqual(
            parsed[".args"][-1][".offset"] + parsed[".args"][-1][".size"], 24
        )

    def test_argument_and_geometry_metadata_guards(self):
        metadata, kernel = self.metadata()
        mutations = [
            (".offset", 20, "argument alignment"),
            (".size", 32, "kernarg extent"),
            (".align", 16, "argument alignment"),
            (".value_kind", "hidden_printf_buffer", "unsupported argument kind"),
        ]
        for field, value, message in mutations:
            changed = copy.deepcopy(metadata)
            changed["amdhsa.kernels"][0][".args"][0][field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(embed.ImageError, message),
            ):
                embed.kernel_metadata(changed, self.symbol, kernel.descriptor)
        changed = copy.deepcopy(metadata)
        changed["amdhsa.kernels"][0][".args"][1][".offset"] = 0
        with self.assertRaisesRegex(embed.ImageError, "overlapping"):
            embed.kernel_metadata(changed, self.symbol, kernel.descriptor)
        changed = copy.deepcopy(metadata)
        changed["amdhsa.kernels"][0][".reqd_workgroup_size"] = [64, 2, 1]
        with self.assertRaisesRegex(embed.ImageError, "exceeds maximum"):
            embed.kernel_metadata(changed, self.symbol, kernel.descriptor)
        changed["amdhsa.kernels"][0][".reqd_workgroup_size"] = None
        with self.assertRaisesRegex(embed.ImageError, "required XYZ"):
            embed.kernel_metadata(changed, self.symbol, kernel.descriptor)

    def test_messagepack_duplicate_truncated_and_trailing_values(self):
        for data, message in (
            (b"\x82\xa1x\x01\xa1x\x02", "duplicate"),
            (b"\xdb\xff\xff\xff\xff", "out-of-bounds"),
            (b"\x91", "truncated"),
            (b"\x01\x02", "trailing"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(embed.ImageError, message),
            ):
                embed.MessagePack(data).decode()


class ImmutableDataEmbedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.products = {target: path.read_bytes() for target, path in cls.variants}
        cls.alternate_products = {
            target: path.read_bytes() for target, path in cls.alternate_variants
        }

    def test_products_preserve_complete_rodata_and_kernel_contract(self):
        self.assertEqual(self.products.keys(), self.alternate_products.keys())
        for target, data in self.products.items():
            with self.subTest(target=target):
                kernel = embed.extract_image(data, self.symbol)
                table = sections(data)
                rodata = table[".rodata"][1]
                text = table[".text"][1]
                phase = rodata[3] % 256
                entry = phase + text[3] - rodata[3]
                self.assertEqual(kernel.descriptor_offset, phase)
                self.assertEqual(kernel.rodata_size, rodata[5])
                self.assertGreater(kernel.rodata_size, 64)
                self.assertEqual(kernel.entry_offset, entry)
                self.assertEqual(kernel.image[:phase], bytes(phase))
                self.assertEqual(
                    kernel.image[phase : phase + rodata[5]],
                    data[rodata[4] : rodata[4] + rodata[5]],
                )
                self.assertEqual(
                    kernel.image[phase + rodata[5] : entry],
                    bytes(entry - phase - rodata[5]),
                )
                self.assertEqual(
                    kernel.image[entry:], data[text[4] : text[4] + text[5]]
                )
                self.assertEqual(kernel.metadata[".kernarg_segment_size"], 8)
                self.assertEqual(kernel.metadata[".reqd_workgroup_size"], [64, 1, 1])
                self.assertEqual(kernel.metadata[".max_flat_workgroup_size"], 64)
                self.assertEqual(kernel.metadata[".group_segment_fixed_size"], 0)
                self.assertEqual(kernel.metadata[".private_segment_fixed_size"], 0)
                self.assertEqual(
                    [
                        (
                            argument[".offset"],
                            argument[".size"],
                            argument[".value_kind"],
                        )
                        for argument in kernel.metadata[".args"]
                    ],
                    [(0, 8, "global_buffer")],
                )
                source = embed.render_variant(kernel, "kernels::immutable_data")
                self.assertIn(f"kRodataByteLength = {kernel.rodata_size}u;", source)

    def test_pair_has_identical_text_and_distinct_immutable_data(self):
        for target, data in self.products.items():
            with self.subTest(target=target):
                alternate_data = self.alternate_products[target]
                kernel = embed.extract_image(data, self.symbol)
                alternate = embed.extract_image(alternate_data, self.symbol)
                self.assertEqual(kernel.descriptor, alternate.descriptor)
                self.assertEqual(kernel.descriptor_offset, alternate.descriptor_offset)
                self.assertEqual(kernel.rodata_size, alternate.rodata_size)
                self.assertEqual(kernel.entry_offset, alternate.entry_offset)
                self.assertEqual(kernel.entry_size, alternate.entry_size)
                self.assertEqual(kernel.text_size, alternate.text_size)
                self.assertEqual(kernel.metadata, alternate.metadata)
                rodata_begin = kernel.descriptor_offset
                rodata_end = rodata_begin + kernel.rodata_size
                self.assertEqual(
                    kernel.image[: rodata_begin + 64],
                    alternate.image[: rodata_begin + 64],
                )
                self.assertNotEqual(
                    kernel.image[rodata_begin + 64 : rodata_end],
                    alternate.image[rodata_begin + 64 : rodata_end],
                )
                self.assertEqual(
                    kernel.image[rodata_end:], alternate.image[rodata_end:]
                )
                self.assertEqual(
                    struct.unpack("<16I", kernel.image[rodata_begin + 64 : rodata_end]),
                    tuple(range(16)),
                )
                self.assertEqual(
                    struct.unpack(
                        "<16I", alternate.image[rodata_begin + 64 : rodata_end]
                    ),
                    tuple(range(100, 116)),
                )

    def test_rejects_unbounded_or_overlapping_rodata_objects(self):
        original = next(iter(self.products.values()))
        table = sections(original)
        rodata = table[".rodata"][1]
        position, record = symbols(original)["immutable_words"]
        mutations = [
            (position + 4, "<B", 0x12, "global object"),
            (position + 6, "<H", section_index(original, ".text"), "outside rodata"),
            (position + 8, "<Q", rodata[3] + rodata[5], "exceeds rodata"),
            (position + 8, "<Q", rodata[3] + 32, "overlapping rodata"),
            (position + 16, "<Q", 0, "exceeds rodata"),
        ]
        self.assertEqual(record[1], 0x11)
        for offset, format_string, value, message in mutations:
            data = bytearray(original)
            struct.pack_into(format_string, data, offset, value)
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(embed.ImageError, message),
            ):
                embed.extract_image(bytes(data), self.symbol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", action="append", required=True, metavar="TARGET=PATH"
    )
    parser.add_argument(
        "--immutable-variant", action="append", required=True, metavar="TARGET=PATH"
    )
    parser.add_argument(
        "--immutable-alternate-variant",
        action="append",
        required=True,
        metavar="TARGET=PATH",
    )
    parser.add_argument("--symbol", default="aql_transform")
    args, unittest_args = parser.parse_known_args()
    EmbedTest.variants = [
        (target, Path(path))
        for target, path in (variant.split("=", 1) for variant in args.variant)
    ]
    EmbedTest.symbol = args.symbol
    ImmutableDataEmbedTest.variants = [
        (target, Path(path))
        for target, path in (
            variant.split("=", 1) for variant in args.immutable_variant
        )
    ]
    ImmutableDataEmbedTest.alternate_variants = [
        (target, Path(path))
        for target, path in (
            variant.split("=", 1) for variant in args.immutable_alternate_variant
        )
    ]
    ImmutableDataEmbedTest.symbol = "aql_immutable_data"
    unittest.main(argv=[sys.argv[0], *unittest_args])
