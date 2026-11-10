# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from libamdf.build_tools import generate_product_names as generator


class ProductNamesTest(unittest.TestCase):
    def test_first_parent_and_first_matching_child_own_names(self):
        devices, children = generator.parse_database(
            "# comment\n1.0.0\n1586, C1, Original\n"
            "\t1028, 0001, Board\n\t1028, 0001, Ignored\n"
            "1586, C1, Ignored parent\n\t1028, 0002, Ignored child\n"
            "1586, C2, Other revision"
        )
        self.assertEqual(devices, {0x1586C1: "Original", 0x1586C2: "Other revision"})
        self.assertEqual(children, {0x1586C110280001: "Board"})

    def test_empty_indented_comments_do_not_end_a_parent(self):
        devices, children = generator.parse_database(
            "1.0.0\n1586, C1, GPU\n\t# board\n\t\n\t1028, 0001, Board\n"
        )
        self.assertEqual(children, {0x1586C110280001: "Board"})
        self.assertEqual(devices, {0x1586C1: "GPU"})

    def test_invalid_input_fails_at_generation(self):
        for text in (
            "",
            "2.0.0\n1586, C1, GPU",
            "1.0.0",
            "1.0.0\n1586, C1",
            "1.0.0\n+1586, C1, GPU",
            "1.0.0\n1586x, C1, GPU",
            "1.0.0\n10000, C1, GPU",
            "1.0.0\n1586, 100, GPU",
            "1.0.0\n1586, C1, ",
            "1.0.0\n1586, C1, bad\0name",
            "1.0.0\n\t1028, 0001, Orphan",
            "1.0.0\n1586, C1, GPU\n\t1028, 10000, Board",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                generator.parse_database(text)

    def test_string_pool_deduplicates_base_and_board_names(self):
        text = generator.render_tables(
            {0x1586C2: "Shared", 0x1586C1: "Shared"},
            {0x1586C110280001: "Shared"},
        )
        self.assertEqual(text.count('"Shared\\000"'), 1)
        self.assertIn("amdf_pci_product_names[7]", text)
        self.assertIn("0x1586C1u, 0x1586C2u", text)
        self.assertIn("UINT64_C(0x1586C110280001)", text)
        self.assertIn("AMDF_PCI_PRODUCT_MAX_NAME_LENGTH 6", text)

    def test_utf8_quotes_backslashes_and_following_digits_are_unambiguous(self):
        self.assertEqual(
            generator.c_string('GPU "\\μ7'.encode() + b"\0"),
            '"GPU \\042\\134\\316\\2747\\000"',
        )
        text = generator.render_tables({0: "μ"}, {})
        self.assertIn("AMDF_PCI_PRODUCT_MAX_NAME_LENGTH 2", text)
        self.assertIn("amdf_pci_product_names[3]", text)
        self.assertIn("AMDF_PCI_PRODUCT_SUBSYSTEM_COUNT 0", text)
        self.assertNotIn("amdf_pci_product_subsystem_keys", text)
        self.assertEqual(generator.c_string(b"??/"), '"\\077\\077/"')

    def test_large_registries_widen_offsets_at_build_time(self):
        names = {key: f"GPU {key:04d} " + "X" * 60 for key in range(1000)}
        text = generator.render_tables(names, {})
        self.assertIn("uint32_t amdf_pci_product_device_offsets", text)


if __name__ == "__main__":
    unittest.main()
