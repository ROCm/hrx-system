# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks function/data ownership in compiler-generated Windows export lists."""

import unittest

from loom.build_tools.shared_exports import export_declarations


class SharedExportsTest(unittest.TestCase):
    def test_nm_archive_symbols(self):
        self.assertEqual(
            export_declarations(
                """
runtime.lib[allocator.obj]:
iree_allocator_free T 0 0
iree_table R 0 80
private_helper T 0 0
mi_malloc T 0 0
runtime.lib[duplicate.obj]:
iree_allocator_free T 0 0
""",
                ["iree_", "mi_"],
            ),
            ["iree_allocator_free", "iree_table DATA", "mi_malloc"],
        )

    def test_dumpbin_definitions_and_imports(self):
        self.assertEqual(
            export_declarations(
                """
00A 00000000 SECT3  notype ()    External     | iree_allocator_free
00B 00000000 UNDEF  notype ()    External     | iree_imported
00C 00000000 SECT4  notype       External     | iree_table
00D 00000000 SECT4  notype       Static       | iree_private
00E 00000000 SECT5  notype ()    External     | private_helper
""",
                ["iree_"],
                dumpbin=True,
            ),
            ["iree_allocator_free", "iree_table DATA"],
        )

    def test_empty_or_conflicting_exports_fail(self):
        for symbols in ("", "iree_value T 0 0\niree_value D 0 0", "iree_value ? 0 0"):
            with self.subTest(symbols=symbols), self.assertRaises(ValueError):
                export_declarations(symbols, ["iree_"])


if __name__ == "__main__":
    unittest.main()
