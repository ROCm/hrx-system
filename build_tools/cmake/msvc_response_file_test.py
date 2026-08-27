# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import tempfile
import unittest
from pathlib import Path

from build_tools.cmake import msvc_response_file


class MsvcResponseFileTest(unittest.TestCase):
    def test_rewraps_unquoted_arguments(self):
        self.assertEqual(
            msvc_response_file.one_argument_per_line(
                b"first.obj  second.lib\tthird.lib\r\nfourth.lib"
            ),
            b"first.obj\r\nsecond.lib\r\nthird.lib\r\nfourth.lib\r\n",
        )

    def test_preserves_whitespace_inside_quotes(self):
        self.assertEqual(
            msvc_response_file.one_argument_per_line(
                b'"path with spaces\\first.obj"  /flag:"value with spaces"'
            ),
            b'"path with spaces\\first.obj"\r\n/flag:"value with spaces"\r\n',
        )

    def test_preserves_escaped_quotes(self):
        self.assertEqual(
            msvc_response_file.one_argument_per_line(
                b'"value with \\"quoted spaces\\" inside" next.lib'
            ),
            b'"value with \\"quoted spaces\\" inside"\r\nnext.lib\r\n',
        )

    def test_rejects_unterminated_quotes(self):
        with self.assertRaisesRegex(ValueError, "unterminated quote"):
            msvc_response_file.one_argument_per_line(b'"first second')

    def test_rewrites_file_atomically_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            response_file = Path(temporary_directory) / "link.rsp"
            response_file.write_bytes(b"first.obj second.lib")

            msvc_response_file.rewrap_response_file(response_file)
            expected_contents = b"first.obj\r\nsecond.lib\r\n"
            self.assertEqual(response_file.read_bytes(), expected_contents)

            msvc_response_file.rewrap_response_file(response_file)
            self.assertEqual(response_file.read_bytes(), expected_contents)


if __name__ == "__main__":
    unittest.main()
