# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import subprocess
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

    def test_launcher_preserves_arguments_and_exit_status(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            response_file = Path(temporary_directory) / "link arguments.rsp"
            response_file.write_bytes(b'first.obj "path with spaces/second.lib"')
            linker = (
                "import json,sys; from pathlib import Path; "
                "print(json.dumps([sys.argv[1:], Path(sys.argv[1][1:]).read_text()])); "
                "sys.exit(23)"
            )
            # The parent launcher has an opaque @argument of its own. It must
            # reach that launcher unchanged, without being opened as a file.
            parent = (
                "import subprocess,sys; "
                "assert sys.argv[1] == '@parent configuration'; "
                "sys.exit(subprocess.call(sys.argv[2:]))"
            )
            parent_arguments = [sys.executable, "-c", parent, "@parent configuration"]
            linker_arguments = ["@" + str(response_file), 'quoted "value" with spaces']
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    msvc_response_file.__file__,
                    *parent_arguments,
                    "--iree-native-linker",
                    sys.executable,
                    "-c",
                    linker,
                    *linker_arguments,
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 23, result.stderr)
            arguments, response = json.loads(result.stdout)
            self.assertEqual(arguments, linker_arguments)
            self.assertEqual(response, 'first.obj\n"path with spaces/second.lib"\n')

    @unittest.skipUnless(sys.platform == "win32", "Windows process status encoding")
    def test_preserves_windows_status_bits(self):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                msvc_response_file.__file__,
                "--iree-native-linker",
                sys.executable,
                "-c",
                "import sys; sys.exit(-1073741819)",
            ]
        )
        self.assertEqual(result.returncode, 0xC0000005)


if __name__ == "__main__":
    unittest.main()
