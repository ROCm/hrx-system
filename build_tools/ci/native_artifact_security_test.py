# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from build_tools.ci import native_artifact_security

BUNDLED_RUNPATH = native_artifact_security.ElfRunpathPolicy.BUNDLED_ORIGIN
NO_RUNPATH = native_artifact_security.ElfRunpathPolicy.ABSENT


def parse_elf(document, *, executable=True, runpath_policy=BUNDLED_RUNPATH):
    return native_artifact_security._parse_elf(
        document,
        executable=executable,
        runpath_policy=runpath_policy,
    )


def elf_document():
    return [
        {
            "ElfHeader": {"Type": "SharedObject (0x3)"},
            "ProgramHeaders": [
                {
                    "ProgramHeader": {
                        "Type": {"Name": "PT_LOAD"},
                        "Flags": {
                            "Flags": [
                                {"Name": "PF_R"},
                                {"Name": "PF_X"},
                            ]
                        },
                    }
                },
                {
                    "ProgramHeader": {
                        "Type": {"Name": "PT_GNU_STACK"},
                        "Flags": {
                            "Flags": [
                                {"Name": "PF_R"},
                                {"Name": "PF_W"},
                            ]
                        },
                    }
                },
                {
                    "ProgramHeader": {
                        "Type": {"Name": "PT_GNU_RELRO"},
                        "Flags": {"Flags": [{"Name": "PF_R"}]},
                    }
                },
            ],
            "DynamicSection": [
                {"Type": "RUNPATH", "Path": ["$ORIGIN"]},
                {"Type": "FLAGS", "Flags": ["BIND_NOW"]},
                {"Type": "FLAGS_1", "Flags": ["NOW", "PIE"]},
            ],
        }
    ]


PE_OUTPUT = """\
Format: COFF-x86-64
ImageOptionalHeader {
  Characteristics [ (0x8160)
    IMAGE_DLL_CHARACTERISTICS_DYNAMIC_BASE (0x40)
    IMAGE_DLL_CHARACTERISTICS_HIGH_ENTROPY_VA (0x20)
    IMAGE_DLL_CHARACTERISTICS_NX_COMPAT (0x100)
  ]
}
LoadConfig [
  SecurityCookie: 0x14000F040
]
Sections [
  Section {
    Name: .text
    Characteristics [
      IMAGE_SCN_MEM_EXECUTE
      IMAGE_SCN_MEM_READ
    ]
  }
  Section {
    Name: .data
    Characteristics [
      IMAGE_SCN_MEM_READ
      IMAGE_SCN_MEM_WRITE
    ]
  }
]
"""


class NativeArtifactSecurityTest(unittest.TestCase):
    def test_elf_requires_complete_position_independent_hardening(self):
        report = parse_elf(elf_document())
        self.assertEqual(report.format, "ELF")
        self.assertIn("pie", report.properties)
        self.assertIn("eager-binding", report.properties)

        executable_stack = copy.deepcopy(elf_document())
        executable_stack[0]["ProgramHeaders"][1]["ProgramHeader"]["Flags"][
            "Flags"
        ].append({"Name": "PF_X"})
        with self.assertRaisesRegex(
            native_artifact_security.NativeArtifactSecurityError,
            "non-executable-stack",
        ):
            parse_elf(executable_stack)

        no_pie = copy.deepcopy(elf_document())
        no_pie[0]["DynamicSection"][2]["Flags"].remove("PIE")
        with self.assertRaisesRegex(
            native_artifact_security.NativeArtifactSecurityError, "pie"
        ):
            parse_elf(no_pie)
        library_report = parse_elf(no_pie, executable=False)
        self.assertNotIn("pie", library_report.properties)

    def test_elf_standalone_artifact_requires_no_runpath(self):
        document = elf_document()
        document[0]["DynamicSection"] = [
            entry
            for entry in document[0]["DynamicSection"]
            if entry["Type"] != "RUNPATH"
        ]
        report = parse_elf(document, runpath_policy=NO_RUNPATH)
        self.assertIn("no-runpath", report.properties)

        with self.assertRaisesRegex(
            native_artifact_security.NativeArtifactSecurityError,
            "origin-runpath",
        ):
            parse_elf(document)
        with self.assertRaisesRegex(
            native_artifact_security.NativeArtifactSecurityError,
            "no-runpath",
        ):
            parse_elf(elf_document(), runpath_policy=NO_RUNPATH)

    def test_elf_rejects_writable_executable_loads(self):
        document = elf_document()
        document[0]["ProgramHeaders"][0]["ProgramHeader"]["Flags"]["Flags"].append(
            {"Name": "PF_W"}
        )
        with self.assertRaisesRegex(
            native_artifact_security.NativeArtifactSecurityError,
            "writable-xor-executable",
        ):
            parse_elf(document)

    def test_pe_requires_aslr_dep_stack_cookie_and_writable_xor_executable(self):
        report = native_artifact_security._parse_pe(PE_OUTPUT)
        self.assertEqual(report.format, "PE")
        self.assertIn("high-entropy-aslr", report.properties)
        self.assertIn("stack-cookie", report.properties)

        for removed, expected in [
            ("IMAGE_DLL_CHARACTERISTICS_NX_COMPAT", "data-execution-prevention"),
            ("0x14000F040", "stack-cookie"),
        ]:
            with self.subTest(removed=removed):
                invalid = PE_OUTPUT.replace(removed, "0x0")
                with self.assertRaisesRegex(
                    native_artifact_security.NativeArtifactSecurityError,
                    expected,
                ):
                    native_artifact_security._parse_pe(invalid)

        writable_text = PE_OUTPUT.replace(
            "IMAGE_SCN_MEM_EXECUTE\n      IMAGE_SCN_MEM_READ",
            "IMAGE_SCN_MEM_EXECUTE\n      IMAGE_SCN_MEM_READ\n      IMAGE_SCN_MEM_WRITE",
        )
        with self.assertRaisesRegex(
            native_artifact_security.NativeArtifactSecurityError,
            "writable-xor-executable",
        ):
            native_artifact_security._parse_pe(writable_text)

    def test_wheel_qualifies_every_native_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            wheel = Path(temporary) / "release.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("pkg.data/scripts/tool", b"\x7fELFtool")
                archive.writestr("pkg.data/scripts/LIBRARY.DLL", b"MZlibrary")
                archive.writestr(
                    "pkg.data/scripts/tool.pdb", b"Microsoft C/C++ MSF 7.00"
                )
                archive.writestr("pkg.data/scripts/helper", b"#!/usr/bin/python3")
                archive.writestr("pkg/extension.so", b"\x7fELFextension")
                archive.writestr("pkg/__init__.py", b"")
            reports = [
                native_artifact_security.NativeArtifactSecurity("ELF", frozenset()),
                native_artifact_security.NativeArtifactSecurity("PE", frozenset()),
                native_artifact_security.NativeArtifactSecurity("ELF", frozenset()),
            ]
            with mock.patch.object(
                native_artifact_security, "inspect_file", side_effect=reports
            ) as inspect_file:
                qualified = native_artifact_security.qualify_wheel(
                    Path("llvm-readobj"), wheel
                )

        self.assertEqual(
            [name for name, _ in qualified],
            [
                "pkg.data/scripts/LIBRARY.DLL",
                "pkg.data/scripts/tool",
                "pkg/extension.so",
            ],
        )
        self.assertEqual(
            [call.kwargs["executable"] for call in inspect_file.call_args_list],
            [False, True, False],
        )
        self.assertTrue(
            all(
                call.kwargs["runpath_policy"] is BUNDLED_RUNPATH
                for call in inspect_file.call_args_list
            )
        )

    def test_wheel_rejects_unrecognized_script_payloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            wheel = Path(temporary) / "release.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("pkg.data/scripts/tool", b"not an executable")
            with self.assertRaisesRegex(
                native_artifact_security.NativeArtifactSecurityError,
                "unsupported native payload",
            ):
                native_artifact_security.qualify_wheel(Path("llvm-readobj"), wheel)


if __name__ == "__main__":
    unittest.main()
