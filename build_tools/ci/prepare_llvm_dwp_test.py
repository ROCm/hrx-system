# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_tools.ci import prepare_llvm_dwp


class PrepareLlvmDwpTest(unittest.TestCase):
    def test_parses_rocm_source_revision(self):
        identity = prepare_llvm_dwp.parse_llvm_source_identity(
            "AMD clang version 23.0.0git "
            "(https://github.com/ROCm/llvm-project.git "
            "0bace1908348b840e6aa1b4b6e12151dae208158)\n"
        )

        self.assertEqual("https://github.com/ROCm/llvm-project", identity.repository)
        self.assertEqual("ROCm/llvm-project", identity.github_repository)
        self.assertEqual(
            "https://raw.githubusercontent.com/ROCm/llvm-project/"
            "0bace1908348b840e6aa1b4b6e12151dae208158/"
            "llvm/tools/llvm-dwp/llvm-dwp.cpp",
            prepare_llvm_dwp.source_url(identity, "llvm/tools/llvm-dwp/llvm-dwp.cpp"),
        )

    def test_rejects_untrusted_source_repository(self):
        with self.assertRaisesRegex(ValueError, "unsupported LLVM source repository"):
            prepare_llvm_dwp.parse_llvm_source_identity(
                "clang version 23.0.0 "
                "(https://github.com/example/llvm-project.git "
                "0123456789abcdef0123456789abcdef01234567)\n"
            )

    def test_requires_immutable_source_revision(self):
        with self.assertRaisesRegex(ValueError, "does not report"):
            prepare_llvm_dwp.parse_llvm_source_identity("clang version 23.0.0\n")

    def test_cached_tool_matches_exact_installation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            llvm_root = root / "llvm"
            llvm_root.mkdir()
            output = root / "llvm-dwp"
            output.write_bytes(b"tool")
            output.chmod(0o755)
            identity = prepare_llvm_dwp.LlvmSourceIdentity(
                repository="https://github.com/llvm/llvm-project",
                github_repository="llvm/llvm-project",
                commit="a" * 40,
            )
            manifest = output.with_name(output.name + ".json")
            manifest.write_text(
                json.dumps(
                    {
                        "cache_version": prepare_llvm_dwp._CACHE_VERSION,
                        "llvm_root": str(llvm_root),
                        "output_sha256": hashlib.sha256(b"tool").hexdigest(),
                        "source_commit": identity.commit,
                        "source_repository": identity.repository,
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(
                prepare_llvm_dwp._cached_tool_matches(
                    output, manifest, identity, llvm_root
                )
            )
            self.assertFalse(
                prepare_llvm_dwp._cached_tool_matches(
                    output, manifest, identity, root / "other-llvm"
                )
            )
            output.chmod(0o644)
            self.assertFalse(
                prepare_llvm_dwp._cached_tool_matches(
                    output, manifest, identity, llvm_root
                )
            )
            output.chmod(0o755)
            output.write_bytes(b"changed tool")
            self.assertFalse(
                prepare_llvm_dwp._cached_tool_matches(
                    output, manifest, identity, llvm_root
                )
            )


if __name__ == "__main__":
    unittest.main()
