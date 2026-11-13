# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import unittest

from build_tools.lefthook import repository_policy


class RepositoryPolicyTest(unittest.TestCase):
    def assert_rules(self, path: str, text: str, expected: set[str]):
        actual = {
            finding.rule_id
            for finding in repository_policy.findings_for_text(path, text)
        }
        self.assertEqual(actual, expected)

    def test_rule_identifiers_are_unique(self):
        identifiers = [rule.rule_id for rule in repository_policy.RULES]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_c_runtime_forbids_goto_error_handling(self):
        rule = {"iree.c.no-goto-error-handling"}
        self.assert_rules("runtime/src/iree/base/example.c", "goto cleanup;", rule)
        self.assert_rules("loom/src/loom/example.c", "goto fail;", rule)
        self.assert_rules("libamdf/src/example.c", "goto done;", rule)
        self.assert_rules("libhrx/src/example.c", "goto cleanup;", set())
        self.assert_rules("runtime/src/iree/base/example.cc", "goto cleanup;", set())

    def test_cpp_forbids_designated_initializer_suppression(self):
        rule = {"iree.cpp.no-designated-initializer-suppression"}
        self.assert_rules(
            "loom/src/loom/example.cc",
            "// NOLINTNEXTLINE(iree-cpp-designated-initializer)",
            rule,
        )
        self.assert_rules(
            "libhrx/src/example.cxx",
            "// NOLINT(another-check,\n// iree-cpp-designated-initializer)",
            rule,
        )
        self.assert_rules(
            "runtime/src/iree/base/example.c",
            "// NOLINT(iree-cpp-designated-initializer)",
            set(),
        )

    def test_target_diagnostics_remain_target_generic(self):
        rule = {"iree.loom.target-diagnostics-are-target-generic"}
        for text in (
            "AIE2P worker placement is invalid.",
            "xDnA routing capacity is exhausted.",
            "Select an RDNA4 profile.",
            "CDNA and gfx942 expose this resource.",
            "The x86-64 ABI cannot represent this value.",
            "AMD64, WASM, and SPIRV use different ABIs.",
            "Select WebAssembly instead of SPIR-V.",
            "Vulkan, CUDA, and NVIDIA PTX use different layouts.",
            "Use an AArch64 or RISC-V target.",
            "ARM64 NEON cannot lower this packet.",
            "Intel AVX512 and SSE4 use different carriers.",
            "Apple Metal rejected the operation.",
        ):
            with self.subTest(text=text):
                self.assert_rules("loom/py/loom/error/target.py", text, rule)
        self.assert_rules(
            "loom/py/loom/error/target.py",
            "The selected GPU target has no compatible lowering.",
            set(),
        )
        self.assert_rules("loom/py/loom/error/xdna.py", "XDNA routing failed.", set())

    def test_architecture_records_use_their_own_error_domain(self):
        rule = {"iree.loom.arch-target-record-diagnostics-use-own-domain"}
        path = "loom/src/loom/target/arch/xdna/ops/target.c"
        self.assert_rules(path, "LOOM_ERR_TARGET_082", rule)
        self.assert_rules(path, "LOOM_ERR_TARGET_082_REF", rule)
        self.assert_rules(path, "LOOM_ERR_XDNA_062", set())

    def test_hardware_reference_vocabulary_and_links(self):
        vocabulary = "amd-reference.no-library-vocabulary"
        local_link = "amd-reference.no-local-implementation-links"
        path = "docs/reference/amd/gpu/packets.md"
        for text in (
            "libamdf selects this packet.",
            "`AMDF_GPU_QUEUE_FLAG_NONE` is public.",
            "`amdf_gpu_queue_create` is public.",
            "A CTS observation is not a native contract.",
        ):
            with self.subTest(text=text):
                self.assert_rules(path, text, {vocabulary})
        for text in (
            "[Queue](../../../../runtime/src/queue.c)",
            "[ABI]: <../../../../include/amdf/gpu.h>",
            "[Storage](../src/storage.c#ownership)",
            "[Notes](../../../../.notes/session.md)",
            "[Source](file:///tmp/src/native.c)",
            "[Header](/include/native.h)",
        ):
            with self.subTest(text=text):
                self.assert_rules(path, text, {local_link})
        for text in (
            "The packet selects objects and affects products.",
            "[Queue](https://example.org/driver/src/queue.c)",
            "[Local section](#include-layout)",
            "[Cache](cache.md#src-fields)",
        ):
            with self.subTest(text=text):
                self.assert_rules(path, text, set())
        self.assert_rules("docs/reference/amd-other/README.md", "libamdf CTS", set())

    def test_library_documentation_is_independent_of_cts(self):
        rule = {"libamdf.docs.no-cts-dependencies"}
        for path in (
            "libamdf/docs/memory.md",
            "libamdf/docs/experimental/example.rst",
            "libamdf/README.md",
        ):
            with self.subTest(path=path):
                self.assert_rules(path, "The CTS qualifies this API.", rule)
        for path in (
            "libamdf/cts/README.md",
            "libamdf/experimental/cts/README.md",
            "libamdf/README.extra.md",
        ):
            with self.subTest(path=path):
                self.assert_rules(path, "CTS", set())

    def test_findings_report_exact_source_position(self):
        findings = repository_policy.findings_for_text(
            "runtime/src/iree/base/example.c",
            "first line\n  goto cleanup;\n",
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual((findings[0].line, findings[0].column), (2, 3))


if __name__ == "__main__":
    unittest.main()
