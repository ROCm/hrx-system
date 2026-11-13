# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Module visibility of dependencies reached by exported testing APIs."""

import unittest
from pathlib import Path

from build_tools.bazel_to_cmake import deps


class ExportedDependenciesTest(unittest.TestCase):
    def test_exported_test_frameworks_survive_non_root_resolution(self):
        # Runfiles can link data to source while retaining a copied test script.
        # The parser's repository root must own the resolved module fragment.
        fragment = Path(__file__).with_name("deps.MODULE.bazel").resolve()
        root = fragment.parents[2]
        dependencies = {
            dependency.name: dependency
            for dependency in deps.ModuleParser(root).parse(fragment)
        }
        # runtime/testing exports GoogleTest and Benchmark helpers; libhrx CTS
        # exports Catch2-based tests. Their frameworks are part of those APIs,
        # not development-only tooling of the enclosing root workspace.
        for name in ("googletest", "google_benchmark", "catch2"):
            with self.subTest(dependency=name):
                self.assertIn(name, dependencies)
                self.assertFalse(dependencies[name].dev_dependency)

    def test_stack_trace_repairs_are_owned_by_the_dependency(self):
        fragment = Path(__file__).with_name("deps.MODULE.bazel").resolve()
        dependencies = deps.ModuleParser(fragment.parents[2]).parse(fragment)
        backtrace = next(dep for dep in dependencies if dep.name == "libbacktrace")
        # Module-owned repository rules run for imported modules too. A root
        # single_version_override would silently drop these correctness repairs.
        self.assertEqual(backtrace.kind, "http_archive")
        self.assertEqual(
            backtrace.patches,
            (
                "//build_tools/third_party/libbacktrace:explicit_integer_conversions.patch",
                "//build_tools/third_party/libbacktrace:threaded_cache_reads.patch",
                "//build_tools/third_party/libbacktrace:ignore_nonruntime_symbols.patch",
                "//build_tools/third_party/libbacktrace:zero_size_vectors.patch",
                "//build_tools/third_party/libbacktrace:iterative_function_scopes.patch",
                "//build_tools/third_party/libbacktrace:portable_test_warnings.patch",
            ),
        )
        self.assertEqual(backtrace.patch_args, ("-p1",))


if __name__ == "__main__":
    unittest.main()
