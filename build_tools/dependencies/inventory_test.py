# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for derived dependency inventory construction."""

import unittest

from build_tools.bazel_to_cmake.deps import Dependency
from build_tools.dependencies.inventory import (
    SCHEMA,
    build_inventory,
    source_package_url,
    validate_inventory,
)


class InventoryTest(unittest.TestCase):
    def test_combines_resolved_modules_and_collapses_source_identity(self):
        graph = {
            "root": True,
            "dependencies": [
                {
                    "name": "rules_cc",
                    "version": "1.2.3",
                    "apparentName": "rules_cc",
                },
                {
                    "name": "zlib",
                    "version": "1.3.2",
                    "apparentName": "zlib",
                },
            ],
        }
        dependencies = [
            Dependency(
                name="zlib",
                kind="bazel_dep",
                owner="shared",
                module_name="zlib",
                repo_name="zlib",
                version="1.3.2",
                dev_dependency=False,
            ),
            Dependency(
                name="linux_first_header",
                kind="http_file",
                owner="shared",
                module_name="linux_first_header",
                repo_name="linux_first_header",
                version="",
                dev_dependency=False,
                urls=(
                    "https://raw.githubusercontent.com/torvalds/linux/"
                    "abc123/include/uapi/linux/first.h",
                ),
                sha256="1" * 64,
                downloaded_file_path="linux/first.h",
            ),
            Dependency(
                name="linux_second_header",
                kind="http_file",
                owner="shared",
                module_name="linux_second_header",
                repo_name="linux_second_header",
                version="",
                dev_dependency=False,
                urls=(
                    "https://raw.githubusercontent.com/torvalds/linux/"
                    "abc123/include/uapi/linux/second.h",
                ),
                sha256="2" * 64,
                downloaded_file_path="linux/second.h",
            ),
        ]

        inventory = build_inventory(graph, dependencies)

        self.assertEqual(inventory["schema"], SCHEMA)
        validate_inventory(inventory)
        entries = inventory["dependencies"]
        self.assertEqual(len(entries), 3)
        self.assertEqual(
            entries["pkg:generic/bazel/rules_cc@1.2.3"]["scope"],
            "development",
        )
        self.assertEqual(
            entries["pkg:generic/bazel/zlib@1.3.2"]["scope"],
            "runtime",
        )
        linux = entries["pkg:github/torvalds/linux@abc123"]
        self.assertEqual(linux["scope"], "runtime")
        self.assertEqual(
            [value["name"] for value in linux["declarations"]],
            ["linux_first_header", "linux_second_header"],
        )

    def test_source_package_urls_cover_tag_commit_and_generic_sources(self):
        base = dict(
            kind="http_archive",
            owner="shared",
            module_name="demo",
            repo_name="demo",
            version="",
            dev_dependency=False,
        )
        tagged = Dependency(
            name="demo",
            urls=("https://github.com/example/demo/archive/refs/tags/v1.2.tar.gz",),
            **base,
        )
        committed = Dependency(
            name="demo",
            urls=("https://github.com/example/demo/archive/abc123.zip",),
            **base,
        )
        libdrm = Dependency(
            name="libdrm_headers",
            urls=("https://example.com/libdrm-2.4.134.tar.xz",),
            strip_prefix="libdrm-2.4.134",
            **base,
        )

        self.assertEqual(source_package_url(tagged), "pkg:github/example/demo@v1.2")
        self.assertEqual(
            source_package_url(committed), "pkg:github/example/demo@abc123"
        )
        self.assertEqual(source_package_url(libdrm), "pkg:generic/libdrm@2.4.134")

    def test_rejects_resolved_module_version_disagreement(self):
        graph = {
            "root": True,
            "dependencies": [{"name": "zlib", "version": "1.3.1"}],
        }
        dependency = Dependency(
            name="zlib",
            kind="bazel_dep",
            owner="shared",
            module_name="zlib",
            repo_name="zlib",
            version="1.3.2",
            dev_dependency=False,
        )

        with self.assertRaisesRegex(ValueError, "Bazel resolved zlib@1.3.1"):
            build_inventory(graph, [dependency])

    def test_rejects_unclassified_source_host(self):
        dependency = Dependency(
            name="demo",
            kind="http_archive",
            owner="shared",
            module_name="demo",
            repo_name="demo",
            version="",
            dev_dependency=False,
            urls=("https://example.com/demo-1.0.tar.gz",),
        )

        with self.assertRaisesRegex(ValueError, "cannot derive a package identity"):
            source_package_url(dependency)


if __name__ == "__main__":
    unittest.main()
