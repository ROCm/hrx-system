# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the GitHub dependency snapshot projection."""

import unittest

from build_tools.dependencies.github_snapshot import build_snapshot
from build_tools.dependencies.inventory import SCHEMA


class GithubSnapshotTest(unittest.TestCase):
    def test_projects_inventory_without_reinterpreting_identity(self):
        package_url = "pkg:github/example/demo@v1.2.3"
        bazel_package_url = "pkg:generic/bazel/rules_demo@1.0.0"
        inventory = {
            "schema": SCHEMA,
            "dependencies": {
                bazel_package_url: {
                    "package_url": bazel_package_url,
                    "relationship": "direct",
                    "scope": "development",
                    "declarations": [
                        {
                            "name": "rules_demo",
                            "kind": "bazel_module",
                            "owner": "root",
                            "source_location": "MODULE.bazel",
                        }
                    ],
                },
                package_url: {
                    "package_url": package_url,
                    "relationship": "direct",
                    "scope": "runtime",
                    "declarations": [
                        {
                            "name": "demo_headers",
                            "kind": "http_archive",
                            "owner": "shared",
                            "source_location": (
                                "build_tools/third_party/deps.MODULE.bazel"
                            ),
                            "patches": ["//build_tools/third_party/demo:fix.patch"],
                        }
                    ],
                },
            },
        }

        snapshot = build_snapshot(
            inventory,
            sha="a" * 40,
            ref="refs/heads/main",
            scanned="2026-11-10T18:00:00+00:00",
            job_id="1234",
            job_url="https://github.com/ROCm/hrx-system/actions/runs/1234",
        )

        self.assertEqual(snapshot["version"], 0)
        self.assertEqual(snapshot["sha"], "a" * 40)
        bazel_resolved = snapshot["manifests"]["MODULE.bazel"]["resolved"]
        self.assertEqual(list(bazel_resolved), [bazel_package_url])
        self.assertEqual(bazel_resolved[bazel_package_url]["scope"], "development")
        resolved = snapshot["manifests"]["build_tools/third_party/deps.MODULE.bazel"][
            "resolved"
        ]
        self.assertEqual(list(resolved), [package_url])
        self.assertEqual(resolved[package_url]["scope"], "runtime")
        self.assertEqual(
            resolved[package_url]["metadata"],
            {
                "hrx_declarations": "demo_headers",
                "hrx_kinds": "http_archive",
                "hrx_patch_count": 1,
                "hrx_source_locations": ("build_tools/third_party/deps.MODULE.bazel"),
            },
        )


if __name__ == "__main__":
    unittest.main()
