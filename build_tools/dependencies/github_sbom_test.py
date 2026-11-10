# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for GitHub dependency graph verification."""

import unittest

from build_tools.dependencies.github_sbom import verify_inventory
from build_tools.dependencies.inventory import SCHEMA


class GithubSbomTest(unittest.TestCase):
    def test_accepts_exact_inventory_with_github_name_normalization(self):
        inventory = _inventory(
            "pkg:github/KhronosGroup/SPIRV-Headers@vulkan-sdk-1.4.357.0",
            "pkg:generic/bazel/rules_demo@1.0.0",
        )
        response = _response(
            "pkg:github/khronosgroup/spirv-headers@vulkan-sdk-1.4.357.0",
            "pkg:generic/bazel/rules_demo@1.0.0",
            "pkg:pypi/unrelated@2.0.0",
        )

        verification = verify_inventory(inventory, response)

        self.assertEqual(verification.dependency_count, 2)
        self.assertEqual(verification.package_count, 3)

    def test_rejects_missing_inventory_package(self):
        inventory = _inventory(
            "pkg:github/example/one@v1",
            "pkg:github/example/two@v2",
        )

        with self.assertRaisesRegex(ValueError, "missing pkg:github/example/two@v2"):
            verify_inventory(inventory, _response("pkg:github/example/one@v1"))

    def test_rejects_duplicate_inventory_package(self):
        package_url = "pkg:github/example/demo@v1"

        with self.assertRaisesRegex(ValueError, "duplicated.*2 copies"):
            verify_inventory(
                _inventory(package_url),
                _response(package_url, package_url),
            )


def _inventory(*package_urls: str) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "dependencies": {
            package_url: {
                "package_url": package_url,
                "relationship": "direct",
                "scope": "runtime",
                "declarations": [
                    {
                        "name": f"demo_{index}",
                        "kind": "http_archive",
                        "owner": "shared",
                        "source_location": (
                            "build_tools/third_party/deps.MODULE.bazel"
                        ),
                    }
                ],
            }
            for index, package_url in enumerate(package_urls)
        },
    }


def _response(*package_urls: str) -> dict[str, object]:
    return {
        "sbom": {
            "packages": [
                {
                    "externalRefs": [
                        {
                            "referenceType": "purl",
                            "referenceLocator": package_url,
                        }
                    ]
                }
                for package_url in package_urls
            ]
        }
    }


if __name__ == "__main__":
    unittest.main()
