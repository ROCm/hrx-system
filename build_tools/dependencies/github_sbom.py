# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Verifies that GitHub retained a derived dependency inventory."""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Mapping
from typing import Any

from build_tools.dependencies.inventory import validate_inventory


@dataclasses.dataclass(frozen=True, slots=True)
class Verification:
    """Summary of an inventory represented exactly once in an exported SBOM."""

    dependency_count: int
    package_count: int


def verify_inventory(
    inventory: Mapping[str, Any], response: Mapping[str, Any]
) -> Verification:
    """Requires every derived package exactly once in GitHub's SPDX document."""

    validate_inventory(inventory)
    sbom = response.get("sbom")
    if not isinstance(sbom, dict):
        raise ValueError("GitHub response has no SPDX document in 'sbom'")
    packages = sbom.get("packages")
    if not isinstance(packages, list):
        raise ValueError("GitHub SPDX document has no package list")

    actual_package_urls: Counter[str] = Counter()
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("GitHub SPDX package must be an object")
        external_references = package.get("externalRefs", [])
        if not isinstance(external_references, list):
            raise ValueError("GitHub SPDX externalRefs must be a list")
        for external_reference in external_references:
            if not isinstance(external_reference, dict):
                raise ValueError("GitHub SPDX external reference must be an object")
            if external_reference.get("referenceType") != "purl":
                continue
            package_url = external_reference.get("referenceLocator")
            if not isinstance(package_url, str) or not package_url:
                raise ValueError("GitHub SPDX purl reference has no locator")
            actual_package_urls[_canonical_package_url(package_url)] += 1

    expected_package_urls: dict[str, str] = {}
    for package_url in inventory["dependencies"]:
        canonical_package_url = _canonical_package_url(package_url)
        previous = expected_package_urls.setdefault(canonical_package_url, package_url)
        if previous != package_url:
            raise ValueError(
                "inventory package identities collide after GitHub normalization: "
                f"{previous!r} and {package_url!r}"
            )

    missing = sorted(
        original
        for canonical, original in expected_package_urls.items()
        if actual_package_urls[canonical] == 0
    )
    duplicates = sorted(
        (original, actual_package_urls[canonical])
        for canonical, original in expected_package_urls.items()
        if actual_package_urls[canonical] > 1
    )
    errors = []
    if missing:
        errors.append("missing " + ", ".join(missing))
    if duplicates:
        errors.append(
            "duplicated "
            + ", ".join(
                f"{package_url} ({count} copies)" for package_url, count in duplicates
            )
        )
    if errors:
        raise ValueError(
            "GitHub dependency graph verification failed: " + "; ".join(errors)
        )

    return Verification(
        dependency_count=len(expected_package_urls),
        package_count=len(packages),
    )


def _canonical_package_url(package_url: str) -> str:
    # GitHub lowercases the owner and repository portions of pkg:github
    # identities in its exported SPDX document while retaining tag/commit case.
    if not package_url.startswith("pkg:github/"):
        return package_url
    package, separator, version = package_url.rpartition("@")
    if not separator:
        return package_url.casefold()
    return f"{package.casefold()}@{version}"
