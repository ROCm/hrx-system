# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Projects the derived inventory into GitHub dependency-submission JSON."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from build_tools.dependencies.inventory import validate_inventory

DETECTOR_NAME = "hrx-bazel-source-dependencies"
DETECTOR_VERSION = "0.1"


def build_snapshot(
    inventory: Mapping[str, Any],
    *,
    sha: str,
    ref: str,
    scanned: str,
    job_id: str,
    job_url: str,
) -> dict[str, object]:
    """Builds a replaceable GitHub dependency snapshot."""

    validate_inventory(inventory)
    bazel_resolved = {}
    source_resolved = {}
    for package_url, dependency in inventory["dependencies"].items():
        declarations = dependency["declarations"]
        resolved = (
            bazel_resolved
            if package_url.startswith("pkg:generic/bazel/")
            else source_resolved
        )
        resolved[package_url] = {
            "package_url": package_url,
            "relationship": dependency["relationship"],
            "scope": dependency["scope"],
            "metadata": {
                "hrx_declarations": ",".join(
                    sorted({value["name"] for value in declarations})
                ),
                "hrx_kinds": ",".join(
                    sorted({value["kind"] for value in declarations})
                ),
                "hrx_patch_count": sum(
                    len(value.get("patches", ())) for value in declarations
                ),
                "hrx_source_locations": ",".join(
                    sorted({value["source_location"] for value in declarations})
                ),
            },
        }

    return {
        "version": 0,
        "sha": sha,
        "ref": ref,
        "job": {
            "correlator": DETECTOR_NAME,
            "id": job_id,
            "html_url": job_url,
        },
        "detector": {
            "name": DETECTOR_NAME,
            "version": DETECTOR_VERSION,
            "url": "https://github.com/ROCm/hrx-system",
        },
        "scanned": scanned,
        "manifests": {
            "MODULE.bazel": {
                "name": "HRX Bazel module dependencies",
                "file": {"source_location": "MODULE.bazel"},
                "resolved": bazel_resolved,
            },
            "build_tools/third_party/deps.MODULE.bazel": {
                "name": "HRX source dependencies",
                "file": {
                    "source_location": ("build_tools/third_party/deps.MODULE.bazel")
                },
                "resolved": source_resolved,
            },
        },
    }
