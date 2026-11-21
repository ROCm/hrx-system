# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Execution admission from the same declarations used by package policies."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from build_tools.bazel_to_cmake.bazel_to_cmake_loads import ModuleLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENT_MODULES = (
    "build_tools/vulkan/requirements/defs.bzl",
    "build_tools/d3d12/requirements/defs.bzl",
    "libamdf/requirements/defs.bzl",
    "loom/requirements/defs.bzl",
    "runtime/requirements/defs.bzl",
)
TAG_PREFIX = "iree-run-requirement="


@lru_cache(maxsize=None)
def declared_ids(repo_root: Path = REPO_ROOT) -> frozenset[str]:
    loader = ModuleLoader(
        repo_root,
        bindings={
            "build_requirement": lambda **kwargs: None,
            "run_requirement": lambda id, **kwargs: id,
        },
    )
    return frozenset(
        str(requirement)
        for module in REQUIREMENT_MODULES
        for requirement in loader.load_file(repo_root / module)["REQUIREMENTS"]
        if requirement is not None
    )


def unavailable_ids(available: tuple[str, ...]) -> tuple[str, ...]:
    known = declared_ids()
    unknown = set(available) - known
    if unknown:
        raise ValueError("Unknown run requirements: " + ", ".join(sorted(unknown)))
    return tuple(sorted(known - set(available)))


def bazel_exclusions(available: tuple[str, ...]) -> tuple[str, ...]:
    return tuple("-" + TAG_PREFIX + value for value in unavailable_ids(available))


def ctest_exclusion_regex(available: tuple[str, ...]) -> str:
    unavailable = unavailable_ids(available)
    if not unavailable:
        return ""
    return "^" + TAG_PREFIX + "(" + "|".join(map(re.escape, unavailable)) + ")$"


def bazel_test_query(targets: tuple[str, ...]) -> str:
    # Preserve ordered target-pattern subtraction, including later re-inclusion.
    expression = "set()"
    for target in targets:
        excluded = target.startswith("-")
        pattern = target[1:] if excluded else target
        # Quoted query words preserve relative labels, wildcards, and separators.
        word = json.dumps(pattern)
        operation = "except" if excluded else "union"
        expression = f"({expression} {operation} tests(set({word})))"
    return expression


def bazel_audit_query_args(targets: tuple[str, ...]) -> tuple[str, ...]:
    # Select by namespace so the audit also sees empty and misspelled IDs.
    return (
        f'attr("tags", "{TAG_PREFIX}", {bazel_test_query(targets)})',
        "--output=streamed_jsonproto",
        "--proto:output_rule_attrs=tags",
        "--noproto:rule_inputs_and_outputs",
    )


def validate_tags(name: str, tags: list[str]) -> None:
    requirements = {
        tag.removeprefix(TAG_PREFIX) for tag in tags if tag.startswith(TAG_PREFIX)
    }
    unknown = requirements - declared_ids()
    if unknown:
        raise ValueError(
            f"{name}: unknown run requirements: " + ", ".join(sorted(unknown))
        )


def audit_bazel_json(payload: str) -> None:
    for line in payload.splitlines():
        rule = json.loads(line)["rule"]
        attributes = {attribute["name"]: attribute for attribute in rule["attribute"]}
        validate_tags(rule["name"], attributes["tags"]["stringListValue"])


def audit_ctest_json(payload: str) -> None:
    model = json.loads(payload)
    if model.get("kind") != "ctestInfo" or model.get("version", {}).get("major") != 1:
        raise ValueError("CTest did not return a version 1 ctestInfo document")
    for test in model["tests"]:
        for prop in test.get("properties", []):
            if prop["name"] == "LABELS":
                validate_tags(test["name"], prop["value"])
