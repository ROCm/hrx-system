# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declarative build and run requirements.

Requirement specs are Starlark values consumed by macros during BUILD loading.
`declare_requirements` emits matching queryable targets for audits and tooling.
"""

_INCOMPATIBLE_TARGET = [Label("@platforms//:incompatible")]

def _requirement_tags(kind, requirements):
    return [
        "iree-%s-requirement=%s" % (kind, requirement.id)
        for requirement in requirements
    ]

def _append(values, appended_values):
    if values == None:
        values = []
    return values + appended_values

def apply_build_requirements(kwargs, requirements):
    """Applies explicit build requirements to target keyword arguments.

    Args:
      kwargs: Rule or macro keyword arguments.
      requirements: Build requirement specs to apply.

    Returns:
      New keyword arguments with compatibility selectors and audit tags.
    """
    result = dict(kwargs)
    target_compatible_with = result.get("target_compatible_with")
    if target_compatible_with == None:
        target_compatible_with = []
    for requirement in requirements:
        if requirement.phase != "build":
            fail("requirement %s is not a build requirement" % requirement.id)
        target_compatible_with = target_compatible_with + select({
            requirement.enabled_by: [],
            "//conditions:default": _INCOMPATIBLE_TARGET,
        })
    result["target_compatible_with"] = target_compatible_with
    result["tags"] = _append(
        result.get("tags"),
        _requirement_tags("build", requirements),
    )
    return result

def apply_test_requirements(
        kwargs,
        build_requirements = [],
        run_requirements = [],
        resource_group = None):
    """Applies explicit build and runtime requirements to test arguments.

    Args:
      kwargs: Test rule or macro keyword arguments.
      build_requirements: Build requirement specs to apply.
      run_requirements: Runtime resource requirement specs to apply.
      resource_group: Optional local resource group for the test.

    Returns:
      New keyword arguments with compatibility, audit, and resource policy.
    """
    result = apply_build_requirements(kwargs, build_requirements)
    for requirement in run_requirements:
        if requirement.phase != "run":
            fail("requirement %s is not a run requirement" % requirement.id)
    result["tags"] = _append(
        result.get("tags"),
        _requirement_tags("run", run_requirements),
    )
    if resource_group and not result.get("resource_group"):
        result["resource_group"] = resource_group
    return result

IreeRequirementInfo = provider(
    doc = "Describes a semantic build or run requirement.",
    fields = {
        "cmake_condition": "CMake condition that enables a build requirement.",
        "cmake_label": "CTest/CMake label generated for a run requirement.",
        "enabled_by": "Bazel config_setting label enabling a build requirement.",
        "id": "Stable semantic requirement id.",
        "phase": "Requirement phase: build or run.",
        "skip_contract": "Runtime skip contract for a run requirement.",
    },
)

def build_requirement(id, label, enabled_by, cmake_condition):
    """Defines a build-time requirement spec.

    Args:
      id: Stable semantic requirement id.
      label: Queryable requirement target label.
      enabled_by: Bazel config_setting label that enables the requirement.
      cmake_condition: CMake condition that enables the requirement.

    Returns:
      Requirement spec consumed by package policy.
    """
    return struct(
        cmake_condition = cmake_condition,
        cmake_label = None,
        enabled_by = enabled_by,
        id = id,
        label = label,
        phase = "build",
        skip_contract = None,
    )

def run_requirement(id, label, cmake_label, skip_contract):
    """Defines a runtime-resource requirement spec.

    Args:
      id: Stable semantic requirement id.
      label: Queryable requirement target label.
      cmake_label: CTest/CMake label emitted for matching tests.
      skip_contract: Description of how tests behave when the resource is absent.

    Returns:
      Requirement spec consumed by package policy.
    """
    return struct(
        cmake_condition = None,
        cmake_label = cmake_label,
        enabled_by = None,
        id = id,
        label = label,
        phase = "run",
        skip_contract = skip_contract,
    )

def _requirement_target_impl(ctx):
    return [
        IreeRequirementInfo(
            cmake_condition = ctx.attr.cmake_condition,
            cmake_label = ctx.attr.cmake_label,
            enabled_by = ctx.attr.enabled_by,
            id = ctx.attr.requirement_id,
            phase = ctx.attr.phase,
            skip_contract = ctx.attr.skip_contract,
        ),
    ]

_requirement_target = rule(
    implementation = _requirement_target_impl,
    attrs = {
        "cmake_condition": attr.string(),
        "cmake_label": attr.string(),
        "enabled_by": attr.string(),
        "phase": attr.string(mandatory = True, values = ["build", "run"]),
        "requirement_id": attr.string(mandatory = True),
        "skip_contract": attr.string(),
    },
)

def _target_name(label):
    label = str(label)
    if ":" in label:
        return label.rsplit(":", 1)[1]
    return label.rsplit("/", 1)[1]

def declare_requirements(name, requirements):
    """Declares queryable requirement targets from requirement specs.

    Args:
      name: Logical group name for the declared requirements.
      requirements: Requirement specs to declare.
    """
    for requirement in requirements:
        _requirement_target(
            name = _target_name(requirement.label),
            cmake_condition = requirement.cmake_condition or "",
            cmake_label = requirement.cmake_label or "",
            enabled_by = str(requirement.enabled_by or ""),
            phase = requirement.phase,
            requirement_id = requirement.id,
            skip_contract = requirement.skip_contract or "",
        )
