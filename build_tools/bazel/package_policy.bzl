# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Prefix-based package policy for build and run requirements."""

_INCOMPATIBLE_TARGET = ["@platforms//:incompatible"]

def package_policy(
        packages,
        build_requirements = [],
        forbidden_deps = [],
        run_requirements = [],
        resource_group = None):
    """Defines requirements that apply to matching package prefixes.

    Args:
      packages: Exact package names or package prefixes ending in "/...".
      build_requirements: Build requirements applied to matching packages.
      forbidden_deps: Direct dependency labels or package prefixes ending in
        "/..." that matching packages must not reference.
      run_requirements: Run requirements applied to matching packages.
      resource_group: Optional local resource group for matching tests.

    Returns:
      Package policy consumed by project-specific macros.
    """
    return struct(
        build_requirements = build_requirements,
        forbidden_deps = forbidden_deps,
        packages = packages,
        resource_group = resource_group,
        run_requirements = run_requirements,
    )

def _matches(pattern, package_name):
    if pattern.endswith("/..."):
        prefix = pattern[:-4]
        return package_name == prefix or package_name.startswith(prefix + "/")
    return package_name == pattern

def _specificity(pattern):
    if pattern.endswith("/..."):
        return len(pattern) - 4
    return len(pattern)

def _append_unique_requirements(target, requirements):
    ids = {requirement.id: True for requirement in target}
    for requirement in requirements:
        if requirement.id not in ids:
            target.append(requirement)
            ids[requirement.id] = True

def _append_unique_strings(target, values):
    seen = {value: True for value in target}
    for value in values:
        if value not in seen:
            target.append(value)
            seen[value] = True

def collect_package_policy(package_name, policies):
    """Returns the merged policy for a package.

    Args:
      package_name: Package name to match against policy prefixes.
      policies: Package policies to merge.

    Returns:
      A merged policy struct with build_requirements, run_requirements, and
      resource_group fields.
    """
    build_requirements = []
    forbidden_deps = []
    run_requirements = []
    resource_group = None
    resource_group_specificity = -1
    for policy in policies:
        matching_patterns = [
            pattern
            for pattern in policy.packages
            if _matches(pattern, package_name)
        ]
        if not matching_patterns:
            continue
        _append_unique_requirements(build_requirements, policy.build_requirements)
        _append_unique_strings(forbidden_deps, policy.forbidden_deps)
        _append_unique_requirements(run_requirements, policy.run_requirements)
        if policy.resource_group:
            specificity = max([_specificity(pattern) for pattern in matching_patterns])
            if specificity >= resource_group_specificity:
                resource_group = policy.resource_group
                resource_group_specificity = specificity
    return struct(
        build_requirements = build_requirements,
        forbidden_deps = forbidden_deps,
        resource_group = resource_group,
        run_requirements = run_requirements,
    )

def _requirement_tags(kind, requirements):
    return [
        "iree-%s-requirement=%s" % (kind, requirement.id)
        for requirement in requirements
    ]

def _append(values, appended_values):
    if values == None:
        values = []
    return values + appended_values

def _canonical_label(label, package_name):
    if label.startswith(":"):
        return "//%s%s" % (package_name, label)
    if not label.startswith("//") or label.endswith("/..."):
        return label
    if ":" in label:
        return label
    package_path = label[2:]
    return "%s:%s" % (label, package_path.rsplit("/", 1)[-1])

def _label_package(canonical_label):
    if not canonical_label.startswith("//"):
        return None
    return canonical_label[2:].split(":", 1)[0]

def _dependency_matches_forbidden_pattern(dependency, pattern, package_name):
    canonical_dependency = _canonical_label(dependency, package_name)
    if pattern.endswith("/..."):
        dependency_package = _label_package(canonical_dependency)
        if dependency_package == None:
            return False
        return _matches(pattern[2:], dependency_package)
    return canonical_dependency == _canonical_label(pattern, package_name)

def _check_forbidden_deps(package_name, name, deps, forbidden_deps):
    if type(deps) != type([]):
        return
    for dependency in deps:
        if type(dependency) != type(""):
            continue
        for forbidden_dep in forbidden_deps:
            if _dependency_matches_forbidden_pattern(
                dependency,
                forbidden_dep,
                package_name,
            ):
                fail(
                    "%s may not depend on %s from package policy pattern %s" %
                    (name, dependency, forbidden_dep),
                )

def apply_target_policy(kwargs, policy, name = None):
    """Applies build requirements to a target kwargs dictionary.

    Args:
      kwargs: Rule or macro keyword arguments.
      policy: Merged package policy for the current package.
      name: Optional target name used in forbidden dependency diagnostics.

    Returns:
      New kwargs dictionary with target compatibility and audit tags applied.
    """
    result = dict(kwargs)
    package_name = native.package_name()
    for deps_attr in ["deps", "implementation_deps"]:
        _check_forbidden_deps(
            package_name,
            name or result.get("name", "<unnamed>"),
            result.get(deps_attr, []),
            policy.forbidden_deps,
        )
    target_compatible_with = result.get("target_compatible_with")
    if target_compatible_with == None:
        target_compatible_with = []
    for requirement in policy.build_requirements:
        target_compatible_with = target_compatible_with + select({
            requirement.enabled_by: [],
            "//conditions:default": _INCOMPATIBLE_TARGET,
        })
    result["target_compatible_with"] = target_compatible_with
    result["tags"] = _append(
        result.get("tags"),
        _requirement_tags("build", policy.build_requirements),
    )
    return result

def apply_test_policy(kwargs, policy, name = None):
    """Applies build and run requirements to a test-like target.

    Args:
      kwargs: Rule or macro keyword arguments.
      policy: Merged package policy for the current package.
      name: Optional target name used in forbidden dependency diagnostics.

    Returns:
      New kwargs dictionary with target compatibility, audit tags, and resource
      group applied.
    """
    result = apply_target_policy(kwargs, policy, name = name)
    result["tags"] = _append(
        result.get("tags"),
        _requirement_tags("run", policy.run_requirements),
    )
    if policy.resource_group and not result.get("resource_group"):
        result["resource_group"] = policy.resource_group
    return result
