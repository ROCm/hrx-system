# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configured dependency checks for repository invariants."""

_DependencyClosureInfo = provider(
    doc = "Configured target labels reachable from a dependency root.",
    fields = {
        "labels": "depset of configured target labels in the dependency closure.",
    },
)

def _collect_dependency_closure_aspect_impl(target, ctx):
    transitive_labels = []
    for attr_name in [
        "actual",
        "data",
        "deps",
        "exports",
        "implementation_deps",
        "runtime_deps",
    ]:
        if not hasattr(ctx.rule.attr, attr_name):
            continue
        attr_value = getattr(ctx.rule.attr, attr_name)
        if type(attr_value) == type([]):
            for dep in attr_value:
                if _DependencyClosureInfo in dep:
                    transitive_labels.append(dep[_DependencyClosureInfo].labels)
        elif attr_value and _DependencyClosureInfo in attr_value:
            transitive_labels.append(attr_value[_DependencyClosureInfo].labels)
    return [_DependencyClosureInfo(
        labels = depset(
            [str(target.label)],
            transitive = transitive_labels,
        ),
    )]

_collect_dependency_closure_aspect = aspect(
    implementation = _collect_dependency_closure_aspect_impl,
    attr_aspects = [
        "actual",
        "data",
        "deps",
        "exports",
        "implementation_deps",
        "runtime_deps",
    ],
)

def _label_is_in_subtree(label, prefix):
    if prefix.endswith("//"):
        return label.startswith(prefix)
    return label.startswith(prefix + ":") or label.startswith(prefix + "/")

def _assert_dependency_boundary_impl(ctx):
    forbidden_dependencies = {
        label: True
        for label in ctx.attr.forbidden_dependency_labels
    }
    allowed_dependencies = {
        label: True
        for label in ctx.attr.allowed_dependency_labels
    }
    violations = []
    for label in ctx.attr.target[_DependencyClosureInfo].labels.to_list():
        if label in forbidden_dependencies:
            violations.append(label)
            continue
        if label in allowed_dependencies:
            continue
        for prefix in ctx.attr.forbidden_dependency_subtree_prefixes:
            if _label_is_in_subtree(label, prefix):
                violations.append(label)
                break

    if not violations:
        return [AnalysisTestResultInfo(
            message = "",
            success = True,
        )]

    message = ctx.attr.message
    if not message:
        message = "%s depends on forbidden configured targets:\n  %s" % (
            ctx.attr.target.label,
            "\n  ".join(sorted(violations)),
        )
    return [AnalysisTestResultInfo(
        message = message,
        success = False,
    )]

def _canonical_label_string(label):
    if type(label) != type(""):
        label = str(label)
    if not (label.startswith(":") or label.startswith("//") or label.startswith("@")):
        label = ":" + label
    return str(Label(label))

def _canonical_subtree_prefix(subtree):
    if type(subtree) != type(""):
        subtree = str(subtree)
    if not subtree.endswith("/..."):
        fail("dependency subtree must end in '/...': %s" % subtree)
    if not (subtree.startswith("//") or subtree.startswith("@")):
        fail("dependency subtree must be repository-absolute: %s" % subtree)
    package = subtree[:-4]
    if package.endswith("//"):
        package = package[:-1]
    marker_label = str(Label(package + ":__iree_dependency_boundary__"))
    return marker_label[:-(len(":__iree_dependency_boundary__"))]

def iree_assert_dependency_boundary(
        name,
        target,
        forbidden_dependencies = None,
        forbidden_subtrees = None,
        allowed_dependencies = None,
        message = None,
        tags = None,
        **kwargs):
    """Asserts a configured transitive dependency boundary for `target`.

    This checks the configured dependency graph so optional `select()` branches
    do not force disabled providers to resolve their dependencies. Exact labels
    and `//package/...` subtrees are compared as strings instead of rule
    dependencies so checks can name labels that are expected to be absent from
    the configured repository.

    Exact allowed labels exempt subtree matches only. They cannot exempt an
    exact forbidden dependency. Keeping exceptions exact makes architectural
    concessions visible and prevents a package-shaped allowlist from silently
    expanding as targets are added.

    Args:
      name: Test target name.
      target: Label of the target whose transitive closure is checked.
      forbidden_dependencies: Exact labels forbidden in the closure.
      forbidden_subtrees: Repository-absolute `//package/...` patterns forbidden
        in the closure.
      allowed_dependencies: Exact labels allowed despite a subtree match.
      message: Optional diagnostic printed when the boundary is crossed.
      tags: Additional tags for the test target.
      **kwargs: Additional attributes forwarded to the analysis test target.
    """
    if forbidden_dependencies == None:
        forbidden_dependencies = []
    if forbidden_subtrees == None:
        forbidden_subtrees = []
    if allowed_dependencies == None:
        allowed_dependencies = []
    if tags == None:
        tags = []
    if not forbidden_dependencies and not forbidden_subtrees:
        fail("dependency boundary requires a forbidden dependency or subtree")

    forbidden_dependency_labels = [
        _canonical_label_string(dependency)
        for dependency in forbidden_dependencies
    ]
    forbidden_dependency_subtree_prefixes = [
        _canonical_subtree_prefix(subtree)
        for subtree in forbidden_subtrees
    ]
    allowed_dependency_labels = [
        _canonical_label_string(dependency)
        for dependency in allowed_dependencies
    ]
    for allowed_dependency in allowed_dependency_labels:
        if not any([
            _label_is_in_subtree(allowed_dependency, prefix)
            for prefix in forbidden_dependency_subtree_prefixes
        ]):
            fail("allowed dependency is outside every forbidden subtree: %s" % allowed_dependency)

    attr_values = {
        "allowed_dependency_labels": allowed_dependency_labels,
        "forbidden_dependency_labels": forbidden_dependency_labels,
        "forbidden_dependency_subtree_prefixes": forbidden_dependency_subtree_prefixes,
        "message": message or "",
        "tags": tags,
        "target": target,
    }
    attr_values.update(kwargs)

    testing.analysis_test(
        name,
        _assert_dependency_boundary_impl,
        attrs = {
            "allowed_dependency_labels": attr.string_list(),
            "forbidden_dependency_labels": attr.string_list(),
            "forbidden_dependency_subtree_prefixes": attr.string_list(),
            "message": attr.string(),
            "target": attr.label(
                aspects = [_collect_dependency_closure_aspect],
                mandatory = True,
            ),
        },
        attr_values = attr_values,
    )

def iree_assert_no_dependency(name, target, dependency, message = None, tags = None, **kwargs):
    """Asserts that `target` does not transitively depend on `dependency`.

    Args:
      name: Test target name.
      target: Label of the target whose transitive closure is checked.
      dependency: Exact label that must not appear in the closure.
      message: Optional diagnostic printed when the dependency is present.
      tags: Additional tags for the test target.
      **kwargs: Additional attributes forwarded to the analysis test target.
    """
    iree_assert_dependency_boundary(
        name = name,
        target = target,
        forbidden_dependencies = [dependency],
        message = message,
        tags = tags,
        **kwargs
    )
