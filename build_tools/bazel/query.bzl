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
    ],
)

def _assert_no_dependency_impl(ctx):
    dependency_label = ctx.attr.dependency_label
    target_labels = ctx.attr.target[_DependencyClosureInfo].labels.to_list()
    if dependency_label not in target_labels:
        return [AnalysisTestResultInfo(
            message = "",
            success = True,
        )]

    message = ctx.attr.message
    if not message:
        message = "%s depends on forbidden dependency %s" % (
            ctx.attr.target.label,
            dependency_label,
        )
    return [AnalysisTestResultInfo(
        message = message,
        success = False,
    )]

def _canonical_label_string(label):
    if type(label) != type(""):
        label = str(label)
    if label.startswith(":"):
        return "//%s%s" % (native.package_name(), label)
    if label.startswith("//") or label.startswith("@"):
        return label
    return "//%s:%s" % (native.package_name(), label)

def iree_assert_no_dependency(name, target, dependency, message = None, tags = None, **kwargs):
    """Asserts that `target` does not transitively depend on `dependency`.

    This checks the configured dependency graph so optional `select()` branches
    do not force disabled providers to resolve their dependencies. The forbidden
    dependency is compared by label string instead of as a rule dependency so
    checks can name labels that are expected to be absent from the configured
    repository.

    Args:
      name: Test target name.
      target: Label of the target whose transitive closure is checked.
      dependency: Label that must not appear in `target`'s transitive closure.
      message: Optional diagnostic printed when the dependency is present.
      tags: Additional tags for the test target.
      **kwargs: Additional attributes forwarded to the analysis test target.
    """
    if tags == None:
        tags = []

    attr_values = {
        "dependency_label": _canonical_label_string(dependency),
        "message": message or "",
        "tags": tags,
        "target": target,
    }
    attr_values.update(kwargs)

    testing.analysis_test(
        name,
        _assert_no_dependency_impl,
        attrs = {
            "dependency_label": attr.string(mandatory = True),
            "message": attr.string(),
            "target": attr.label(
                aspects = [_collect_dependency_closure_aspect],
                mandatory = True,
            ),
        },
        attr_values = attr_values,
    )
