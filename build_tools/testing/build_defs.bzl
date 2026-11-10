# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bazel rules for public command-line interface tests."""

load("@rules_python//python:defs.bzl", "py_test")
load("//build_tools/bazel:cc_attrs.bzl", "cc_attrs")
load("//build_tools/bazel:runfiles.bzl", "IreeRunfilesEnvironmentInfo")

def _tool_environment_impl(ctx):
    environments = {}
    for name, tool in zip(ctx.attr.names, ctx.attr.tools):
        environment = dict(tool[RunEnvironmentInfo].environment) if RunEnvironmentInfo in tool else {}
        if IreeRunfilesEnvironmentInfo in tool:
            environment.update(tool[IreeRunfilesEnvironmentInfo].environment)
        if environment:
            environments[name] = environment
    output = ctx.actions.declare_file(ctx.label.name + ".json")
    ctx.actions.write(output, json.encode(environments))
    return [DefaultInfo(files = depset([output]))]

_tool_environment = rule(
    implementation = _tool_environment_impl,
    attrs = {
        "names": attr.string_list(),
        "tools": attr.label_list(),
    },
)

def iree_cli_test(
        name,
        manifests,
        tools,
        data = None,
        args = None,
        resource_group = None,
        **kwargs):
    """Declares a test of a public command-line interface.

    Use this rule only for behavior introduced by process entry: critical
    public flag combinations and conflicts, process exit, process-owned output,
    output publication, and a minimal successful invocation of a materially
    distinct configured executable. Compiler, linker, transform, target,
    testbench, runtime, and structured-format semantics belong to their API or
    component owners. Expected results here should change only with the public
    CLI contract.

    Prefer iree_executable_test when successful process loading is the complete
    contract. CLI manifest cases should invoke one declared tool once against
    checked-in inputs.

    Args:
      name: Bazel target name.
      manifests: JSON manifest files to run.
      tools: Dictionary mapping manifest tool names to executable labels.
      data: Additional runtime data files made available to manifests.
      args: Additional runner arguments.
      resource_group: Local resource name used to serialize competing tests.
      **kwargs: Extra py_test attributes.
    """
    tags = kwargs.pop("tags", None)
    data = list(data or [])
    kwargs["tags"] = cc_attrs.with_resource_group_tags(tags, resource_group)
    args = list(args or [])
    tool_labels = []
    runner_args = []
    for manifest in manifests:
        runner_args.append("--manifest=$(rootpath %s)" % manifest)
    for tool_name, tool_label in sorted(tools.items()):
        tool_labels.append(tool_label)
        runner_args.append("--tool=%s=$(rootpath %s)" % (tool_name, tool_label))
    environment_name = name + "_tool_environment"
    _tool_environment(
        name = environment_name,
        names = sorted(tools),
        tools = [tools[name] for name in sorted(tools)],
        testonly = True,
        tags = ["manual"],
    )
    data.append(":" + environment_name)
    runner_args.append("--tool-environment=$(rootpath :%s)" % environment_name)
    py_test(
        name = name,
        srcs = [
            "//build_tools/testing:execution.py",
            "//build_tools/testing:execution_main.py",
        ],
        args = runner_args + args,
        data = list(manifests) + data + tool_labels,
        main = "//build_tools/testing:execution_main.py",
        **kwargs
    )
