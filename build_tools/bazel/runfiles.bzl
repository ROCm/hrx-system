# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configured runfile paths that survive host-side execution handoffs."""

IreeRunfilesArgumentsInfo = provider(
    doc = "Configured executable arguments with graph-proven runfile path spans.",
    fields = {
        "arguments": "Configured target argument tokens before path resolution.",
        "marked_arguments": "Argument tokens with runfile path spans delimited for host resolution.",
    },
)

IreeRunfilesEnvironmentInfo = provider(
    doc = "RunEnvironmentInfo entries whose values are rooted in target runfiles.",
    fields = {
        "environment": "Environment variable names mapped to their runfile targets.",
    },
)

RUNFILES_PATH_BEGIN = "__IREE_BAZEL_RUNFILE_PATH_BEGIN__"
RUNFILES_PATH_END = "__IREE_BAZEL_RUNFILE_PATH_END__"

_RUNFILES_LOCATION_KINDS = [
    "location",
    "locations",
    "rootpath",
    "rootpaths",
]

def _mark_runfiles_locations(argument):
    if RUNFILES_PATH_BEGIN in argument or RUNFILES_PATH_END in argument:
        fail("Executable argument contains a reserved IREE runfile path marker")

    result = ""
    cursor = 0
    done = False
    has_runfiles_paths = False
    for _ in range(len(argument) + 1):
        if not done:
            start = argument.find("$(", cursor)
            if start == -1:
                result += argument[cursor:]
                done = True
            else:
                end = argument.find(")", start + 2)
                if end == -1:
                    result += argument[cursor:]
                    done = True
                else:
                    token = argument[start:end + 1]
                    is_runfiles_path = False
                    for kind in _RUNFILES_LOCATION_KINDS:
                        if token.startswith("$(%s " % kind):
                            is_runfiles_path = True
                    result += argument[cursor:start]
                    if is_runfiles_path:
                        result += RUNFILES_PATH_BEGIN + token + RUNFILES_PATH_END
                        has_runfiles_paths = True
                    else:
                        result += token
                    cursor = end + 1
    return struct(
        argument = result,
        has_runfiles_paths = has_runfiles_paths,
    )

def create_runfiles_arguments_info(ctx, location_targets):
    """Creates configured runfile argument metadata for an executable rule.

    Args:
      ctx: Executable rule context whose args are being described.
      location_targets: Targets available to the rule's location expansion.

    Returns:
      IreeRunfilesArgumentsInfo when an argument contains a runfile location,
      otherwise None.
    """
    if not hasattr(ctx.attr, "args") or not ctx.attr.args:
        return None

    arguments = []
    marked_arguments = []
    has_runfiles_paths = False
    for argument in ctx.attr.args:
        marked = _mark_runfiles_locations(argument)
        configured_arguments = ctx.tokenize(
            ctx.expand_location(argument, targets = location_targets),
        )
        configured_marked_arguments = ctx.tokenize(
            ctx.expand_location(marked.argument, targets = location_targets),
        )
        if len(configured_arguments) != len(configured_marked_arguments):
            fail("Runfile markers changed executable argument tokenization")
        arguments.extend(configured_arguments)
        marked_arguments.extend(configured_marked_arguments)
        has_runfiles_paths = has_runfiles_paths or marked.has_runfiles_paths

    if not has_runfiles_paths:
        return None
    return IreeRunfilesArgumentsInfo(
        arguments = arguments,
        marked_arguments = marked_arguments,
    )
