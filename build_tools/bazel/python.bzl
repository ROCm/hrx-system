# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared Python Bazel rules for IREE repositories."""

load(
    "@rules_python//python:defs.bzl",
    _py_binary = "py_binary",
    _py_library = "py_library",
    _py_test = "py_test",
)
load(
    "//build_tools/bazel:test_resources.bzl",
    _with_resource_group_tags = "with_resource_group_tags",
)

def iree_py_library(**kwargs):
    _py_library(**kwargs)

def iree_py_binary(**kwargs):
    if "python_version" not in kwargs:
        kwargs["python_version"] = "3.12"
    _py_binary(**kwargs)

def iree_py_test(resource_group = None, **kwargs):
    """Defines a Python test with the shared execution-resource policy.

    Args:
      resource_group: Local resource name used to serialize competing tests.
      **kwargs: Attributes forwarded to rules_python's py_test.
    """
    if "python_version" not in kwargs:
        kwargs["python_version"] = "3.12"
    kwargs["deps"] = kwargs.get("deps", []) + [
        Label("//build_tools/bazel:python_test_environment"),
    ]
    kwargs["tags"] = _with_resource_group_tags(
        kwargs.get("tags"),
        resource_group,
    )
    _py_test(**kwargs)
