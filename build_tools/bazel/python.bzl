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

def iree_py_library(**kwargs):
    _py_library(**kwargs)

def iree_py_binary(**kwargs):
    if "python_version" not in kwargs:
        kwargs["python_version"] = "3.12"
    _py_binary(**kwargs)

def iree_py_test(**kwargs):
    if "python_version" not in kwargs:
        kwargs["python_version"] = "3.12"
    _py_test(**kwargs)
