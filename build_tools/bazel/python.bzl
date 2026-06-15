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
    _py_binary(**kwargs)

def iree_py_test(**kwargs):
    _py_test(**kwargs)
