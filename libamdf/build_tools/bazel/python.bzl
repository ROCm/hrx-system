# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python test declarations with libamdf's package and resource policy."""

load("//build_tools/bazel:python.bzl", "iree_py_test")
load("//libamdf/requirements:package_policy.bzl", "apply_amdf_test_policy")

def amdf_py_test(**kwargs):
    iree_py_test(**apply_amdf_test_policy(kwargs))
