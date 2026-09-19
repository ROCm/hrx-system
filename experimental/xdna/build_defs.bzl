# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build rules for the temporary libamdf XDNA consumers."""

load("//build_tools/bazel:package_policy.bzl", "apply_test_policy", "collect_package_policy")
load("//build_tools/bazel:requirements.bzl", "apply_build_requirements")
load("//build_tools/testing:build_defs.bzl", "iree_execution_test_suite")
load("//experimental/xdna/requirements:defs.bzl", "REQUIREMENTS")
load("//experimental/xdna/requirements:package_policy.bzl", "PACKAGE_POLICIES")
load("//runtime/build_tools/bazel:cc.bzl", "iree_runtime_cc_binary", "iree_runtime_cc_library")
load("//runtime/build_tools/bazel:cc_benchmark.bzl", "iree_runtime_cc_benchmark")
load("//runtime/build_tools/bazel:cc_test.bzl", "iree_runtime_cc_test")

def xdna_cc_library(name, **kwargs):
    iree_runtime_cc_library(name = name, **apply_build_requirements(kwargs, REQUIREMENTS))

def xdna_cc_binary(name, **kwargs):
    iree_runtime_cc_binary(name = name, **apply_build_requirements(kwargs, REQUIREMENTS))

def xdna_cc_test(name, **kwargs):
    policy = collect_package_policy(native.package_name(), PACKAGE_POLICIES)
    iree_runtime_cc_test(name = name, **apply_test_policy(kwargs, policy, name = name))

def xdna_cc_benchmark(name, **kwargs):
    policy = collect_package_policy(native.package_name(), PACKAGE_POLICIES)
    iree_runtime_cc_benchmark(name = name, **apply_test_policy(kwargs, policy, name = name))

def xdna_execution_test_suite(name, **kwargs):
    policy = collect_package_policy(native.package_name(), PACKAGE_POLICIES)
    iree_execution_test_suite(name = name, **apply_test_policy(kwargs, policy, name = name))
