# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HRX-specific C/C++ Bazel macros."""

load("//build_tools/bazel:cc.bzl", "iree_cc_binary", "iree_cc_library")
load("//build_tools/bazel:cc_benchmark.bzl", "iree_cc_benchmark")
load("//build_tools/bazel:cc_test.bzl", "iree_cc_test")
load(":cc_attrs.bzl", "hrx_cc_attrs")

_GOOGLE_BENCHMARK_DEP = Label("//third_party:google_benchmark")

def _pop_hrx_compiler_options(kwargs):
    compiler_options = hrx_cc_attrs.with_hrx_compiler_options(
        copts = kwargs.pop("copts", None),
        conlyopts = kwargs.pop("conlyopts", None),
        cxxopts = kwargs.pop("cxxopts", None),
        features = kwargs.pop("features", None),
    )
    kwargs["copts"] = compiler_options.copts
    kwargs["conlyopts"] = compiler_options.conlyopts
    kwargs["cxxopts"] = compiler_options.cxxopts
    kwargs["features"] = compiler_options.features

def hrx_cc_library(name, deps = None, **kwargs):
    _pop_hrx_compiler_options(kwargs)
    iree_cc_library(
        name = name,
        deps = hrx_cc_attrs.with_hrx_deps(deps),
        **kwargs
    )

def hrx_cc_binary(name, deps = None, **kwargs):
    _pop_hrx_compiler_options(kwargs)
    iree_cc_binary(
        name = name,
        deps = hrx_cc_attrs.with_hrx_deps(deps),
        **kwargs
    )

def hrx_cc_test(name, deps = None, **kwargs):
    _pop_hrx_compiler_options(kwargs)
    iree_cc_test(
        name = name,
        deps = hrx_cc_attrs.with_hrx_deps(deps),
        **kwargs
    )

def hrx_cc_benchmark(name, deps = None, **kwargs):
    if deps == None:
        deps = []
    _pop_hrx_compiler_options(kwargs)
    iree_cc_benchmark(
        name = name,
        deps = hrx_cc_attrs.with_hrx_deps(deps + [_GOOGLE_BENCHMARK_DEP]),
        **kwargs
    )

def hrx_cc_shared_library(
        name,
        srcs = None,
        hdrs = None,
        textual_hdrs = None,
        deps = None,
        copts = None,
        **kwargs):
    """Builds an HRX shared-library artifact under Bazel.

    bazel_to_cmake has a matching handler that emits an iree_cc_library(SHARED)
    target from the same BUILD call.

    Args:
      name: Target name.
      srcs: C/C++ source files.
      hdrs: Public or private headers compiled into the shared library.
      textual_hdrs: Textual headers compiled into the shared library.
      deps: Additional target dependencies.
      copts: Additional C/C++ compiler options.
      **kwargs: Additional attributes forwarded to the shared C/C++ binary
        macro.
    """
    if srcs == None:
        srcs = []
    if hdrs == None:
        hdrs = []
    if textual_hdrs == None:
        textual_hdrs = []
    if copts == None:
        copts = []
    compiler_options = hrx_cc_attrs.with_hrx_compiler_options(
        copts = copts,
        conlyopts = kwargs.pop("conlyopts", None),
        cxxopts = kwargs.pop("cxxopts", None),
        features = kwargs.pop("features", None),
    )
    iree_cc_binary(
        name = name,
        srcs = srcs + hdrs + textual_hdrs,
        deps = hrx_cc_attrs.with_hrx_deps(deps),
        copts = compiler_options.copts,
        conlyopts = compiler_options.conlyopts,
        cxxopts = compiler_options.cxxopts,
        features = compiler_options.features,
        linkshared = True,
        **kwargs
    )
