# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public Bazel API for Loom source repositories."""

load(
    ":loom_binary.bzl",
    _LoomBinaryInfo = "LoomBinaryInfo",
    _loom_kernel_binary = "loom_kernel_binary",
)
load(
    ":loom_corpus_build.bzl",
    _LoomCorpusBuildInfo = "LoomCorpusBuildInfo",
    _loom_corpus_build = "loom_corpus_build",
)
load(
    ":loom_corpus_catalog.bzl",
    _loom_corpus_catalog = "loom_corpus_catalog",
    _loom_corpus_manifest = "loom_corpus_manifest",
    _loom_corpus_sources = "loom_corpus_sources",
    _loom_legacy_case_corpus = "loom_legacy_case_corpus",
    _loom_scenario_corpus = "loom_scenario_corpus",
)
load(
    ":loom_corpus_execution.bzl",
    _loom_corpus_test = "loom_corpus_test",
    _loom_legacy_case_test = "loom_legacy_case_test",
    _loom_scenario_test = "loom_scenario_test",
)
load(
    ":loom_library.bzl",
    _LoomExecutionTestInfo = "LoomExecutionTestInfo",
    _LoomLibraryInfo = "LoomLibraryInfo",
    _loom_execution_profile = "loom_execution_profile",
    _loom_kernel_library = "loom_kernel_library",
    _loom_library = "loom_library",
    _loom_test = "loom_test",
    _loom_test_module = "loom_test_module",
)
load(
    ":loom_module.bzl",
    _loom_module = "loom_module",
)
load(
    ":loom_target_profile.bzl",
    _LoomTargetProfileInfo = "LoomTargetProfileInfo",
    _LoomTargetSetInfo = "LoomTargetSetInfo",
    _loom_amdgpu_target_profile = "loom_amdgpu_target_profile",
    _loom_target_profile = "loom_target_profile",
    _loom_target_set = "loom_target_set",
)
load(
    ":loom_toolchain.bzl",
    _loom_tools_toolchains = "loom_tools_toolchains",
)

LoomBinaryInfo = _LoomBinaryInfo
LoomCorpusBuildInfo = _LoomCorpusBuildInfo
LoomExecutionTestInfo = _LoomExecutionTestInfo
LoomLibraryInfo = _LoomLibraryInfo
LoomTargetProfileInfo = _LoomTargetProfileInfo
LoomTargetSetInfo = _LoomTargetSetInfo
loom_amdgpu_target_profile = _loom_amdgpu_target_profile
loom_corpus_build = _loom_corpus_build
loom_corpus_catalog = _loom_corpus_catalog
loom_legacy_case_corpus = _loom_legacy_case_corpus
loom_corpus_manifest = _loom_corpus_manifest
loom_corpus_sources = _loom_corpus_sources
loom_corpus_test = _loom_corpus_test
loom_legacy_case_test = _loom_legacy_case_test
loom_scenario_corpus = _loom_scenario_corpus
loom_scenario_test = _loom_scenario_test
loom_execution_profile = _loom_execution_profile
loom_kernel_binary = _loom_kernel_binary
loom_kernel_library = _loom_kernel_library
loom_library = _loom_library
loom_module = _loom_module
loom_target_profile = _loom_target_profile
loom_target_set = _loom_target_set
loom_test = _loom_test
loom_test_module = _loom_test_module
loom_tools_toolchains = _loom_tools_toolchains
