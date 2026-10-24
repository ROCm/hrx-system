# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public native link requirements across shared implementation boundaries."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_cc//cc/common:cc_shared_library_info.bzl", "CcSharedLibraryInfo")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")
load("//build_tools/bazel:cc_shared_library.bzl", "iree_cc_shared_library")

def _public_requirements_impl(env, target):
    exported = target[CcSharedLibraryInfo].linker_input
    component = target[TestingAspectInfo].attrs.deps[0]
    public_inputs = component[CcInfo].linking_context.linker_inputs.to_list()
    env.expect.that_collection(exported.user_link_flags).contains_exactly([
        flag
        for linker_input in public_inputs
        for flag in linker_input.user_link_flags
    ])
    env.expect.that_collection(exported.additional_inputs).contains_exactly([
        file
        for linker_input in public_inputs
        for file in linker_input.additional_inputs
    ])
    env.expect.that_collection(exported.linkstamps).contains_exactly([])
    env.expect.that_collection([file.basename for file in exported.additional_inputs]).not_contains("cc_shared_exports.lds")
    env.expect.that_int(len(exported.libraries)).equals(1)
    env.expect.that_int(len(exported.additional_inputs)).equals(1)
    for flag in exported.user_link_flags:
        env.expect.that_bool("version-script" in flag).equals(False)

def _middle_link_impl(env, target):
    _public_requirements_impl(env, target)
    actions = target[TestingAspectInfo].actions
    env.expect.that_collection([action for action in actions if action.mnemonic == "CppLinkstampCompile"]).contains_exactly([])
    link = [action for action in actions if action.mnemonic == "CppLink"][0]
    for file in target[CcSharedLibraryInfo].linker_input.additional_inputs:
        env.expect.that_collection(link.inputs.to_list()).contains(file)
        env.expect.that_collection(link.argv).contains(file.path)

def cc_shared_library_test_suite(name):
    """Defines static and two-DSO consumers of a public inline native call.

    Args:
        name: Suite name and fixture prefix.
    """
    fixture = name + "_fixture"
    common = dict(testonly = True, tags = ["manual"])
    cc_library(name = fixture + "_headers", hdrs = ["cc_shared_fixture.h"], **common)
    cc_library(
        name = fixture + "_native",
        srcs = ["cc_shared_native.c"],
        features = ["-thin_lto", "-thin_lto_link"],
        deps = [":" + fixture + "_headers"],
        **common
    )
    native.filegroup(
        name = fixture + "_native_archive",
        srcs = [":" + fixture + "_native"],
        output_group = "archive",
        **common
    )
    cc_library(
        name = fixture + "_native_link",
        linkopts = ["$(location :" + fixture + "_native_archive)"],
        additional_linker_inputs = [":" + fixture + "_native_archive"],
        **common
    )
    cc_library(name = fixture + "_stamp", linkstamp = "cc_archive_stamp.cc", **common)
    cc_library(
        name = fixture + "_component",
        srcs = ["cc_shared_component.c"],
        deps = [
            ":" + fixture + "_headers",
            ":" + fixture + "_native_link",
            ":" + fixture + "_stamp",
        ],
        **common
    )
    cc_library(
        name = fixture + "_middle",
        srcs = ["cc_shared_middle.c"],
        deps = [":" + fixture + "_component", ":" + fixture + "_headers"],
        **common
    )
    for layer in ["component", "middle"]:
        iree_cc_shared_library(
            name = fixture + "_" + layer + "_shared",
            deps = [":" + fixture + "_" + layer],
            dynamic_deps = [] if layer == "component" else [":" + fixture + "_component_shared"],
            # Public header callers must retain the native archive; they cannot
            # reach its implementation through a private dependency's symbols.
            user_link_flags = ["-Wl,--version-script=$(location cc_shared_exports.lds)"],
            additional_linker_inputs = ["cc_shared_exports.lds"],
            **common
        )
    tests = []
    for layer, implementation in [("component", _public_requirements_impl), ("middle", _middle_link_impl)]:
        test_name = name + "_" + layer + "_metadata"
        analysis_test(
            name = test_name,
            target = ":" + fixture + "_" + layer + "_shared",
            impl = implementation,
            attr_values = {"target_compatible_with": ["@platforms//os:linux"]},
        )
        tests.append(":" + test_name)
    for linkage in ["static", "shared"]:
        test_name = name + "_" + linkage
        cc_test(
            name = test_name,
            srcs = ["cc_shared_test.c"],
            deps = [":" + fixture + "_component", ":" + fixture + "_middle", ":" + fixture + "_headers"],
            dynamic_deps = [] if linkage == "static" else [":" + fixture + "_component_shared", ":" + fixture + "_middle_shared"],
            target_compatible_with = ["@platforms//os:linux"],
            size = "small",
        )
        tests.append(":" + test_name)
    native.test_suite(name = name, tests = tests)
