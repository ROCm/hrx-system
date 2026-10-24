# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Archive metadata and actual native-library/linkstamp composition tests."""

load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_cc//cc/common:cc_shared_library_info.bzl", "CcSharedLibraryInfo")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test")
load("@rules_testing//lib:truth.bzl", "matching")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")
load("//build_tools/bazel:cc_shared_library.bzl", "iree_cc_shared_library")
load("//build_tools/bazel:cc_static_library.bzl", "iree_cc_static_library")

def _archive_metadata_impl(env, target):
    inputs = target[CcInfo].linking_context.linker_inputs.to_list()
    env.expect.that_int(len(inputs)).equals(1)
    archive_input = inputs[0]
    archive = archive_input.libraries[0]
    component = target[TestingAspectInfo].attrs.deps[0]
    originals = component[CcInfo].linking_context.linker_inputs.to_list()
    original = originals[0]
    selected = original.libraries[0]
    objects = selected.pic_objects or selected.objects
    env.expect.that_collection(archive.pic_objects or archive.objects).contains_exactly(objects)
    env.expect.that_collection(archive_input.user_link_flags).contains_exactly([flag for input in originals for flag in input.user_link_flags])
    env.expect.that_collection(archive_input.linkstamps).contains_exactly([stamp for input in originals for stamp in input.linkstamps])
    env.expect.that_collection(archive_input.additional_inputs).contains_exactly([file for input in originals for file in input.additional_inputs])
    env.expect.that_int(len(archive_input.linkstamps)).equals(1)

    # The precompiled native archive is retained, not folded or lost.
    env.expect.that_int(len(archive_input.libraries)).equals(2)
    env.expect.that_collection(target[CcInfo].compilation_context.headers.to_list()).contains_exactly([])
    env.expect.that_collection(target[DefaultInfo].default_runfiles.files.to_list()).contains_exactly(component[DefaultInfo].default_runfiles.files.to_list())
    action = [action for action in target[TestingAspectInfo].actions if action.mnemonic == "CppTransitiveArchive"][0]
    for object_file in objects:
        env.expect.that_collection(action.inputs.to_list()).contains(object_file)
    if selected.pic_objects:
        for object_file in selected.objects:
            env.expect.that_collection(action.inputs.to_list()).not_contains(object_file)
    if archive._pic_lto_compilation_context != selected._pic_lto_compilation_context:
        env.fail("archive changed producer ThinLTO bitcode/flags")
    if archive._pic_shared_non_lto_backends != selected._pic_shared_non_lto_backends:
        env.fail("archive changed producer backend identities")

def _archive_lto_metadata_impl(env, target):
    _archive_metadata_impl(env, target)
    archive = target[CcInfo].linking_context.linker_inputs.to_list()[0].libraries[0]
    env.expect.that_bool(bool(archive.pic_lto_bitcode_files)).equals(True)

def _shared_metadata_impl(env, target):
    exported = target[CcSharedLibraryInfo].linker_input
    env.expect.that_collection(exported.user_link_flags).contains("-pthread")
    env.expect.that_collection([file.basename for file in exported.additional_inputs]).contains("empty_suppressions.txt")
    env.expect.that_collection(exported.linkstamps).contains_exactly([])
    actions = target[TestingAspectInfo].actions
    stamps = [action for action in actions if action.mnemonic == "CppLinkstampCompile"]
    env.expect.that_int(len(stamps)).equals(1)
    link = [action for action in actions if action.mnemonic == "CppLink"][0]
    env.expect.that_collection(link.argv).contains("-pthread")
    env.expect.that_collection([file.basename for file in link.inputs.to_list()]).contains("empty_suppressions.txt")
    for action in stamps:
        for output in action.outputs.to_list():
            env.expect.that_collection(link.inputs.to_list()).contains(output)

def _duplicate_export_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(
        matching.contains("Trying to export a library already exported by a different shared library"),
    )

def cc_static_library_test_suite(name):
    """Registers real archive consumers plus producer-identity analysis checks.

    Args:
        name: Suite name and fixture prefix.
    """
    fixture = name + "_fixture"
    common = dict(testonly = True, tags = ["manual"])
    cc_library(name = fixture + "_headers", hdrs = ["cc_archive_fixture.h"], **common)
    cc_library(
        name = fixture + "_native",
        srcs = ["cc_archive_native.c"],
        # Model a precompiled native SDK archive. Bitcode imported without its
        # producer metadata cannot participate in Bazel's split backend graph.
        features = ["-thin_lto", "-thin_lto_link"],
        deps = [":" + fixture + "_headers"],
        **common
    )
    native.filegroup(
        name = fixture + "_native_file",
        srcs = [":" + fixture + "_native"],
        output_group = "archive",
        **common
    )
    cc_import(name = fixture + "_import", static_library = ":" + fixture + "_native_file", **common)
    cc_library(
        name = fixture + "_metadata",
        linkstamp = "cc_archive_stamp.cc",
        linkopts = ["-pthread"],
        additional_linker_inputs = ["empty_suppressions.txt"],
        **common
    )
    cc_library(
        name = fixture + "_component",
        srcs = ["cc_archive_component.c"],
        data = ["empty_suppressions.txt"],
        deps = [":" + fixture + "_headers", ":" + fixture + "_import", ":" + fixture + "_metadata"],
        **common
    )
    iree_cc_static_library(name = fixture, features = ["-symbol_check"], deps = [":" + fixture + "_component"], **common)
    iree_cc_shared_library(name = fixture + "_shared", deps = [":" + fixture], **common)
    iree_cc_shared_library(name = fixture + "_direct_shared", deps = [":" + fixture + "_component"], **common)
    iree_cc_shared_library(
        name = fixture + "_duplicate_export",
        deps = [":" + fixture],
        dynamic_deps = [":" + fixture + "_shared"],
        **common
    )
    tests = []
    for suffix, implementation, features in [
        ("metadata", _archive_metadata_impl, []),
        ("lto_metadata", _archive_lto_metadata_impl, ["thin_lto"]),
    ]:
        test_name = name + "_" + suffix
        analysis_test(
            name = test_name,
            target = ":" + fixture,
            impl = implementation,
            config_settings = {"//command_line_option:features": features},
            attr_values = {"target_compatible_with": ["@platforms//os:linux"]},
        )
        tests.append(":" + test_name)
    for linkage in ["static", "shared", "direct_shared"]:
        test_name = name + "_" + linkage
        cc_test(
            name = test_name,
            srcs = ["cc_archive_test.c"],
            deps = [":" + fixture + "_component" if linkage == "direct_shared" else ":" + fixture, ":" + fixture + "_headers"],
            dynamic_deps = [] if linkage == "static" else [":" + fixture + "_" + linkage],
            target_compatible_with = ["@platforms//os:linux"],
            size = "small",
        )
        tests.append(":" + test_name)
    for linkage in ["shared", "direct_shared"]:
        test_name = name + "_" + linkage + "_metadata"
        analysis_test(
            name = test_name,
            target = ":" + fixture + "_" + linkage,
            impl = _shared_metadata_impl,
            attr_values = {"target_compatible_with": ["@platforms//os:linux"]},
        )
        tests.append(":" + test_name)
    test_name = name + "_duplicate_export"
    analysis_test(
        name = test_name,
        target = ":" + fixture + "_duplicate_export",
        impl = _duplicate_export_impl,
        expect_failure = True,
        attr_values = {"target_compatible_with": ["@platforms//os:linux"]},
    )
    tests.append(":" + test_name)
    native.test_suite(name = name, tests = tests)
