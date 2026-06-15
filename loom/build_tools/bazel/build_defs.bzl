# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom-specific Bazel build macros.

The rules in this file are intentionally scoped to Loom instead of the shared
IREE build layer. They describe compact generated C tables derived from
checked-in Loom Python descriptions and, for some targets, fetched vendor
machine-readable data.
"""

load("//build_tools/bazel:cmake.bzl", _iree_cmake_extra_content = "iree_cmake_extra_content")
load("//build_tools/bazel:generate.bzl", "iree_generated_files")
load("//build_tools/bazel:select.bzl", _iree_select = "iree_select")
load(
    "//loom/requirements:package_policy.bzl",
    "apply_loom_target_policy",
)
load(":cc.bzl", _loom_cc_binary = "loom_cc_binary", _loom_cc_library = "loom_cc_library")
load(":cc_benchmark.bzl", _loom_cc_benchmark = "loom_cc_benchmark")
load(":cc_fuzz.bzl", _loom_cc_fuzz = "loom_cc_fuzz")
load(":cc_test.bzl", _loom_cc_test = "loom_cc_test")

_INCOMPATIBLE_TARGET = ["@platforms//:incompatible"]

iree_cmake_extra_content = _iree_cmake_extra_content
iree_select = _iree_select
loom_cc_benchmark = _loom_cc_benchmark
loom_cc_binary = _loom_cc_binary
loom_cc_fuzz = _loom_cc_fuzz
loom_cc_library = _loom_cc_library
loom_cc_test = _loom_cc_test

def _loom_bazel_generator_args(args):
    if type(args) != type([]):
        return args
    return [
        arg.replace("$(rootpath ", "$(location ").replace("$(execpath ", "$(location ")
        for arg in args
    ]

def loom_config_compatible_with(config_labels):
    """Returns target compatibility selectors for Loom configuration labels.

    Args:
      config_labels: Loom config setting or config setting group labels that
        must all be enabled for a target to be compatible.

    Returns:
      A target_compatible_with value that marks the target incompatible when any
      config label is disabled.
    """
    target_compatible_with = []
    for config_label in config_labels:
        target_compatible_with = target_compatible_with + select({
            config_label: [],
            "//conditions:default": _INCOMPATIBLE_TARGET,
        })
    return target_compatible_with

def _loom_output_basename(path):
    return path.split("/")[-1]

def _loom_output_arg(flag):
    if "{path}" in flag:
        return flag
    if flag.endswith("="):
        return flag + "{path}"
    if "=" in flag:
        return flag + "={path}"
    return flag

def _loom_output_args(flags, outputs):
    output_args = {}
    for i in range(len(outputs)):
        output_args[_loom_output_basename(outputs[i])] = _loom_output_arg(flags[i])
    return output_args

def loom_generated_textual_header(
        name,
        generator,
        output,
        output_flag,
        args = [],
        inputs = [],
        tags = [],
        visibility = None,
        comment = None):
    """Generates one textual header consumed by a Loom C/C++ target.

    Args:
      name: Generator action target name.
      generator: Executable label that writes the textual header.
      output: Generated textual header filename.
      output_flag: Generator flag paired with the output path.
      args: Generator arguments before the output flag.
      inputs: Source data labels consumed by the generator.
      tags: Additional Bazel tags for the generator action.
      visibility: Passed through to the generator action.
      comment: Optional progress message for the generator action.
    """

    rule_kwargs = {}
    if visibility != None:
        rule_kwargs["visibility"] = visibility
    if comment != None:
        rule_kwargs["progress_message"] = comment
    rule_kwargs["tags"] = tags + ["skip-bazel_to_cmake"]
    rule_kwargs = apply_loom_target_policy(rule_kwargs)

    iree_generated_files(
        name = name,
        srcs = inputs,
        outs = [output],
        args = _loom_bazel_generator_args(args),
        output_args = _loom_output_args([output_flag], [output]),
        tool = generator,
        **rule_kwargs
    )

def loom_low_descriptor_data_archive(
        name,
        repo_name,
        urls,
        sha256,
        **kwargs):
    """Declares a fetched data archive used by low descriptor generators.

    Bazel repository fetching is still owned by MODULE.bazel/module extensions.
    This declaration exists so bazel_to_cmake has the same target-local fetch
    metadata without each backend embedding CMake ExternalProject snippets.

    Args:
      name: Package-local archive name used for generated CMake targets.
      repo_name: Bazel external repository name without a leading '@'.
      urls: Archive URLs.
      sha256: Archive SHA-256 digest.
      **kwargs: Reserved for CMake-converted metadata. Ignored by Bazel.
    """

    # The Bazel repository rule is declared from MODULE.bazel; this macro is a
    # target-local CMake conversion contract.
    _ = (repo_name, urls, sha256, kwargs)  # @unused

def loom_target_table_cc_library(
        name,
        generator = None,
        source = None,
        header = None,
        generated_hdr_flags = [],
        generated_hdrs = [],
        header_only = False,
        args = [],
        inputs = [],
        deps = [],
        exclude_from_cmake_all = False,
        tags = [],
        testonly = False,
        visibility = None,
        **kwargs):
    """Generates a target-owned C/H table and wraps it in a runtime library.

    This is the common build-system contract for compact target-owned generated
    data. It deliberately abstracts only generation and library wiring. Target
    packages still own vendor schema parsing, overlay selection, semantic tests,
    registry composition, and which shards are selected by default.

    Args:
      name: Runtime C library target name.
      generator: Python executable label that writes C/H outputs. Bazel uses it
        as the generator tool; bazel_to_cmake resolves the Python target's main
        script and transitive sources for generated CMake dependencies.
      source: Generated C source filename. Defaults to <name>.c.
      header: Generated C header filename. Defaults to <name>.h.
      generated_hdr_flags: Generator flags paired with generated_hdrs. Each
        flag is emitted as <flag>=<path>, allowing selected view names such as
        --view-header=rdna4_gfx125x to prefix the generated path.
      generated_hdrs: Additional generated headers produced by this action for
        sibling header_only targets.
      header_only: Declares a generated-header-only library. The header must be
        produced by another target in the package, usually a family table action
        that also emits arch-overlay view headers.
      args: Generator arguments before the output flags. Arguments may use
        $(rootpath <label>) for declared inputs; the Bazel action rewrites
        those to $(execpath <label>) while CMake conversion maps them to source
        paths.
      inputs: Source/vendor data labels consumed by the generator.
      deps: Runtime C library dependencies.
      exclude_from_cmake_all: Excludes the generated C library from CMake all.
      tags: Additional Bazel tags for the internal generator action.
      testonly: Passed through to the runtime C library.
      visibility: Passed through to the generator action and runtime library.
      **kwargs: Additional arguments passed through to loom_cc_library.
    """
    if len(generated_hdr_flags) != len(generated_hdrs):
        fail("generated_hdr_flags and generated_hdrs must have the same length")
    _ = exclude_from_cmake_all  # @unused

    package_name = native.package_name()
    header = header or (name + ".h")
    generated_header = "//%s:%s" % (package_name, header)

    if header_only:
        if generator != None:
            fail("header-only target table %s must not specify generator" % name)
        if source != None:
            fail("header-only target table %s must not specify source" % name)
        if args:
            fail("header-only target table %s must not specify args" % name)
        if inputs:
            fail("header-only target table %s must not specify inputs" % name)
        if generated_hdrs:
            fail("header-only target table %s must not specify generated_hdrs" % name)
        loom_cc_library(
            name = name,
            hdrs = [generated_header],
            deps = deps,
            testonly = testonly,
            visibility = visibility,
            **kwargs
        )
        return

    if generator == None:
        fail("loom_target_table_cc_library %s requires generator" % name)

    rule_kwargs = {}
    if visibility != None:
        rule_kwargs["visibility"] = visibility
    rule_kwargs["tags"] = tags + ["skip-bazel_to_cmake"]
    rule_kwargs = apply_loom_target_policy(rule_kwargs)

    source = source or (name + ".c")

    # BUILD files use rootpath so the CMake converter can map external data
    # labels to their fetched source directories. The Bazel action expands the
    # same labels through declared inputs.
    bazel_args = _loom_bazel_generator_args(args)
    output_flags = ["--source", "--header"] + generated_hdr_flags
    outputs = [source, header] + generated_hdrs

    iree_generated_files(
        name = name + "_gen",
        srcs = inputs,
        outs = outputs,
        args = bazel_args,
        output_args = _loom_output_args(output_flags, outputs),
        tool = generator,
        **rule_kwargs
    )

    generated_source = "//%s:%s" % (package_name, source)
    loom_cc_library(
        name = name,
        srcs = [generated_source],
        hdrs = [generated_header],
        deps = deps,
        testonly = testonly,
        visibility = visibility,
        **kwargs
    )

def loom_generated_cc_library(
        name,
        generator,
        source = None,
        srcs = [],
        generated_src_flags = [],
        generated_srcs = [],
        hdrs = [],
        generated_hdr_flags = [],
        generated_hdrs = [],
        args = [],
        inputs = [],
        extra_output_flags = [],
        extra_outputs = [],
        deps = [],
        tags = [],
        testonly = False,
        visibility = None,
        **kwargs):
    """Generates C source tables and wraps them in a runtime library.

    This is the common build-system contract for generated C data. Generators
    receive --source=<path> for the primary C output by default, optional
    flag/output pairs for additional generated sources, and optional
    flag/output pairs for generated headers and sidecar generated artifacts.

    Args:
      name: Runtime C library target name.
      generator: Python executable label that writes the generated outputs.
      source: Generated C source filename. Defaults to <name>.c when no
        generated_srcs are specified.
      srcs: Checked-in C source filenames compiled into the same library.
      generated_src_flags: Generator flags paired with generated_srcs.
      generated_srcs: Additional generated C source filenames.
      hdrs: Checked-in public/private headers for the C library.
      generated_hdr_flags: Generator flags paired with generated_hdrs.
      generated_hdrs: Generated public/private headers for the C library.
      args: Generator arguments before the output flags.
      inputs: Source data labels consumed by the generator.
      extra_output_flags: Generator flags paired with extra_outputs.
      extra_outputs: Additional generated files produced by the action.
      deps: Runtime C library dependencies.
      tags: Additional Bazel tags for the internal generator action.
      testonly: Passed through to the runtime C library.
      visibility: Passed through to the generator action and runtime library.
      **kwargs: Additional arguments passed through to loom_cc_library.
    """
    if len(generated_src_flags) != len(generated_srcs):
        fail("generated_src_flags and generated_srcs must have the same length")
    if len(generated_hdr_flags) != len(generated_hdrs):
        fail("generated_hdr_flags and generated_hdrs must have the same length")
    if len(extra_output_flags) != len(extra_outputs):
        fail("extra_output_flags and extra_outputs must have the same length")

    rule_kwargs = {}
    if visibility != None:
        rule_kwargs["visibility"] = visibility
    rule_kwargs["tags"] = tags + ["skip-bazel_to_cmake"]
    rule_kwargs = apply_loom_target_policy(rule_kwargs)

    if source == None and not generated_srcs:
        source = name + ".c"

    source_flags = []
    sources = []
    if source != None:
        source_flags.append("--source")
        sources.append(source)
    source_flags += generated_src_flags
    sources += generated_srcs
    if not sources:
        fail("loom_generated_cc_library requires at least one generated source")

    bazel_args = _loom_bazel_generator_args(args)
    output_flags = source_flags + generated_hdr_flags + extra_output_flags
    outputs = sources + generated_hdrs + extra_outputs

    iree_generated_files(
        name = name + "_gen",
        srcs = inputs,
        outs = outputs,
        args = bazel_args,
        output_args = _loom_output_args(output_flags, outputs),
        tool = generator,
        **rule_kwargs
    )

    package_name = native.package_name()
    generated_sources = [
        "//%s:%s" % (package_name, source)
        for source in sources
    ]
    generated_headers = [
        "//%s:%s" % (package_name, header)
        for header in generated_hdrs
    ]
    loom_cc_library(
        name = name,
        srcs = srcs + generated_sources,
        hdrs = hdrs + generated_headers,
        deps = deps,
        testonly = testonly,
        visibility = visibility,
        **kwargs
    )

def loom_low_descriptor_cc_library(
        name,
        generator = None,
        source = None,
        header = None,
        generated_hdr_flags = [],
        generated_hdrs = [],
        header_only = False,
        args = [],
        inputs = [],
        deps = [],
        ids_deps = None,
        ids_visibility = None,
        exclude_from_cmake_all = False,
        tags = [],
        testonly = False,
        visibility = None,
        **kwargs):
    """Generates a low descriptor C/H shard and wraps it in a runtime library.

    Args:
      name: Runtime C library target name.
      generator: Python executable label that writes C/H outputs.
      source: Generated C source filename. Defaults to <name>.c for generated
        targets.
      header: Generated C header filename. Defaults to <name>.h.
      generated_hdr_flags: Generator flags paired with generated_hdrs.
      generated_hdrs: Additional generated headers produced by this action for
        sibling header_only targets.
      header_only: Declares a generated-header-only library.
      args: Generator arguments before the output flags.
      inputs: Source/vendor data labels consumed by the generator.
      deps: Runtime C library dependencies.
      ids_deps: Runtime C library dependencies for the generated _ids target.
        Defaults to deps.
      ids_visibility: Visibility for the generated _ids target. Defaults to
        visibility.
      exclude_from_cmake_all: Excludes the generated C library from CMake all.
      tags: Additional Bazel tags for the internal generator action.
      testonly: Passed through to the runtime C libraries.
      visibility: Passed through to the generator action and runtime libraries.
      **kwargs: Additional arguments passed through to loom_cc_library.
    """
    if not header_only:
        source = source or (name + ".c")
    header = header or (name + ".h")
    loom_target_table_cc_library(
        name = name,
        generator = generator,
        source = source,
        header = header,
        generated_hdr_flags = generated_hdr_flags,
        generated_hdrs = generated_hdrs,
        header_only = header_only,
        args = args,
        inputs = inputs,
        deps = deps,
        exclude_from_cmake_all = exclude_from_cmake_all,
        tags = tags,
        testonly = testonly,
        visibility = visibility,
        **kwargs
    )

    package_name = native.package_name()
    generated_header = "//%s:%s" % (package_name, header)
    loom_cc_library(
        name = name + "_ids",
        hdrs = [generated_header],
        deps = ids_deps if ids_deps != None else deps,
        testonly = testonly,
        visibility = ids_visibility if ids_visibility != None else visibility,
    )

def loom_low_descriptor_exclude_from_cmake_all(
        cc_libraries = [],
        targets = []):
    """Excludes descriptor-dependent CMake targets from the default all target.

    Descriptor registries and backend-specific tests often depend on large
    target tables. Keeping those targets declared but outside CMake all lets
    explicit builds and tests work without making every default build fetch and
    compile every backend descriptor package.

    Args:
      cc_libraries: iree_cc_library names in the current package.
      targets: Other CMake target names in the current package, usually tests.
    """
    _ = (cc_libraries, targets)  # @unused
