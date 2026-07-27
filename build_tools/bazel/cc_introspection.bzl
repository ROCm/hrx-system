# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared C/C++ build introspection helpers for source-analysis tools."""

load("@rules_cc//cc:action_names.bzl", "CPP_COMPILE_ACTION_NAME", "C_COMPILE_ACTION_NAME")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

_C_SOURCE_EXTENSIONS = (".c",)
_CXX_SOURCE_EXTENSIONS = (".cc", ".cpp", ".cxx", ".c++", ".C")

def iree_cc_sanitize_label(label):
    """Returns a filesystem-safe identifier for a Bazel label.

    Args:
      label: Label to sanitize.

    Returns:
      Label text converted to a stable filesystem-safe identifier.
    """
    result = str(label)
    for old, new in [
        ("@", "external_"),
        ("//", ""),
        (":", "__"),
        ("/", "_"),
        ("+", "_"),
        ("-", "_"),
        (".", "_"),
    ]:
        result = result.replace(old, new)
    return result

def iree_cc_source_language(source):
    """Returns the source language for a C/C++ source file or None.

    Args:
      source: Source file to classify.

    Returns:
      "c", "c++", or None when the file extension is not a supported C/C++
      source extension.
    """
    if source.extension:
        extension = "." + source.extension
    else:
        extension = ""
    if extension in _C_SOURCE_EXTENSIONS:
        return "c"
    if extension in _CXX_SOURCE_EXTENSIONS:
        return "c++"
    return None

def iree_cc_is_main_workspace_source(source):
    """Returns true for source files owned by the main workspace.

    Args:
      source: Source file to inspect.

    Returns:
      True when the source is a checked-in main-workspace file.
    """
    return source.is_source and source.owner.workspace_name == ""

def iree_cc_source_files(ctx):
    """Returns main-workspace C/C++ source files from the current rule.

    Args:
      ctx: Aspect context whose rule attributes should be inspected.

    Returns:
      List of checked-in C/C++ source files from the rule's srcs.
    """
    if not hasattr(ctx.rule.attr, "srcs"):
        return []
    result = []
    for src_target in ctx.rule.attr.srcs:
        for src in src_target.files.to_list():
            if iree_cc_is_main_workspace_source(src) and iree_cc_source_language(src) != None:
                result.append(src)
    return result

def _string_list_attr(ctx, name):
    if not hasattr(ctx.rule.attr, name):
        return []
    value = getattr(ctx.rule.attr, name)
    if value == None:
        return []
    return value

def _user_compile_flags(ctx, language):
    flags = []
    flags.extend(_string_list_attr(ctx, "copts"))
    if language == "c":
        flags.extend(_string_list_attr(ctx, "conlyopts"))
    elif language == "c++":
        flags.extend(_string_list_attr(ctx, "cxxopts"))
    return flags

def _compile_action_name(language):
    if language == "c":
        return C_COMPILE_ACTION_NAME
    if language == "c++":
        return CPP_COMPILE_ACTION_NAME
    fail("unsupported source language: %s" % language)

def _compile_variables_extension(feature_configuration, compilation_context):
    variables = {}

    # Bazel/rules_cc currently exposes module-map data through private
    # compilation-context fields while public cc_common.create_compile_variables
    # still needs these variables when the toolchain enables module maps.
    module_maps_enabled = cc_common.is_enabled(
        feature_configuration = feature_configuration,
        feature_name = "module_maps",
    )
    layering_check_enabled = cc_common.is_enabled(
        feature_configuration = feature_configuration,
        feature_name = "layering_check",
    )
    if module_maps_enabled and hasattr(compilation_context, "_module_map"):
        module_map = compilation_context._module_map
        if module_map:
            variables["module_name"] = module_map.name
            variables["module_map_file"] = module_map.file.path

    if layering_check_enabled and hasattr(compilation_context, "_direct_module_maps"):
        variables["dependent_module_map_files"] = [
            module_map.path
            for module_map in compilation_context._direct_module_maps.to_list()
        ]

    return variables

def iree_cc_feature_configuration(ctx, cc_toolchain):
    """Returns the configured C/C++ feature set for the current rule.

    Args:
      ctx: Aspect context providing requested and disabled features.
      cc_toolchain: C/C++ toolchain for the current configuration.

    Returns:
      Feature configuration for compile-action introspection.
    """
    return cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

def iree_cc_compile_command(ctx, target, cc_toolchain, feature_configuration, source):
    """Returns compiler, language, and argument metadata for one source file.

    Args:
      ctx: Aspect context for the rule being inspected.
      target: Configured C/C++ target containing the source file.
      cc_toolchain: C/C++ toolchain for the current configuration.
      feature_configuration: Configured C/C++ feature set.
      source: Source file to model.

    Returns:
      Struct containing compiler, language, full arguments, and compiler
      arguments without the compiler executable.
    """
    language = iree_cc_source_language(source)
    action_name = _compile_action_name(language)
    compilation_context = target[CcInfo].compilation_context
    variables = cc_common.create_compile_variables(
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        source_file = source.path,
        user_compile_flags = _user_compile_flags(ctx, language),
        include_directories = compilation_context.includes,
        quote_include_directories = compilation_context.quote_includes,
        system_include_directories = compilation_context.system_includes,
        framework_include_directories = compilation_context.framework_includes,
        preprocessor_defines = depset(transitive = [
            compilation_context.defines,
            compilation_context.local_defines,
        ]),
        variables_extension = _compile_variables_extension(
            feature_configuration,
            compilation_context,
        ),
    )
    compiler = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = action_name,
    )
    compile_args = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = action_name,
        variables = variables,
    )
    return struct(
        arguments = [compiler] + compile_args,
        compiler = compiler,
        compile_args = compile_args,
        language = language,
    )
