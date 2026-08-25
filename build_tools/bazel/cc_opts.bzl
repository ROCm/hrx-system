# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared C/C++ compiler option policy for IREE-owned code.

These options are private implementation policy for targets using IREE's C/C++
macro layer. They must not be attached globally in `.bazelrc`, because doing so
would leak first-party warning and ABI decisions into external repositories,
generator tools, and embedders' own code.
"""

_CLANG_COPTS = [
    "-Werror",
    "-Wno-error=deprecated-declarations",
    "-Wno-error=unused-command-line-argument",
    "-fno-lax-vector-conversions",
    "-Wall",
    "-Wno-extern-c-compat",
    "-Wno-unused-const-variable",
    "-Wno-unused-function",
    "-Wno-pointer-sign",
    "-Wno-char-subscripts",
    "-Wfloat-overflow-conversion",
    "-Wfloat-zero-conversion",
    "-Wfor-loop-analysis",
    "-Wformat-security",
    "-Wgnu-redeclared-enum",
    "-Wimplicit-fallthrough",
    "-Winfinite-recursion",
    "-Wliteral-conversion",
    "-Wlogical-op-parentheses",
    "-Wpointer-arith",
    "-Wself-assign",
    "-Wstring-conversion",
    "-Wtautological-overlap-compare",
    "-Wthread-safety",
    "-Wthread-safety-beta",
    "-Wunused-comparison",
    "-Wvla",
]

_CLANG_CONLYOPTS = []

_CLANG_CXXOPTS = [
    "-std=c++17",
    "-fno-exceptions",
    "-fno-rtti",
    "-Wno-c++20-extensions",
    "-Wno-ambiguous-member-template",
    "-Wno-invalid-offsetof",
    "-Wno-unused-lambda-capture",
    "-Wno-unused-private-field",
    "-Wctad-maybe-unsupported",
    "-Wnon-virtual-dtor",
    "-Woverloaded-virtual",
]

_GCC_COPTS = [
    "-Werror",
    "-Wno-error=deprecated-declarations",
    "-Wall",
    "-Wno-address",
    "-Wno-address-of-packed-member",
    "-Wno-comment",
    "-Wno-format-zero-length",
    "-Wno-misleading-indentation",
    "-Wno-sign-compare",
    "-Wno-unknown-pragmas",
    "-Wno-uninitialized",
    "-Wno-unused-but-set-variable",
    "-Wno-unused-function",
]

_GCC_CONLYOPTS = [
    "-Wno-pointer-sign",
]

_GCC_CXXOPTS = [
    "-std=c++17",
    "-fno-exceptions",
    "-fno-rtti",
    "-Wno-invalid-offsetof",
    "-Wno-overloaded-virtual",
]

_MSVC_COPTS = [
    "/W3",
    "/WX",
    "/utf-8",
    "/DWIN32_LEAN_AND_MEAN",
    "/DNOMINMAX",
    "/D_USE_MATH_DEFINES",
    "/D_CRT_SECURE_NO_WARNINGS",
    "/D_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES",
    "/EHsc",
    "/bigobj",
    "/wd4200",  # Nonstandard extension used: zero-sized array in struct/union.
    "/wd4005",  # Macro redefinition.
    "/wd4018",  # Signed/unsigned mismatch.
    "/wd4065",  # Switch statement contains 'default' but no 'case' labels.
    "/wd4141",  # Inline used more than once.
    "/wd4146",  # Unary minus operator applied to unsigned type.
    "/wd4244",  # Possible loss of data.
    "/wd4267",  # Possible loss of data converting from size_t.
    "/wd4505",  # Unreferenced local function removed.
    "/wd4576",  # Non-standard explicit type conversion syntax.
    "/wd4624",  # Destructor was implicitly defined as deleted.
    "/wd5105",  # Macro expansion producing 'defined' has undefined behavior.
]

_CLANG_CL_COPTS = _MSVC_COPTS + [
    "-Wno-unused-function",
    "-Wno-unused-lambda-capture",
    "-Wno-unused-variable",
]

_MSVC_ONLY_COPTS = [
    # Enable the standards-conforming preprocessor required by __VA_OPT__.
    # clang-cl is already conforming and diagnoses this option as unused.
    "/Zc:preprocessor",
]

# The rules_cc Windows toolchain enables its standards-conforming C17 mode.
# Do not append another /std option: MSVC warns whenever the later option
# overrides the toolchain default, and C17 preserves the C11 language surface
# required by IREE code.
_MSVC_CONLYOPTS = []

_MSVC_CXXOPTS = [
    "/GR-",
    "/std:c++17",
    "/Zc:__cplusplus",
]

_CLANG_CL_CXXOPTS = _MSVC_CXXOPTS + [
    "-Wno-invalid-offsetof",
]

def _append(values, appended_values):
    if values == None:
        values = []
    if appended_values == None:
        appended_values = []
    return values + appended_values

def _compiler_options(
        clang_options,
        gcc_options,
        clang_cl_options,
        msvc_options):
    return select({
        "//build_tools/bazel:cc_compiler_clang": clang_options,
        "//build_tools/bazel:cc_compiler_clang_cl": clang_cl_options,
        "//build_tools/bazel:cc_compiler_gcc": gcc_options,
        "//build_tools/bazel:cc_compiler_msvc": msvc_options,
        "//conditions:default": clang_options,
    })

def _iree_code_compiler_options(
        copts = None,
        conlyopts = None,
        cxxopts = None):
    """Returns compiler options for first-party IREE C/C++ targets.

    Callers pass through target-specific options exactly as they would on a
    native C/C++ rule. This helper prepends IREE's compiler-conditioned policy
    while preserving configurable values such as `select()` expressions.
    """
    return struct(
        copts = _append(
            _compiler_options(
                _CLANG_COPTS,
                _GCC_COPTS,
                _CLANG_CL_COPTS,
                _MSVC_COPTS + _MSVC_ONLY_COPTS,
            ),
            copts,
        ),
        conlyopts = _append(
            _compiler_options(
                _CLANG_CONLYOPTS,
                _GCC_CONLYOPTS,
                _MSVC_CONLYOPTS,
                _MSVC_CONLYOPTS,
            ),
            conlyopts,
        ),
        cxxopts = _append(
            _compiler_options(
                _CLANG_CXXOPTS,
                _GCC_CXXOPTS,
                _CLANG_CL_CXXOPTS,
                _MSVC_CXXOPTS,
            ),
            cxxopts,
        ),
    )

cc_opts = struct(
    iree_code_compiler_options = _iree_code_compiler_options,
)
