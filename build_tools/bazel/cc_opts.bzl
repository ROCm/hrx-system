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
    "/Zc:__cplusplus",
]

_CLANG_CL_CXXOPTS = _MSVC_CXXOPTS + [
    "-Wno-invalid-offsetof",
]

# Sanitizer stack traces require frame pointers. Bazel sanitizer configs are
# shared across hosts, so select the spelling understood by the active compiler
# instead of leaking a Unix driver flag through a global --copt.
_SANITIZER_FRAME_POINTER_COPTS = select({
    "//build_tools/bazel:sanitizer_cc_compiler_clang_cl": ["/Oy-"],
    "//build_tools/bazel:sanitizer_cc_compiler_msvc": ["/Oy-"],
    "//build_tools/bazel:sanitizer_enabled": ["-fno-omit-frame-pointer"],
    "//conditions:default": [],
})

# Unix links AddressSanitizer through the compiler driver. Clang's C driver
# also needs the C++ runtime to intercept new/delete in C++ dependencies.
# Windows runtime dependencies come from the selected C/C++ runtimes toolchain.
_ADDRESS_SANITIZER_LINKOPTS = select({
    "//build_tools/bazel:address_sanitizer_cc_compiler_clang": [
        "-fsanitize=address",
        "-fsanitize-link-c++-runtime",
    ],
    "//build_tools/bazel:address_sanitizer_cc_compiler_gcc": ["-fsanitize=address"],
    "//conditions:default": [],
})

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

def _iree_cxx_standard_options(cxx_standard = "c++20"):
    """Returns compiler options selecting a first-party C++ language mode."""
    if cxx_standard not in ["c++17", "c++20", "c++23"]:
        fail("unsupported C++ standard: " + cxx_standard)

    windows_cxx_options = []
    windows_features = []

    # rules_cc owns the Windows C++17 default. A target selecting another
    # standard replaces that feature instead of overriding its command line.
    if cxx_standard != "c++17":
        windows_standard = "c++latest" if cxx_standard == "c++23" else cxx_standard
        windows_cxx_options.append("/std:" + windows_standard)
        windows_features.append("-default_cpp_std")
    return struct(
        features = _compiler_options([], [], windows_features, windows_features),
        cxxopts = _compiler_options(
            ["-std=" + cxx_standard],
            ["-std=" + cxx_standard],
            windows_cxx_options,
            windows_cxx_options,
        ),
    )

def _iree_code_compiler_options(
        copts = None,
        conlyopts = None,
        cxxopts = None,
        cxx_standard = "c++20",
        cxx_features = None,
        features = None):
    """Returns compiler options for first-party IREE C/C++ targets.

    Callers pass through target-specific options exactly as they would on a
    native C/C++ rule. This helper prepends IREE's compiler-conditioned policy
    while preserving configurable values such as `select()` expressions.
    Project wrappers select the C++ standard and optional exceptions/RTTI before
    flags are assembled. They forward the returned toolchain features along
    with the compiler options when selecting a non-default language mode.
    """
    if cxx_features == None:
        cxx_features = []
    for cxx_feature in cxx_features:
        if cxx_feature not in ["exceptions", "rtti"]:
            fail("unsupported C++ feature: " + cxx_feature)
    standard_options = _iree_cxx_standard_options(cxx_standard)
    unix_cxx_options = [
        "-fexceptions" if "exceptions" in cxx_features else "-fno-exceptions",
        "-frtti" if "rtti" in cxx_features else "-fno-rtti",
    ]
    windows_cxx_options = ["/GR" if "rtti" in cxx_features else "/GR-"]
    return struct(
        features = _append(
            features,
            standard_options.features,
        ),
        copts = _append(
            _compiler_options(
                _CLANG_COPTS,
                _GCC_COPTS,
                _CLANG_CL_COPTS,
                _MSVC_COPTS + _MSVC_ONLY_COPTS,
            ) + _SANITIZER_FRAME_POINTER_COPTS,
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
            standard_options.cxxopts + _compiler_options(
                unix_cxx_options + _CLANG_CXXOPTS,
                unix_cxx_options + _GCC_CXXOPTS,
                windows_cxx_options + _CLANG_CL_CXXOPTS,
                windows_cxx_options + _MSVC_CXXOPTS,
            ),
            cxxopts,
        ),
    )

def _iree_code_link_options(linkopts = None):
    """Returns linker options for first-party IREE executable targets."""
    return _append(_ADDRESS_SANITIZER_LINKOPTS, linkopts)

cc_opts = struct(
    iree_code_compiler_options = _iree_code_compiler_options,
    iree_code_link_options = _iree_code_link_options,
    iree_cxx_standard_options = _iree_cxx_standard_options,
)
