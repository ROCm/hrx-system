# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sanitizer suppression sets and Bazel test helpers."""

rocm_suppressions = {
    "lsan": "//build_tools/sanitizer:lsan_suppressions_rocm.txt",
}

vulkan_suppressions = {
    "lsan": "//build_tools/sanitizer:lsan_suppressions_vulkan.txt",
}

_SANITIZER_OPTION_ENV = {
    "asan": "ASAN_OPTIONS",
    "lsan": "LSAN_OPTIONS",
    "msan": "MSAN_OPTIONS",
    "tsan": "TSAN_OPTIONS",
    "ubsan": "UBSAN_OPTIONS",
}

_SANITIZER_EXTRA_OPTIONS = {
    # LSAN suppression matching depends on symbolized stack frames. Bazel test
    # sandboxes may not expose llvm-symbolizer on PATH, but addr2line is
    # available in normal GCC/Clang development environments and is enough for
    # suppressions keyed to system functions such as pthread_once.
    "lsan": ["allow_addr2line=1"],
}

def iree_sanitizer_suppression_files(sanitizer_suppressions):
    """Returns suppression file labels from a sanitizer suppression map.

    Args:
      sanitizer_suppressions: Dictionary from sanitizer name to suppression
        file label.

    Returns:
      List of suppression file labels.
    """
    return [
        label
        for _, label in sorted((sanitizer_suppressions or {}).items())
    ]

def iree_sanitizer_suppression_data(data, sanitizer_suppressions):
    """Adds sanitizer suppression files to a Bazel test data list.

    Args:
      data: Existing Bazel test data list.
      sanitizer_suppressions: Dictionary from sanitizer name to suppression
        file label.

    Returns:
      Test data list containing the original data plus suppression files.
    """
    suppression_files = iree_sanitizer_suppression_files(
        sanitizer_suppressions,
    )
    if not suppression_files:
        return data
    if data == None:
        return suppression_files
    return data + suppression_files

def iree_sanitizer_suppression_env(env, sanitizer_suppressions):
    """Adds sanitizer suppression environment variables to a Bazel test env.

    Args:
      env: Existing Bazel test environment dictionary.
      sanitizer_suppressions: Dictionary from sanitizer name to suppression
        file label.

    Returns:
      Test environment with sanitizer suppression variables.
    """
    if not sanitizer_suppressions:
        return env
    result = dict(env or {})
    for sanitizer, label in sorted((sanitizer_suppressions or {}).items()):
        option_env = _SANITIZER_OPTION_ENV.get(sanitizer)
        if not option_env:
            fail("unknown sanitizer suppression kind: %s" % sanitizer)
        if option_env in result:
            fail(
                "%s is set directly and by sanitizer_suppressions" %
                option_env,
            )
        options = ["suppressions=$(location %s)" % label]
        options.extend(_SANITIZER_EXTRA_OPTIONS.get(sanitizer, []))
        result[option_env] = ":".join(options)
    return result
