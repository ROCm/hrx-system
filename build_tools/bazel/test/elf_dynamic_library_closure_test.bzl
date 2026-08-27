# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for relocatable ELF dynamic-library closure resolution."""

load(
    "@bazel_skylib//lib:unittest.bzl",
    "analysistest",
    "asserts",
)
load(
    "//build_tools/bazel:elf_dynamic_library_closure.bzl",
    "resolve_elf_dynamic_library_closure",
    "try_resolve_elf_dynamic_library_closure",
)

_ElfClosureInfo = provider(
    doc = "Resolved ELF dynamic-library closure under test.",
    fields = {
        "error": "Non-fatal resolution error.",
        "libraries": "Resolved library map.",
    },
)

_EXPECTED_LOADER_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "LD_AUDIT": "",
    "LD_LIBRARY_PATH": "/sdk/lib:/sdk/sysdeps/lib",
    "LD_PRELOAD": "",
}

def _fake_which(name):
    if name == "ldd":
        return "/usr/bin/ldd"
    return None

def _missing_ldd_which(name):
    if name != "ldd":
        fail("unexpected executable lookup: {}".format(name))
    return None

def _fake_watch(path):
    return path

def _fake_path(path):
    path = str(path)
    canonical_paths = {
        "/fixtures/alias_a/libfixture.so": "/fixtures/runtime/libfixture.so",
        "/fixtures/alias_b/libfixture.so": "/fixtures/runtime/libfixture.so",
        "/fixtures/tool_a": "/fixtures/tool_a.real",
        "/usr/bin/ldd": "/usr/bin/ldd.real",
    }
    canonical_path = canonical_paths.get(path, path)
    return struct(
        exists = not path.startswith("/missing/"),
        realpath = canonical_path,
    )

def _execute_result(stdout = "", stderr = "", return_code = 0):
    return struct(
        return_code = return_code,
        stderr = stderr,
        stdout = stdout,
    )

def _success_execute(args, environment = None, quiet = False):
    if environment != _EXPECTED_LOADER_ENVIRONMENT:
        fail("unexpected loader environment: {}".format(environment))
    if not quiet:
        fail("ELF dependency inspection must be quiet")
    if args[0] != "/usr/bin/ldd.real":
        fail("unexpected ldd path: {}".format(args[0]))
    if args[1] == "/fixtures/tool_a.real":
        return _execute_result(stdout = """
linux-vdso.so.1 (0x00007ffff7fc9000)
libfixture.so => /fixtures/alias_a/libfixture.so (0x00007ffff7000000)
libstdc++.so.6 => /sdk/sysdeps/lib/libstdc++.so.6 (0x00007ffff6800000)
libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007ffff6700000)
libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007ffff6500000)
/lib64/ld-linux-x86-64.so.2 (0x00007ffff7fcb000)
""")
    if args[1] == "/fixtures/tool_b":
        return _execute_result(stdout = """
libfixture.so => /fixtures/alias_b/libfixture.so (0x00007ffff7000000)
libz.so.1 => /sdk/sysdeps/lib/libz.so.1 (0x00007ffff6800000)
""")
    if args[1] == "/fixtures/static_tool":
        return _execute_result(
            return_code = 1,
            stdout = "statically linked\n",
        )
    fail("unexpected artifact: {}".format(args[1]))

def _missing_library_execute(args, environment = None, quiet = False):
    if len(args) != 2 or environment == None or not quiet:
        fail("unexpected ldd invocation")
    return _execute_result(
        stdout = "libmissing.so => not found\n",
    )

def _inconsistent_library_execute(args, environment = None, quiet = False):
    if environment == None or not quiet:
        fail("unexpected ldd invocation")
    library_path = "/fixtures/distinct/a/libfixture.so"
    if args[1] == "/fixtures/tool_b":
        library_path = "/fixtures/distinct/b/libfixture.so"
    return _execute_result(
        stdout = "libfixture.so => {} (0x00000000)\n".format(library_path),
    )

def _fake_repository_ctx(execute = _success_execute, which = _fake_which):
    return struct(
        execute = execute,
        path = _fake_path,
        watch = _fake_watch,
        which = which,
    )

def _success_subject_impl(ctx):
    if ctx.label == None:
        fail("test subject requires a label")
    libraries = resolve_elf_dynamic_library_closure(
        _fake_repository_ctx(),
        {
            "static": "/fixtures/static_tool",
            "tool_a": "/fixtures/tool_a",
            "tool_b": "/fixtures/tool_b",
        },
        library_search_path = "/sdk/lib:/sdk/sysdeps/lib",
    )
    return [_ElfClosureInfo(error = "", libraries = libraries)]

_success_subject = rule(implementation = _success_subject_impl)

def _success_test_impl(ctx):
    env = analysistest.begin(ctx)
    target = analysistest.target_under_test(env)
    asserts.equals(
        env,
        {
            "libfixture.so": "/fixtures/runtime/libfixture.so",
            "libstdc++.so.6": "/sdk/sysdeps/lib/libstdc++.so.6",
            "libz.so.1": "/sdk/sysdeps/lib/libz.so.1",
        },
        target[_ElfClosureInfo].libraries,
    )
    return analysistest.end(env)

_success_test = analysistest.make(_success_test_impl)

def _missing_library_subject_impl(ctx):
    if ctx.label == None:
        fail("test subject requires a label")
    resolve_elf_dynamic_library_closure(
        _fake_repository_ctx(execute = _missing_library_execute),
        {"tool": "/fixtures/tool_a"},
    )
    return []

_missing_library_subject = rule(implementation = _missing_library_subject_impl)

def _missing_library_test_impl(ctx):
    env = analysistest.begin(ctx)
    asserts.expect_failure(env, "cannot load required dynamic library libmissing.so")
    return analysistest.end(env)

_missing_library_test = analysistest.make(
    _missing_library_test_impl,
    expect_failure = True,
)

def _inconsistent_library_subject_impl(ctx):
    if ctx.label == None:
        fail("test subject requires a label")
    resolve_elf_dynamic_library_closure(
        _fake_repository_ctx(execute = _inconsistent_library_execute),
        {
            "tool_a": "/fixtures/tool_a",
            "tool_b": "/fixtures/tool_b",
        },
    )
    return []

_inconsistent_library_subject = rule(
    implementation = _inconsistent_library_subject_impl,
)

def _inconsistent_library_test_impl(ctx):
    env = analysistest.begin(ctx)
    asserts.expect_failure(env, "resolve libfixture.so inconsistently")
    return analysistest.end(env)

_inconsistent_library_test = analysistest.make(
    _inconsistent_library_test_impl,
    expect_failure = True,
)

def _missing_ldd_subject_impl(ctx):
    if ctx.label == None:
        fail("test subject requires a label")
    resolve_elf_dynamic_library_closure(
        _fake_repository_ctx(which = _missing_ldd_which),
        {"tool": "/fixtures/tool_a"},
    )
    return []

_missing_ldd_subject = rule(implementation = _missing_ldd_subject_impl)

def _missing_ldd_test_impl(ctx):
    env = analysistest.begin(ctx)
    asserts.expect_failure(env, "ldd was not found on PATH")
    return analysistest.end(env)

_missing_ldd_test = analysistest.make(
    _missing_ldd_test_impl,
    expect_failure = True,
)

def _try_missing_ldd_subject_impl(ctx):
    if ctx.label == None:
        fail("test subject requires a label")
    result = try_resolve_elf_dynamic_library_closure(
        _fake_repository_ctx(which = _missing_ldd_which),
        {"tool": "/fixtures/tool_a"},
    )
    return [_ElfClosureInfo(
        error = result.error,
        libraries = result.libraries,
    )]

_try_missing_ldd_subject = rule(implementation = _try_missing_ldd_subject_impl)

def _try_missing_ldd_test_impl(ctx):
    env = analysistest.begin(ctx)
    target = analysistest.target_under_test(env)
    asserts.equals(
        env,
        "Could not inspect ELF dynamic libraries: ldd was not found on PATH.",
        target[_ElfClosureInfo].error,
    )
    asserts.equals(env, {}, target[_ElfClosureInfo].libraries)
    return analysistest.end(env)

_try_missing_ldd_test = analysistest.make(_try_missing_ldd_test_impl)

def elf_dynamic_library_closure_test_suite(name):
    """Defines the ELF dynamic-library closure resolution tests.

    Args:
      name: Test suite target name.
    """
    success_subject = name + "_success_subject"
    _success_subject(name = success_subject, tags = ["manual"])
    _success_test(
        name = name + "_success_test",
        target_under_test = ":" + success_subject,
    )

    missing_library_subject = name + "_missing_library_subject"
    _missing_library_subject(name = missing_library_subject, tags = ["manual"])
    _missing_library_test(
        name = name + "_missing_library_test",
        target_under_test = ":" + missing_library_subject,
    )

    inconsistent_library_subject = name + "_inconsistent_library_subject"
    _inconsistent_library_subject(
        name = inconsistent_library_subject,
        tags = ["manual"],
    )
    _inconsistent_library_test(
        name = name + "_inconsistent_library_test",
        target_under_test = ":" + inconsistent_library_subject,
    )

    missing_ldd_subject = name + "_missing_ldd_subject"
    _missing_ldd_subject(name = missing_ldd_subject, tags = ["manual"])
    _missing_ldd_test(
        name = name + "_missing_ldd_test",
        target_under_test = ":" + missing_ldd_subject,
    )

    try_missing_ldd_subject = name + "_try_missing_ldd_subject"
    _try_missing_ldd_subject(name = try_missing_ldd_subject, tags = ["manual"])
    _try_missing_ldd_test(
        name = name + "_try_missing_ldd_test",
        target_under_test = ":" + try_missing_ldd_subject,
    )

    native.test_suite(
        name = name,
        tests = [
            ":" + name + "_inconsistent_library_test",
            ":" + name + "_missing_ldd_test",
            ":" + name + "_missing_library_test",
            ":" + name + "_success_test",
            ":" + name + "_try_missing_ldd_test",
        ],
    )
