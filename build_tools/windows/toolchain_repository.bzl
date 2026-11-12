# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Projects a selected local LLVM and xwin sysroot into a Windows toolchain."""

load(
    "//build_tools/bazel:elf_dynamic_library_closure.bzl",
    "elf_loader_environment",
    "resolve_elf_dynamic_library_closure",
)

_TOOLS = [
    "clang-cl",
    "lld-link",
    "llvm-lib",
    "llvm-ml",
    "llvm-cov",
    "llvm-nm",
    "llvm-objcopy",
    "llvm-objdump",
    "llvm-strip",
]

_WRAPPER = """#!/bin/sh
set -eu
tool_root="${0%/*}/.."
export LD_LIBRARY_PATH="$tool_root/host_dynamic_libraries"
unset LD_PRELOAD LD_AUDIT
exec "$tool_root/bin/{tool}" "$@"
"""

_MISSING_LLVM_ML = """#!/bin/sh
echo 'The selected clang-cl Windows toolchain requires LLVM_ROOT/bin/llvm-ml.' >&2
exit 1
"""

_MSVC_ENV = """#!/usr/bin/env bash
tool_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINDIR="$tool_root/msvc/bin"
export PATH="$tool_root/tools:$PATH"
export INCLUDE="z:$tool_root/include/crt;z:$tool_root/include/ucrt;z:$tool_root/include/shared;z:$tool_root/include/um;z:$tool_root/include/winrt;z:$tool_root/include/cppwinrt"
export LIB="z:$tool_root/lib/crt;z:$tool_root/lib/ucrt;z:$tool_root/lib/um"
export LIBPATH="$LIB"
export WINEPATH="z:$BINDIR"
: "${WINEPREFIX:?Linux-hosted MSVC requires WINEPREFIX to name a user-owned directory.}"
export WINEPREFIX
export WINEDLLOVERRIDES="vcruntime140=n;vcruntime140_1=n"
"""

_MISSING_MSVC_WRAPPER = """#!/bin/sh
echo 'Linux-hosted MSVC requires MSVC_WINE_ROOT; run .github/scripts/setup_windows_msvc.sh and export its output path.' >&2
exit 1
"""

_MSVC_DUMPBIN_WRAPPER = """#!/usr/bin/env bash
. "$(dirname "$0")/msvcenv.sh"
"$(dirname "$0")/wine-msvc.sh" "$BINDIR/dumpbin.exe" "$@"
"""

def _required_path(repository_ctx, path):
    path = repository_ctx.path(path)
    if not path.exists:
        fail("Windows toolchain requires %s; check the selected LLVM, Windows SDK, and compiler-rt inputs in BUILDING.md." % path)
    repository_ctx.watch(path)
    canonical_path = path.realpath
    repository_ctx.watch(canonical_path)
    return canonical_path

def _project(repository_ctx, source, destination):
    repository_ctx.symlink(_required_path(repository_ctx, source), destination)

def _version_directory(repository_ctx, parent):
    parent = _required_path(repository_ctx, parent)
    versions = [path for path in parent.readdir() if path.is_dir]
    if len(versions) != 1:
        fail("Windows toolchain requires exactly one version under %s; found %s. Select a versioned xwin sysroot with WINSDK_ROOT." % (parent, versions))
    return _required_path(repository_ctx, versions[0])

def _windows_toolchain_repository_impl(repository_ctx):
    if repository_ctx.os.name != "linux":
        fail("The Windows cross toolchain requires a Linux host; native Windows uses local_config_cc. Host is %s." % repository_ctx.os.name)
    architecture = repository_ctx.os.arch
    if architecture in ["amd64", "x86_64"]:
        architecture = "x86_64"
    elif architecture in ["aarch64", "arm64"]:
        architecture = "aarch64"
    else:
        fail("Unsupported Linux LLVM execution architecture: %s" % architecture)

    llvm_root = repository_ctx.getenv("LLVM_ROOT")
    sdk_root = repository_ctx.getenv("WINSDK_ROOT")
    if not llvm_root or not sdk_root:
        fail("Windows cross-compilation requires LLVM_ROOT and WINSDK_ROOT. Set these environment variables or pass --repo_env=LLVM_ROOT=/path/to/llvm --repo_env=WINSDK_ROOT=/path/to/windows-sysroot. See BUILDING.md.")
    llvm_root = _required_path(repository_ctx, llvm_root)
    sdk_root = _required_path(repository_ctx, sdk_root)
    artifacts = {}
    for tool in _TOOLS:
        tool_path = llvm_root.get_child("bin", tool)
        if tool == "llvm-ml" and not tool_path.exists:
            # Genuine MSVC supplies ml64.exe. Keep the parallel clang-cl
            # toolchain definable, but fail clearly if it is selected for a
            # MASM action with an LLVM distribution that omits llvm-ml.
            repository_ctx.file("bin/" + tool, _MISSING_LLVM_ML, executable = True)
        else:
            artifacts[tool] = _required_path(repository_ctx, tool_path)
            repository_ctx.symlink(artifacts[tool], "bin/" + tool)
        repository_ctx.file("tools/" + tool, _WRAPPER.replace("{tool}", tool), executable = True)

    libraries = resolve_elf_dynamic_library_closure(repository_ctx, artifacts)
    for name, path in libraries.items():
        repository_ctx.symlink(path, "host_dynamic_libraries/" + name)

    resource_result = repository_ctx.execute(
        [str(artifacts["clang-cl"]), "--no-default-config", "-print-resource-dir"],
        environment = elf_loader_environment(),
        quiet = True,
    )
    if resource_result.return_code:
        fail("Could not query LLVM resource directory:\n" + resource_result.stderr)
    resource_root = _required_path(repository_ctx, resource_result.stdout.strip())
    _project(repository_ctx, resource_root.get_child("include"), "resource/include")
    _project(
        repository_ctx,
        resource_root.get_child("lib", "windows", "clang_rt.builtins-x86_64.lib"),
        "runtime/clang_rt.builtins-x86_64.lib",
    )

    runtime_root = repository_ctx.getenv("WINDOWS_COMPILER_RT_ROOT")
    if runtime_root:
        runtime_root = _required_path(repository_ctx, runtime_root)
        for name in [
            "clang_rt.asan_dynamic-x86_64.dll",
            "clang_rt.asan_dynamic-x86_64.lib",
            "clang_rt.asan_dynamic_runtime_thunk-x86_64.lib",
            "clang_rt.asan_static_runtime_thunk-x86_64.lib",
        ]:
            _project(repository_ctx, runtime_root.get_child(name), "runtime/" + name)

    crt_root = _version_directory(repository_ctx, sdk_root.get_child("VC", "Tools", "MSVC"))
    sdk_include_root = _version_directory(repository_ctx, sdk_root.get_child("Windows Kits", "10", "Include"))
    sdk_library_root = sdk_root.get_child("Windows Kits", "10", "Lib", sdk_include_root.basename)
    _project(repository_ctx, crt_root.get_child("include"), "include/crt")
    for component in ["ucrt", "shared", "um", "winrt", "cppwinrt"]:
        _project(repository_ctx, sdk_include_root.get_child(component), "include/" + component)
    _project(repository_ctx, crt_root.get_child("lib", "x64"), "lib/crt")
    for component in ["ucrt", "um"]:
        _project(repository_ctx, sdk_library_root.get_child(component, "x64"), "lib/" + component)

    msvc_root = repository_ctx.getenv("MSVC_WINE_ROOT")
    if msvc_root:
        wine = repository_ctx.which("wine64") or repository_ctx.which("wine")
        if not wine:
            system_wine64 = repository_ctx.path("/usr/lib/wine/wine64")
            if system_wine64.exists:
                wine = system_wine64
        if not wine:
            fail("Linux-hosted MSVC requires a Wine64 loader when MSVC_WINE_ROOT is set. Install wine64 or place wine64 or wine on PATH.")
        msvc_root = _required_path(repository_ctx, msvc_root)
        msvc_version_root = _version_directory(repository_ctx, msvc_root.get_child("VC", "Tools", "MSVC"))
        _project(repository_ctx, msvc_version_root.get_child("bin", "Hostx64", "x64"), "msvc/bin")
        _project(repository_ctx, wine, "tools/wine64")
        for source, destination in {
            "cl": "msvc-cl",
            "lib": "msvc-lib",
            "link": "msvc-link",
            "ml64": "msvc-ml64",
            "wine-msvc.sh": "wine-msvc.sh",
        }.items():
            _project(repository_ctx, msvc_root.get_child("wrappers", source), "tools/" + destination)
        repository_ctx.file("tools/msvc-dumpbin", _MSVC_DUMPBIN_WRAPPER, executable = True)
    else:
        for destination in ["msvc-cl", "msvc-dumpbin", "msvc-lib", "msvc-link", "msvc-ml64", "wine-msvc.sh", "wine64"]:
            repository_ctx.file("tools/" + destination, _MISSING_MSVC_WRAPPER, executable = True)
    repository_ctx.file("tools/msvcenv.sh", _MSVC_ENV, executable = True)

    repository_ctx.file("BUILD.bazel", """
load({toolchain_bzl}, "windows_cross_toolchain")
package(default_visibility = ["//visibility:public"])
windows_cross_toolchain(
    name = "toolchain",
    repository_path = {repository_path},
    execution_architecture = {execution_architecture},
    msvc_version = {msvc_version},
)
""".format(
        toolchain_bzl = repr(str(repository_ctx.attr._toolchain_bzl)),
        repository_path = repr("external/" + repository_ctx.name),
        execution_architecture = repr(architecture),
        msvc_version = repr(crt_root.basename),
    ))

windows_toolchain_repository = repository_rule(
    implementation = _windows_toolchain_repository_impl,
    attrs = {
        "_toolchain_bzl": attr.label(default = Label("//build_tools/windows:toolchain.bzl")),
    },
    local = True,
    configure = True,
)
