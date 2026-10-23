# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declares native Windows compiler artifacts and their installation identity."""

# Use the same discovery and environment contract as local_config_cc.
# buildifier: disable=bzl-visibility
load("@rules_cc//cc/private/toolchain:cc_configure.bzl", "MSVC_ENVVARS")

# buildifier: disable=bzl-visibility
load("@rules_cc//cc/private/toolchain:windows_cc_configure.bzl", "find_llvm_path", "find_llvm_tool", "find_msvc_tool", "find_vc_path", "setup_vc_env_vars")

NativeWindowsRuntimeInfo = provider("Native compiler identity and its runtime artifacts.", fields = {
    # Matching ASAN files present in this compiler installation.
    "address_sanitizer_files": "Runtime DLL, import library and thunks.",
    # Clang compiler builtins; MSVC does not need a separate builtins archive.
    "builtins": "Optional compiler builtins File.",
    # Exact compiler path selected by the upstream discovery helpers.
    "compiler_executable": "Compiler executable path.",
    # Upstream configuration directory owning its relative helper paths.
    "configuration_directory": "Native local_config_cc directory.",
    # Absolute native symbol-reader path, as used for other upstream tools.
    "nm_executable": "llvm-nm executable path, or empty for MSVC.",
    # Resource data directory read implicitly by the selected Clang driver.
    "resource_directory": "Clang resource directory, or empty for MSVC.",
})

def _native_windows_runtime_impl(ctx):
    return [NativeWindowsRuntimeInfo(
        compiler_executable = ctx.attr.compiler_executable,
        builtins = ctx.file.builtins,
        address_sanitizer_files = ctx.files.address_sanitizer_files,
        resource_directory = ctx.attr.resource_directory,
        nm_executable = ctx.attr.nm_executable,
        configuration_directory = ctx.attr.configuration_directory,
    ), DefaultInfo(files = depset([file for file in [ctx.file.nm, ctx.file.address_sanitizer_ignorelist, ctx.file.builtins] if file])), OutputGroupInfo(
        compiler_files = depset([ctx.file.address_sanitizer_ignorelist] if ctx.file.address_sanitizer_ignorelist else []),
        linker_files = depset([ctx.file.builtins] if ctx.file.builtins else []),
    )]

native_windows_runtime = rule(
    implementation = _native_windows_runtime_impl,
    attrs = {
        "address_sanitizer_files": attr.label_list(allow_files = True),
        "address_sanitizer_ignorelist": attr.label(allow_single_file = [".txt"]),
        "builtins": attr.label(allow_single_file = [".lib"]),
        "compiler_executable": attr.string(mandatory = True),
        "configuration_directory": attr.string(mandatory = True),
        "nm": attr.label(allow_single_file = [".exe"]),
        "nm_executable": attr.string(),
        "resource_directory": attr.string(),
    },
    provides = [NativeWindowsRuntimeInfo],
)

def _project_runtime(repository_ctx, path, name):
    path = repository_ctx.path(path)
    repository_ctx.watch(path)
    if path.exists:
        repository_ctx.symlink(path, "runtime/" + name)
        return True
    return False

def _native_windows_runtime_repository_impl(repository_ctx):
    if "windows" not in repository_ctx.os.name.lower():
        fail("Native Windows runtimes require a Windows host.")

    # USE_CLANG_CL also changes the compiler in the upstream MSVC toolchain.
    clang = repository_ctx.attr.compiler == "clang-cl" or repository_ctx.getenv("USE_CLANG_CL") == "1"
    runtime_names = [
        "clang_rt.asan_dynamic-x86_64.dll",
        "clang_rt.asan_dynamic-x86_64.lib",
        "clang_rt.asan_dynamic_runtime_thunk-x86_64.lib",
        "clang_rt.asan_static_runtime_thunk-x86_64.lib",
    ]
    builtins = None
    ignorelist = None
    resource_directory = ""
    nm = None
    nm_path = ""
    if clang:
        llvm_path = find_llvm_path(repository_ctx)
        compiler = find_llvm_tool(repository_ctx, llvm_path, "clang-cl.exe") if llvm_path else None
        if not compiler:
            fail("The native Windows Clang runtime requires clang-cl from the installation selected by BAZEL_LLVM or LLVM discovery. See BUILDING.md.")
        repository_ctx.watch(compiler)
        resource = repository_ctx.execute([compiler, "--no-default-config", "-print-resource-dir"], quiet = True)
        if resource.return_code:
            fail("Could not query the selected clang-cl resource directory:\n" + resource.stderr)
        resource_root = repository_ctx.path(resource.stdout.strip())
        resource_directory = str(resource_root)
        runtime_root = resource_root.get_child("lib", "windows")
        builtins_name = "clang_rt.builtins-x86_64.lib"
        if not _project_runtime(repository_ctx, runtime_root.get_child(builtins_name), builtins_name):
            fail("The selected clang-cl installation is missing %s. See BUILDING.md." % runtime_root.get_child(builtins_name))
        builtins = "runtime/" + builtins_name
        for name in runtime_names:
            _project_runtime(repository_ctx, runtime_root.get_child(name), name)
        if _project_runtime(repository_ctx, resource_root.get_child("share", "asan_ignorelist.txt"), "asan_ignorelist.txt"):
            ignorelist = "runtime/asan_ignorelist.txt"
        nm_path = find_llvm_tool(repository_ctx, llvm_path, "llvm-nm.exe")
        if not nm_path:
            fail("The selected clang-cl installation is missing llvm-nm.exe. See BUILDING.md.")
        _project_runtime(repository_ctx, nm_path, "llvm-nm.exe")
        nm = "runtime/llvm-nm.exe"
    else:
        vc_path = find_vc_path(repository_ctx)
        compiler = find_msvc_tool(repository_ctx, vc_path, "cl.exe") if vc_path else None
        if not compiler:
            fail("The native Windows MSVC runtime requires cl.exe from the Visual Studio installation selected by Bazel. See BUILDING.md.")
        repository_ctx.watch(compiler)
        environment = setup_vc_env_vars(repository_ctx, vc_path, envvars = ["LIB"], escape = False)
        for name in runtime_names:
            if name.endswith(".dll"):
                path = find_msvc_tool(repository_ctx, vc_path, name)
                if path:
                    _project_runtime(repository_ctx, path, name)
            else:
                for directory in environment["LIB"].split(";"):
                    if directory and _project_runtime(repository_ctx, repository_ctx.path(directory).get_child(name), name):
                        break

    repository_ctx.file("BUILD.bazel", """
load({runtime_bzl}, "native_windows_runtime")
native_windows_runtime(
    name = "runtime",
    compiler_executable = {compiler},
    builtins = {builtins},
    address_sanitizer_files = glob(["runtime/*asan*.dll", "runtime/*asan*.lib"], allow_empty = True),
    address_sanitizer_ignorelist = {ignorelist},
    resource_directory = {resource_directory},
    nm = {nm},
    nm_executable = {nm_executable},
    configuration_directory = {configuration_directory},
    visibility = ["//visibility:public"],
)
""".format(
        runtime_bzl = repr(str(repository_ctx.attr._runtime_bzl)),
        compiler = repr(compiler),
        builtins = repr(builtins),
        ignorelist = repr(ignorelist),
        resource_directory = repr(resource_directory),
        nm = repr(nm),
        nm_executable = repr(nm_path),
        configuration_directory = repr(str(repository_ctx.path(repository_ctx.attr._configuration).dirname)),
    ))

native_windows_runtime_repository = repository_rule(
    implementation = _native_windows_runtime_repository_impl,
    attrs = {
        "compiler": attr.string(mandatory = True, values = ["clang-cl", "msvc"]),
        "_configuration": attr.label(default = Label("@local_config_cc//:BUILD")),
        "_runtime_bzl": attr.label(default = Label("//build_tools/windows:native_runtime.bzl")),
    },
    environ = MSVC_ENVVARS + ["BAZEL_LLVM", "CC_CONFIGURE_DEBUG", "PATH", "PROGRAMFILES", "PROGRAMFILES(X86)", "SYSTEMROOT", "USE_CLANG_CL"],
    configure = True,
    local = True,
)
