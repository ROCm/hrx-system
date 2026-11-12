# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Windows MSVC-ABI actions using locally provisioned Linux host tools."""

load("@rules_cc//cc/toolchains:cc_toolchain.bzl", "cc_toolchain")
load(":cc_toolchain.bzl", "windows_cc_toolchain")
load(":toolchain_config.bzl", "windows_cross_toolchain_config")

def windows_cross_toolchain(name, repository_path, execution_architecture, msvc_version):
    """Defines the x86-64 Windows compiler and its action-specific file inputs.

    Args:
      name: Registered C/C++ toolchain target name.
      repository_path: Stable execroot-relative path of the generated repository.
      execution_architecture: CPU architecture of the selected Linux LLVM tools.
      msvc_version: Version of the selected MSVC toolset in the xwin sysroot.
    """
    native.filegroup(name = "empty")
    native.filegroup(
        name = "host_dynamic_libraries",
        srcs = native.glob(["host_dynamic_libraries/*"], allow_empty = True),
    )
    native.filegroup(
        name = "sdk_headers",
        srcs = native.glob(["include/**"]),
    )
    native.filegroup(
        name = "headers",
        srcs = [":sdk_headers"] + native.glob(["resource/include/**"]),
    )
    native.filegroup(
        name = "libraries",
        srcs = native.glob(["lib/**/*.lib"]),
    )
    native.filegroup(
        name = "compiler_files",
        srcs = [
            ":headers",
            "bin/clang-cl",
            "tools/clang-cl",
            ":host_dynamic_libraries",
        ],
    )
    native.filegroup(
        name = "linker_files",
        srcs = [
            ":libraries",
            "bin/lld-link",
            "runtime/clang_rt.builtins-x86_64.lib",
            "tools/lld-link",
            ":host_dynamic_libraries",
        ],
    )
    native.filegroup(
        name = "archiver_files",
        srcs = ["bin/llvm-lib", "tools/llvm-lib", ":host_dynamic_libraries"],
    )
    native.filegroup(
        name = "all_files",
        srcs = [":compiler_files", ":linker_files", ":archiver_files"] + native.glob(
            ["bin/*", "tools/*"],
            exclude = ["tools/flock", "tools/msvc-*", "tools/msvcenv.sh", "tools/wine-msvc.sh", "tools/wine64", "tools/wineserver"],
        ),
    )
    native.filegroup(
        name = "address_sanitizer_files",
        srcs = native.glob(["runtime/*"], allow_empty = True),
    )
    native.filegroup(
        name = "llvm_utility_files",
        srcs = [
            "bin/llvm-cov",
            "bin/llvm-nm",
            "bin/llvm-objcopy",
            "bin/llvm-objdump",
            "bin/llvm-strip",
            "tools/llvm-cov",
            "tools/llvm-nm",
            "tools/llvm-objcopy",
            "tools/llvm-objdump",
            "tools/llvm-strip",
            ":host_dynamic_libraries",
        ],
    )
    native.filegroup(
        name = "msvc_tool_files",
        srcs = native.glob(["msvc/bin/**"], allow_empty = True) + [
            "tools/flock",
            "tools/msvc-cl",
            "tools/msvc-cl-upstream",
            "tools/msvc-dumpbin",
            "tools/msvc-lib",
            "tools/msvc-link",
            "tools/msvc-ml64",
            "tools/msvcenv.sh",
            "tools/wine-msvc.sh",
            "tools/wine64",
            "tools/wineserver",
        ],
    )
    native.filegroup(
        name = "msvc_compiler_files",
        srcs = [":sdk_headers", ":msvc_tool_files"],
    )
    native.filegroup(
        name = "msvc_linker_files",
        srcs = [":libraries", ":msvc_tool_files"],
    )
    native.filegroup(
        name = "msvc_all_files",
        srcs = [
            ":msvc_compiler_files",
            ":msvc_linker_files",
            ":llvm_utility_files",
        ],
    )

    msvc_include_directories = [
        repository_path + "/include/" + component
        for component in ["crt", "ucrt", "shared", "um", "winrt", "cppwinrt"]
    ]
    include_directories = [repository_path + "/resource/include"] + msvc_include_directories
    msvc_library_directories = [repository_path + "/lib/" + component for component in ["crt", "ucrt", "um"]]
    library_directories = msvc_library_directories + [
        repository_path + "/runtime",
    ]
    windows_cross_toolchain_config(
        name = "config",
        cpu = "x64_windows",
        compiler = "clang-cl",
        host_system_name = execution_architecture + "-unknown-linux-gnu",
        target_system_name = "x86_64-pc-windows-msvc",
        target_libc = "msvcrt",
        abi_version = "msvc",
        abi_libc_version = "msvcrt",
        toolchain_identifier = "llvm_windows_x86_64",
        msvc_env_tmp = "/tmp",
        msvc_env_path = "/usr/bin:/bin",
        msvc_env_include = ";".join(include_directories),
        msvc_env_lib = ";".join(library_directories),
        msvc_cl_path = "tools/clang-cl",
        msvc_ml_path = "tools/clang-cl",
        msvc_link_path = "tools/lld-link",
        msvc_lib_path = "tools/llvm-lib",
        cxx_builtin_include_directories = include_directories,
        tool_paths = {
            "ar": "tools/llvm-lib",
            "cpp": "tools/clang-cl",
            "gcc": "tools/clang-cl",
            "gcov": "tools/llvm-cov",
            "ld": "tools/lld-link",
            "nm": "tools/llvm-nm",
            "objcopy": "tools/llvm-objcopy",
            "objdump": "tools/llvm-objdump",
            "strip": "tools/llvm-strip",
        },
        default_compile_flags = [
            "--no-default-config",
            "--target=x86_64-pc-windows-msvc",
            # MSVC toolset 14.x implements compiler version 19.x. The STL uses
            # this compiler version to select its supported builtin surface.
            "-fms-compatibility-version=" + str(5 + int(msvc_version.split(".")[0])) + "." + msvc_version.split(".")[1],
            "/Z7",
            "/X",
            "-resource-dir=" + repository_path + "/resource",
            "/clang:-fdebug-compilation-dir=.",
        ] + ["/imsvc" + directory for directory in include_directories],
        default_link_flags = [
            "/MACHINE:X64",
            "/DEFAULTLIB:clang_rt.builtins-x86_64.lib",
        ] + ["/LIBPATH:" + directory for directory in library_directories],
        archiver_flags = ["/MACHINE:X64"],
        dbg_mode_debug_flag = "/DEBUG",
        fastbuild_mode_debug_flag = "/DEBUG",
        supports_parse_showincludes = True,
    )
    cc_toolchain(
        name = "cc_toolchain",
        toolchain_identifier = "llvm_windows_x86_64",
        toolchain_config = ":config",
        all_files = ":all_files",
        ar_files = ":archiver_files",
        as_files = ":compiler_files",
        compiler_files = ":compiler_files",
        dwp_files = ":empty",
        linker_files = ":linker_files",
        objcopy_files = ":all_files",
        strip_files = ":all_files",
        supports_param_files = 1,
    )
    windows_cc_toolchain(
        name = name,
        cc = ":cc_toolchain",
        masm = "tools/llvm-ml",
        masm_arguments = ["-m64", "/c"],
        masm_files = ["bin/llvm-ml", ":host_dynamic_libraries"],
        address_sanitizer_files = select({
            Label("//build_tools/bazel:address_sanitizer_target"): [":address_sanitizer_files"],
            "//conditions:default": [],
        }),
    )

    windows_cross_toolchain_config(
        name = "msvc_config",
        cpu = "x64_windows",
        compiler = "msvc-cl",
        host_system_name = execution_architecture + "-unknown-linux-gnu",
        target_system_name = "x86_64-pc-windows-msvc",
        target_libc = "msvcrt",
        abi_version = "msvc",
        abi_libc_version = "msvcrt",
        toolchain_identifier = "msvc_wine_windows_x86_64",
        msvc_env_tmp = "/tmp",
        msvc_env_path = "/usr/bin:/bin",
        msvc_env_include = ";".join(msvc_include_directories),
        msvc_env_lib = ";".join(msvc_library_directories),
        msvc_cl_path = "tools/msvc-cl",
        msvc_ml_path = "tools/msvc-ml64",
        msvc_link_path = "tools/msvc-link",
        msvc_lib_path = "tools/msvc-lib",
        cxx_builtin_include_directories = msvc_include_directories,
        tool_paths = {
            "ar": "tools/msvc-lib",
            "cpp": "tools/msvc-cl",
            "dumpbin": "tools/msvc-dumpbin",
            "gcc": "tools/msvc-cl",
            "gcov": "tools/llvm-cov",
            "ld": "tools/msvc-link",
            "nm": "tools/llvm-nm",
            "objcopy": "tools/llvm-objcopy",
            "objdump": "tools/llvm-objdump",
            "strip": "tools/llvm-strip",
        },
        default_compile_flags = [
            "/Z7",
            "/X",
        ] + ["/I" + directory for directory in msvc_include_directories],
        default_link_flags = ["/MACHINE:X64"],
        archiver_flags = ["/MACHINE:X64"],
        dbg_mode_debug_flag = "/DEBUG",
        fastbuild_mode_debug_flag = "/DEBUG",
        supports_parse_showincludes = True,
    )
    cc_toolchain(
        name = "msvc_cc_toolchain",
        toolchain_identifier = "msvc_wine_windows_x86_64",
        toolchain_config = ":msvc_config",
        all_files = ":msvc_all_files",
        ar_files = ":msvc_tool_files",
        as_files = ":msvc_compiler_files",
        compiler_files = ":msvc_compiler_files",
        dwp_files = ":empty",
        linker_files = ":msvc_linker_files",
        objcopy_files = ":llvm_utility_files",
        strip_files = ":llvm_utility_files",
        supports_param_files = 1,
    )
    windows_cc_toolchain(
        name = "msvc_toolchain",
        cc = ":msvc_cc_toolchain",
        dumpbin = "tools/msvc-dumpbin",
        masm = "tools/msvc-ml64",
        masm_arguments = ["/nologo", "/Zi", "/c"],
        masm_files = [":msvc_tool_files"],
    )
