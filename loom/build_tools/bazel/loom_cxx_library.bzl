# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C/C++ source admission and exported header interfaces for Loom libraries."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("//loom/requirements:package_policy.bzl", "apply_loom_target_policy")
load(":loom_linking.bzl", "LoomLibraryInfo", "loom_linking")

_CXX_TOOLCHAIN = Label("//loom/build_tools/bazel:cxx_import_toolchain_type")
_LINK_TOOLCHAIN = Label("//loom/build_tools/bazel:link_toolchain_type")
_HEADER_SUFFIXES = [".h", ".hh", ".hpp", ".hxx", ".inc", ".inl", ".ipp", ".tcc", ".def"]

LoomCxxInfo = provider(
    doc = "Exported C/C++ source interface, independent of compiled Loom modules.",
    fields = {
        "defines": "Depset of preprocessor definitions exported to dependents.",
        "headers": "Depset of public headers, including transitive interfaces.",
        "includes": "Depset of include directories with execution and publication paths.",
    },
)

def _relative_path(path, directory):
    path_parts = paths.normalize(path).split("/")
    directory_parts = paths.normalize(directory).split("/") if directory and directory != "." else []
    common = 0
    for index in range(min(len(path_parts), len(directory_parts))):
        if path_parts[index] != directory_parts[index]:
            break
        common += 1
    return paths.normalize("/".join([".."] * (len(directory_parts) - common) + path_parts[common:]))

def _quote_option(value):
    # The public provider-option tokenizer accepts quoted tokens, not shell
    # expansion. Preserve macro replacement strings and paths byte-for-byte.
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"

def _option_string(copts, defines, directories, source_directory):
    options = list(copts) + ["D=" + value for value in defines]
    options.extend(["I=" + _relative_path(path, source_directory) for path in {directory: True for directory in directories}])
    return " ".join([_quote_option(option) for option in options])

def _include_directories(ctx):
    repository = ctx.label.workspace_root
    publication_root = "../" + ctx.label.workspace_name if repository else ""
    directories = [""]
    for include in ctx.attr.includes:
        if paths.is_absolute(include):
            fail("%s includes must be repository-relative, got %r" % (ctx.label, include))
        path = paths.normalize(paths.join(ctx.label.package, include))
        if path == ".." or path.startswith("../"):
            fail("%s include directory escapes its repository: %r" % (ctx.label, include))
        directories.append(path)
    return [struct(
        path = paths.join(root, repository, directory),
        publication_path = paths.join(publication_root, directory),
    ) for directory in directories for root in ["", ctx.bin_dir.path]]

def _declare_import(ctx, source, ordinal, inputs, defines, includes, tool):
    output = ctx.actions.declare_file("%s.imports/%d-%s.loombc" % (ctx.label.name, ordinal, source.basename))
    args = ctx.actions.args()
    args.add(source)
    args.add("--to=bc")
    args.add("--module-name=%s/unit%d" % (ctx.label, ordinal))
    if ctx.var["COMPILATION_MODE"] != "dbg":
        args.add("--strip-debug")
    args.add("--output=%s" % output.path)
    args.add("--options=" + _option_string(ctx.attr.copts, defines, [directory.path for directory in includes], source.dirname))
    ctx.actions.run(
        arguments = [args],
        executable = tool.files_to_run,
        inputs = depset(direct = [source], transitive = [inputs]),
        mnemonic = "LoomCxxImport",
        outputs = [output],
        progress_message = "Importing C/C++ source %s" % source.path,
    )
    return struct(
        file = source,
        format = "cxx",
        options = ("cxx:" + _option_string(ctx.attr.copts, defines, [directory.publication_path for directory in includes], paths.dirname(source.short_path)),),
        inputs = inputs,
        normalized_file = output,
        normalized_format = "loombc",
    )

def _loom_cxx_library_impl(ctx):
    dependency_infos = [dependency[LoomLibraryInfo] for dependency in ctx.attr.deps]
    interfaces = [dependency[LoomCxxInfo] for dependency in ctx.attr.deps if LoomCxxInfo in dependency]
    headers = []
    sources = []
    for file in ctx.files.srcs:
        if any([file.path.endswith(suffix) for suffix in _HEADER_SUFFIXES]):
            headers.append(file)
        else:
            sources.append(file)
    if not sources and not ctx.files.hdrs and not ctx.attr.deps:
        fail("%s requires translation units, public headers, or dependencies" % ctx.label)
    interface = LoomCxxInfo(
        headers = depset(direct = ctx.files.hdrs, transitive = [info.headers for info in interfaces]),
        includes = depset(direct = _include_directories(ctx), transitive = [info.includes for info in interfaces]),
        defines = depset(direct = ctx.attr.defines, transitive = [info.defines for info in interfaces]),
    )
    inputs = depset(direct = headers, transitive = [interface.headers])
    defines = interface.defines.to_list() + ctx.attr.local_defines
    includes = interface.includes.to_list()
    records = [
        _declare_import(ctx, source, ordinal, inputs, defines, includes, ctx.toolchains[_CXX_TOOLCHAIN].tool)
        for ordinal, source in enumerate(sources)
    ]
    module = None
    reports = []
    if records:
        artifacts = loom_linking.declare_relocatable_module(
            ctx = ctx,
            sources = [source.normalized_file for source in records],
            dependency_infos = dependency_infos,
            output_stem = ctx.label.name,
            mnemonic = "LoomLibrary",
            progress_message = "Linking Loom library %s" % ctx.label,
            tool = ctx.toolchains[_LINK_TOOLCHAIN].tool,
        )
        module = artifacts.module
        reports.append(artifacts.dependency_report)
    return [
        DefaultInfo(
            files = depset([module] if module else []),
            runfiles = ctx.runfiles(files = ctx.files.data).merge_all([target[DefaultInfo].default_runfiles for target in ctx.attr.data + ctx.attr.deps]),
        ),
        OutputGroupInfo(dependency_reports = depset(reports)),
        loom_linking.library_info(ctx.label, module, records, ctx.files.data, dependency_infos),
        interface,
    ]

_loom_cxx_library = rule(
    implementation = _loom_cxx_library_impl,
    attrs = {
        "copts": attr.string_list(doc = "Local C++ source options in key=value form, such as std=c++23 or fp-model=fast."),
        "data": attr.label_list(allow_files = True, doc = "Runtime fixtures. Compiler header inputs belong in hdrs or srcs."),
        "defines": attr.string_list(doc = "NAME[=VALUE] preprocessor definitions exported to C++ dependents."),
        "deps": attr.label_list(providers = [LoomLibraryInfo], doc = "Direct C++ header interfaces and independent Loom link dependencies."),
        "hdrs": attr.label_list(allow_files = True, doc = "Public headers available to this component and its C++ dependents."),
        "includes": attr.string_list(doc = "Package-relative include directories exported to C++ dependents. Source and generated directories are searched."),
        "local_defines": attr.string_list(doc = "NAME[=VALUE] preprocessor definitions used only by this component."),
        "srcs": attr.label_list(allow_files = True, doc = "Translation units and private headers. Header files are inputs, not separate translation units."),
    },
    toolchains = [_CXX_TOOLCHAIN, _LINK_TOOLCHAIN],
)

def loom_cxx_library(name, srcs = [], hdrs = [], deps = [], copts = [], defines = [], local_defines = [], includes = [], data = [], **kwargs):
    """Imports C/C++ once and exposes its module and public header interface.

    Header-only components forward declared interfaces without emitting an empty
    module. Other components merge their imported translation units into one
    relocatable bytecode module; dependency modules stay independent. Source
    options use the public C++ importer's key=value spelling, not host compiler
    flags. Diagnostic metadata follows Bazel's compilation mode.

    Args:
      name: Component name and logical bytecode identity.
      srcs: Translation units and private headers.
      hdrs: Public headers exported to C++ consumers.
      deps: Direct library dependencies, including header-only components.
      copts: Local C++ source options, such as std=c++23 or fp-model=fast.
      defines: Exported preprocessor definitions.
      local_defines: Private preprocessor definitions.
      includes: Exported include directories relative to this package.
      data: Runtime fixtures, independent of compiler header inputs.
      **kwargs: Standard Bazel attributes, including testonly and visibility.
    """
    _loom_cxx_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        copts = copts,
        defines = defines,
        local_defines = local_defines,
        includes = includes,
        data = data,
        **apply_loom_target_policy(kwargs, name = name)
    )
