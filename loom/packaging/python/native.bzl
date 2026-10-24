# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The native command and shared-library closure installed by the Loom wheel."""

NATIVE_COMMANDS = [
    "loom-benchmark",
    "loom-check",
    "loom-compile",
    "loom-format",
    "loom-import-cxx",
    "loom-link",
    "loom-opt",
    "loom-run",
    "loom-serve",
    "loom-test",
]

_CONFIGURATION = {
    str(Label("//loom/config:shared_tools")): True,
    str(Label("//loom/config/import:enable")): ["cxx"],
    str(Label("//loom/config/import/cxx:embed_includes")): True,
}

def _configuration_impl(_settings, _attr):
    return _CONFIGURATION

_configuration = transition(
    implementation = _configuration_impl,
    inputs = [],
    outputs = _CONFIGURATION.keys(),
)

_SymbolsInfo = provider(
    doc = "Symbols paired with their native outputs before runfiles flattening.",
    fields = {
        "pairs": "Depset of (native File, PDB File) pairs from their link owners.",
    },
)

_SYMBOL_DEPENDENCIES = [
    "actual",
    "data",
    "deps",
    "dynamic_deps",
    "environment",
    "implementation_deps",
    "runtime_deps",
    "src",
    "srcs",
]

def _symbols_impl(target, ctx):
    pairs = []
    if OutputGroupInfo in target and hasattr(target[OutputGroupInfo], "pdb_file"):
        for file in target[DefaultInfo].files.to_list():
            if file.extension in ["exe", "dll"]:
                for pdb in target[OutputGroupInfo].pdb_file.to_list():
                    pairs.append((file, pdb))
    transitive = []
    if ctx.rule:
        for name in _SYMBOL_DEPENDENCIES:
            value = getattr(ctx.rule.attr, name, None)

            # Runtime bundles name their root libraries with a label-keyed
            # dictionary. Other edges are labels or label lists.
            dependencies = value.keys() if type(value) == "dict" else value
            if type(dependencies) != "list":
                dependencies = [dependencies] if dependencies else []
            for dependency in dependencies:
                if type(dependency) == "Target" and _SymbolsInfo in dependency:
                    transitive.append(dependency[_SymbolsInfo].pairs)
    return [_SymbolsInfo(pairs = depset(pairs, transitive = transitive))]

_symbols = aspect(
    implementation = _symbols_impl,
    attr_aspects = _SYMBOL_DEPENDENCIES,
    doc = "Retains declared symbols through link and runtime edges, excluding build tools.",
)

def _native_files_impl(ctx):
    files = {}
    for library in ctx.attr.libraries:
        for file in library[DefaultInfo].files.to_list():
            if file.basename in files:
                fail("Duplicate native library destination: " + file.basename)
            files[file.basename] = file
    for tool in ctx.attr.tools:
        executable = tool[DefaultInfo].files_to_run.executable
        if executable == None:
            fail("Native command has no executable: " + str(tool.label))
        if executable.basename in files:
            fail("Duplicate native command destination: " + executable.basename)
        files[executable.basename] = executable
    for tool in ctx.attr.tools:
        # C++ runfiles include libraries and their loader aliases: solib links
        # on Unix and per-executable DLL copies on Windows. Install each name
        # once, retaining toolchain runtimes such as Windows AddressSanitizer.
        # Other runfiles serve the Bazel launcher, including sanitizer
        # suppression metadata; they are not native loader inputs.
        for file in tool[DefaultInfo].default_runfiles.files.to_list():
            name = file.basename.lower()
            if name.endswith(".dll") or name.endswith(".so") or ".so." in name:
                files.setdefault(file.basename, file)

    native_files = {file: True for file in files.values()}
    symbols = depset(transitive = [
        target[_SymbolsInfo].pairs
        for target in ctx.attr.libraries + ctx.attr.tools
    ])
    for file, pdb in symbols.to_list():
        if file in native_files:
            existing = files.get(pdb.basename)
            if existing != None and existing != pdb:
                fail("Duplicate native symbol destination: " + pdb.basename)
            files[pdb.basename] = pdb

    outputs = files.values()
    if ctx.executable.elf_patcher:
        # Bazel's solib paths serve build-tree execution. The wheel owns a flat
        # installation with matching companions beside every native command.
        outputs = []
        for file in files.values():
            output = ctx.actions.declare_file(ctx.label.name + "/" + file.basename)
            ctx.actions.run(
                executable = ctx.executable.elf_patcher,
                arguments = ["--set-rpath", "$ORIGIN", "--output", output.path, file.path],
                inputs = [file],
                outputs = [output],
                mnemonic = "LoomWheelRelocate",
                progress_message = "Relocating Loom wheel native file %s" % file.basename,
            )
            outputs.append(output)
    return [DefaultInfo(files = depset(outputs))]

loom_native_files = rule(
    implementation = _native_files_impl,
    cfg = _configuration,
    attrs = {
        "elf_patcher": attr.label(
            executable = True,
            cfg = "exec",
        ),
        "libraries": attr.label_list(
            aspects = [_symbols],
            default = [
                "//loom/binding/c:iree_shared",
                "//loom/binding/c:loomc_shared",
            ],
        ),
        "tools": attr.label_list(
            aspects = [_symbols],
            default = ["//loom/src/loom/tools/" + name for name in NATIVE_COMMANDS],
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)
