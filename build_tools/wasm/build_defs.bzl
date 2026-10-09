# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""WebAssembly JavaScript companion and bundling rules."""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//build_tools/bazel:cc_opts.bzl", "cc_opts")

_CXX_OPTIONS = cc_opts.iree_cxx_standard_options()

IreeWasmJsInfo = provider(
    doc = "JavaScript companion sources for a wasm import module.",
    fields = {
        "js_files": "Depset of JavaScript source files.",
        "modules": "Depset of structs with `module` and `js_files` fields.",
    },
)

IreeWasmEntryInfo = provider(
    doc = "JavaScript entry point for a wasm binary.",
    fields = {
        "main": "Main JavaScript entry point file.",
        "srcs": "Depset of local JavaScript imports used by the entry point.",
    },
)

IreeWasmJsCollectionInfo = provider(
    doc = "Transitive JavaScript companion sources collected from C/C++ deps.",
    fields = {
        "js_files": "Depset of collected JavaScript source files.",
        "modules": "Depset of collected wasm import module structs.",
    },
)

IreeWasmEntryCollectionInfo = provider(
    doc = "Transitive wasm entry points collected from C/C++ deps.",
    fields = {
        "entries": "Depset of structs with `main` and `srcs` fields.",
    },
)

def _iree_wasm_cc_library_impl(ctx):
    transitive_js_files = []
    transitive_modules = []
    for dep in ctx.attr.deps:
        transitive_js_files.append(dep[IreeWasmJsInfo].js_files)
        transitive_modules.append(dep[IreeWasmJsInfo].modules)

    module = struct(
        module = ctx.attr.module,
        js_files = tuple(ctx.files.srcs),
    )
    return [
        IreeWasmJsInfo(
            js_files = depset(ctx.files.srcs, transitive = transitive_js_files),
            modules = depset([module], transitive = transitive_modules),
        ),
        CcInfo(),
    ]

iree_wasm_cc_library = rule(
    implementation = _iree_wasm_cc_library_impl,
    attrs = {
        "deps": attr.label_list(
            providers = [IreeWasmJsInfo],
            doc = "Other wasm companion libraries this module depends on.",
        ),
        "module": attr.string(
            mandatory = True,
            doc = "Wasm import module name implemented by `srcs`.",
        ),
        "srcs": attr.label_list(
            allow_files = [".js", ".mjs"],
            doc = "JavaScript files implementing this wasm import module.",
        ),
    },
    doc = "Declares JavaScript companion files for C/C++ code compiled to wasm.",
    provides = [IreeWasmJsInfo, CcInfo],
)

def _iree_wasm_entry_impl(ctx):
    return [
        IreeWasmEntryInfo(
            main = ctx.file.main,
            srcs = depset(ctx.files.srcs),
        ),
        CcInfo(),
    ]

iree_wasm_entry = rule(
    implementation = _iree_wasm_entry_impl,
    attrs = {
        "main": attr.label(
            allow_single_file = [".js", ".mjs"],
            mandatory = True,
            doc = "JavaScript entry point that instantiates the wasm binary.",
        ),
        "srcs": attr.label_list(
            allow_files = [".js", ".mjs"],
            doc = "Local JavaScript imports used by the entry point.",
        ),
    },
    doc = "Declares the JavaScript entry point for a wasm binary.",
    provides = [IreeWasmEntryInfo, CcInfo],
)

def _collect_wasm_js_impl(target, ctx):
    transitive_js_files = []
    transitive_modules = []
    transitive_entries = []

    if IreeWasmJsInfo in target:
        transitive_js_files.append(target[IreeWasmJsInfo].js_files)
        transitive_modules.append(target[IreeWasmJsInfo].modules)
    if IreeWasmEntryInfo in target:
        entry = struct(
            main = target[IreeWasmEntryInfo].main,
            srcs = tuple(target[IreeWasmEntryInfo].srcs.to_list()),
        )
        transitive_entries.append(depset([entry]))

    if hasattr(ctx.rule.attr, "deps"):
        for dep in ctx.rule.attr.deps:
            if IreeWasmJsCollectionInfo in dep:
                transitive_js_files.append(dep[IreeWasmJsCollectionInfo].js_files)
                transitive_modules.append(dep[IreeWasmJsCollectionInfo].modules)
            if IreeWasmEntryCollectionInfo in dep:
                transitive_entries.append(dep[IreeWasmEntryCollectionInfo].entries)

    return [
        IreeWasmJsCollectionInfo(
            js_files = depset(transitive = transitive_js_files),
            modules = depset(transitive = transitive_modules),
        ),
        IreeWasmEntryCollectionInfo(
            entries = depset(transitive = transitive_entries),
        ),
    ]

collect_wasm_js = aspect(
    implementation = _collect_wasm_js_impl,
    attr_aspects = ["deps"],
    doc = "Collects wasm JavaScript companions and entries through C/C++ deps.",
)

_WASM32_COMPATIBLE_WITH = ["@platforms//cpu:wasm32"]

def _with_wasm_target_compatibility(kwargs):
    kwargs = dict(kwargs)
    kwargs["cxxopts"] = _CXX_OPTIONS.cxxopts + kwargs.get("cxxopts", [])
    kwargs["features"] = _CXX_OPTIONS.features + kwargs.get("features", [])
    kwargs["target_compatible_with"] = kwargs.get("target_compatible_with", []) + _WASM32_COMPATIBLE_WITH
    return kwargs

def discover_wasm_entry(targets):
    """Finds a single `iree_wasm_entry` reachable from `targets`.

    Args:
      targets: Targets with the wasm collection aspect applied.

    Returns:
      A struct with `main` and `srcs` fields, or `None` when no entry exists.
    """
    entries = []
    for target in targets:
        if IreeWasmEntryCollectionInfo in target:
            entries.extend(target[IreeWasmEntryCollectionInfo].entries.to_list())
    if len(entries) > 1:
        fail("multiple iree_wasm_entry targets found; expected at most one")
    return entries[0] if entries else None

def collect_and_bundle_wasm(ctx, wasm_binary, main_js, cc_deps, bundler, main_srcs = []):
    """Bundles transitive JavaScript companions with a wasm binary.

    Args:
      ctx: Rule context used to register actions.
      wasm_binary: Raw wasm binary file.
      main_js: JavaScript entry point file.
      cc_deps: Targets with the wasm collection aspect applied.
      bundler: Executable bundler tool.
      main_srcs: Local JavaScript imports used by `main_js`.

    Returns:
      A struct containing the generated JavaScript bundle and its adjacent
      staged WebAssembly binary.
    """
    transitive_modules = []
    transitive_js_files = []
    for dep in cc_deps:
        if IreeWasmJsCollectionInfo in dep:
            transitive_modules.append(dep[IreeWasmJsCollectionInfo].modules)
            transitive_js_files.append(dep[IreeWasmJsCollectionInfo].js_files)

    modules_file = ctx.actions.declare_file(ctx.label.name + "_wasm_modules.json")
    module_entries = []
    for module in depset(transitive = transitive_modules).to_list():
        for js_file in module.js_files:
            module_entries.append(json.encode({
                "module": module.module,
                "path": js_file.path,
            }))
    ctx.actions.write(
        output = modules_file,
        content = "[" + ",".join(module_entries) + "]",
    )

    output_mjs = ctx.actions.declare_file(ctx.label.name + ".mjs")
    output_wasm = ctx.actions.declare_file(ctx.label.name + ".wasm")
    ctx.actions.symlink(
        output = output_wasm,
        target_file = wasm_binary,
    )
    arguments = ctx.actions.args()
    arguments.add("--wasm", wasm_binary)
    arguments.add("--wasm-filename", output_wasm.basename)
    arguments.add("--main", main_js)
    arguments.add("--modules", modules_file)
    arguments.add("--output", output_mjs)
    ctx.actions.run(
        executable = bundler,
        arguments = [arguments],
        inputs = depset(
            [wasm_binary, main_js, modules_file] + list(main_srcs),
            transitive = transitive_js_files,
        ),
        mnemonic = "IreeWasmBundle",
        outputs = [output_mjs],
        progress_message = "Bundling wasm JavaScript companions for %{label}",
    )
    return struct(
        binary = output_wasm,
        main = output_mjs,
    )

def _iree_wasm_bundle_impl(ctx):
    main_js = ctx.file.main
    main_srcs = []
    if main_js == None:
        entry = discover_wasm_entry(ctx.attr.cc_deps)
        if entry == None:
            fail("no wasm entry point found; provide `main` or depend on `iree_wasm_entry`")
        main_js = entry.main
        main_srcs = list(entry.srcs)

    bundle = collect_and_bundle_wasm(
        ctx = ctx,
        wasm_binary = ctx.file.binary,
        main_js = main_js,
        cc_deps = ctx.attr.cc_deps,
        bundler = ctx.executable._bundler,
        main_srcs = main_srcs,
    )
    return [DefaultInfo(
        files = depset([bundle.main]),
        runfiles = ctx.runfiles(files = [bundle.binary, bundle.main]),
    )]

_iree_wasm_bundle = rule(
    implementation = _iree_wasm_bundle_impl,
    attrs = {
        "binary": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "Raw wasm binary to bundle.",
        ),
        "cc_deps": attr.label_list(
            aspects = [collect_wasm_js],
            doc = "C/C++ deps used to collect wasm JavaScript companions.",
        ),
        "main": attr.label(
            allow_single_file = [".js", ".mjs"],
            doc = "Explicit JavaScript entry point. If omitted, an `iree_wasm_entry` is discovered from `cc_deps`.",
        ),
        "_bundler": attr.label(
            cfg = "exec",
            default = "//build_tools/wasm:wasm_binary_bundler",
            executable = True,
        ),
    },
    doc = "Bundles a wasm binary with JavaScript companions.",
)

def iree_wasm_cc_binary(name, main = None, srcs = None, deps = None, **kwargs):
    """Defines a wasm C/C++ binary, JS bundle, and Node runner.

    Args:
      name: Public runner target name.
      main: Optional JavaScript entry point label.
      srcs: C/C++ source files for the raw wasm binary.
      deps: C/C++ and wasm companion dependencies.
      **kwargs: Additional attributes forwarded to the raw `cc_binary`.
    """
    kwargs = _with_wasm_target_compatibility(kwargs)
    cc_binary(
        name = name + "_wasm",
        srcs = srcs or [],
        deps = deps or [],
        **kwargs
    )

    bundle_kwargs = {
        "binary": ":" + name + "_wasm",
        "cc_deps": deps or [],
    }
    if main != None:
        bundle_kwargs["main"] = main
    _iree_wasm_bundle(
        name = name + "_bundle",
        target_compatible_with = kwargs["target_compatible_with"],
        **bundle_kwargs
    )

    sh_binary(
        name = name,
        args = ["$(rootpath :" + name + "_bundle)"],
        data = [":" + name + "_bundle"],
        srcs = ["//build_tools/wasm:wasm_node_test_runner.sh"],
        target_compatible_with = kwargs["target_compatible_with"],
    )

def iree_wasm_cc_test(name, main = None, srcs = None, deps = None, **kwargs):
    """Defines a wasm C/C++ test, JS bundle, and Node test runner.

    Args:
      name: Public test target name.
      main: Optional JavaScript entry point label.
      srcs: C/C++ source files for the raw wasm binary.
      deps: C/C++ and wasm companion dependencies.
      **kwargs: Additional attributes forwarded to the raw `cc_binary`.
    """
    kwargs.pop("testonly", None)
    kwargs = _with_wasm_target_compatibility(kwargs)
    cc_binary(
        name = name + "_wasm",
        srcs = srcs or [],
        deps = deps or [],
        testonly = True,
        **kwargs
    )

    bundle_kwargs = {
        "binary": ":" + name + "_wasm",
        "cc_deps": deps or [],
        "testonly": True,
    }
    if main != None:
        bundle_kwargs["main"] = main
    _iree_wasm_bundle(
        name = name + "_bundle",
        target_compatible_with = kwargs["target_compatible_with"],
        **bundle_kwargs
    )

    sh_test(
        name = name,
        args = ["$(rootpath :" + name + "_bundle)"],
        data = [":" + name + "_bundle"],
        env_inherit = [
            "IREE_WASM_NODE",
            "PATH",
        ],
        srcs = ["//build_tools/wasm:wasm_node_test_runner.sh"],
        target_compatible_with = kwargs["target_compatible_with"],
    )
