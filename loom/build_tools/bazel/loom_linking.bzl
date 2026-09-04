# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared link actions for Loom libraries and final products."""

_LOOM_LINK_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:link_toolchain_type")

LoomLibraryInfo = provider(
    doc = "One relocatable Loom module and its independent dependencies.",
    fields = {
        "module": "Relocatable Loom bytecode module owned by this library.",
        "transitive_dependencies": "Depset of separate dependency modules, excluding module.",
    },
)

def _collect_dependency_modules(dependency_infos):
    direct_modules = [info.module for info in dependency_infos]
    dependency_closure = depset(
        transitive = [info.transitive_dependencies for info in dependency_infos],
    )
    direct_module_paths = {module.path: True for module in direct_modules}
    transitive_modules = [
        module
        for module in dependency_closure.to_list()
        if module.path not in direct_module_paths
    ]
    return struct(
        direct = direct_modules,
        transitive = transitive_modules,
    )

def _declare_relocatable_module(
        ctx,
        sources,
        dependency_infos,
        output_stem,
        mnemonic,
        progress_message):
    dependencies = _collect_dependency_modules(dependency_infos)
    transitive_dependencies = depset(
        direct = dependencies.direct,
        transitive = [info.transitive_dependencies for info in dependency_infos],
    )
    module = ctx.actions.declare_file(output_stem + ".loombc")
    dependency_report = ctx.actions.declare_file(
        output_stem + ".dependencies.json",
    )
    args = ctx.actions.args()
    args.add("--mode=merge")
    args.add("--strict-deps")
    args.add("--dependency-component=%s" % ctx.label)
    args.add("--dependency-report=%s" % dependency_report.path)
    args.add("--to=bc")
    args.add("--output=%s" % module.path)
    args.add_all(sources)
    args.add_all(dependencies.direct, format_each = "--library=%s")
    args.add_all(
        dependencies.transitive,
        format_each = "--transitive-library=%s",
    )

    tool = ctx.toolchains[_LOOM_LINK_TOOLCHAIN_TYPE].tool
    ctx.actions.run(
        arguments = [args],
        executable = tool.files_to_run,
        inputs = depset(
            direct = sources + dependencies.direct + dependencies.transitive,
        ),
        mnemonic = mnemonic,
        outputs = [module, dependency_report],
        progress_message = progress_message,
    )
    return struct(
        dependency_report = dependency_report,
        module = module,
        transitive_dependencies = transitive_dependencies,
    )

def _declare_linked_module(
        ctx,
        direct_modules,
        transitive_modules,
        roots,
        configs,
        target_profile,
        output_stem,
        mnemonic,
        progress_message):
    module = ctx.actions.declare_file(output_stem + ".loombc")
    args = ctx.actions.args()
    args.add("--mode=link")
    args.add("--strip-check")
    args.add("--require-resolved-config")
    args.add("--to=bc")
    args.add("--output=%s" % module.path)
    if roots:
        args.add_all(direct_modules, format_each = "--library=%s")
        args.add_all(roots, format_each = "--root=%s")
    else:
        args.add_all(direct_modules, format_each = "--root-library=%s")
    args.add_all(
        transitive_modules,
        format_each = "--transitive-library=%s",
    )
    for key in sorted(configs.keys()):
        args.add("--config=%s=%s" % (key, configs[key]))
    if target_profile:
        args.add(
            "--target=%s:%s" % (
                target_profile.family,
                target_profile.selector,
            ),
        )

    tool = ctx.toolchains[_LOOM_LINK_TOOLCHAIN_TYPE].tool
    ctx.actions.run(
        arguments = [args],
        executable = tool.files_to_run,
        inputs = depset(direct = direct_modules + transitive_modules),
        mnemonic = mnemonic,
        outputs = [module],
        progress_message = progress_message,
    )
    return module

def _declare_test_module(
        ctx,
        root_module,
        dependency_infos,
        output_stem,
        mnemonic,
        progress_message):
    dependencies = _collect_dependency_modules(dependency_infos)
    module = ctx.actions.declare_file(output_stem + ".loombc")
    args = ctx.actions.args()
    args.add("--mode=link")
    args.add("--include-input-tests")
    args.add("--to=bc")
    args.add("--output=%s" % module.path)
    args.add(root_module)
    args.add_all(dependencies.direct, format_each = "--library=%s")
    args.add_all(
        dependencies.transitive,
        format_each = "--transitive-library=%s",
    )

    tool = ctx.toolchains[_LOOM_LINK_TOOLCHAIN_TYPE].tool
    ctx.actions.run(
        arguments = [args],
        executable = tool.files_to_run,
        inputs = depset(
            direct = [root_module] + dependencies.direct + dependencies.transitive,
        ),
        mnemonic = mnemonic,
        outputs = [module],
        progress_message = progress_message,
    )
    return module

loom_linking = struct(
    collect_dependency_modules = _collect_dependency_modules,
    declare_linked_module = _declare_linked_module,
    declare_relocatable_module = _declare_relocatable_module,
    declare_test_module = _declare_test_module,
)
