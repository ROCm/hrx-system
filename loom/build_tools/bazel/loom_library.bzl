# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public rules for reusable Loom source and kernel libraries."""

_LOOM_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:loom_toolchain_type")

LoomLibraryInfo = provider(
    doc = "A verified Loom source closure and its linked archive.",
    fields = {
        "module": "Linked Loom bytecode archive.",
        "sources": "Ordered depset of source modules in the archive.",
    },
)

LoomCompileTargetInfo = provider(
    doc = "A named Loom compiler backend and target qualification profile.",
    fields = {
        "artifact_extension": "Filename extension for compiled artifacts.",
        "backend": "loom-compile backend name.",
        "target": "Optional loom-compile target name.",
    },
)

LoomCompilationInfo = provider(
    doc = "One target-qualified Loom compilation and its report.",
    fields = {
        "artifact": "Compiled target artifact.",
        "library": "Source Loom library used for compilation.",
        "report": "Summary compile report for the target artifact.",
        "target": "Target qualification profile used for compilation.",
    },
)

def _loom_compile_target_impl(ctx):
    if not ctx.attr.backend:
        fail("%s requires a non-empty backend" % ctx.label)
    if not ctx.attr.artifact_extension:
        fail("%s requires a non-empty artifact_extension" % ctx.label)
    if "/" in ctx.attr.artifact_extension or ctx.attr.artifact_extension.startswith("."):
        fail("%s artifact_extension must be a bare filename extension" % ctx.label)
    return [
        LoomCompileTargetInfo(
            artifact_extension = ctx.attr.artifact_extension,
            backend = ctx.attr.backend,
            target = ctx.attr.target,
        ),
    ]

loom_compile_target = rule(
    implementation = _loom_compile_target_impl,
    attrs = {
        "artifact_extension": attr.string(
            mandatory = True,
            doc = "Bare filename extension for compiled artifacts.",
        ),
        "backend": attr.string(
            mandatory = True,
            doc = "loom-compile backend name, such as amdgpu-hal.",
        ),
        "target": attr.string(
            doc = "Optional backend target name, such as gfx11-generic.",
        ),
    },
    doc = """Declares one reusable Loom compiler target profile.

    Target packages should use Bazel's target_compatible_with attribute to name
    the configuration capabilities required by their backend target. Generated
    compilation targets inherit incompatibility through this profile dependency
    and are skipped when the selected compiler tool omits that implementation.
    """,
)

def _loom_library_impl(ctx):
    if not ctx.files.srcs and not ctx.attr.deps:
        fail("%s requires at least one source across srcs and deps" % ctx.label)
    transitive_sources = [
        dep[LoomLibraryInfo].sources
        for dep in ctx.attr.deps
    ]
    sources = depset(
        direct = ctx.files.srcs,
        transitive = transitive_sources,
        order = "preorder",
    )
    output = ctx.actions.declare_file(ctx.label.name + ".loombc")
    args = ctx.actions.args()
    args.add("--mode=archive")
    args.add("--to=bc")
    args.add("--output=%s" % output.path)
    args.add_all(sources)

    toolchain = ctx.toolchains[_LOOM_TOOLCHAIN_TYPE]
    ctx.actions.run(
        arguments = [args],
        executable = toolchain.link.files_to_run,
        inputs = sources,
        mnemonic = "LoomLibrary",
        outputs = [output],
        progress_message = "Linking Loom library %s" % ctx.label,
    )

    return [
        DefaultInfo(files = depset([output])),
        LoomLibraryInfo(
            module = output,
            sources = sources,
        ),
    ]

_loom_library = rule(
    implementation = _loom_library_impl,
    attrs = {
        "deps": attr.label_list(
            providers = [LoomLibraryInfo],
            doc = "Loom libraries supplying transitive source modules.",
        ),
        "srcs": attr.label_list(
            allow_files = [".loom", ".loombc"],
            doc = "Ordered Loom text or bytecode source modules.",
        ),
    },
    doc = "Links one reusable Loom source closure into a bytecode archive.",
    toolchains = [_LOOM_TOOLCHAIN_TYPE],
)

def _loom_compile_impl(ctx):
    library = ctx.attr.library[LoomLibraryInfo]
    target = ctx.attr.target[LoomCompileTargetInfo]
    artifact = ctx.actions.declare_file(
        "%s.%s" % (ctx.label.name, target.artifact_extension),
    )
    report = ctx.actions.declare_file(ctx.label.name + ".compile.json")
    args = ctx.actions.args()
    args.add(library.module)
    args.add("--backend=%s" % target.backend)
    if target.target:
        args.add("--target=%s" % target.target)
    args.add_all(ctx.attr.roots, format_each = "--root=%s")
    args.add_all(ctx.attr.configs, format_each = "--config=%s")
    args.add("--output=%s" % artifact.path)
    args.add("--compile-report=summary")
    args.add("--compile-report-output=%s" % report.path)

    toolchain = ctx.toolchains[_LOOM_TOOLCHAIN_TYPE]
    ctx.actions.run(
        arguments = [args],
        executable = toolchain.compile.files_to_run,
        inputs = depset([library.module]),
        mnemonic = "LoomCompile",
        outputs = [artifact, report],
        progress_message = "Compiling Loom library %s for %s" % (
            ctx.attr.library.label,
            ctx.attr.target.label,
        ),
    )

    return [
        DefaultInfo(files = depset([artifact, report])),
        LoomCompilationInfo(
            artifact = artifact,
            library = library,
            report = report,
            target = target,
        ),
    ]

_loom_compile = rule(
    implementation = _loom_compile_impl,
    attrs = {
        "configs": attr.string_list(
            doc = "Compile-time key=value bindings.",
        ),
        "library": attr.label(
            mandatory = True,
            providers = [LoomLibraryInfo],
            doc = "Loom library to compile.",
        ),
        "roots": attr.string_list(
            doc = "Optional entry-point symbols selected from the library.",
        ),
        "target": attr.label(
            mandatory = True,
            providers = [LoomCompileTargetInfo],
            doc = "Compiler backend and target qualification profile.",
        ),
    },
    doc = "Compiles a Loom library and emits a target artifact and report.",
    toolchains = [_LOOM_TOOLCHAIN_TYPE],
)

def _tool_runfiles(ctx, tool, files):
    runfiles = ctx.runfiles(files = files + [tool.executable])
    if tool.runfiles:
        runfiles = runfiles.merge(tool.runfiles)
    return runfiles

def _write_test_launcher(ctx, tool, input_file, tool_args):
    output = ctx.actions.declare_file(ctx.label.name + ".sh")
    command_args = "".join([
        " \\\n  \"%s\"" % arg
        for arg in tool_args
    ])
    content = (
        "#!/usr/bin/env bash\n" +
        "set -euo pipefail\n" +
        "RUNFILES=\"${{RUNFILES_DIR:-$0.runfiles}}\"\n" +
        "cd \"${{RUNFILES}}/{workspace}\"\n" +
        "exec \"${{PWD}}/{tool}\" \"${{PWD}}/{input}\"{args}\n"
    ).format(
        workspace = ctx.workspace_name,
        tool = tool.executable.short_path,
        input = input_file.short_path,
        args = command_args,
    )
    ctx.actions.write(
        content = content,
        is_executable = True,
        output = output,
    )
    return output

def _loom_format_test_impl(ctx):
    tool = ctx.toolchains[_LOOM_TOOLCHAIN_TYPE].format
    output = _write_test_launcher(ctx, tool, ctx.file.src, ["--check"])
    return [
        DefaultInfo(
            executable = output,
            files = depset([output]),
            runfiles = _tool_runfiles(ctx, tool, [ctx.file.src]),
        ),
    ]

_loom_format_test = rule(
    implementation = _loom_format_test_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = [".loom"],
            mandatory = True,
            doc = "Canonical Loom text source to verify.",
        ),
    },
    doc = "Verifies that one Loom source file is canonical and valid.",
    test = True,
    toolchains = [_LOOM_TOOLCHAIN_TYPE],
)

def _loom_plan_test_impl(ctx):
    tool = ctx.toolchains[_LOOM_TOOLCHAIN_TYPE].benchmark
    module = ctx.attr.library[LoomLibraryInfo].module
    output = _write_test_launcher(
        ctx,
        tool,
        module,
        [
            "--dry-run",
            "--output-format=jsonl",
            "--compile-report=none",
        ],
    )
    return [
        DefaultInfo(
            executable = output,
            files = depset([output]),
            runfiles = _tool_runfiles(ctx, tool, [module]),
        ),
    ]

_loom_plan_test = rule(
    implementation = _loom_plan_test_impl,
    attrs = {
        "library": attr.label(
            mandatory = True,
            providers = [LoomLibraryInfo],
            doc = "Kernel library whose check.benchmark work graph is planned.",
        ),
    },
    doc = "Plans every benchmark declared by a Loom kernel library.",
    test = True,
    toolchains = [_LOOM_TOOLCHAIN_TYPE],
)

def _loom_compilation_test_impl(ctx):
    compilation = ctx.attr.compilation[LoomCompilationInfo]
    output = ctx.actions.declare_file(ctx.label.name + ".sh")
    content = (
        "#!/usr/bin/env bash\n" +
        "set -euo pipefail\n" +
        "RUNFILES=\"${RUNFILES_DIR:-$0.runfiles}\"\n" +
        "cd \"${RUNFILES}/%s\"\n" % ctx.workspace_name +
        "test -s \"${PWD}/%s\"\n" % compilation.artifact.short_path +
        "test -s \"${PWD}/%s\"\n" % compilation.report.short_path
    )
    ctx.actions.write(
        content = content,
        is_executable = True,
        output = output,
    )
    return [
        DefaultInfo(
            executable = output,
            files = depset([output]),
            runfiles = ctx.runfiles(files = [
                compilation.artifact,
                compilation.report,
            ]),
        ),
    ]

_loom_compilation_test = rule(
    implementation = _loom_compilation_test_impl,
    attrs = {
        "compilation": attr.label(
            mandatory = True,
            providers = [LoomCompilationInfo],
            doc = "Target compilation whose outputs must be non-empty.",
        ),
    },
    doc = "Verifies that a target compilation emitted its promised outputs.",
    test = True,
)

def loom_compile(
        name,
        library,
        target,
        configs = [],
        roots = [],
        tags = [],
        visibility = None):
    """Compiles a Loom library for one reusable target profile."""
    _loom_compile(
        name = name,
        configs = configs,
        library = library,
        roots = roots,
        tags = tags,
        target = target,
        visibility = visibility,
    )

def _compile_target_suffix(target):
    target_text = str(target)
    if ":" in target_text:
        target_text = target_text.split(":")[-1]
    elif "/" in target_text:
        target_text = target_text.split("/")[-1]
    return target_text.replace("-", "_").replace(".", "_").replace("+", "_")

def _declare_library(
        name,
        srcs,
        deps,
        compile_targets,
        plan_benchmarks,
        tags,
        visibility):
    _loom_library(
        name = name,
        srcs = srcs,
        deps = deps,
        tags = tags,
        visibility = visibility,
    )

    tests = []
    for index, src in enumerate(srcs):
        test_name = "%s_format_%d_test" % (name, index)
        _loom_format_test(
            name = test_name,
            src = src,
            tags = tags + ["hostonly"],
            visibility = ["//visibility:private"],
        )
        tests.append(test_name)

    if plan_benchmarks:
        plan_test_name = name + "_plan_test"
        _loom_plan_test(
            name = plan_test_name,
            library = ":" + name,
            tags = tags + ["hostonly"],
            visibility = ["//visibility:private"],
        )
        tests.append(plan_test_name)

    compile_names = {}
    for target in compile_targets:
        compile_name = "%s_compile_%s" % (name, _compile_target_suffix(target))
        if compile_name in compile_names:
            fail(
                "%s compile target %s has the same generated name as %s" % (
                    name,
                    target,
                    compile_names[compile_name],
                ),
            )
        compile_names[compile_name] = target
        _loom_compile(
            name = compile_name,
            library = ":" + name,
            tags = tags,
            target = target,
            visibility = ["//visibility:private"],
        )
        compile_test_name = compile_name + "_test"
        _loom_compilation_test(
            name = compile_test_name,
            compilation = ":" + compile_name,
            tags = tags + ["hostonly"],
            visibility = ["//visibility:private"],
        )
        tests.append(compile_test_name)

    suite_kwargs = {
        "name": name + "_test",
        "tags": tags + ["hostonly"],
        "tests": tests,
    }
    if visibility != None:
        suite_kwargs["visibility"] = visibility
    native.test_suite(**suite_kwargs)

def loom_library(
        name,
        srcs,
        deps = [],
        tags = [],
        visibility = None):
    """Declares a reusable function/template library and format coverage."""
    _declare_library(
        name = name,
        srcs = srcs,
        deps = deps,
        compile_targets = [],
        plan_benchmarks = False,
        tags = tags,
        visibility = visibility,
    )

def loom_kernel_library(
        name,
        srcs,
        deps = [],
        compile_targets = [],
        tags = [],
        visibility = None):
    """Declares launchable kernels with format, plan, and compile coverage."""
    _declare_library(
        name = name,
        srcs = srcs,
        deps = deps,
        compile_targets = compile_targets,
        plan_benchmarks = True,
        tags = tags,
        visibility = visibility,
    )
