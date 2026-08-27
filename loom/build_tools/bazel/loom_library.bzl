# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public rules for reusable Loom source and kernel libraries."""

load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//build_tools/bazel:cc_attrs.bzl", "cc_attrs")
load("//build_tools/bazel:requirements.bzl", "apply_test_requirements")
load(
    ":loom_linking.bzl",
    "loom_linking",
    _LoomLibraryInfo = "LoomLibraryInfo",
)

_LOOM_BENCHMARK_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:benchmark_toolchain_type")
_LOOM_COMPILE_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:compile_toolchain_type")
_LOOM_FORMAT_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:format_toolchain_type")
_LOOM_LINT_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:lint_toolchain_type")
_LOOM_LINK_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:link_toolchain_type")
_LOOM_TEST_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:test_toolchain_type")

LoomLibraryInfo = _LoomLibraryInfo

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

LoomExecutionTestInfo = provider(
    doc = "A correctness test executing one authored Loom source through one profile.",
    fields = {
        "libraries": "Ordered Loom libraries supplied to the correctness runner.",
        "profile_name": "Stable execution profile name.",
        "runner": "Resolved correctness runner executable.",
        "runner_args": "Profile arguments passed to the correctness runner.",
        "source": "Authored Loom source consumed as the primary input.",
    },
)

def loom_execution_profile(
        name,
        target_family,
        target_class,
        executor,
        runner_args = [],
        build_requirements = [],
        run_requirements = [],
        resource_group = None,
        tags = []):
    """Defines immutable execution policy expanded by Loom library macros.

    Args:
      name: Stable profile name used in generated target names and query tags.
      target_family: Compiler target family, such as amdgpu or spirv.
      target_class: Broad target class, such as gpu or cpu.
      executor: Execution environment, such as hardware or reference.
      runner_args: Arguments appended after explicit libraries passed to
        iree-test-loom.
      build_requirements: Build requirements needed by the correctness runner.
      run_requirements: Runtime resources needed to execute the test.
      resource_group: Optional local resource group serializing competing tests.
      tags: Additional stable tags applied to generated tests.

    Returns:
      An immutable profile value suitable for execution_profiles lists.
    """
    for field_name, value in [
        ("name", name),
        ("target_family", target_family),
        ("target_class", target_class),
        ("executor", executor),
    ]:
        if type(value) != type("") or not value:
            fail("loom execution profile %s must be a non-empty string" % field_name)
    requirement_ids = {}
    for phase, requirements in [
        ("build", build_requirements),
        ("run", run_requirements),
    ]:
        for requirement in requirements:
            if requirement.phase != phase:
                fail(
                    "execution profile %s received %s requirement %s as %s" % (
                        name,
                        requirement.phase,
                        requirement.id,
                        phase,
                    ),
                )
            if requirement.id in requirement_ids:
                fail(
                    "execution profile %s repeats requirement %s" %
                    (name, requirement.id),
                )
            requirement_ids[requirement.id] = True
    return struct(
        build_requirements = build_requirements,
        executor = executor,
        kind = "loom_execution_profile",
        name = name,
        resource_group = resource_group,
        run_requirements = run_requirements,
        runner_args = runner_args,
        tags = tags,
        target_class = target_class,
        target_family = target_family,
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
    dependency_infos = [dep[LoomLibraryInfo] for dep in ctx.attr.deps]
    artifacts = loom_linking.declare_relocatable_module(
        ctx = ctx,
        sources = ctx.files.srcs,
        dependency_infos = dependency_infos,
        output_stem = ctx.label.name,
        mnemonic = "LoomLibrary",
        progress_message = "Linking Loom library %s" % ctx.label,
    )

    return [
        DefaultInfo(files = depset([artifacts.module])),
        OutputGroupInfo(
            dependency_reports = depset([artifacts.dependency_report]),
        ),
        LoomLibraryInfo(
            module = artifacts.module,
            transitive_dependencies = artifacts.transitive_dependencies,
        ),
    ]

_loom_library = rule(
    implementation = _loom_library_impl,
    attrs = {
        "deps": attr.label_list(
            providers = [LoomLibraryInfo],
            doc = "Direct Loom library dependencies kept as separate modules.",
        ),
        "srcs": attr.label_list(
            allow_files = [".loom", ".loombc"],
            doc = "Ordered Loom text or bytecode source modules.",
        ),
    },
    doc = "Merges direct sources into one relocatable Loom bytecode module.",
    toolchains = [_LOOM_LINK_TOOLCHAIN_TYPE],
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

    tool = ctx.toolchains[_LOOM_COMPILE_TOOLCHAIN_TYPE].tool
    ctx.actions.run(
        arguments = [args],
        executable = tool.files_to_run,
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
    toolchains = [_LOOM_COMPILE_TOOLCHAIN_TYPE],
)

def _tool_runfiles(ctx, tool, files):
    runfiles = ctx.runfiles(files = files + [tool.executable])
    if tool.runfiles:
        runfiles = runfiles.merge(tool.runfiles)
    return runfiles

def _shell_quote(value):
    return "'" + value.replace("'", "'\"'\"'") + "'"

def _write_test_launcher(ctx, tool, input_file, tool_args):
    output = ctx.actions.declare_file(ctx.label.name + ".sh")
    command_args = "".join([
        " \\\n  %s" % _shell_quote(arg)
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

def _loom_execution_test_launcher_impl(ctx):
    tool = ctx.toolchains[_LOOM_TEST_TOOLCHAIN_TYPE].tool
    libraries = [dep[LoomLibraryInfo] for dep in ctx.attr.libraries]
    library_modules = [library.module for library in libraries]
    output = _write_test_launcher(
        ctx,
        tool,
        ctx.file.src,
        ["--library=%s" % module.short_path for module in library_modules] +
        ctx.attr.runner_args,
    )
    return [
        DefaultInfo(
            executable = output,
            files = depset([output]),
            runfiles = _tool_runfiles(
                ctx,
                tool,
                [ctx.file.src] + library_modules,
            ),
        ),
        LoomExecutionTestInfo(
            libraries = libraries,
            profile_name = ctx.attr.profile_name,
            runner = tool.executable,
            runner_args = ctx.attr.runner_args,
            source = ctx.file.src,
        ),
    ]

_loom_execution_test_launcher = rule(
    implementation = _loom_execution_test_launcher_impl,
    attrs = {
        "libraries": attr.label_list(
            providers = [LoomLibraryInfo],
            doc = "Loom libraries linked into the authored test source.",
        ),
        "profile_name": attr.string(
            mandatory = True,
            doc = "Stable execution profile name.",
        ),
        "runner_args": attr.string_list(
            doc = "Profile arguments appended after explicit libraries.",
        ),
        "src": attr.label(
            allow_single_file = [".loom"],
            mandatory = True,
            doc = "Authored Loom source containing correctness cases.",
        ),
    },
    doc = "Generates a launcher for one authored Loom execution profile.",
    executable = True,
    toolchains = [_LOOM_TEST_TOOLCHAIN_TYPE],
)

def _loom_format_test_launcher_impl(ctx):
    tool = ctx.toolchains[_LOOM_FORMAT_TOOLCHAIN_TYPE].tool
    output = _write_test_launcher(ctx, tool, ctx.file.src, ["--check"])
    return [
        DefaultInfo(
            executable = output,
            files = depset([output]),
            runfiles = _tool_runfiles(ctx, tool, [ctx.file.src]),
        ),
    ]

_loom_format_test_launcher = rule(
    implementation = _loom_format_test_launcher_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = [".loom"],
            mandatory = True,
            doc = "Canonical Loom text source to verify.",
        ),
    },
    doc = "Generates a launcher that verifies one canonical Loom source.",
    executable = True,
    toolchains = [_LOOM_FORMAT_TOOLCHAIN_TYPE],
)

def _loom_lint_test_launcher_impl(ctx):
    tool = ctx.toolchains[_LOOM_LINT_TOOLCHAIN_TYPE].tool
    output = _write_test_launcher(ctx, tool, ctx.file.src, [])
    return [
        DefaultInfo(
            executable = output,
            files = depset([output]),
            runfiles = _tool_runfiles(ctx, tool, [ctx.file.src]),
        ),
    ]

_loom_lint_test_launcher = rule(
    implementation = _loom_lint_test_launcher_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = [".loom"],
            mandatory = True,
            doc = "Authored Loom text source to check.",
        ),
    },
    doc = "Generates a launcher that checks one Loom source file.",
    executable = True,
    toolchains = [_LOOM_LINT_TOOLCHAIN_TYPE],
)

def _loom_plan_test_launcher_impl(ctx):
    tool = ctx.toolchains[_LOOM_BENCHMARK_TOOLCHAIN_TYPE].tool
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

_loom_plan_test_launcher = rule(
    implementation = _loom_plan_test_launcher_impl,
    attrs = {
        "library": attr.label(
            mandatory = True,
            providers = [LoomLibraryInfo],
            doc = "Kernel library whose check.benchmark work graph is planned.",
        ),
    },
    doc = "Generates a launcher that plans every declared Loom benchmark.",
    executable = True,
    toolchains = [_LOOM_BENCHMARK_TOOLCHAIN_TYPE],
)

def _loom_compilation_test_launcher_impl(ctx):
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

_loom_compilation_test_launcher = rule(
    implementation = _loom_compilation_test_launcher_impl,
    attrs = {
        "compilation": attr.label(
            mandatory = True,
            providers = [LoomCompilationInfo],
            doc = "Target compilation whose outputs must be non-empty.",
        ),
    },
    doc = "Generates a launcher that verifies target compilation outputs.",
    executable = True,
)

def _declare_launcher_test(name, launcher_rule, launcher_attrs, test_kwargs):
    """Wraps a generated shell launcher in rules_shell's platform runner."""
    launcher_name = name + "_launcher"
    launcher_kwargs = dict(launcher_attrs)
    launcher_kwargs.update({
        "name": launcher_name,
        "tags": test_kwargs.get("tags", []) + ["manual"],
        "testonly": True,
        "visibility": ["//visibility:private"],
    })
    target_compatible_with = test_kwargs.get("target_compatible_with")
    if target_compatible_with != None:
        launcher_kwargs["target_compatible_with"] = target_compatible_with
    launcher_rule(**launcher_kwargs)

    sh_test(
        name = name,
        srcs = [":" + launcher_name],
        data = [":" + launcher_name],
        **test_kwargs
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

def _execution_profile_tags(profile):
    return [
        "loom-execution-profile=%s" % profile.name,
        "loom-target-family=%s" % profile.target_family,
        "loom-target-class=%s" % profile.target_class,
        "loom-executor=%s" % profile.executor,
    ]

def _declare_execution_test(name, src, libraries, profile, tags):
    if getattr(profile, "kind", None) != "loom_execution_profile":
        fail("%s execution profile was not created by loom_execution_profile" % name)
    test_kwargs = apply_test_requirements(
        {
            "tags": tags + profile.tags + _execution_profile_tags(profile),
        },
        build_requirements = profile.build_requirements,
        run_requirements = profile.run_requirements,
        resource_group = profile.resource_group,
    )
    resource_group = test_kwargs.pop("resource_group", None)
    test_kwargs["tags"] = cc_attrs.with_resource_group_tags(
        test_kwargs.get("tags"),
        resource_group,
    )
    test_kwargs["visibility"] = ["//visibility:private"]
    _declare_launcher_test(
        name = name,
        launcher_rule = _loom_execution_test_launcher,
        launcher_attrs = {
            "libraries": libraries,
            "profile_name": profile.name,
            "runner_args": profile.runner_args,
            "src": src,
        },
        test_kwargs = test_kwargs,
    )

def _declare_library(
        name,
        srcs,
        deps,
        compile_targets,
        execution_profiles,
        plan_benchmarks,
        module_testonly,
        module_visibility,
        tags,
        visibility):
    _loom_library(
        name = name,
        srcs = srcs,
        deps = deps,
        tags = tags,
        testonly = module_testonly,
        visibility = module_visibility,
    )

    tests = []
    for index, src in enumerate(srcs):
        lint_test_name = "%s_lint_%d_test" % (name, index)
        _declare_launcher_test(
            name = lint_test_name,
            launcher_rule = _loom_lint_test_launcher,
            launcher_attrs = {"src": src},
            test_kwargs = {
                "tags": tags + ["hostonly"],
                "visibility": ["//visibility:private"],
            },
        )
        tests.append(lint_test_name)

        test_name = "%s_format_%d_test" % (name, index)
        _declare_launcher_test(
            name = test_name,
            launcher_rule = _loom_format_test_launcher,
            launcher_attrs = {"src": src},
            test_kwargs = {
                "tags": tags + ["hostonly"],
                "visibility": ["//visibility:private"],
            },
        )
        tests.append(test_name)

    if plan_benchmarks:
        plan_test_name = name + "_plan_test"
        _declare_launcher_test(
            name = plan_test_name,
            launcher_rule = _loom_plan_test_launcher,
            launcher_attrs = {"library": ":" + name},
            test_kwargs = {
                "tags": tags + ["hostonly"],
                "visibility": ["//visibility:private"],
            },
        )
        tests.append(plan_test_name)

    if execution_profiles and not srcs:
        fail("%s requires authored srcs for execution profiles" % name)
    execution_names = {}
    for profile in execution_profiles:
        profile_suffix = _compile_target_suffix(profile.name)
        for source_index, src in enumerate(srcs):
            if len(srcs) == 1:
                execution_name = "%s_execute_%s_test" % (name, profile_suffix)
            else:
                execution_name = "%s_execute_%s_%d_test" % (
                    name,
                    profile_suffix,
                    source_index,
                )
            if execution_name in execution_names:
                fail(
                    "%s execution profile %s has the same generated name as %s" % (
                        name,
                        profile.name,
                        execution_names[execution_name],
                    ),
                )
            execution_names[execution_name] = profile.name
            _declare_execution_test(
                name = execution_name,
                src = src,
                libraries = deps,
                profile = profile,
                tags = tags,
            )
            tests.append(execution_name)

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
            testonly = module_testonly,
            visibility = ["//visibility:private"],
        )
        compile_test_name = compile_name + "_test"
        _declare_launcher_test(
            name = compile_test_name,
            launcher_rule = _loom_compilation_test_launcher,
            launcher_attrs = {"compilation": ":" + compile_name},
            test_kwargs = {
                "tags": tags + ["hostonly"],
                "visibility": ["//visibility:private"],
            },
        )
        tests.append(compile_test_name)

    suite_kwargs = {
        "name": name + "_test",
        "tags": tags,
        "tests": tests,
    }
    if visibility != None:
        suite_kwargs["visibility"] = visibility
    native.test_suite(**suite_kwargs)

def loom_library(
        name,
        srcs = [],
        deps = [],
        tags = [],
        visibility = None):
    """Declares a reusable Loom function or template library.

    The rule merges only ``srcs`` into its relocatable bytecode module. Direct
    ``deps`` and their transitive closure remain independent linker inputs, and
    strict dependency analysis rejects source references satisfied only by the
    transitive closure. The ``dependency_reports`` output group contains the
    schema-versioned JSON analysis report.
    """
    _declare_library(
        name = name,
        srcs = srcs,
        deps = deps,
        compile_targets = [],
        execution_profiles = [],
        plan_benchmarks = False,
        module_testonly = False,
        module_visibility = visibility,
        tags = tags,
        visibility = visibility,
    )

def loom_test_library(
        name,
        srcs,
        deps = [],
        compile_targets = [],
        execution_profiles = [],
        tags = [],
        visibility = None):
    """Declares test-only Loom wrappers and their generated test suite.

    The wrapper module is private and test-only. Its ``<name>_test`` suite
    formats every source, plans declared benchmarks, compiles requested target
    qualifications, and executes each authored source through every requested
    profile with ``deps`` supplied as explicit libraries.
    """
    if not srcs:
        fail("%s requires at least one authored test source" % name)
    _declare_library(
        name = name,
        srcs = srcs,
        deps = deps,
        compile_targets = compile_targets,
        execution_profiles = execution_profiles,
        plan_benchmarks = True,
        module_testonly = True,
        module_visibility = ["//visibility:private"],
        tags = tags,
        visibility = visibility,
    )

def loom_kernel_library(
        name,
        srcs,
        deps = [],
        compile_targets = [],
        execution_profiles = [],
        tags = [],
        visibility = None):
    """Declares launchable kernels with format, plan, and compile coverage."""
    _declare_library(
        name = name,
        srcs = srcs,
        deps = deps,
        compile_targets = compile_targets,
        execution_profiles = execution_profiles,
        plan_benchmarks = True,
        module_testonly = False,
        module_visibility = visibility,
        tags = tags,
        visibility = visibility,
    )
