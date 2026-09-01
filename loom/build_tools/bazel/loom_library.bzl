# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public rules for Loom libraries, kernels, and correctness tests."""

load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//build_tools/bazel:cc_attrs.bzl", "cc_attrs")
load("//build_tools/bazel:requirements.bzl", "apply_test_requirements")
load(
    "//build_tools/sanitizer:suppressions.bzl",
    "iree_sanitizer_suppression_data",
    "iree_sanitizer_suppression_env",
)
load(":loom_binary.bzl", "LoomBinaryInfo", "loom_kernel_binary")
load(
    ":loom_linking.bzl",
    "loom_linking",
    _LoomLibraryInfo = "LoomLibraryInfo",
)

_LOOM_BENCHMARK_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:benchmark_toolchain_type")
_LOOM_FORMAT_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:format_toolchain_type")
_LOOM_LINT_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:lint_toolchain_type")
_LOOM_LINK_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:link_toolchain_type")
_LOOM_TEST_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:test_toolchain_type")

LoomLibraryInfo = _LoomLibraryInfo

LoomExecutionTestInfo = provider(
    doc = "A correctness test executing one authored Loom source.",
    fields = {
        "libraries": "Ordered Loom libraries supplied to the correctness runner.",
        "profile_name": "Stable execution profile name, or empty for an unprofiled test.",
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
        sanitizer_suppressions = None,
        tags = []):
    """Defines immutable policy for Loom correctness-test execution.

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
      sanitizer_suppressions: Sanitizer suppression files keyed by sanitizer
        name and required by the execution environment.
      tags: Additional stable tags applied to generated tests.

    Returns:
      An immutable profile value suitable for Loom test declarations.
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
        sanitizer_suppressions = sanitizer_suppressions,
        tags = tags,
        target_class = target_class,
        target_family = target_family,
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

def _loom_binary_test_launcher_impl(ctx):
    binary = ctx.attr.binary[LoomBinaryInfo]
    files = [binary.primary_artifact] + binary.reports.to_list()
    output = ctx.actions.declare_file(ctx.label.name + ".sh")
    content = (
        "#!/usr/bin/env bash\n" +
        "set -euo pipefail\n" +
        "RUNFILES=\"${RUNFILES_DIR:-$0.runfiles}\"\n" +
        "cd \"${RUNFILES}/%s\"\n" % ctx.workspace_name +
        "".join([
            "test -s \"${PWD}/%s\"\n" % file.short_path
            for file in files
        ])
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
            runfiles = ctx.runfiles(files = files),
        ),
    ]

_loom_binary_test_launcher = rule(
    implementation = _loom_binary_test_launcher_impl,
    attrs = {
        "binary": attr.label(
            mandatory = True,
            providers = [LoomBinaryInfo],
            doc = "Deployment product whose primary artifact and reports must be non-empty.",
        ),
    },
    doc = "Generates a launcher that verifies deployment product outputs.",
    executable = True,
)

def _declare_launcher_test(name, launcher_rule, launcher_attrs, test_kwargs):
    """Wraps a generated shell launcher in rules_shell's platform runner."""
    test_kwargs = dict(test_kwargs)
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
        data = [":" + launcher_name] + test_kwargs.pop("data", []),
        **test_kwargs
    )

def _name_suffix(value):
    value_text = str(value)
    if ":" in value_text:
        value_text = value_text.split(":")[-1]
    elif "/" in value_text:
        value_text = value_text.split("/")[-1]
    return value_text.replace("-", "_").replace(".", "_").replace("+", "_")

def _execution_profile_tags(profile):
    return [
        "loom-execution-profile=%s" % profile.name,
        "loom-target-family=%s" % profile.target_family,
        "loom-target-class=%s" % profile.target_class,
        "loom-executor=%s" % profile.executor,
    ]

def _declare_execution_test(
        name,
        src,
        libraries,
        profile,
        runner_args,
        size,
        tags,
        visibility):
    profile_name = ""
    profile_runner_args = []
    test_kwargs = {
        "size": size,
        "tags": tags,
    }
    if profile != None:
        if getattr(profile, "kind", None) != "loom_execution_profile":
            fail("%s execution profile was not created by loom_execution_profile" % name)
        profile_name = profile.name
        profile_runner_args = profile.runner_args
        test_kwargs = apply_test_requirements(
            {
                "size": size,
                "tags": tags + profile.tags + _execution_profile_tags(profile),
            },
            build_requirements = profile.build_requirements,
            run_requirements = profile.run_requirements,
            resource_group = profile.resource_group,
        )
        test_kwargs["data"] = iree_sanitizer_suppression_data(
            [],
            profile.sanitizer_suppressions,
        )
        test_env = iree_sanitizer_suppression_env(
            None,
            profile.sanitizer_suppressions,
        )
        if test_env:
            test_kwargs["env"] = test_env
    resource_group = test_kwargs.pop("resource_group", None)
    test_kwargs["tags"] = cc_attrs.with_resource_group_tags(
        test_kwargs.get("tags"),
        resource_group,
    )
    if visibility != None:
        test_kwargs["visibility"] = visibility
    _declare_launcher_test(
        name = name,
        launcher_rule = _loom_execution_test_launcher,
        launcher_attrs = {
            "libraries": libraries,
            "profile_name": profile_name,
            "runner_args": profile_runner_args + runner_args,
            "src": src,
        },
        test_kwargs = test_kwargs,
    )

def _declare_library(
        name,
        srcs,
        deps,
        execution_profiles,
        kernel_targets,
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
        profile_suffix = _name_suffix(profile.name)
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
                runner_args = [],
                size = "small",
                tags = tags,
                visibility = ["//visibility:private"],
            )
            tests.append(execution_name)

    binary_names = {}
    for target in kernel_targets:
        binary_name = "%s_kernel_binary_%s" % (name, _name_suffix(target))
        if binary_name in binary_names:
            fail(
                "%s kernel target %s has the same generated name as %s" % (
                    name,
                    target,
                    binary_names[binary_name],
                ),
            )
        binary_names[binary_name] = target
        loom_kernel_binary(
            name = binary_name,
            deps = [":" + name],
            tags = tags + ["manual"],
            target = target,
            testonly = True,
            visibility = ["//visibility:private"],
        )
        binary_test_name = binary_name + "_test"
        _declare_launcher_test(
            name = binary_test_name,
            launcher_rule = _loom_binary_test_launcher,
            launcher_attrs = {"binary": ":" + binary_name},
            test_kwargs = {
                "tags": tags + ["hostonly"],
                "visibility": ["//visibility:private"],
            },
        )
        tests.append(binary_test_name)

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
        execution_profiles = [],
        kernel_targets = [],
        plan_benchmarks = False,
        module_testonly = False,
        module_visibility = visibility,
        tags = tags,
        visibility = visibility,
    )

def loom_test(
        name,
        src,
        deps = [],
        args = [],
        execution_profile = None,
        size = "small",
        tags = [],
        visibility = None):
    """Executes one authored Loom source as a correctness test.

    The test runs ``iree-test-loom`` directly on ``src`` and supplies ``deps``
    as explicit Loom bytecode libraries. Each selected ``check.case`` sample is
    executed once. ``check.benchmark`` records do not trigger benchmark
    planning, timing, or repetition.

    Args:
      name: Name of the generated test target.
      src: Authored ``.loom`` source containing correctness cases.
      deps: Loom libraries linked into the source before execution.
      args: Additional arguments passed to ``iree-test-loom``.
      execution_profile: Optional execution environment and requirement policy.
      size: Bazel test size.
      tags: Additional tags applied to the test.
      visibility: Bazel visibility of the generated test target.
    """
    if not src:
        fail("%s requires one authored test source" % name)
    _declare_execution_test(
        name = name,
        src = src,
        libraries = deps,
        profile = execution_profile,
        runner_args = args,
        size = size,
        tags = tags,
        visibility = visibility,
    )

def _loom_test_suite_source_name(src):
    source_name = str(src)
    if ":" in source_name:
        source_name = source_name.split(":")[-1]
    if "/" in source_name:
        source_name = source_name.split("/")[-1]
    if source_name.endswith(".loom"):
        source_name = source_name[:-len(".loom")]
    return _name_suffix(source_name)

def loom_test_suite(
        name,
        srcs,
        deps = [],
        args = [],
        execution_profile = None,
        size = "small",
        tags = [],
        visibility = None):
    """Executes each authored Loom source as an independent test.

    This is a naming and aggregation convenience around ``loom_test``. It does
    not merge sources or create a test-only Loom library.

    Args:
      name: Name of the generated test suite.
      srcs: Authored ``.loom`` sources, each executed as its own test.
      deps: Loom libraries supplied to every test.
      args: Additional ``iree-test-loom`` arguments supplied to every test.
      execution_profile: Optional execution profile shared by every test.
      size: Bazel test size applied to every test.
      tags: Additional tags applied to every test and the suite.
      visibility: Bazel visibility of the generated test suite.
    """
    if not srcs:
        fail("%s requires at least one authored test source" % name)
    tests = []
    source_names = {}
    for src in srcs:
        source_name = _loom_test_suite_source_name(src)
        if source_name in source_names:
            fail(
                "%s sources %s and %s have the same test name" %
                (name, source_names[source_name], src),
            )
        source_names[source_name] = src
        test_name = "%s_%s_test" % (name, source_name)
        loom_test(
            name = test_name,
            src = src,
            deps = deps,
            args = args,
            execution_profile = execution_profile,
            size = size,
            tags = tags,
            visibility = ["//visibility:private"],
        )
        tests.append(test_name)
    suite_kwargs = {
        "name": name,
        "tags": tags,
        "tests": tests,
    }
    if visibility != None:
        suite_kwargs["visibility"] = visibility
    native.test_suite(**suite_kwargs)

def loom_kernel_library(
        name,
        srcs,
        deps = [],
        execution_profiles = [],
        targets = [],
        tags = [],
        visibility = None):
    """Declares launchable kernels with format, plan, and target coverage.

    Each entry in ``targets`` generates a private ``loom_kernel_binary`` and a
    test that checks its loader artifact and reports. The generated binary uses
    this library as its direct input, so only this library's exports become
    roots while dependencies remain closure-only candidates.
    """
    _declare_library(
        name = name,
        srcs = srcs,
        deps = deps,
        execution_profiles = execution_profiles,
        kernel_targets = targets,
        plan_benchmarks = True,
        module_testonly = False,
        module_visibility = visibility,
        tags = tags,
        visibility = visibility,
    )
