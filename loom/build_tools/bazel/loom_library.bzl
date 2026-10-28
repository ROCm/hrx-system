# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public rules for Loom libraries, kernels, and tests."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//build_tools/bazel:cc_attrs.bzl", "cc_attrs")
load("//build_tools/bazel:executable.bzl", "iree_executable_test")
load("//build_tools/bazel:requirements.bzl", "apply_test_requirements")
load(
    "//build_tools/bazel:runfiles.bzl",
    "RUNFILES_PATH_BEGIN",
    "RUNFILES_PATH_END",
)
load(
    "//loom/requirements:package_policy.bzl",
    "apply_loom_target_policy",
)
load(":loom_binary.bzl", "LoomBinaryInfo", "loom_kernel_binary")
load(":loom_check.bzl", "loom_check_compile_tests")
load(
    ":loom_linking.bzl",
    "loom_linking",
    _LoomLibraryInfo = "LoomLibraryInfo",
)

_LOOM_BENCHMARK_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:benchmark_toolchain_type")
_LOOM_FORMAT_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:format_toolchain_type")
_LOOM_LINK_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:link_toolchain_type")
_LOOM_TEST_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:test_toolchain_type")

_LOOM_BENCHMARK_SMOKE_ARGS = [
    "--iterations=1",
    "--warmup-iterations=0",
    "--min-time-ms=0",
    "--warmup-time-ms=0",
    "--max-batches=1",
    "--output-format=jsonl",
    "--compile-report=none",
]

LoomLibraryInfo = _LoomLibraryInfo

LoomExecutionTestInfo = provider(
    doc = "A test executing one linked Loom test module.",
    fields = {
        "benchmark_runner": "Resolved benchmark runner, or None for correctness-only launchers.",
        "benchmark_runner_args": "Smoke, profile, and workload arguments passed to the benchmark runner, or an empty list.",
        "module": "Linked Loom module containing root-owned cases and benchmarks.",
        "profile_name": "Stable execution profile name.",
        "test_runner": "Resolved correctness runner executable.",
        "test_runner_args": "Profile, workload, and test arguments passed to the correctness runner.",
    },
)

_LoomTestModuleInfo = provider(
    doc = "A linked module retaining tests owned by one root library.",
    fields = {
        "module": "Linked Loom bytecode module consumed by execution and plan runners.",
    },
)

def _apply_package_policy(name, tags, target_compatible_with):
    attributes = apply_loom_target_policy({
        "tags": tags,
        "target_compatible_with": target_compatible_with,
    }, name = name)
    return struct(
        tags = attributes["tags"],
        target_compatible_with = attributes["target_compatible_with"],
    )

def _reject_workload_args(name, args):
    for arg in args:
        if arg.split("=")[0] in ["--config", "--case"]:
            fail("%s: use loom_test configs and case for workload selection, not %s" % (name, arg))

def loom_execution_profile(
        name,
        target_family,
        target_class,
        executor,
        runner_args = [],
        runner = None,
        build_requirements = [],
        run_requirements = [],
        resource_group = None,
        tags = []):
    """Defines immutable policy for Loom test execution.

    Args:
      name: Stable profile name used in generated target names and query tags.
      target_family: Compiler target family, such as amdgpu or spirv.
      target_class: Broad target class, such as gpu or cpu.
      executor: Execution environment, such as hardware or reference.
      runner_args: Profile arguments passed to the selected execution runner or
          runner sequence. Workload configs and case selection belong on
          loom_test.
      runner: Optional executable label owning this profile's execution. When
          present, it receives the linked module and correctness arguments
          directly instead of using the registered correctness and benchmark
          tools.
      build_requirements: Build requirements needed by the execution runners.
      run_requirements: Runtime resources needed to execute the test.
      resource_group: Optional local resource group serializing competing tests.
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
    _reject_workload_args(name, runner_args)
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
        runner = runner,
        runner_args = runner_args,
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
        data = ctx.files.data,
        input_format = ctx.attr.input_format,
        inputopts = [
            ctx.expand_location(option, targets = ctx.attr.srcs + ctx.attr.data)
            for option in ctx.attr.inputopts
        ],
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
            component = struct(
                label = ctx.label,
                sources = tuple(ctx.files.srcs),
                data = tuple(ctx.files.data),
                input_format = ctx.attr.input_format,
                inputopts = tuple(ctx.attr.inputopts),
                module = artifacts.module,
            ),
            module = artifacts.module,
            transitive_components = depset(
                direct = [info.component for info in dependency_infos],
                transitive = [info.transitive_components for info in dependency_infos],
                order = "preorder",
            ),
            transitive_dependencies = artifacts.transitive_dependencies,
        ),
    ]

_loom_library = rule(
    implementation = _loom_library_impl,
    attrs = {
        "data": attr.label_list(
            allow_files = True,
            doc = "Declared source-admission inputs, such as included headers.",
        ),
        "deps": attr.label_list(
            providers = [LoomLibraryInfo],
            doc = "Direct Loom library dependencies kept as separate modules.",
        ),
        "input_format": attr.string(
            doc = "Source provider override; empty selects each source by filename.",
        ),
        "inputopts": attr.string_list(
            doc = "Provider-scoped options (format:options), with location expansion.",
        ),
        "srcs": attr.label_list(
            allow_files = True,
            doc = "Ordered source modules accepted by the configured input providers, or bytecode.",
        ),
    },
    doc = "Merges direct sources into one relocatable Loom bytecode module.",
    toolchains = [_LOOM_LINK_TOOLCHAIN_TYPE],
)

def _loom_test_module_impl(ctx):
    root_library = ctx.attr.root_library[LoomLibraryInfo]
    dependency_infos = [dep[LoomLibraryInfo] for dep in ctx.attr.deps]
    module = loom_linking.declare_test_module(
        ctx = ctx,
        root_module = root_library.module,
        dependency_infos = dependency_infos,
        output_stem = ctx.label.name,
        mnemonic = "LoomTestModule",
        progress_message = "Linking Loom test module %s" % ctx.label,
    )
    runfiles = ctx.runfiles(files = [module] + ctx.files.data)
    for data in ctx.attr.data:
        runfiles = runfiles.merge(data[DefaultInfo].default_runfiles)
    return [
        DefaultInfo(files = depset([module]), runfiles = runfiles),
        _LoomTestModuleInfo(module = module),
    ]

_loom_test_module = rule(
    implementation = _loom_test_module_impl,
    attrs = {
        "data": attr.label_list(
            allow_files = True,
            doc = "Runtime fixtures retained with the linked module.",
        ),
        "deps": attr.label_list(
            providers = [LoomLibraryInfo],
            doc = "Direct libraries available only for dependency resolution.",
        ),
        "root_library": attr.label(
            mandatory = True,
            providers = [LoomLibraryInfo],
            doc = "Root library whose test-only symbols become link roots.",
        ),
    },
    doc = "Links one root-owned Loom test module.",
    toolchains = [_LOOM_LINK_TOOLCHAIN_TYPE],
)

def _tool_runfiles(ctx, tool, files):
    runfiles = ctx.runfiles(files = files + [tool.executable])
    if tool.runfiles:
        runfiles = runfiles.merge(tool.runfiles)
    return runfiles

def _shell_quote(value):
    return "'" + value.replace("'", "'\"'\"'") + "'"

def _tool_environment_exports(tool, indent = ""):
    # The launcher has entered the workspace runfiles directory. Resolve only
    # graph-proven path spans so each child receives its own runtime policy.
    entries = []
    for name, value in sorted(tool.environment.items()):
        if name in tool.runfiles_environment:
            value = _shell_quote(tool.runfiles_environment[name])
            value = value.replace(RUNFILES_PATH_BEGIN, "'\"${PWD}/")
            value = value.replace(RUNFILES_PATH_END, "\"'")
        else:
            value = _shell_quote(value)
        entries.append(indent + "export " + name + "=" + value + "\n")
    return "".join(entries)

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
        "{environment}" +
        "exec \"${{PWD}}/{tool}\" \"${{PWD}}/{input}\"{args} \"$@\"\n"
    ).format(
        workspace = ctx.workspace_name,
        tool = tool.executable.short_path,
        environment = _tool_environment_exports(tool),
        input = input_file.short_path,
        args = command_args,
    )
    ctx.actions.write(
        content = content,
        is_executable = True,
        output = output,
    )
    return output

def _write_execution_test_launcher(
        ctx,
        test_tool,
        benchmark_tool,
        module,
        test_runner_args,
        benchmark_runner_args):
    output = ctx.actions.declare_file(ctx.label.name + ".sh")
    test_args = "".join([
        " \\\n  %s" % _shell_quote(arg)
        for arg in test_runner_args
    ])
    benchmark_args = "".join([
        " \\\n  %s" % _shell_quote(arg)
        for arg in benchmark_runner_args
    ])
    if benchmark_tool != None:
        test_command = (
            "(\n" +
            "{test_environment}" +
            "  exec \"${{PWD}}/{test_tool}\" " +
            "\"${{PWD}}/{module}\"{test_args} \"$@\"\n" +
            ")\n"
        ).format(
            module = module.short_path,
            test_args = test_args,
            test_environment = _tool_environment_exports(test_tool, indent = "  "),
            test_tool = test_tool.executable.short_path,
        )
    else:
        test_command = (
            "{test_environment}" +
            "exec \"${{PWD}}/{test_tool}\" " +
            "\"${{PWD}}/{module}\"{test_args} \"$@\"\n"
        ).format(
            module = module.short_path,
            test_args = test_args,
            test_environment = _tool_environment_exports(test_tool),
            test_tool = test_tool.executable.short_path,
        )
    benchmark_command = ""
    if benchmark_tool != None:
        # Both tools own the execution options, except these correctness-only
        # inputs. Keep shared overrides in order and let each real parser reject
        # unsupported arguments; the launcher does not parse option values.
        benchmark_command = (
            "benchmark_command=(\"${{PWD}}/{benchmark_tool}\" " +
            "\"${{PWD}}/{module}\"{benchmark_args})\n" +
            "for argument in \"$@\"; do\n" +
            "  case \"${{argument%%=*}}\" in\n" +
            "    --library|--xfail|--allow-failure) ;;\n" +
            "    *) benchmark_command+=(\"$argument\") ;;\n" +
            "  esac\n" +
            "done\n" +
            "{benchmark_environment}" +
            "exec \"${{benchmark_command[@]}}\"\n"
        ).format(
            benchmark_args = benchmark_args,
            benchmark_environment = _tool_environment_exports(benchmark_tool),
            benchmark_tool = benchmark_tool.executable.short_path,
            module = module.short_path,
        )
    content = (
        "#!/usr/bin/env bash\n" +
        "set -euo pipefail\n" +
        "RUNFILES=\"${{RUNFILES_DIR:-$0.runfiles}}\"\n" +
        "cd \"${{RUNFILES}}/{workspace}\"\n"
    ).format(
        workspace = ctx.workspace_name,
    ) + test_command + benchmark_command
    ctx.actions.write(
        content = content,
        is_executable = True,
        output = output,
    )
    return output

def _execution_test_launcher_providers(ctx, test_tool, benchmark_tool):
    module = ctx.attr.module[_LoomTestModuleInfo].module
    sanitizers = [
        mode
        for mode, setting in [
            ("asan", ctx.attr._address_sanitizer),
            ("tsan", ctx.attr._thread_sanitizer),
            ("ubsan", ctx.attr._undefined_sanitizer),
        ]
        if setting[BuildSettingInfo].value
    ]
    sanitizer_args = ["--sanitizer=" + "|".join(sanitizers)] if sanitizers else []
    runner_args = sanitizer_args + ctx.attr.profile_args + ctx.attr.workload_args
    test_runner_args = runner_args + ctx.attr.test_args
    benchmark_runner_args = (
        _LOOM_BENCHMARK_SMOKE_ARGS + runner_args if benchmark_tool != None else []
    )
    output = _write_execution_test_launcher(
        ctx,
        test_tool,
        benchmark_tool,
        module,
        test_runner_args,
        benchmark_runner_args,
    )
    runfiles = _tool_runfiles(ctx, test_tool, [])
    runfiles = runfiles.merge(ctx.attr.module[DefaultInfo].default_runfiles)
    if benchmark_tool != None:
        runfiles = runfiles.merge(_tool_runfiles(ctx, benchmark_tool, []))
    return [
        DefaultInfo(
            executable = output,
            files = depset([output]),
            runfiles = runfiles,
        ),
        LoomExecutionTestInfo(
            benchmark_runner = benchmark_tool.executable if benchmark_tool != None else None,
            benchmark_runner_args = benchmark_runner_args,
            module = module,
            profile_name = ctx.attr.profile_name,
            test_runner = test_tool.executable,
            test_runner_args = test_runner_args,
        ),
    ]

def _loom_execution_test_launcher_impl(ctx):
    return _execution_test_launcher_providers(
        ctx,
        ctx.toolchains[_LOOM_TEST_TOOLCHAIN_TYPE].tool,
        ctx.toolchains[_LOOM_BENCHMARK_TOOLCHAIN_TYPE].tool,
    )

def _loom_correctness_test_launcher_impl(ctx):
    return _execution_test_launcher_providers(
        ctx,
        ctx.toolchains[_LOOM_TEST_TOOLCHAIN_TYPE].tool,
        None,
    )

_EXECUTION_TEST_LAUNCHER_ATTRS = {
    "module": attr.label(
        mandatory = True,
        providers = [_LoomTestModuleInfo],
        doc = "Linked Loom module containing root-owned cases and benchmarks.",
    ),
    "profile_args": attr.string_list(
        doc = "Profile arguments appended after the module for each runner.",
    ),
    "profile_name": attr.string(
        mandatory = True,
        doc = "Stable execution profile name.",
    ),
    "test_args": attr.string_list(
        doc = "Arguments appended only to the correctness runner.",
    ),
    "workload_args": attr.string_list(
        doc = "Configuration bindings and case selection shared by each runner.",
    ),
    "_address_sanitizer": attr.label(
        default = Label("//loom/config/sanitizer:address"),
        providers = [BuildSettingInfo],
    ),
    "_thread_sanitizer": attr.label(
        default = Label("//loom/config/sanitizer:thread"),
        providers = [BuildSettingInfo],
    ),
    "_undefined_sanitizer": attr.label(
        default = Label("//loom/config/sanitizer:undefined"),
        providers = [BuildSettingInfo],
    ),
}

_loom_execution_test_launcher = rule(
    implementation = _loom_execution_test_launcher_impl,
    attrs = _EXECUTION_TEST_LAUNCHER_ATTRS,
    doc = "Generates a launcher for one linked Loom test profile.",
    executable = True,
    toolchains = [
        _LOOM_BENCHMARK_TOOLCHAIN_TYPE,
        _LOOM_TEST_TOOLCHAIN_TYPE,
    ],
)

_loom_correctness_test_launcher = rule(
    implementation = _loom_correctness_test_launcher_impl,
    attrs = _EXECUTION_TEST_LAUNCHER_ATTRS,
    doc = "Generates a correctness-only launcher for one Loom execution profile.",
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

def _loom_plan_test_launcher_impl(ctx):
    tool = ctx.toolchains[_LOOM_BENCHMARK_TOOLCHAIN_TYPE].tool
    module = ctx.attr.module[_LoomTestModuleInfo].module
    output = _write_test_launcher(
        ctx,
        tool,
        module,
        [
            "--dry-run",
            "--output-format=jsonl",
            "--compile-report=none",
        ] + ctx.attr.workload_args,
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
        "module": attr.label(
            mandatory = True,
            providers = [_LoomTestModuleInfo],
            doc = "Linked root-owned test module whose benchmarks are planned.",
        ),
        "workload_args": attr.string_list(doc = "Workload configuration and case selection."),
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
        "if [[ \"$#\" -ne 0 ]]; then\n" +
        "  echo >&2 \"artifact checks do not accept arguments\"\n" +
        "  exit 2\n" +
        "fi\n" +
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
        module,
        profile,
        test_runner_args,
        size,
        tags,
        visibility,
        benchmark_smoke = True,
        target_compatible_with = [],
        workload_args = []):
    test_kwargs = apply_test_requirements(
        {
            "size": size,
            "tags": tags + profile.tags + _execution_profile_tags(profile),
        },
        build_requirements = profile.build_requirements,
        run_requirements = profile.run_requirements,
        resource_group = profile.resource_group,
    )
    if target_compatible_with:
        test_kwargs["target_compatible_with"] = test_kwargs.get("target_compatible_with", []) + target_compatible_with
    resource_group = test_kwargs.pop("resource_group", None)
    test_kwargs["tags"] = cc_attrs.with_resource_group_tags(
        test_kwargs.get("tags"),
        resource_group,
    )
    if visibility != None:
        test_kwargs["visibility"] = visibility
    runner_args = profile.runner_args + workload_args + test_runner_args
    if profile.runner != None:
        iree_executable_test(
            name = name,
            args = ["$(rootpath %s)" % module] + runner_args,
            data = [module],
            src = profile.runner,
            **test_kwargs
        )
        return
    launcher_attrs = {
        "module": module,
        "profile_args": profile.runner_args,
        "profile_name": profile.name,
        "test_args": test_runner_args,
        "workload_args": workload_args,
    }
    if benchmark_smoke:
        launcher_rule = _loom_execution_test_launcher
    else:
        launcher_rule = _loom_correctness_test_launcher
    _declare_launcher_test(
        name = name,
        launcher_rule = launcher_rule,
        launcher_attrs = launcher_attrs,
        test_kwargs = test_kwargs,
    )

def _declare_execution_tests(name, module, profiles, **kwargs):
    """Expands independent execution policies over one source-owned module."""
    tests = {}
    for profile in profiles:
        if getattr(profile, "kind", None) != "loom_execution_profile":
            fail("%s execution profile was not created by loom_execution_profile" % name)
        test_name = "%s_execute_%s_test" % (name, _name_suffix(profile.name))
        if test_name in tests:
            fail(
                "%s execution profile %s has the same generated name as %s" % (
                    name,
                    profile.name,
                    tests[test_name],
                ),
            )
        tests[test_name] = profile.name
        _declare_execution_test(
            name = test_name,
            module = module,
            profile = profile,
            **kwargs
        )
    return tests.keys()

def _declare_source_format_tests(name, srcs, tags, target_compatible_with):
    """Exercises the native formatter; presubmit owns authoring-policy checks."""
    tests = []
    for index, src in enumerate(srcs):
        if not str(src).endswith(".loom"):
            continue
        test_name = "%s_format_%d_test" % (name, index)
        _declare_launcher_test(
            name = test_name,
            launcher_rule = _loom_format_test_launcher,
            launcher_attrs = {"src": src},
            test_kwargs = {
                "tags": tags + ["hostonly"],
                "target_compatible_with": target_compatible_with,
                "visibility": ["//visibility:private"],
            },
        )
        tests.append(test_name)
    return tests

def _declare_library(
        name,
        srcs,
        deps,
        data,
        input_format,
        inputopts,
        execution_profiles,
        kernel_targets,
        benchmark_smoke,
        module_testonly,
        module_visibility,
        tags,
        visibility,
        target_compatible_with = []):
    _loom_library(
        name = name,
        srcs = srcs,
        deps = deps,
        data = data,
        input_format = input_format,
        inputopts = inputopts,
        tags = tags,
        testonly = module_testonly,
        target_compatible_with = target_compatible_with,
        visibility = module_visibility,
    )
    tests = _declare_source_format_tests(name, srcs, tags, target_compatible_with)

    test_module = None
    if benchmark_smoke or execution_profiles:
        test_module = name + "_test_module"
        _loom_test_module(
            name = test_module,
            data = data,
            deps = deps,
            root_library = ":" + name,
            tags = tags + ["manual"],
            testonly = True,
            visibility = ["//visibility:private"],
        )

    if benchmark_smoke:
        plan_test_name = name + "_plan_test"
        _declare_launcher_test(
            name = plan_test_name,
            launcher_rule = _loom_plan_test_launcher,
            launcher_attrs = {"module": ":" + test_module},
            test_kwargs = {
                "tags": tags + ["hostonly"],
                "visibility": ["//visibility:private"],
            },
        )
        tests.append(plan_test_name)

    if execution_profiles and not srcs:
        fail("%s requires authored srcs for execution profiles" % name)
    if execution_profiles:
        tests.extend(_declare_execution_tests(
            name = name,
            benchmark_smoke = benchmark_smoke,
            module = ":" + test_module,
            profiles = execution_profiles,
            test_runner_args = [],
            size = "small",
            tags = tags,
            visibility = ["//visibility:private"],
            target_compatible_with = target_compatible_with,
        ))

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

    if tests:
        native.test_suite(
            name = name + "_test",
            tests = tests,
            tags = tags,
            visibility = visibility,
        )

def loom_library(
        name,
        srcs = [],
        deps = [],
        data = [],
        input_format = "",
        inputopts = [],
        tags = [],
        visibility = None,
        target_compatible_with = []):
    """Declares a reusable Loom function or template library.

    The rule merges only ``srcs`` into its relocatable bytecode module. Direct
    ``deps`` and their transitive closure remain independent linker inputs, and
    strict dependency analysis rejects source references satisfied only by the
    transitive closure. The ``dependency_reports`` output group contains the
    schema-versioned JSON analysis report.
    """
    package_attributes = _apply_package_policy(
        name,
        tags,
        target_compatible_with,
    )
    _declare_library(
        name = name,
        srcs = srcs,
        deps = deps,
        data = data,
        input_format = input_format,
        inputopts = inputopts,
        execution_profiles = [],
        kernel_targets = [],
        benchmark_smoke = False,
        module_testonly = False,
        module_visibility = visibility,
        tags = package_attributes.tags,
        target_compatible_with = package_attributes.target_compatible_with,
        visibility = visibility,
    )

def _test_variants(name, configs, case, variants):
    if variants == None:
        return {name: struct(configs = configs, case = case)}
    if not variants:
        fail("%s variants must contain at least one named workload" % name)
    workloads = {}
    for variant_name, variant in variants.items():
        if type(variant_name) != "string" or not variant_name:
            fail("%s variant names must be non-empty strings" % name)
        for key in variant:
            if key not in ["configs", "case"]:
                fail("%s variant %s has unknown field %s" % (name, variant_name, key))
        workload_name = name + "_" + _name_suffix(variant_name)
        if workload_name in workloads:
            fail("%s has colliding variant names: %s" % (name, variant_name))
        workload_configs = dict(configs)
        workload_configs.update(variant.get("configs", {}))
        workloads[workload_name] = struct(
            configs = workload_configs,
            case = variant.get("case", case),
        )
    return workloads

def _test_config_args(name, configs):
    for key, value in configs.items():
        if type(key) != "string" or type(value) != "string":
            fail("%s configs must map symbol names to string values" % name)
    return ["--config=%s=%s" % (key, configs[key]) for key in sorted(configs.keys())]

def _declare_test_module(
        name,
        srcs,
        deps,
        data,
        input_format,
        inputopts,
        tags,
        visibility,
        target_compatible_with):
    if not srcs:
        fail("%s requires at least one authored test source" % name)
    library_name = name + "_library"
    _loom_library(
        name = library_name,
        srcs = srcs,
        deps = deps,
        data = data,
        input_format = input_format,
        inputopts = inputopts,
        tags = tags + ["manual"],
        testonly = True,
        visibility = ["//visibility:private"],
        target_compatible_with = target_compatible_with,
    )
    _loom_test_module(
        name = name,
        data = data,
        deps = deps,
        root_library = ":" + library_name,
        tags = tags,
        testonly = True,
        visibility = visibility,
        target_compatible_with = target_compatible_with,
    )

def loom_test_module(
        name,
        srcs,
        deps = [],
        data = [],
        input_format = "",
        inputopts = [],
        tags = [],
        visibility = None,
        target_compatible_with = []):
    """Owns a reusable linked test module and its runtime fixtures.

    Direct sources jointly own all check cases and benchmarks in the module.
    Dependencies supply reachable definitions, not additional test roots.
    Import options and data belong here; consumers select their workload and
    execution or compiler profiles with ``loom_test(module = ...)``.
    The module requires only its source providers, independently of consumers'
    device requirements. Its default output is the linked bytecode file.
    Authored ``.loom`` inputs also produce native format checks in a
    ``<name>_test`` suite. These belong to this source owner, not to each
    workload or profile consuming the module. Other input formats do not
    produce Loom-text checks or a source suite.

    Args:
      name: Public module target name.
      srcs: Authored source modules jointly owning the test roots.
      deps: Libraries available only for dependency resolution.
      data: Headers and runtime fixtures retained by the source owner.
      input_format: Source provider override, or empty for filename selection.
      inputopts: Provider-scoped import options, with location expansion.
      tags: Additional tags for the source module and its source checks.
      visibility: Bazel visibility of the reusable module.
      target_compatible_with: Source-provider build configuration constraints.
    """
    package_attributes = _apply_package_policy(
        name,
        tags,
        target_compatible_with,
    )
    _declare_test_module(
        name = name,
        srcs = srcs,
        deps = deps,
        data = data,
        input_format = input_format,
        inputopts = inputopts,
        tags = package_attributes.tags,
        visibility = visibility,
        target_compatible_with = package_attributes.target_compatible_with,
    )
    tests = _declare_source_format_tests(
        name,
        srcs,
        package_attributes.tags,
        package_attributes.target_compatible_with,
    )
    if tests:
        native.test_suite(
            name = name + "_test",
            tests = tests,
            tags = package_attributes.tags,
            visibility = visibility,
        )

def loom_test(
        name,
        srcs = [],
        deps = [],
        data = [],
        input_format = "",
        inputopts = [],
        module = None,
        args = [],
        configs = {},
        case = "",
        variants = None,
        execution_profiles = [],
        compile_targets = [],
        benchmark_smoke = True,
        size = "small",
        tags = [],
        visibility = None,
        target_compatible_with = []):
    """Qualifies authored sources or a reusable Loom test module.

    The rule merges ``srcs`` into one relocatable root library and links a test
    module containing every root-owned ``check.case`` and ``check.benchmark``
    plus their reachable dependencies. Test-only symbols from ``deps`` are not
    selected. The linked module is the only Loom input to the runners.
    Default execution profiles run ``loom-test`` for correctness. When
    benchmark smoke is enabled, ``loom-benchmark`` then executes each
    benchmark with one measured iteration and no warmup. Each workload also
    plans its benchmarks once in a host-only test, independent of execution
    profiles. A profile with an explicit runner delegates the linked module to
    that runner once and owns its benchmark policy; exclusively custom runners
    do not create an implicit host benchmark plan.

    Runtime test arguments follow declared arguments, so scalar options can
    override their defaults. Combined launchers pass runtime arguments to both
    tools except ``--library``, ``--xfail`` and ``--allow-failure``, which belong
    only to correctness. A ``--flagfile`` is read by both tools and therefore
    contains shared options. Other single-tool launchers forward arguments
    directly, while artifact-presence checks accept none.

    Built-in runners select program instrumentation through the address, thread,
    and undefined flags in //loom/config/sanitizer. These combine independently
    of native C/C++ sanitizers and do not add test targets. Explicit profile and
    runtime sanitizer options override the configured checks. Custom runners
    own their argument contract and do not receive implicit Loom options.

    Args:
      name: Name of the suite containing source, compilation, and execution checks.
      srcs: Authored source modules jointly owning the test module.
      data: Headers and runtime fixtures available during import and execution.
      input_format: Source provider override, or empty for filename selection.
      inputopts: Provider-scoped options, such as ``cxx:std=c++20``.
      deps: Loom libraries available only for dependency resolution.
      module: Existing loom_test_module to qualify without importing or linking
          again. Exclusive with srcs, deps, data, input_format, and inputopts;
          the source owner retains those inputs, its source checks, and its
          runtime fixtures.
      args: Additional arguments passed to the correctness runner.
      configs: String-valued configuration bindings shared by compiler checks
          and selected execution runners.
      case: Optional case selector passed to selected execution runners.
          Compiler checks continue to qualify the entire owned module.
      variants: Complete mapping of named workloads, each with optional configs
          and case fields overriding the common values. Omitted variants create
          one default workload; an explicitly empty mapping is invalid.
      execution_profiles: Independent execution environments and requirement
          policies. An empty list declares no execution children.
      compile_targets: Typed compiler profiles qualifying the same linked test
          module offline, without execution profiles' device requirements.
          At least one execution or compiler profile is required.
      benchmark_smoke: Whether workloads plan benchmarks on the host and
          correctness tests also run one benchmark iteration.
      size: Bazel test size.
      tags: Additional tags applied to the test.
      visibility: Bazel visibility of the generated test target.
      target_compatible_with: Build configuration constraints for all actions.
    """
    if module and (srcs or deps or data or input_format or inputopts):
        fail("%s module is exclusive with source and import options" % name)
    if not module and not srcs:
        fail("%s requires srcs or a test module" % name)
    if not execution_profiles and not compile_targets:
        fail("%s requires execution_profiles or compile_targets" % name)
    _reject_workload_args(name, args)
    requested_tags = tags
    requested_target_compatible_with = target_compatible_with
    package_attributes = _apply_package_policy(
        name,
        tags,
        target_compatible_with,
    )
    tags = package_attributes.tags
    target_compatible_with = package_attributes.target_compatible_with
    workloads = _test_variants(name, configs, case, variants)
    tests = []
    if not module:
        module_name = name + "_module"
        _declare_test_module(
            name = module_name,
            srcs = srcs,
            deps = deps,
            data = data,
            input_format = input_format,
            inputopts = inputopts,
            tags = tags + ["manual"],
            visibility = ["//visibility:private"],
            target_compatible_with = target_compatible_with,
        )
        module = ":" + module_name
        tests.extend(_declare_source_format_tests(name, srcs, tags, target_compatible_with))
    for workload_name, workload in workloads.items():
        config_args = _test_config_args(workload_name, workload.configs)
        if type(workload.case) != "string":
            fail("%s case must be a string" % workload_name)
        workload_args = config_args + (["--case=" + workload.case] if workload.case else [])
        tests.extend(_declare_execution_tests(
            name = workload_name,
            benchmark_smoke = benchmark_smoke,
            module = module,
            profiles = execution_profiles,
            workload_args = workload_args,
            test_runner_args = args,
            size = size,
            tags = tags,
            visibility = visibility,
            target_compatible_with = target_compatible_with,
        ))
        if benchmark_smoke and (
            not execution_profiles or any([profile.runner == None for profile in execution_profiles])
        ):
            plan_test_name = workload_name + "_plan_test"
            _declare_launcher_test(
                name = plan_test_name,
                launcher_rule = _loom_plan_test_launcher,
                launcher_attrs = {"module": module, "workload_args": workload_args},
                test_kwargs = {
                    "size": size,
                    "tags": tags + ["hostonly"],
                    "target_compatible_with": target_compatible_with,
                    "visibility": visibility,
                },
            )
            tests.append(plan_test_name)
        tests.extend(loom_check_compile_tests(
            name = workload_name,
            src = module,
            targets = compile_targets,
            args = config_args,
            size = size,
            tags = requested_tags,
            visibility = visibility,
            target_compatible_with = requested_target_compatible_with,
        ))
    native.test_suite(
        name = name,
        tests = tests,
        tags = tags,
        visibility = visibility,
    )

def loom_kernel_library(
        name,
        srcs,
        deps = [],
        data = [],
        input_format = "",
        inputopts = [],
        execution_profiles = [],
        targets = [],
        benchmark_smoke = True,
        tags = [],
        visibility = None):
    """Declares launchable kernels with format, plan, and target coverage.

    Cases and benchmarks in ``srcs`` form one root-owned test module;
    test-only symbols from ``deps`` are excluded. Each execution profile runs
    that module for correctness and, when enabled, benchmark smoke. Each entry in
    ``targets`` generates a private ``loom_kernel_binary`` and a test that
    checks its loader artifact and reports. The generated binary uses this
    library as its direct input, so only this library's exports become roots
    while dependencies remain closure-only candidates.

    Inline cases and benchmarks support single-file iteration. Authors may also
    put checks in a separate ``loom_test`` depending on this library. Set
    ``benchmark_smoke = False`` when benchmark planning and execution belong to
    that separate test; correctness execution profiles remain available.
    """
    package_attributes = _apply_package_policy(name, tags, [])
    _declare_library(
        name = name,
        srcs = srcs,
        deps = deps,
        data = data,
        input_format = input_format,
        inputopts = inputopts,
        execution_profiles = execution_profiles,
        kernel_targets = targets,
        benchmark_smoke = benchmark_smoke,
        module_testonly = False,
        module_visibility = visibility,
        tags = package_attributes.tags,
        target_compatible_with = package_attributes.target_compatible_with,
        visibility = visibility,
    )
