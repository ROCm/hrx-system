# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Linux process invocations with an explicit rocjitsu topology."""

load("@bazel_skylib//lib:shell.bzl", "shell")
load("//build_tools/bazel:executable.bzl", "IreeExecutableInfo", "executable_launcher_transition")
load("//build_tools/bazel:execution_requirements.bzl", "collect_execution_requirements", "inject_execution_requirements")
load(":runtime.bzl", "RocjitsuRuntimeInfo")

def _runfile_path(ctx, file):
    path = file.short_path
    if path.startswith("../"):
        path = path[3:]
    else:
        path = ctx.workspace_name + "/" + path
    return path

def _runfile(ctx, file):
    return '"$RUNFILES_DIR"/' + shell.quote(_runfile_path(ctx, file))

def _single_file(target):
    files = target[DefaultInfo].files.to_list()
    if len(files) != 1:
        fail("%s must provide exactly one bootstrap file" % target.label)
    return files[0]

def _option(name):
    if not name.startswith("--") or len(name) < 3:
        fail("Bootstrap option must start with --: %r" % name)
    for character in name[2:].elems():
        if character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-":
            fail("Invalid bootstrap option: %r" % name)
    return name

def _rocjitsu_executable_impl(ctx):
    runtime = ctx.attr.runtime[RocjitsuRuntimeInfo]
    if runtime.sanitizer == "address" and not ctx.attr.address_sanitizer:
        fail("This rocjitsu runtime requires an address-sanitized consumer (--config=asan)")
    if not ctx.attr.configurations:
        fail("A rocjitsu invocation requires at least one machine configuration")

    arguments = []
    options = {}
    for argument in ctx.attr.bootstrap_args:
        name, separator, _ = argument.partition("=")
        _option(name)
        if not separator:
            fail("Bootstrap arguments use --option=value: %r" % argument)
        options[name] = True
        arguments.append(shell.quote(argument))
    for target, name in ctx.attr.bootstrap_files.items():
        _option(name)
        options[name] = True
        arguments.append(shell.quote(name + "=") + _runfile(ctx, _single_file(target)))
    library_option = _option(ctx.attr.library_option)
    configuration_option = _option(ctx.attr.configuration_option)
    if library_option == configuration_option or library_option in options or configuration_option in options:
        fail("Simulator library/configuration options must have distinct bootstrap bindings")
    arguments.append(shell.quote(library_option + "=") + _runfile(ctx, runtime.library))
    for index in ctx.attr.machine_indices:
        target = ctx.attr.configurations[index]
        arguments.append(shell.quote(configuration_option + "=") + _runfile(ctx, _single_file(target)))
    if ctx.attr.bootstrap_lock_option:
        lock_option = _option(ctx.attr.bootstrap_lock_option)
        if lock_option in options or lock_option in [library_option, configuration_option]:
            fail("Bootstrap lock option must be distinct from the values it locks")
        for option in options.keys() + [library_option, configuration_option]:
            arguments.append(shell.quote(lock_option + "=" + option[2:]))

    output = ctx.actions.declare_file(ctx.label.name)
    library_directory = '"$RUNFILES_DIR"/' + shell.quote(_runfile_path(ctx, runtime.library).rsplit("/", 1)[0])
    ctx.actions.write(
        output = output,
        is_executable = True,
        content = "\n".join([
            "#!/bin/sh",
            "set -eu",
            'RUNFILES_DIR="${RUNFILES_DIR:-${TEST_SRCDIR:-$0.runfiles}}"',
            "export RUNFILES_DIR",
            "LD_LIBRARY_PATH=" + library_directory,
            "export LD_LIBRARY_PATH",
            "exec %s %s %s %s -- \"$@\"" % (
                _runfile(ctx, ctx.executable._launcher),
                shell.quote(runtime.host_cpu),
                _runfile(ctx, ctx.executable.src),
                " ".join(arguments),
            ),
            "",
        ]),
    )
    dependencies = [ctx.attr.src, ctx.attr.runtime] + ctx.attr.data + ctx.attr.configurations + ctx.attr.bootstrap_files.keys()
    runfiles = ctx.runfiles(files = [ctx.executable.src, ctx.executable._launcher])
    for dependency in dependencies:
        runfiles = runfiles.merge(dependency[DefaultInfo].default_runfiles)
        runfiles = runfiles.merge(ctx.runfiles(transitive_files = dependency[DefaultInfo].files))
    providers = [
        DefaultInfo(executable = output, runfiles = runfiles),
        IreeExecutableInfo(
            data = depset(transitive = [dependency[DefaultInfo].files for dependency in dependencies[1:]]),
            env = {},
            output = output,
            src = ctx.attr.src.label,
        ),
    ]
    if RunEnvironmentInfo in ctx.attr.src:
        providers.append(ctx.attr.src[RunEnvironmentInfo])
    return inject_execution_requirements(ctx, providers, dependencies, output)

_rocjitsu_executable = rule(
    implementation = _rocjitsu_executable_impl,
    executable = True,
    attrs = {
        "address_sanitizer": attr.bool(doc = "Whether the configured consumer uses the address sanitizer ABI."),
        "bootstrap_args": attr.string_list(doc = "Fixed --option=value tokens, preserved across executable/test wrapping."),
        "bootstrap_files": attr.label_keyed_string_dict(allow_files = True, doc = "Single-file labels bound to fixed --option=file arguments."),
        "bootstrap_lock_option": attr.string(doc = "Consumer option locking one named flag after bootstrap; the consumer enforces aliases and flagfiles."),
        "configuration_option": attr.string(mandatory = True, doc = "Consumer option receiving each machine JSON path."),
        "configurations": attr.label_list(allow_files = [".json"], doc = "Distinct machine configuration inputs."),
        "data": attr.label_list(allow_files = True, aspects = [collect_execution_requirements]),
        "library_option": attr.string(mandatory = True, doc = "Consumer option receiving the explicit simulator library path."),
        "machine_indices": attr.int_list(doc = "Ordered indices into configurations, one per machine."),
        "runtime": attr.label(default = "@rocjitsu_runtime//:runtime", providers = [RocjitsuRuntimeInfo], aspects = [collect_execution_requirements]),
        "src": attr.label(mandatory = True, executable = True, cfg = "target", aspects = [collect_execution_requirements]),
        "_allowlist_function_transition": attr.label(default = "@bazel_tools//tools/allowlists/function_transition_allowlist"),
        "_launcher": attr.label(default = "//build_tools/third_party/rocjitsu:launcher", executable = True, cfg = executable_launcher_transition),
    },
)

def rocjitsu_executable(name, configurations, target_compatible_with = [], **kwargs):
    """Declares one runnable simulator topology over an ordinary executable.

    The consumer's existing library/configuration options are bound explicitly.
    The launcher rejects caller arguments naming a fixed bootstrap option.
    Machine ownership remains entirely within the consumer.

    Args:
      name: Concrete invocation target name.
      configurations: Ordered machine configuration labels; repetitions create separate machines.
      target_compatible_with: Additional caller build constraints.
      **kwargs: Source, bootstrap bindings, runtime inputs, and standard attributes.
    """
    inputs = depset(configurations).to_list()
    _rocjitsu_executable(
        name = name,
        configurations = inputs,
        machine_indices = [inputs.index(configuration) for configuration in configurations],
        target_compatible_with = target_compatible_with + [
            "@platforms//os:linux",
            "@platforms//cpu:x86_64",
        ],
        address_sanitizer = select({
            "//build_tools/bazel:address_sanitizer_target": True,
            "//conditions:default": False,
        }),
        **kwargs
    )
