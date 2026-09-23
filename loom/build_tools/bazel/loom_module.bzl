# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 WITH LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for linking Loom source and bytecode modules."""

load(
    "//loom/requirements:package_policy.bzl",
    "apply_loom_target_policy",
)

_LOOM_LINK_MODES = ["merge", "link"]
_LOOM_LINK_OUTPUT_FORMATS = ["text", "bc"]
_LOOM_LINK_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:link_toolchain_type")

def _loom_module_impl(ctx):
    tool = ctx.toolchains[_LOOM_LINK_TOOLCHAIN_TYPE].tool
    args = ctx.actions.args()
    args.add_all([
        ctx.expand_location(arg, targets = ctx.attr.srcs)
        for arg in ctx.attr.args
    ])
    args.add("--output=%s" % ctx.outputs.output.path)
    ctx.actions.run(
        arguments = [args],
        executable = tool.files_to_run,
        inputs = depset(ctx.files.srcs),
        mnemonic = "LoomLink",
        outputs = [ctx.outputs.output],
        progress_message = "Linking Loom module %s" % ctx.outputs.output.short_path,
    )
    return [DefaultInfo(files = depset([ctx.outputs.output]))]

_loom_module = rule(
    implementation = _loom_module_impl,
    attrs = {
        "args": attr.string_list(),
        "output": attr.output(mandatory = True),
        "srcs": attr.label_list(allow_files = True),
    },
    toolchains = [_LOOM_LINK_TOOLCHAIN_TYPE],
)

def loom_module(
        name,
        srcs,
        libraries = [],
        roots = [],
        configs = [],
        mode = "merge",
        output = None,
        output_format = "text",
        include_input_exports = False,
        include_input_tests = False,
        strip_check = False,
        require_resolved_config = False,
        strict_deps = False,
        tags = [],
        target_compatible_with = [],
        visibility = None):
    """Builds one Loom text or bytecode module.

    Args:
      name: Name of the generated module target.
      srcs: Ordered primary Loom source or bytecode module labels.
      libraries: Ordered separate library modules. Merge mode leaves their
        symbols out of the product; link mode may select reachable providers.
      roots: Optional root symbol names used by link mode.
      configs: Optional compile-time config bindings as key=value strings.
      mode: Module construction mode: merge or link.
      output: Generated module filename. Defaults to <name>.loom or .loombc.
      output_format: Generated representation: text or bc.
      include_input_exports: Whether exported input symbols are implicit roots.
      include_input_tests: Whether root-owned checks and benchmarks are roots.
      strip_check: Whether check.case and check.benchmark symbols are removed.
      require_resolved_config: Whether unresolved config.decl symbols fail.
      strict_deps: Whether source references must resolve from direct libraries.
      tags: Additional tags for the generator action.
      target_compatible_with: Optional compatibility constraints for the
        generated module.
      visibility: Visibility of the generated module target.
    """
    if not srcs and not libraries:
        fail("loom_module %s requires at least one source or library" % name)
    if mode not in _LOOM_LINK_MODES:
        fail("loom_module %s has unsupported mode %r" % (name, mode))
    if mode == "merge" and (roots or include_input_exports or include_input_tests):
        fail("loom_module %s merge mode does not accept roots" % name)
    if mode == "link" and not roots and not include_input_exports and not include_input_tests:
        fail("loom_module %s link mode requires roots, include_input_exports, or include_input_tests" % name)
    if output_format not in _LOOM_LINK_OUTPUT_FORMATS:
        fail("loom_module %s has unsupported output format %r" %
             (name, output_format))

    output = output or (name + (".loombc" if output_format == "bc" else ".loom"))
    args = [
        "--mode=%s" % mode,
        "--to=%s" % output_format,
    ]
    args.extend(["$(location %s)" % src for src in srcs])
    args.extend(["--library=$(location %s)" % library for library in libraries])
    args.extend(["--root=%s" % root for root in roots])
    args.extend(["--config=%s" % config for config in configs])
    if include_input_exports:
        args.append("--include-input-exports=true")
    if include_input_tests:
        args.append("--include-input-tests=true")
    if strip_check:
        args.append("--strip-check=true")
    if require_resolved_config:
        args.append("--require-resolved-config=true")
    if strict_deps:
        args.append("--strict-deps")

    rule_kwargs = {
        "tags": tags + ["skip-bazel_to_cmake"],
        "target_compatible_with": target_compatible_with,
    }
    if visibility != None:
        rule_kwargs["visibility"] = visibility
    rule_kwargs = apply_loom_target_policy(rule_kwargs)

    _loom_module(
        name = name,
        srcs = srcs + libraries,
        output = output,
        args = args,
        **rule_kwargs
    )
