# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 WITH LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for linking Loom source and bytecode modules."""

load("//build_tools/bazel:generate.bzl", "iree_generated_files")
load(
    "//loom/requirements:package_policy.bzl",
    "apply_loom_target_policy",
)

_LOOM_LINK_MODES = ["merge", "link"]
_LOOM_LINK_OUTPUT_FORMATS = ["text", "bc"]

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
        strip_check = False,
        require_resolved_config = False,
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
      strip_check: Whether check.case and check.benchmark symbols are removed.
      require_resolved_config: Whether unresolved config.decl symbols fail.
      tags: Additional tags for the generator action.
      target_compatible_with: Optional compatibility constraints for the
        generated module.
      visibility: Visibility of the generated module target.
    """
    if not srcs:
        fail("loom_module %s requires at least one primary source" % name)
    if mode not in _LOOM_LINK_MODES:
        fail("loom_module %s has unsupported mode %r" % (name, mode))
    if mode == "merge" and (roots or include_input_exports):
        fail("loom_module %s merge mode does not accept roots" % name)
    if mode == "link" and not roots and not include_input_exports:
        fail("loom_module %s link mode requires roots or include_input_exports" % name)
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
    if strip_check:
        args.append("--strip-check=true")
    if require_resolved_config:
        args.append("--require-resolved-config=true")

    rule_kwargs = {
        "tags": tags + ["skip-bazel_to_cmake"],
        "target_compatible_with": target_compatible_with,
    }
    if visibility != None:
        rule_kwargs["visibility"] = visibility
    rule_kwargs = apply_loom_target_policy(rule_kwargs)

    iree_generated_files(
        name = name,
        srcs = srcs + libraries,
        outs = [output],
        args = args,
        output_args = {
            output.split("/")[-1]: "--output={path}",
        },
        tool = "//loom/src/loom/tools/loom-link",
        mnemonic = "LoomLink",
        progress_message = "Linking Loom module %s" % output,
        **rule_kwargs
    )
