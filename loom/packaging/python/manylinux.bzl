# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable wheel tags backed by an ELF compatibility audit."""

def _manylinux_wheel_impl(ctx):
    source = ctx.file.wheel
    output = ctx.actions.declare_file(
        ctx.label.name + "/" + source.basename.rsplit("-", 1)[0] + "-" + ctx.attr.platform + ".whl",
    )
    ctx.actions.run(
        executable = ctx.executable._auditor,
        arguments = [source.path, output.path, ctx.attr.platform],
        inputs = [source],
        outputs = [output],
        env = {"SOURCE_DATE_EPOCH": "315532800"},
        mnemonic = "LoomWheelAudit",
        progress_message = "Auditing Loom wheel for %s" % ctx.attr.platform,
    )
    return [DefaultInfo(files = depset([output]))]

manylinux_wheel = rule(
    implementation = _manylinux_wheel_impl,
    attrs = {
        "platform": attr.string(mandatory = True),
        "wheel": attr.label(mandatory = True, allow_single_file = [".whl"]),
        "_auditor": attr.label(
            default = "//loom/packaging/python:manylinux",
            executable = True,
            cfg = "exec",
        ),
    },
)
