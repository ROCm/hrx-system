# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Constructs a Loom kernel from source-independent importer facts."""

# --8<-- [start:importer]
from loom.importers import (
    KernelArgumentSpec,
    KernelLaunchConfigSpec,
    KernelModuleSpec,
    create_kernel_module,
    kernel_module_ops,
    print_loom_module,
)
from loom.ir import INDEX, OFFSET, Module
from loom.verify import verify_module

TARGET_PRESET = "reference"
MODULE_OPS = kernel_module_ops(TARGET_PRESET)


def build_copy_module() -> Module:
    """Builds a kernel that copies one byte per workitem."""

    shell = create_kernel_module(
        KernelModuleSpec(
            target_preset=TARGET_PRESET,
            export_symbol="copy_64_bytes",
            callee="copy_64_bytes",
            arguments=(
                KernelArgumentSpec(ordinal=0, name="source"),
                KernelArgumentSpec(ordinal=1, name="target"),
            ),
            launch_config=KernelLaunchConfigSpec(workgroup_size=(64, 1, 1)),
        )
    )
    source = shell.body_arguments_by_ordinal[0]
    target = shell.body_arguments_by_ordinal[1]

    with shell.builder.insertion_block(shell.body_block):
        workitem_index = shell.builder.kernel.workitem_dispatch_id(
            dimension="x",
            results=[INDEX],
            name="workitem_index",
        )
        byte_offset = shell.builder.index.cast(
            input=workitem_index,
            results=[OFFSET],
            name="byte_offset",
        )
        value = shell.builder.buffer.load_i8_u(
            source=source,
            byte_offset=byte_offset,
            name="value",
        )
        shell.builder.buffer.store_i8(
            value=value,
            target=target,
            byte_offset=byte_offset,
        )
        shell.builder.kernel.return_()

    verify_module(shell.module, ops=MODULE_OPS).raise_if_errors()
    return shell.module


# --8<-- [end:importer]


def main() -> None:
    print(print_loom_module(build_copy_module(), ops=MODULE_OPS), end="")


if __name__ == "__main__":
    main()
