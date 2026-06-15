# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.importers.core import (
    KernelArgumentSpec,
    KernelConfigArgumentSpec,
    KernelModuleSpec,
    create_kernel_module,
    kernel_module_ops,
    print_loom_module,
)
from loom.ir import I32
from loom.verify import verify_module


def _printed_kernel_module_for_target_preset(target_preset: str) -> str:
    shell = create_kernel_module(
        KernelModuleSpec(
            target_preset=target_preset,
            export_symbol="kernel",
            callee="kernel",
            arguments=[],
        )
    )
    with shell.builder.insertion_block(shell.body_block):
        shell.builder.kernel.return_()

    diagnostics = verify_module(
        shell.module,
        ops=kernel_module_ops(target_preset),
    )
    diagnostics.raise_if_errors()
    return print_loom_module(
        shell.module,
        ops=kernel_module_ops(target_preset),
    )


def test_create_kernel_module_splits_config_and_body_arguments() -> None:
    shell = create_kernel_module(
        KernelModuleSpec(
            target_preset="hip -mcpu=gfx1100",
            export_symbol="kernel",
            callee="kernel",
            config_arguments=[
                KernelConfigArgumentSpec(
                    ordinal=0,
                    name="n",
                    type=I32,
                )
            ],
            arguments=[
                KernelArgumentSpec(
                    ordinal=0,
                    name="n",
                    type=I32,
                )
            ],
        )
    )

    config_arg = shell.config_arguments_by_ordinal[0]
    body_arg = shell.body_arguments_by_ordinal[0]
    assert config_arg.id != body_arg.id
    assert config_arg.name == "n"
    assert body_arg.name == "n"
    assert config_arg.type == I32
    assert body_arg.type == I32


def test_create_kernel_module_uses_supported_amdgpu_target_record() -> None:
    shell = create_kernel_module(
        KernelModuleSpec(
            target_preset="hip -mcpu=gfx1100",
            export_symbol="kernel",
            callee="kernel",
            arguments=[],
        )
    )
    with shell.builder.insertion_block(shell.body_block):
        shell.builder.kernel.return_()

    diagnostics = verify_module(
        shell.module,
        ops=kernel_module_ops("hip -mcpu=gfx1100"),
    )
    diagnostics.raise_if_errors()
    assert (
        print_loom_module(
            shell.module,
            ops=kernel_module_ops("hip -mcpu=gfx1100"),
        )
        == """amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export(\"kernel\") @kernel() {
  %wg_count_x = index.constant 1 : index
  %wg_count_y = index.constant 1 : index
  %wg_count_z = index.constant 1 : index
  %wg_size_x = index.constant 1 : index
  %wg_size_y = index.constant 1 : index
  %wg_size_z = index.constant 1 : index
  kernel.launch.config workgroups(%wg_count_x, %wg_count_y, %wg_count_z) workgroup_size(%wg_size_x, %wg_size_y, %wg_size_z) : index
} launch() {
  kernel.return
}
"""
    )


def test_create_kernel_module_uses_gfx942_amdgpu_target_record() -> None:
    shell = create_kernel_module(
        KernelModuleSpec(
            target_preset="hip -mcpu=gfx942",
            export_symbol="kernel",
            callee="kernel",
            arguments=[],
        )
    )
    with shell.builder.insertion_block(shell.body_block):
        shell.builder.kernel.return_()

    diagnostics = verify_module(
        shell.module,
        ops=kernel_module_ops("hip -mcpu=gfx942"),
    )
    diagnostics.raise_if_errors()
    assert (
        print_loom_module(
            shell.module,
            ops=kernel_module_ops("hip -mcpu=gfx942"),
        )
        == """amdgpu.target<gfx942> @hip_mcpu_gfx942

kernel.def target(@hip_mcpu_gfx942) export(\"kernel\") @kernel() {
  %wg_count_x = index.constant 1 : index
  %wg_count_y = index.constant 1 : index
  %wg_count_z = index.constant 1 : index
  %wg_size_x = index.constant 1 : index
  %wg_size_y = index.constant 1 : index
  %wg_size_z = index.constant 1 : index
  kernel.launch.config workgroups(%wg_count_x, %wg_count_y, %wg_count_z) workgroup_size(%wg_size_x, %wg_size_y, %wg_size_z) : index
} launch() {
  kernel.return
}
"""
    )


def test_create_kernel_module_uses_amdgpu_processor_override() -> None:
    assert _printed_kernel_module_for_target_preset("hip -mcpu=gfx1101").startswith(
        'amdgpu.target<gfx1100> @hip_mcpu_gfx1101 {processor = "gfx1101"}\n'
    )


def test_create_kernel_module_uses_generic_amdgpu_target_record() -> None:
    assert _printed_kernel_module_for_target_preset(
        "hip -mcpu=gfx11-generic"
    ).startswith("amdgpu.target<gfx11-generic> @hip_mcpu_gfx11_generic\n")
