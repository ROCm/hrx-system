# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 WITH LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ID4 Loom kernel test and benchmark definitions."""

def id4_vae_group_norm_apply_case(
        name,
        width,
        height,
        channel_count,
        group_count,
        output_channel_tile_width,
        apply_silu):
    """Builds one configured VAE group-norm apply case."""
    output_element_count = width * height * channel_count
    return {
        "actual_case": "@%s_case" % name,
        "benchmark": "@%s" % name,
        "configs": [
            "--config=id4.vae.group_norm.width=%d" % width,
            "--config=id4.vae.group_norm.height=%d" % height,
            "--config=id4.vae.group_norm.channel_count=%d" % channel_count,
            "--config=id4.vae.group_norm.group_count=%d" % group_count,
            "--config=id4.vae.group_norm.channels_per_group=%d" % (channel_count // group_count),
            "--config=id4.vae.group_norm.batch_count=1",
            "--config=id4.vae.group_norm.output_element_count=%d" % output_element_count,
            "--config=id4.vae.group_norm.output_channel_tile_width=%d" % output_channel_tile_width,
            "--config=id4.vae.group_norm.output_channel_tile_count=%d" % (channel_count // output_channel_tile_width),
            "--config=id4.vae.group_norm.output_tile_element_count=%d" % (output_element_count // output_channel_tile_width),
            "--config=id4.vae.group_norm.apply_silu=%d" % apply_silu,
        ],
        "source": "vae/group_norm_apply_bf16.loom",
        "target_name": name[4:],
    }
