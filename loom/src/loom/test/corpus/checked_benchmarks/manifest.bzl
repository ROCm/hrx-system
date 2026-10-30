# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checked application kernels shared with benchmark workloads."""

load("//loom/build_tools/bazel:defs.bzl", "loom_corpus_manifest")

CHECKED_BENCHMARK_CORPUS = loom_corpus_manifest(
    name = "checked_benchmarks",
    package = "//loom/src/loom/test/corpus/checked_benchmarks",
    srcs = [
        "cooperative_paged_attention_f32.loom",
        "grouped_paged_attention_f32.loom",
        "moe_routed_combine_loop_f32.loom",
        "online_softmax_tail_attention_tile_f32.loom",
        "packed_s8_fields_f16scale_f32dot.loom",
        "paged_decode_attention_tile_f32.loom",
        "paged_kv_block_gather_score_tile_f32.loom",
        "paged_online_softmax_loop_f32.loom",
        "rope_pair_rotate_tile_f32.loom",
        "routed_row_combine_f32.loom",
        "signed_i8_tile_quantize_f32.loom",
        "sparse_token_attention_f32.loom",
        "split_kv_online_softmax_combine_f32.loom",
        "streaming_packed_s8_dot.loom",
        "top2_routing_expert_gather_f32.loom",
        "weighted_expert_route_gather_f32.loom",
        "workgroup_rank_sort_f32_i32.loom",
    ],
)
