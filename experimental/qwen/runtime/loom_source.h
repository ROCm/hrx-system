// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_RUNTIME_LOOM_SOURCE_H_
#define EXPERIMENTAL_QWEN_RUNTIME_LOOM_SOURCE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Stable runtime path for the temporary raw-Q4_K embedding producer.
#define QWEN_LOOM_SOURCE_TOKEN_EMBEDDING_Q4K_BRINGUP_WORKAROUND \
  "qwen3_moe/token_embedding_q4k_bringup_workaround"

// Stable runtime path for the temporary partial-pair token selector.
#define QWEN_LOOM_SOURCE_GREEDY_ARGMAX_PARTIALS_BRINGUP_WORKAROUND \
  "qwen3_moe/greedy_argmax_partials_bringup_workaround"

// Stable runtime path for the device-owned attention metadata producer.
#define QWEN_LOOM_SOURCE_ATTENTION_METADATA "qwen3_moe/attention_metadata"

// Stable runtime path for the attention preparation module.
#define QWEN_LOOM_SOURCE_ATTENTION_PREPARE_QUANTIZED \
  "qwen3_moe/attention_prepare_quantized"

// Stable runtime path for the fused quantized Q/K/V module.
#define QWEN_LOOM_SOURCE_ATTENTION_QKV_QUANTIZED \
  "qwen3_moe/attention_qkv_quantized"

// Stable runtime path for fused decode Q/K/V and head postprocessing.
#define QWEN_LOOM_SOURCE_ATTENTION_QKV_POSTPROCESS_FUSED \
  "qwen3_moe/attention_qkv_postprocess_fused"

// Stable runtime path for the attention RoPE and cache publication module.
#define QWEN_LOOM_SOURCE_ATTENTION_POSTPROCESS_F32_F16 \
  "qwen3_moe/attention_postprocess_f32_f16"

// Stable runtime path for the prefill FlashAttention module.
#define QWEN_LOOM_SOURCE_FLASH_ATTENTION_PREFILL_F32_F16 \
  "qwen3_moe/flash_attention_prefill_f32_f16"

// Stable runtime path for fused split-K decode FlashAttention.
#define QWEN_LOOM_SOURCE_FLASH_ATTENTION_DECODE_SPLIT_F32_F16 \
  "qwen3_moe/flash_attention_decode_split_f32_f16"

// Stable runtime path for dense quantized F16-WMMA projections.
#define QWEN_LOOM_SOURCE_DENSE_LINEAR_QUANTIZED_F16 \
  "qwen3_moe/dense_linear_quantized_f16"

// Stable runtime path for the Q6_K vocabulary projection.
#define QWEN_LOOM_SOURCE_VOCABULARY_PROJECTION_Q6 \
  "qwen3_moe/vocabulary_projection_q6"

// Stable runtime path for the temporary F32-activation Q8_1 packer.
#define QWEN_LOOM_SOURCE_QUANTIZE_Q8_1_X4_BRINGUP_WORKAROUND \
  "qwen3_moe/quantize_q8_1_x4_bringup_workaround"

// Stable runtime path for the F32 router projection module.
#define QWEN_LOOM_SOURCE_ROUTER_PROJECTION_F32 "qwen3_moe/router_projection_f32"

// Stable runtime path for fused decode projection and normalized top-8 routing.
#define QWEN_LOOM_SOURCE_ROUTER_PROJECTION_TOP8_FUSED_F32 \
  "qwen3_moe/router_projection_top8_fused_f32"

// Stable runtime path for the normalized top-8 router module.
#define QWEN_LOOM_SOURCE_ROUTER_TOP8_F32 "qwen3_moe/router_top8_f32"

// Stable runtime path for fused prefill expert and partition table
// construction.
#define QWEN_LOOM_SOURCE_EXPERT_TABLE_PARTITION_FUSED \
  "qwen3_moe/expert_table_partition_fused"

// Stable runtime path for grouped routed gate/up projection.
#define QWEN_LOOM_SOURCE_ROUTED_GATE_UP_F16 "qwen3_moe/routed_gate_up_f16"

// Stable runtime path for direct Q8_1 routed gate/up projection.
#define QWEN_LOOM_SOURCE_ROUTED_GATE_UP_Q8 "qwen3_moe/routed_gate_up_q8"

// Stable runtime path for grouped routed down projection and reduction.
#define QWEN_LOOM_SOURCE_ROUTED_DOWN_F16 "qwen3_moe/routed_down_f16"

// Stable runtime path for fused prefill residual and next-layer normalization.
#define QWEN_LOOM_SOURCE_ROUTED_DOWN_NEXT_RMSNORM_F32 \
  "qwen3_moe/routed_down_next_rmsnorm_f32"

// Stable runtime path for direct Q4_K routed down and residual publication.
#define QWEN_LOOM_SOURCE_ROUTED_DOWN_Q4_Q8 "qwen3_moe/routed_down_q4_q8"

// Stable runtime path for direct Q6_K routed down and residual publication.
#define QWEN_LOOM_SOURCE_ROUTED_DOWN_Q6_Q8 "qwen3_moe/routed_down_q6_q8"

// Borrowed embedded Loom module source.
typedef struct qwen_loom_source_module_t {
  // Stable runtime path independent of the generated Bazel output name.
  iree_string_view_t module_path;
  // Source identifier reported in compiler diagnostics.
  iree_string_view_t source_identifier;
  // Complete linked textual Loom module contents.
  iree_const_byte_span_t source_contents;
} qwen_loom_source_module_t;

// Looks up an embedded linked module by its stable runtime path.
//
// The returned views borrow from process-static embedded storage.
iree_status_t qwen_loom_source_lookup(
    iree_string_view_t module_path,
    qwen_loom_source_module_t* out_source_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_RUNTIME_LOOM_SOURCE_H_
