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

// Stable runtime path for the device-owned attention metadata producer.
#define QWEN_LOOM_SOURCE_ATTENTION_METADATA "qwen3_moe/attention_metadata"

// Stable runtime path for the attention preparation module.
#define QWEN_LOOM_SOURCE_ATTENTION_PREPARE_QUANTIZED \
  "qwen3_moe/attention_prepare_quantized"

// Stable runtime path for the fused quantized Q/K/V module.
#define QWEN_LOOM_SOURCE_ATTENTION_QKV_QUANTIZED \
  "qwen3_moe/attention_qkv_quantized"

// Stable runtime path for the attention RoPE and cache publication module.
#define QWEN_LOOM_SOURCE_ATTENTION_POSTPROCESS_F32_F16 \
  "qwen3_moe/attention_postprocess_f32_f16"

// Stable runtime path for the prefill FlashAttention module.
#define QWEN_LOOM_SOURCE_FLASH_ATTENTION_PREFILL_F32_F16 \
  "qwen3_moe/flash_attention_prefill_f32_f16"

// Stable runtime path for dense quantized F16-WMMA projections.
#define QWEN_LOOM_SOURCE_DENSE_LINEAR_QUANTIZED_F16 \
  "qwen3_moe/dense_linear_quantized_f16"

// Stable runtime path for the F32 router projection module.
#define QWEN_LOOM_SOURCE_ROUTER_PROJECTION_F32 "qwen3_moe/router_projection_f32"

// Stable runtime path for the normalized top-8 router module.
#define QWEN_LOOM_SOURCE_ROUTER_TOP8_F32 "qwen3_moe/router_top8_f32"

// Stable runtime path for grouped routed gate/up projection.
#define QWEN_LOOM_SOURCE_ROUTED_GATE_UP_F16 "qwen3_moe/routed_gate_up_f16"

// Stable runtime path for grouped routed down projection and reduction.
#define QWEN_LOOM_SOURCE_ROUTED_DOWN_F16 "qwen3_moe/routed_down_f16"

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
