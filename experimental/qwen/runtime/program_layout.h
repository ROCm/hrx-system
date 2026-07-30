// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_RUNTIME_PROGRAM_LAYOUT_H_
#define EXPERIMENTAL_QWEN_RUNTIME_PROGRAM_LAYOUT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Byte span within per-issue transient program storage.
typedef struct qwen_program_span_t {
  // Byte offset from the transient allocation base.
  iree_device_size_t offset;
  // Exact accessible byte length.
  iree_device_size_t length;
} qwen_program_span_t;

// Named transient spans used by one complete layer program.
//
// The first implementation assigns distinct storage to every semantic value.
// This makes the correctness witness independent of a lifetime planner.
// Evidence-backed reuse can later merge spans only across explicit dependency
// barriers that prove the preceding value dead.
typedef struct qwen_layer_program_layout_t {
  // Attention RMSNorm output sized for either F32 or GGML Q8_1 x4 storage.
  qwen_program_span_t attention_projection_input;
  // Raw F32 query projection.
  qwen_program_span_t raw_query;
  // Raw F32 key projection.
  qwen_program_span_t raw_key;
  // Raw F32 value projection.
  qwen_program_span_t raw_value;
  // Per-head normalized and RoPE-rotated F32 query.
  qwen_program_span_t rotated_query;
  // F32 grouped-query attention result.
  qwen_program_span_t attention_output;
  // Materialized F32 feed-forward RMSNorm output.
  qwen_program_span_t feed_forward_norm;
  // F32 router logits.
  qwen_program_span_t router_logits;
  // I32 selected expert IDs.
  qwen_program_span_t route_ids;
  // F32 normalized selected-expert weights.
  qwen_program_span_t route_weights;
  // I32 expert assignment table.
  qwen_program_span_t expert_table;
  // I32 grouped-dispatch partition table.
  qwen_program_span_t partition_table;
  // F32 routed SwiGLU values.
  qwen_program_span_t swiglu;
  // F16 routed down-projection values.
  qwen_program_span_t routed_down;
  // Complete transient allocation size.
  iree_device_size_t transient_byte_length;
} qwen_layer_program_layout_t;

// Calculates the exact complete-layer transient layout.
iree_status_t qwen_layer_program_layout_calculate(
    iree_host_size_t token_count, qwen_layer_program_layout_t* out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_RUNTIME_PROGRAM_LAYOUT_H_
