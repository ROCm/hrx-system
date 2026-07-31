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
// Most semantic values have distinct storage, keeping the correctness witness
// independent of a general lifetime planner. The projection scratch is the one
// explicit reuse: the QKV input is dead after its dispatch barrier, before the
// attention-output quantizer writes the same storage.
typedef struct qwen_layer_program_layout_t {
  // Phased QKV-input and attention-output Q8_1 x4 projection scratch.
  qwen_program_span_t attention_projection_scratch;
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

// Named transient spans used by one complete full-model program.
//
// The reusable layer owns scratch for the program token count. The terminal
// layer owns distinct one-token scratch for the final-token layer-47 workload.
// Both nested layouts use offsets from the full allocation base, while each
// nested transient_byte_length remains the exact size of its own region.
typedef struct qwen_full_program_layout_t {
  // Reusable complete-layer scratch for the program token count.
  qwen_layer_program_layout_t layer;
  // Complete one-token scratch for the terminal layer-47 workload.
  qwen_layer_program_layout_t terminal_layer;
  // Final RMSNorm output containing one F32 hidden-state row.
  qwen_program_span_t final_normalized_hidden_state;
  // GGML Q8_1 x4 packing of the final normalized hidden-state row.
  qwen_program_span_t final_quantized_hidden_state;
  // Output projection result containing one F32 vocabulary row.
  qwen_program_span_t vocabulary_logits;
  // Complete transient allocation size.
  iree_device_size_t transient_byte_length;
} qwen_full_program_layout_t;

// Calculates the exact complete-layer transient layout.
iree_status_t qwen_layer_program_layout_calculate(
    iree_host_size_t token_count, qwen_layer_program_layout_t* out_layout);

// Calculates the exact complete full-model transient layout.
iree_status_t qwen_full_program_layout_calculate(
    iree_host_size_t token_count, qwen_full_program_layout_t* out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_RUNTIME_PROGRAM_LAYOUT_H_
