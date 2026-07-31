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

// Decode-only completion-counter storage initialized before command execution.
typedef struct qwen_decode_completion_layout_t {
  // Contiguous byte range covered by the issue-time initialization fill.
  qwen_program_span_t initialization;
  // Per-KV-head counters reset by every split-attention dispatch.
  qwen_program_span_t attention;
  // Per-route physical-group counters reset by every fused gate/up dispatch.
  qwen_program_span_t gate_up;
  // Counter reset after each sequential fused decode-stage publication.
  qwen_program_span_t shared;
} qwen_decode_completion_layout_t;

// Compact vocabulary candidates finalized into one selected token.
typedef struct qwen_vocabulary_argmax_layout_t {
  // One F32 maximum logit for each endpoint projection workgroup.
  qwen_program_span_t partial_logits;
  // I32 vocabulary ID associated with each partial maximum.
  qwen_program_span_t partial_ids;
} qwen_vocabulary_argmax_layout_t;

// Full-model transient layout features.
typedef uint32_t qwen_full_program_layout_flags_t;
enum qwen_full_program_layout_flag_bits_e {
  QWEN_FULL_PROGRAM_LAYOUT_FLAG_NONE = 0u,

  // Reserves partial and completion storage for fused split-K decode attention.
  QWEN_FULL_PROGRAM_LAYOUT_FLAG_DECODE_SPLIT_ATTENTION = 1u << 0,

  // Reserves completion storage for fused decode-stage publications.
  QWEN_FULL_PROGRAM_LAYOUT_FLAG_DECODE_FUSED_STAGE_COMPLETION = 1u << 1,
};

// Named transient spans used by one complete layer program.
//
// Most semantic values have distinct storage, keeping the correctness witness
// independent of a general lifetime planner. The two explicitly phased spans
// carry projection inputs whose consumers are separated by dispatch barriers.
typedef struct qwen_layer_program_layout_t {
  // Phased F32 or Q8_1 x4 input for attention and feed-forward projections.
  qwen_program_span_t projection_input_scratch;
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
  // Phased F16 routed-down output or Q8_1 x4 routed-down input.
  qwen_program_span_t routed_projection_scratch;
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
  // Per-KV-block F32 online-softmax maximums reused by every decode layer.
  qwen_program_span_t attention_partial_maximums;
  // Per-KV-block F32 online-softmax sums reused by every decode layer.
  qwen_program_span_t attention_partial_sums;
  // Per-KV-block F16 attention outputs reused by every decode layer.
  qwen_program_span_t attention_partial_outputs;
  // Decode-only completion counters reused sequentially by every layer.
  qwen_decode_completion_layout_t decode_completion;
  // GGML Q8_1 x4 packing of the final normalized hidden-state row.
  qwen_program_span_t final_quantized_hidden_state;
  // Partial projection maxima consumed by compact token finalization.
  qwen_vocabulary_argmax_layout_t vocabulary_argmax;
  // Complete transient allocation size.
  iree_device_size_t transient_byte_length;
} qwen_full_program_layout_t;

// Calculates the exact complete-layer transient layout.
iree_status_t qwen_layer_program_layout_calculate(
    iree_host_size_t token_count, qwen_layer_program_layout_t* out_layout);

// Calculates the exact complete full-model transient layout.
iree_status_t qwen_full_program_layout_calculate(
    iree_host_size_t token_count, iree_host_size_t context_count,
    qwen_full_program_layout_flags_t flags,
    qwen_full_program_layout_t* out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_RUNTIME_PROGRAM_LAYOUT_H_
