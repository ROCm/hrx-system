// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_PROGRAM_BLOCK_H_
#define EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_PROGRAM_BLOCK_H_

#include <stdbool.h>
#include <stdint.h>

#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/stages/ideogram4_dit_program.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY \
  ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY

enum {
  ID4_IDEOGRAM4_DIT_ACTIVATION_IDENTITY = 0,
  ID4_IDEOGRAM4_DIT_ACTIVATION_SILU = 1,
};

typedef struct id4_ideogram4_dit_program_dense_options_t {
  // Stable operation name used for diagnostics.
  iree_string_view_t operation_name;
  // Stable output tensor name used for diagnostics.
  iree_string_view_t output_name;
  // Parameter source policy used when loading dense parameters.
  id4_ideogram4_dit_parameter_sources_t parameter_sources;
  // Parameter key prefix that owns ".weight" and ".bias" tensors.
  iree_string_view_t parameter_prefix;
  // Execution storage strategy selected for dense weights.
  id4_ideogram4_dit_weight_execution_format_t weight_execution_format;
  // Dense input vector.
  id4_pipeline_program_tensor_t input;
  // Dense input channel count.
  uint32_t input_size;
  // Dense output channel count.
  uint32_t output_size;
  // Compile-time activation kind applied after bias addition.
  uint32_t activation_kind;
} id4_ideogram4_dit_program_dense_options_t;

typedef struct id4_ideogram4_dit_program_block_options_t {
  // Program builder receiving the authored block operations.
  id4_pipeline_program_builder_t* builder;
  // Branch diagnostic prefix, such as "ideogram4.cond".
  iree_string_view_t branch_name;
  // Parameter source policy used when loading DiT block parameters.
  id4_ideogram4_dit_parameter_sources_t parameter_sources;
  // Transformer layer ordinal used for parameter keys and diagnostic names.
  uint32_t layer_ordinal;
  // AdaLN conditioning vector channel count.
  uint32_t adaln_size;
  // Transformer hidden-state channel count.
  uint32_t hidden_size;
  // Feed-forward intermediate channel count.
  uint32_t intermediate_size;
  // Transformer attention head count.
  uint32_t attention_head_count;
  // Combined text and image token count.
  uint32_t total_token_count;
  // Initialized input hidden-state tensor for this block.
  id4_pipeline_program_tensor_t hidden_input;
  // Initialized AdaLN conditioning tensor shared by all blocks.
  id4_pipeline_program_tensor_t adaln_input;
  // Initialized rotary position embedding tensor.
  id4_pipeline_program_tensor_t position_embedding;
  // Activation storage format for internal linear-input producers.
  id4_ideogram4_dit_activation_format_t activation_format;
  // Execution storage strategy selected for block linear weights.
  id4_ideogram4_dit_weight_execution_format_t weight_execution_format;
  // Attention implementation selected for this transformer block.
  id4_ideogram4_dit_attention_implementation_t attention_implementation;
  // Feed-forward implementation selected for this transformer block.
  id4_ideogram4_dit_feed_forward_implementation_t feed_forward_implementation;
  // Diagnostic tap names requested by the stage plan.
  iree_string_view_list_t diagnostic_tap_names;
} id4_ideogram4_dit_program_block_options_t;

iree_status_t id4_ideogram4_dit_program_format(char* buffer,
                                               iree_host_size_t buffer_capacity,
                                               iree_string_view_t* out_string,
                                               const char* format, ...);

iree_status_t id4_ideogram4_dit_program_format_branch_layer_name(
    iree_string_view_t branch_name, uint32_t layer_ordinal,
    iree_string_view_t suffix, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string);

bool id4_ideogram4_dit_program_checked_mul_u64(uint64_t lhs, uint64_t rhs,
                                               uint64_t* out_result);

bool id4_ideogram4_dit_program_has_diagnostic_tap(
    iree_string_view_list_t diagnostic_tap_names, iree_string_view_t name);

iree_status_t id4_ideogram4_dit_program_parameter(
    id4_pipeline_program_builder_t* builder, iree_string_view_t source_scope,
    iree_string_view_t key, id4_pipeline_program_dtype_t dtype,
    id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor);

iree_status_t id4_ideogram4_dit_program_parameter_fp8_e4m3_scaled_to_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t source_scope,
    iree_string_view_t weight_key, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t* out_tensor);

iree_status_t id4_ideogram4_dit_program_parameter_bf16_linear_rhs_tile(
    id4_pipeline_program_builder_t* builder, iree_string_view_t source_scope,
    iree_string_view_t weight_key, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t* out_tensor);

iree_status_t
id4_ideogram4_dit_program_parameter_fp8_e4m3_scaled_to_bf16_linear_rhs_tile(
    id4_pipeline_program_builder_t* builder, iree_string_view_t source_scope,
    iree_string_view_t weight_key, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t* out_tensor);

iree_status_t id4_ideogram4_dit_program_parameter_fp8_e4m3_scaled(
    id4_pipeline_program_builder_t* builder, iree_string_view_t source_scope,
    iree_string_view_t weight_key, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t* out_weight,
    id4_pipeline_program_tensor_t* out_scale);

iree_status_t id4_ideogram4_dit_program_parameter_source_resolve(
    id4_ideogram4_dit_parameter_sources_t sources, iree_string_view_t key,
    id4_ideogram4_dit_parameter_source_rule_t* out_source);

iree_status_t id4_ideogram4_dit_program_format_parameter_scale_key(
    iree_string_view_t weight_key, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string);

iree_status_t id4_ideogram4_dit_program_acquire_tensor(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_dtype_t dtype, id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor);

iree_status_t id4_ideogram4_dit_program_barrier(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name);

iree_status_t id4_ideogram4_dit_program_tap(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_tensor_t tensor);

iree_status_t id4_ideogram4_dit_program_tap_linear_input_bf16_as_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    id4_pipeline_program_tensor_t packed_input);

iree_status_t id4_ideogram4_dit_program_tap_token_major_f32_as_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size,
    id4_pipeline_program_tensor_t token_major_input);

iree_status_t id4_ideogram4_dit_program_tap_bf16_as_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_shape_t shape, id4_pipeline_program_tensor_t input);

iree_status_t id4_ideogram4_dit_program_dispatch_cast_bf16_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_shape_t shape, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_zero_tail_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t logical_element_count, uint32_t dispatch_element_count,
    id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dense_bf16(
    id4_pipeline_program_builder_t* builder,
    const id4_ideogram4_dit_program_dense_options_t* options,
    id4_pipeline_program_tensor_t* out_output);

// Authors a dense vector projection that preserves BF16 output rounding while
// storing the rounded value in an F32 tensor for an immediate F32 consumer.
iree_status_t id4_ideogram4_dit_program_dense_bf16_rounded_f32(
    id4_pipeline_program_builder_t* builder,
    const id4_ideogram4_dit_program_dense_options_t* options,
    id4_pipeline_program_tensor_t* out_output);

iree_status_t id4_ideogram4_dit_program_dispatch_adaln_split(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t hidden_size, id4_pipeline_program_tensor_t raw_modulation,
    id4_pipeline_program_tensor_t scale_msa,
    id4_pipeline_program_tensor_t gate_msa,
    id4_pipeline_program_tensor_t scale_mlp,
    id4_pipeline_program_tensor_t gate_mlp);

iree_status_t id4_ideogram4_dit_program_dispatch_modulated_rmsnorm(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t hidden_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t scale, id4_pipeline_program_tensor_t output);

iree_status_t
id4_ideogram4_dit_program_dispatch_modulated_rmsnorm_token_major_linear_input_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t hidden_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t scale, id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_rmsnorm_gated_residual(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t hidden_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t gate, id4_pipeline_program_tensor_t residual,
    id4_pipeline_program_tensor_t output);

iree_status_t
id4_ideogram4_dit_program_dispatch_rmsnorm_gated_residual_input_bf16_token_major(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t hidden_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t gate, id4_pipeline_program_tensor_t residual,
    id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_mlp_gate_up_silu(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size, uint32_t intermediate_size,
    id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t gate_weight,
    id4_pipeline_program_tensor_t up_weight,
    id4_pipeline_program_tensor_t output);

iree_status_t
id4_ideogram4_dit_program_dispatch_mlp_gate_up_silu_linear_input_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size, uint32_t intermediate_size,
    id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t gate_weight,
    id4_pipeline_program_tensor_t up_weight,
    id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_silu_product_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t intermediate_size,
    id4_pipeline_program_tensor_t gate, id4_pipeline_program_tensor_t up,
    id4_pipeline_program_tensor_t output);

iree_status_t
id4_ideogram4_dit_program_dispatch_mlp_up_silu_product_packed_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t intermediate_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t up_weight, id4_pipeline_program_tensor_t gate,
    id4_pipeline_program_tensor_t output);

iree_status_t
id4_ideogram4_dit_program_dispatch_mlp_up_silu_product_packed_bf16_compact_rhs_tile(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t intermediate_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t up_weight, id4_pipeline_program_tensor_t gate,
    id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_linear_packed_bf16_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t weight, id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_linear_packed_bf16_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t weight, id4_pipeline_program_tensor_t output);

iree_status_t
id4_ideogram4_dit_program_dispatch_linear_packed_bf16_bf16_compact_rhs_tile(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t weight, id4_pipeline_program_tensor_t output);

iree_status_t
id4_ideogram4_dit_program_dispatch_linear_packed_bf16_bf16_compact_rhs_tile_workgroup_staged(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t weight, id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_linear_packed_fp8_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t weight, id4_pipeline_program_tensor_t scale,
    id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_linear_bf16_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_linear_bf16_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_linear_fp8_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t weight, id4_pipeline_program_tensor_t scale,
    id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_linear_fp8_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t scale, id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_linear_input_unpack_bf16_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_linear_input_unpack_f32_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_qkv_split(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t hidden_size,
    id4_pipeline_program_tensor_t qkv, id4_pipeline_program_tensor_t query,
    id4_pipeline_program_tensor_t key, id4_pipeline_program_tensor_t value);

iree_status_t id4_ideogram4_dit_program_dispatch_qkv_norm_rotary(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t attention_head_count, uint32_t head_size,
    id4_ideogram4_dit_activation_format_t activation_format,
    id4_pipeline_program_tensor_t qkv,
    id4_pipeline_program_tensor_t norm_q_weight,
    id4_pipeline_program_tensor_t norm_k_weight,
    id4_pipeline_program_tensor_t position_embedding,
    id4_pipeline_program_tensor_t rotated_query,
    id4_pipeline_program_tensor_t rotated_key,
    id4_pipeline_program_tensor_t value);

iree_status_t id4_ideogram4_dit_program_dispatch_qkv_norm_rotary_packed_value(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity,
    uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t qkv,
    id4_pipeline_program_tensor_t norm_q_weight,
    id4_pipeline_program_tensor_t norm_k_weight,
    id4_pipeline_program_tensor_t position_embedding,
    id4_pipeline_program_tensor_t rotated_query,
    id4_pipeline_program_tensor_t rotated_key,
    id4_pipeline_program_tensor_t value);

iree_status_t id4_ideogram4_dit_program_dispatch_head_rmsnorm(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_rotary_apply(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t position_embedding,
    id4_pipeline_program_tensor_t rotated_query,
    id4_pipeline_program_tensor_t rotated_key);

iree_status_t id4_ideogram4_dit_program_dispatch_attention(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t value, id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_attention_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t value, id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_attention_bf16_linear_input(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t value, id4_pipeline_program_tensor_t output);

iree_status_t
id4_ideogram4_dit_program_dispatch_attention_qk_scores_all_heads_bf16_f32_wmma(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t valid_token_count, uint32_t padded_token_count,
    uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t scores);

iree_status_t
id4_ideogram4_dit_program_dispatch_attention_softmax_all_heads_f32_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t valid_token_count, uint32_t padded_token_count,
    uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t scores,
    id4_pipeline_program_tensor_t probabilities);

iree_status_t
id4_ideogram4_dit_program_dispatch_attention_pv_all_heads_bf16_bf16_wmma(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t valid_token_count, uint32_t padded_token_count,
    uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t probabilities,
    id4_pipeline_program_tensor_t value, id4_pipeline_program_tensor_t output);

iree_status_t
id4_ideogram4_dit_program_dispatch_attention_blocked_qk_scores_bf16_f32_wmma(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t valid_token_count, uint32_t padded_token_count,
    uint32_t query_block_offset, uint32_t query_block_token_count,
    uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t scores);

iree_status_t
id4_ideogram4_dit_program_dispatch_attention_blocked_softmax_f32_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t valid_token_count, uint32_t padded_token_count,
    uint32_t query_block_offset, uint32_t query_block_token_count,
    uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t scores,
    id4_pipeline_program_tensor_t probabilities);

iree_status_t
id4_ideogram4_dit_program_dispatch_attention_blocked_pv_bf16_bf16_wmma(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t valid_token_count, uint32_t padded_token_count,
    uint32_t query_block_offset, uint32_t query_block_token_count,
    uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t probabilities,
    id4_pipeline_program_tensor_t value, id4_pipeline_program_tensor_t output);

iree_status_t
id4_ideogram4_dit_program_dispatch_attention_online_bf16_bf16_wmma(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t valid_token_count, uint32_t padded_token_count,
    uint32_t query_block_offset, uint32_t query_block_token_count,
    uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t value, id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dense_f32(
    id4_pipeline_program_builder_t* builder,
    const id4_ideogram4_dit_program_dense_options_t* options,
    id4_pipeline_program_tensor_t* out_output);

iree_status_t id4_ideogram4_dit_program_author_transformer_block(
    const id4_ideogram4_dit_program_block_options_t* options,
    id4_pipeline_program_tensor_t* out_hidden);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_PROGRAM_BLOCK_H_
