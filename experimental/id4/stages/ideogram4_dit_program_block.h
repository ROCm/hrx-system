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
#include "iree/base/api.h"
#include "iree/hal/command_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY 192

enum {
  ID4_IDEOGRAM4_DIT_ACTIVATION_IDENTITY = 0,
  ID4_IDEOGRAM4_DIT_ACTIVATION_SILU = 1,
};

typedef struct id4_ideogram4_dit_program_dense_options_t {
  // Stable operation name used for diagnostics.
  iree_string_view_t operation_name;
  // Stable output tensor name used for diagnostics.
  iree_string_view_t output_name;
  // Parameter key prefix that owns ".weight" and ".bias" tensors.
  iree_string_view_t parameter_prefix;
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
} id4_ideogram4_dit_program_block_options_t;

iree_status_t id4_ideogram4_dit_program_format(char* buffer,
                                               iree_host_size_t buffer_capacity,
                                               iree_string_view_t* out_string,
                                               const char* format, ...);

iree_status_t id4_ideogram4_dit_program_format_layer_parameter(
    uint32_t layer_ordinal, iree_string_view_t suffix, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string);

iree_status_t id4_ideogram4_dit_program_format_branch_layer_name(
    iree_string_view_t branch_name, uint32_t layer_ordinal,
    iree_string_view_t suffix, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string);

bool id4_ideogram4_dit_program_checked_mul_u64(uint64_t lhs, uint64_t rhs,
                                               uint64_t* out_result);

iree_hal_dispatch_config_t id4_ideogram4_dit_program_make_dispatch_config(
    uint32_t workgroup_count_x, uint32_t workgroup_count_y,
    uint32_t workgroup_count_z);

iree_status_t id4_ideogram4_dit_program_make_element_dispatch_config(
    uint32_t element_count, iree_hal_dispatch_config_t* out_dispatch_config);

iree_status_t id4_ideogram4_dit_program_parameter(
    id4_pipeline_program_builder_t* builder, iree_string_view_t key,
    id4_pipeline_program_dtype_t dtype, id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor);

iree_status_t id4_ideogram4_dit_program_layer_parameter(
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    iree_string_view_t suffix, id4_pipeline_program_dtype_t dtype,
    id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor);

iree_status_t id4_ideogram4_dit_program_acquire_tensor(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_dtype_t dtype, id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor);

iree_status_t id4_ideogram4_dit_program_barrier(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name);

iree_status_t id4_ideogram4_dit_program_tap(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_tensor_t tensor);

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

iree_status_t id4_ideogram4_dit_program_dispatch_rmsnorm_gated_residual(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t hidden_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t gate, id4_pipeline_program_tensor_t residual,
    id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_silu_product(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t intermediate_size,
    id4_pipeline_program_tensor_t gate, id4_pipeline_program_tensor_t up,
    id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_linear_bf16_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t output);

iree_status_t id4_ideogram4_dit_program_dispatch_qkv_split(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t hidden_size,
    id4_pipeline_program_tensor_t qkv, id4_pipeline_program_tensor_t query,
    id4_pipeline_program_tensor_t key, id4_pipeline_program_tensor_t value);

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

iree_status_t id4_ideogram4_dit_program_dense_bf16_f32(
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
