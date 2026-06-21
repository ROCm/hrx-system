// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_dit_program_block.h"

iree_status_t id4_ideogram4_dit_program_author_transformer_block(
    const id4_ideogram4_dit_program_block_options_t* options,
    id4_pipeline_program_tensor_t* out_hidden) {
  if (!options) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT transformer block options are required");
  }
  if (!options->builder) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT transformer block builder is required");
  }
  if (!out_hidden) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT transformer block output is required");
  }
  *out_hidden = id4_pipeline_program_tensor_invalid();

  id4_pipeline_program_builder_t* builder = options->builder;
  const iree_string_view_t branch_name = options->branch_name;
  const uint32_t layer_ordinal = options->layer_ordinal;
  const uint32_t adaln_size = options->adaln_size;
  const uint32_t hidden_size = options->hidden_size;
  const uint32_t intermediate_size = options->intermediate_size;
  const uint32_t attention_head_count = options->attention_head_count;
  const uint32_t total_token_count = options->total_token_count;
  const id4_pipeline_program_tensor_t hidden_input = options->hidden_input;
  const id4_pipeline_program_tensor_t adaln_input = options->adaln_input;
  const id4_pipeline_program_tensor_t position_embedding =
      options->position_embedding;
  const uint32_t head_size = hidden_size / attention_head_count;
  uint64_t modulation_output_size64 = 0;
  if (!id4_ideogram4_dit_program_checked_mul_u64(hidden_size, 4,
                                                 &modulation_output_size64) ||
      modulation_output_size64 > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 DiT AdaLN modulation size overflow");
  }
  const uint32_t modulation_output_size = (uint32_t)modulation_output_size64;

  char adaln_modulation_prefix_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char adaln_modulation_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t adaln_modulation_prefix = iree_string_view_empty();
  iree_string_view_t adaln_modulation_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_layer_parameter(
      layer_ordinal, IREE_SV("adaln_modulation"),
      adaln_modulation_prefix_buffer,
      IREE_ARRAYSIZE(adaln_modulation_prefix_buffer),
      &adaln_modulation_prefix));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("adaln_modulation"),
      adaln_modulation_name_buffer,
      IREE_ARRAYSIZE(adaln_modulation_name_buffer), &adaln_modulation_name));
  id4_pipeline_program_tensor_t raw_modulation =
      id4_pipeline_program_tensor_invalid();
  const id4_ideogram4_dit_program_dense_options_t adaln_modulation_options = {
      .operation_name = adaln_modulation_name,
      .output_name = adaln_modulation_name,
      .parameter_prefix = adaln_modulation_prefix,
      .input = adaln_input,
      .input_size = adaln_size,
      .output_size = modulation_output_size,
      .activation_kind = ID4_IDEOGRAM4_DIT_ACTIVATION_IDENTITY,
  };
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dense_bf16_f32(
      builder, &adaln_modulation_options, &raw_modulation));

  char after_adaln_modulation_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_adaln_modulation_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_adaln_modulation"),
      after_adaln_modulation_name_buffer,
      IREE_ARRAYSIZE(after_adaln_modulation_name_buffer),
      &after_adaln_modulation_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_adaln_modulation_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
      builder, adaln_modulation_name, raw_modulation));

  char scale_msa_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char gate_msa_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char scale_mlp_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char gate_mlp_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t scale_msa_name = iree_string_view_empty();
  iree_string_view_t gate_msa_name = iree_string_view_empty();
  iree_string_view_t scale_mlp_name = iree_string_view_empty();
  iree_string_view_t gate_mlp_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("scale_msa"), scale_msa_name_buffer,
      IREE_ARRAYSIZE(scale_msa_name_buffer), &scale_msa_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("gate_msa"), gate_msa_name_buffer,
      IREE_ARRAYSIZE(gate_msa_name_buffer), &gate_msa_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("scale_mlp"), scale_mlp_name_buffer,
      IREE_ARRAYSIZE(scale_mlp_name_buffer), &scale_mlp_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("gate_mlp"), gate_mlp_name_buffer,
      IREE_ARRAYSIZE(gate_mlp_name_buffer), &gate_mlp_name));

  id4_pipeline_program_tensor_t scale_msa =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t gate_msa =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t scale_mlp =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t gate_mlp =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, scale_msa_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(hidden_size), &scale_msa));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, gate_msa_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(hidden_size), &gate_msa));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, scale_mlp_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(hidden_size), &scale_mlp));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, gate_mlp_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(hidden_size), &gate_mlp));

  char adaln_split_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t adaln_split_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("adaln_split"),
      adaln_split_name_buffer, IREE_ARRAYSIZE(adaln_split_name_buffer),
      &adaln_split_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_adaln_split(
      builder, adaln_split_name, hidden_size, raw_modulation, scale_msa,
      gate_msa, scale_mlp, gate_mlp));

  char after_adaln_split_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_adaln_split_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_adaln_split"),
      after_adaln_split_name_buffer,
      IREE_ARRAYSIZE(after_adaln_split_name_buffer), &after_adaln_split_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_adaln_split_name));

  char attention_input_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char attention_input_dispatch_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t attention_input_name = iree_string_view_empty();
  iree_string_view_t attention_input_dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention_input"),
      attention_input_name_buffer, IREE_ARRAYSIZE(attention_input_name_buffer),
      &attention_input_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention_norm1_modulated"),
      attention_input_dispatch_name_buffer,
      IREE_ARRAYSIZE(attention_input_dispatch_name_buffer),
      &attention_input_dispatch_name));

  id4_pipeline_program_tensor_t attention_norm1_weight =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter(
      builder, layer_ordinal, IREE_SV("attention_norm1.weight"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(hidden_size),
      &attention_norm1_weight));
  id4_pipeline_program_tensor_t attention_input =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, attention_input_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(hidden_size, total_token_count),
      &attention_input));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_modulated_rmsnorm(
      builder, attention_input_dispatch_name, total_token_count, hidden_size,
      hidden_input, attention_norm1_weight, scale_msa, attention_input));

  char after_attention_input_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_attention_input_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_attention_input"),
      after_attention_input_name_buffer,
      IREE_ARRAYSIZE(after_attention_input_name_buffer),
      &after_attention_input_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_attention_input_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
      builder, attention_input_name, attention_input));

  uint64_t qkv_size64 = 0;
  if (!id4_ideogram4_dit_program_checked_mul_u64(hidden_size, 3, &qkv_size64) ||
      qkv_size64 > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 DiT QKV size overflow");
  }
  const uint32_t qkv_size = (uint32_t)qkv_size64;
  char qkv_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char qkv_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t qkv_name = iree_string_view_empty();
  iree_string_view_t qkv_dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.qkv"), qkv_name_buffer,
      IREE_ARRAYSIZE(qkv_name_buffer), &qkv_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.qkv_projection"),
      qkv_dispatch_name_buffer, IREE_ARRAYSIZE(qkv_dispatch_name_buffer),
      &qkv_dispatch_name));

  id4_pipeline_program_tensor_t qkv_weight =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter(
      builder, layer_ordinal, IREE_SV("attention.qkv.weight"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(qkv_size, hidden_size),
      &qkv_weight));
  id4_pipeline_program_tensor_t qkv = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, qkv_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(qkv_size, total_token_count),
      &qkv));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_linear_bf16_f32(
      builder, qkv_dispatch_name, total_token_count, hidden_size, qkv_size,
      attention_input, qkv_weight, qkv));

  char after_qkv_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_qkv_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_attention_qkv"),
      after_qkv_name_buffer, IREE_ARRAYSIZE(after_qkv_name_buffer),
      &after_qkv_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_qkv_name));

  char query_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char key_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char value_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char qkv_split_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t query_name = iree_string_view_empty();
  iree_string_view_t key_name = iree_string_view_empty();
  iree_string_view_t value_name = iree_string_view_empty();
  iree_string_view_t qkv_split_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.query"), query_name_buffer,
      IREE_ARRAYSIZE(query_name_buffer), &query_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.key"), key_name_buffer,
      IREE_ARRAYSIZE(key_name_buffer), &key_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.value"), value_name_buffer,
      IREE_ARRAYSIZE(value_name_buffer), &value_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.qkv_split"),
      qkv_split_name_buffer, IREE_ARRAYSIZE(qkv_split_name_buffer),
      &qkv_split_name));

  id4_pipeline_program_tensor_t query = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t key = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t value = id4_pipeline_program_tensor_invalid();
  const id4_pipeline_program_shape_t attention_stream_shape =
      id4_pipeline_program_make_shape_rank2(total_token_count, hidden_size);
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, query_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      attention_stream_shape, &query));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, key_name, ID4_PIPELINE_PROGRAM_DTYPE_F32, attention_stream_shape,
      &key));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, value_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      attention_stream_shape, &value));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_qkv_split(
      builder, qkv_split_name, total_token_count, hidden_size, qkv, query, key,
      value));

  char after_qkv_split_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_qkv_split_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_attention_qkv_split"),
      after_qkv_split_name_buffer, IREE_ARRAYSIZE(after_qkv_split_name_buffer),
      &after_qkv_split_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_qkv_split_name));

  char norm_query_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char norm_key_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char
      norm_query_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char norm_key_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t norm_query_name = iree_string_view_empty();
  iree_string_view_t norm_key_name = iree_string_view_empty();
  iree_string_view_t norm_query_dispatch_name = iree_string_view_empty();
  iree_string_view_t norm_key_dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.query_norm"),
      norm_query_name_buffer, IREE_ARRAYSIZE(norm_query_name_buffer),
      &norm_query_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.key_norm"),
      norm_key_name_buffer, IREE_ARRAYSIZE(norm_key_name_buffer),
      &norm_key_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.norm_q"),
      norm_query_dispatch_name_buffer,
      IREE_ARRAYSIZE(norm_query_dispatch_name_buffer),
      &norm_query_dispatch_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.norm_k"),
      norm_key_dispatch_name_buffer,
      IREE_ARRAYSIZE(norm_key_dispatch_name_buffer), &norm_key_dispatch_name));

  id4_pipeline_program_tensor_t norm_q_weight =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t norm_k_weight =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter(
      builder, layer_ordinal, IREE_SV("attention.norm_q.weight"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(head_size), &norm_q_weight));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter(
      builder, layer_ordinal, IREE_SV("attention.norm_k.weight"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(head_size), &norm_k_weight));

  id4_pipeline_program_tensor_t norm_query =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t norm_key =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, norm_query_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      attention_stream_shape, &norm_query));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, norm_key_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      attention_stream_shape, &norm_key));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_head_rmsnorm(
      builder, norm_query_dispatch_name, total_token_count,
      attention_head_count, head_size, query, norm_q_weight, norm_query));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_head_rmsnorm(
      builder, norm_key_dispatch_name, total_token_count, attention_head_count,
      head_size, key, norm_k_weight, norm_key));

  char after_attention_norm_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_attention_norm_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_attention_qk_norm"),
      after_attention_norm_name_buffer,
      IREE_ARRAYSIZE(after_attention_norm_name_buffer),
      &after_attention_norm_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_attention_norm_name));

  char rotated_query_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char rotated_key_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char rotary_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t rotated_query_name = iree_string_view_empty();
  iree_string_view_t rotated_key_name = iree_string_view_empty();
  iree_string_view_t rotary_dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.query_rotary"),
      rotated_query_name_buffer, IREE_ARRAYSIZE(rotated_query_name_buffer),
      &rotated_query_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.key_rotary"),
      rotated_key_name_buffer, IREE_ARRAYSIZE(rotated_key_name_buffer),
      &rotated_key_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.rotary_apply"),
      rotary_dispatch_name_buffer, IREE_ARRAYSIZE(rotary_dispatch_name_buffer),
      &rotary_dispatch_name));

  id4_pipeline_program_tensor_t rotated_query =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t rotated_key =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, rotated_query_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      attention_stream_shape, &rotated_query));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, rotated_key_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      attention_stream_shape, &rotated_key));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_rotary_apply(
      builder, rotary_dispatch_name, total_token_count, attention_head_count,
      head_size, norm_query, norm_key, position_embedding, rotated_query,
      rotated_key));

  char after_attention_rotary_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_attention_rotary_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_attention_rotary"),
      after_attention_rotary_name_buffer,
      IREE_ARRAYSIZE(after_attention_rotary_name_buffer),
      &after_attention_rotary_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_attention_rotary_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(builder, qkv_name, qkv));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
      builder, rotated_query_name, rotated_query));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_tap(builder, rotated_key_name, rotated_key));

  char attention_context_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char attention_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t attention_context_name = iree_string_view_empty();
  iree_string_view_t attention_dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.context"),
      attention_context_name_buffer,
      IREE_ARRAYSIZE(attention_context_name_buffer), &attention_context_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention"),
      attention_dispatch_name_buffer,
      IREE_ARRAYSIZE(attention_dispatch_name_buffer),
      &attention_dispatch_name));

  id4_pipeline_program_tensor_t attention_context =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, attention_context_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(hidden_size, total_token_count),
      &attention_context));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_attention(
      builder, attention_dispatch_name, total_token_count, attention_head_count,
      head_size, rotated_query, rotated_key, value, attention_context));

  char after_attention_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_attention_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_attention"),
      after_attention_name_buffer, IREE_ARRAYSIZE(after_attention_name_buffer),
      &after_attention_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_attention_name));

  char attention_output_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char attention_output_dispatch_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t attention_output_name = iree_string_view_empty();
  iree_string_view_t attention_output_dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.output"),
      attention_output_name_buffer,
      IREE_ARRAYSIZE(attention_output_name_buffer), &attention_output_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.output_projection"),
      attention_output_dispatch_name_buffer,
      IREE_ARRAYSIZE(attention_output_dispatch_name_buffer),
      &attention_output_dispatch_name));

  id4_pipeline_program_tensor_t attention_output_weight =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter(
      builder, layer_ordinal, IREE_SV("attention.o.weight"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(hidden_size, hidden_size),
      &attention_output_weight));
  id4_pipeline_program_tensor_t attention_output =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, attention_output_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(hidden_size, total_token_count),
      &attention_output));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_linear_bf16_f32(
      builder, attention_output_dispatch_name, total_token_count, hidden_size,
      hidden_size, attention_context, attention_output_weight,
      attention_output));

  char after_attention_output_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_attention_output_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_attention_output"),
      after_attention_output_name_buffer,
      IREE_ARRAYSIZE(after_attention_output_name_buffer),
      &after_attention_output_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_attention_output_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
      builder, attention_output_name, attention_output));

  char post_attention_hidden_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char attention_residual_dispatch_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t post_attention_hidden_name = iree_string_view_empty();
  iree_string_view_t attention_residual_dispatch_name =
      iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("post_attention_hidden"),
      post_attention_hidden_name_buffer,
      IREE_ARRAYSIZE(post_attention_hidden_name_buffer),
      &post_attention_hidden_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.gated_residual"),
      attention_residual_dispatch_name_buffer,
      IREE_ARRAYSIZE(attention_residual_dispatch_name_buffer),
      &attention_residual_dispatch_name));

  id4_pipeline_program_tensor_t attention_norm2_weight =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter(
      builder, layer_ordinal, IREE_SV("attention_norm2.weight"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(hidden_size),
      &attention_norm2_weight));
  id4_pipeline_program_tensor_t post_attention_hidden =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, post_attention_hidden_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(hidden_size, total_token_count),
      &post_attention_hidden));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_dispatch_rmsnorm_gated_residual(
          builder, attention_residual_dispatch_name, total_token_count,
          hidden_size, attention_output, attention_norm2_weight, gate_msa,
          hidden_input, post_attention_hidden));

  char after_attention_residual_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_attention_residual_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_attention_residual"),
      after_attention_residual_name_buffer,
      IREE_ARRAYSIZE(after_attention_residual_name_buffer),
      &after_attention_residual_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_barrier(
      builder, after_attention_residual_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
      builder, post_attention_hidden_name, post_attention_hidden));

  char ffn_input_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char ffn_input_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t ffn_input_name = iree_string_view_empty();
  iree_string_view_t ffn_input_dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.input"), ffn_input_name_buffer,
      IREE_ARRAYSIZE(ffn_input_name_buffer), &ffn_input_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.norm1_modulated"),
      ffn_input_dispatch_name_buffer,
      IREE_ARRAYSIZE(ffn_input_dispatch_name_buffer),
      &ffn_input_dispatch_name));

  id4_pipeline_program_tensor_t ffn_norm1_weight =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter(
      builder, layer_ordinal, IREE_SV("ffn_norm1.weight"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(hidden_size), &ffn_norm1_weight));
  id4_pipeline_program_tensor_t ffn_input =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, ffn_input_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(hidden_size, total_token_count),
      &ffn_input));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_modulated_rmsnorm(
      builder, ffn_input_dispatch_name, total_token_count, hidden_size,
      post_attention_hidden, ffn_norm1_weight, scale_mlp, ffn_input));

  char after_ffn_input_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_ffn_input_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_ffn_input"),
      after_ffn_input_name_buffer, IREE_ARRAYSIZE(after_ffn_input_name_buffer),
      &after_ffn_input_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_ffn_input_name));

  const id4_pipeline_program_shape_t intermediate_stream_shape =
      id4_pipeline_program_make_shape_rank2(intermediate_size,
                                            total_token_count);
  id4_pipeline_program_tensor_t feed_forward_w1_weight =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t feed_forward_w3_weight =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t feed_forward_w2_weight =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter(
      builder, layer_ordinal, IREE_SV("feed_forward.w1.weight"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(intermediate_size, hidden_size),
      &feed_forward_w1_weight));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter(
      builder, layer_ordinal, IREE_SV("feed_forward.w3.weight"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(intermediate_size, hidden_size),
      &feed_forward_w3_weight));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter(
      builder, layer_ordinal, IREE_SV("feed_forward.w2.weight"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(hidden_size, intermediate_size),
      &feed_forward_w2_weight));

  char mlp_gate_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char mlp_up_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char mlp_gate_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char mlp_up_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t mlp_gate_name = iree_string_view_empty();
  iree_string_view_t mlp_up_name = iree_string_view_empty();
  iree_string_view_t mlp_gate_dispatch_name = iree_string_view_empty();
  iree_string_view_t mlp_up_dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.gate"), mlp_gate_name_buffer,
      IREE_ARRAYSIZE(mlp_gate_name_buffer), &mlp_gate_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.up"), mlp_up_name_buffer,
      IREE_ARRAYSIZE(mlp_up_name_buffer), &mlp_up_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.w1_projection"),
      mlp_gate_dispatch_name_buffer,
      IREE_ARRAYSIZE(mlp_gate_dispatch_name_buffer), &mlp_gate_dispatch_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.w3_projection"),
      mlp_up_dispatch_name_buffer, IREE_ARRAYSIZE(mlp_up_dispatch_name_buffer),
      &mlp_up_dispatch_name));

  id4_pipeline_program_tensor_t mlp_gate =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t mlp_up = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, mlp_gate_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      intermediate_stream_shape, &mlp_gate));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, mlp_up_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      intermediate_stream_shape, &mlp_up));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_linear_bf16_f32(
      builder, mlp_gate_dispatch_name, total_token_count, hidden_size,
      intermediate_size, ffn_input, feed_forward_w1_weight, mlp_gate));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_linear_bf16_f32(
      builder, mlp_up_dispatch_name, total_token_count, hidden_size,
      intermediate_size, ffn_input, feed_forward_w3_weight, mlp_up));

  char after_mlp_projection_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_mlp_projection_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_ffn_projections"),
      after_mlp_projection_name_buffer,
      IREE_ARRAYSIZE(after_mlp_projection_name_buffer),
      &after_mlp_projection_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_mlp_projection_name));

  char mlp_hidden_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char
      mlp_hidden_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t mlp_hidden_name = iree_string_view_empty();
  iree_string_view_t mlp_hidden_dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.hidden"), mlp_hidden_name_buffer,
      IREE_ARRAYSIZE(mlp_hidden_name_buffer), &mlp_hidden_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.silu_product"),
      mlp_hidden_dispatch_name_buffer,
      IREE_ARRAYSIZE(mlp_hidden_dispatch_name_buffer),
      &mlp_hidden_dispatch_name));

  id4_pipeline_program_tensor_t mlp_hidden =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, mlp_hidden_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      intermediate_stream_shape, &mlp_hidden));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_silu_product(
      builder, mlp_hidden_dispatch_name, total_token_count, intermediate_size,
      mlp_gate, mlp_up, mlp_hidden));

  char after_mlp_hidden_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_mlp_hidden_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_ffn_hidden"),
      after_mlp_hidden_name_buffer,
      IREE_ARRAYSIZE(after_mlp_hidden_name_buffer), &after_mlp_hidden_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_mlp_hidden_name));

  char mlp_output_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char
      mlp_output_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t mlp_output_name = iree_string_view_empty();
  iree_string_view_t mlp_output_dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.output"), mlp_output_name_buffer,
      IREE_ARRAYSIZE(mlp_output_name_buffer), &mlp_output_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.w2_projection"),
      mlp_output_dispatch_name_buffer,
      IREE_ARRAYSIZE(mlp_output_dispatch_name_buffer),
      &mlp_output_dispatch_name));

  id4_pipeline_program_tensor_t mlp_output =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, mlp_output_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(hidden_size, total_token_count),
      &mlp_output));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_linear_bf16_f32(
      builder, mlp_output_dispatch_name, total_token_count, intermediate_size,
      hidden_size, mlp_hidden, feed_forward_w2_weight, mlp_output));

  char after_mlp_output_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_mlp_output_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_ffn_output"),
      after_mlp_output_name_buffer,
      IREE_ARRAYSIZE(after_mlp_output_name_buffer), &after_mlp_output_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_mlp_output_name));

  char layer_output_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char mlp_residual_dispatch_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t layer_output_name = iree_string_view_empty();
  iree_string_view_t mlp_residual_dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("hidden"), layer_output_name_buffer,
      IREE_ARRAYSIZE(layer_output_name_buffer), &layer_output_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.gated_residual"),
      mlp_residual_dispatch_name_buffer,
      IREE_ARRAYSIZE(mlp_residual_dispatch_name_buffer),
      &mlp_residual_dispatch_name));

  id4_pipeline_program_tensor_t ffn_norm2_weight =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter(
      builder, layer_ordinal, IREE_SV("ffn_norm2.weight"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(hidden_size), &ffn_norm2_weight));
  id4_pipeline_program_tensor_t layer_output =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, layer_output_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(hidden_size, total_token_count),
      &layer_output));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_dispatch_rmsnorm_gated_residual(
          builder, mlp_residual_dispatch_name, total_token_count, hidden_size,
          mlp_output, ffn_norm2_weight, gate_mlp, post_attention_hidden,
          layer_output));

  char after_layer_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_layer_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_layer"),
      after_layer_name_buffer, IREE_ARRAYSIZE(after_layer_name_buffer),
      &after_layer_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_layer_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_tap(builder, layer_output_name, layer_output));
  *out_hidden = layer_output;
  return iree_ok_status();
}
