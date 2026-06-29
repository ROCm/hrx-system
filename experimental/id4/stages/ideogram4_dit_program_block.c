// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_dit_program_block.h"

enum {
  ID4_IDEOGRAM4_DIT_ATTENTION_WMMA_TOKEN_BLOCK = 16,
  ID4_IDEOGRAM4_DIT_BLOCKED_ATTENTION_QUERY_TOKEN_COUNT = 512,
  ID4_IDEOGRAM4_DIT_ONLINE_ATTENTION_QUERY_TOKEN_COUNT = 4096,
  ID4_IDEOGRAM4_DIT_COMPACT_RHS_TILE_INPUT_BLOCK = 16,
  ID4_IDEOGRAM4_DIT_COMPACT_RHS_TILE_OUTPUT_BLOCK = 128,
};

typedef enum id4_ideogram4_dit_program_linear_weight_layout_e {
  ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_INVALID = 0,
  ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_DENSE = 1,
  ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_COMPACT_RHS_TILE = 2,
} id4_ideogram4_dit_program_linear_weight_layout_t;

typedef struct id4_ideogram4_dit_program_linear_parameter_t {
  // Physical storage format used for the logical weight parameter.
  id4_ideogram4_dit_parameter_storage_t storage;
  // Physical execution layout used inside the resident weight tensor.
  id4_ideogram4_dit_program_linear_weight_layout_t layout;
  // Weight tensor loaded from the selected parameter source.
  id4_pipeline_program_tensor_t weight;
  // Row-scale tensor for scaled FP8 weights; invalid for BF16 weights.
  id4_pipeline_program_tensor_t scale;
} id4_ideogram4_dit_program_linear_parameter_t;

static iree_status_t id4_ideogram4_dit_program_require_bf16_parameter(
    iree_string_view_t key, id4_ideogram4_dit_parameter_source_rule_t source) {
  if (source.storage == ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "Ideogram4 DiT parameter `%.*s` storage %" PRIu32
                          " is not supported at this authoring site",
                          (int)key.size, key.data, (uint32_t)source.storage);
}

static iree_status_t id4_ideogram4_dit_program_layer_parameter_bf16(
    id4_pipeline_program_builder_t* builder,
    id4_ideogram4_dit_parameter_sources_t sources, uint32_t layer_ordinal,
    iree_string_view_t suffix, id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  char key_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_layer_parameter(
      layer_ordinal, suffix, key_buffer, IREE_ARRAYSIZE(key_buffer), &key));
  id4_ideogram4_dit_parameter_source_rule_t source;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter_source_resolve(
      sources, key, &source));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_require_bf16_parameter(key, source));
  return id4_ideogram4_dit_program_parameter(builder, source.source_scope, key,
                                             ID4_PIPELINE_PROGRAM_DTYPE_BF16,
                                             shape, out_tensor);
}

static iree_status_t id4_ideogram4_dit_program_layer_linear_parameter(
    id4_pipeline_program_builder_t* builder,
    id4_ideogram4_dit_parameter_sources_t sources, uint32_t layer_ordinal,
    iree_string_view_t suffix, uint32_t input_size, uint32_t output_size,
    id4_ideogram4_dit_program_linear_weight_layout_t bf16_resident_layout,
    id4_ideogram4_dit_weight_execution_format_t weight_execution_format,
    id4_ideogram4_dit_program_linear_parameter_t* out_parameter) {
  IREE_ASSERT_ARGUMENT(out_parameter);
  out_parameter->storage = ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_INVALID;
  out_parameter->layout =
      ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_INVALID;
  out_parameter->weight = id4_pipeline_program_tensor_invalid();
  out_parameter->scale = id4_pipeline_program_tensor_invalid();
  switch (bf16_resident_layout) {
    case ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_DENSE:
    case ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_COMPACT_RHS_TILE:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram4 DiT BF16 resident weight layout %" PRIu32 " is invalid",
          (uint32_t)bf16_resident_layout);
  }

  char weight_key_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t weight_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_layer_parameter(
      layer_ordinal, suffix, weight_key_buffer,
      IREE_ARRAYSIZE(weight_key_buffer), &weight_key));
  id4_ideogram4_dit_parameter_source_rule_t source;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter_source_resolve(
      sources, weight_key, &source));

  const id4_pipeline_program_shape_t weight_shape =
      id4_pipeline_program_make_shape_rank2(output_size, input_size);
  switch (source.storage) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16: {
      if (weight_execution_format !=
          ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram4 DiT BF16 source parameter `%.*s` cannot use weight "
            "execution format %" PRIu32,
            (int)weight_key.size, weight_key.data,
            (uint32_t)weight_execution_format);
      }
      if (bf16_resident_layout ==
          ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_COMPACT_RHS_TILE) {
        IREE_RETURN_IF_ERROR(
            id4_ideogram4_dit_program_parameter_bf16_linear_rhs_tile(
                builder, source.source_scope, weight_key, input_size,
                output_size, &out_parameter->weight));
      } else {
        IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter(
            builder, source.source_scope, weight_key,
            ID4_PIPELINE_PROGRAM_DTYPE_BF16, weight_shape,
            &out_parameter->weight));
      }
      out_parameter->storage = ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16;
      out_parameter->layout = bf16_resident_layout;
      break;
    }
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED: {
      switch (weight_execution_format) {
        case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT: {
          if (bf16_resident_layout ==
              ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_COMPACT_RHS_TILE) {
            IREE_RETURN_IF_ERROR(
                id4_ideogram4_dit_program_parameter_fp8_e4m3_scaled_to_bf16_linear_rhs_tile(
                    builder, source.source_scope, weight_key, input_size,
                    output_size, &out_parameter->weight));
          } else {
            IREE_RETURN_IF_ERROR(
                id4_ideogram4_dit_program_parameter_fp8_e4m3_scaled_to_bf16(
                    builder, source.source_scope, weight_key, input_size,
                    output_size, &out_parameter->weight));
          }
          out_parameter->storage = ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16;
          out_parameter->layout = bf16_resident_layout;
          break;
        }
        case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_DIRECT: {
          IREE_RETURN_IF_ERROR(
              id4_ideogram4_dit_program_parameter_fp8_e4m3_scaled(
                  builder, source.source_scope, weight_key, input_size,
                  output_size, &out_parameter->weight, &out_parameter->scale));
          out_parameter->storage =
              ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED;
          out_parameter->layout =
              ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_DENSE;
          break;
        }
        default:
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "Ideogram4 DiT weight execution format %" PRIu32 " is invalid",
              (uint32_t)weight_execution_format);
      }
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "Ideogram4 DiT linear parameter `%.*s` storage "
                              "%" PRIu32 " is not supported",
                              (int)weight_key.size, weight_key.data,
                              (uint32_t)source.storage);
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_format_child_name(
    iree_string_view_t parent, iree_string_view_t suffix, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string) {
  return id4_ideogram4_dit_program_format(
      buffer, buffer_capacity, out_string, "%.*s.%.*s", (int)parent.size,
      parent.data, (int)suffix.size, suffix.data);
}

iree_status_t id4_ideogram4_dit_program_tap_linear_input_bf16_as_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    id4_pipeline_program_tensor_t packed_input) {
  char canonical_tensor_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char barrier_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t canonical_tensor_name = iree_string_view_empty();
  iree_string_view_t dispatch_name = iree_string_view_empty();
  iree_string_view_t barrier_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      name, IREE_SV("canonical"), canonical_tensor_name_buffer,
      IREE_ARRAYSIZE(canonical_tensor_name_buffer), &canonical_tensor_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      name, IREE_SV("unpack"), dispatch_name_buffer,
      IREE_ARRAYSIZE(dispatch_name_buffer), &dispatch_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      name, IREE_SV("after_unpack"), barrier_name_buffer,
      IREE_ARRAYSIZE(barrier_name_buffer), &barrier_name));
  id4_pipeline_program_tensor_t canonical_tensor =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, canonical_tensor_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(input_size, token_count),
      &canonical_tensor));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_dispatch_linear_input_unpack_bf16_f32(
          builder, dispatch_name, token_count, token_capacity, input_size,
          packed_input, canonical_tensor));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, barrier_name));
  return id4_ideogram4_dit_program_tap(builder, name, canonical_tensor);
}

iree_status_t id4_ideogram4_dit_program_tap_token_major_f32_as_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size,
    id4_pipeline_program_tensor_t token_major_input) {
  char canonical_tensor_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char barrier_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t canonical_tensor_name = iree_string_view_empty();
  iree_string_view_t dispatch_name = iree_string_view_empty();
  iree_string_view_t barrier_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      name, IREE_SV("canonical"), canonical_tensor_name_buffer,
      IREE_ARRAYSIZE(canonical_tensor_name_buffer), &canonical_tensor_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      name, IREE_SV("unpack"), dispatch_name_buffer,
      IREE_ARRAYSIZE(dispatch_name_buffer), &dispatch_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      name, IREE_SV("after_unpack"), barrier_name_buffer,
      IREE_ARRAYSIZE(barrier_name_buffer), &barrier_name));
  id4_pipeline_program_tensor_t canonical_tensor =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, canonical_tensor_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(input_size, token_count),
      &canonical_tensor));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_dispatch_linear_input_unpack_f32_f32(
          builder, dispatch_name, token_count, input_size, token_major_input,
          canonical_tensor));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, barrier_name));
  return id4_ideogram4_dit_program_tap(builder, name, canonical_tensor);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_linear_parameter_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t input,
    id4_ideogram4_dit_program_linear_parameter_t parameter,
    id4_pipeline_program_tensor_t output) {
  switch (parameter.storage) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16:
      if (parameter.layout !=
          ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_DENSE) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "Ideogram4 DiT F32 linear dispatch does not support weight layout "
            "%" PRIu32,
            (uint32_t)parameter.layout);
      }
      return id4_ideogram4_dit_program_dispatch_linear_bf16_f32(
          builder, name, token_count, input_size, output_size, input,
          parameter.weight, output);
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED:
      return id4_ideogram4_dit_program_dispatch_linear_fp8_f32(
          builder, name, token_count, input_size, output_size, input,
          parameter.weight, parameter.scale, output);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "Ideogram4 DiT linear dispatch storage %" PRIu32
                              " is not supported",
                              (uint32_t)parameter.storage);
  }
}

static iree_status_t id4_ideogram4_dit_program_dispatch_linear_parameter_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_ideogram4_dit_program_linear_parameter_t parameter,
    id4_pipeline_program_tensor_t output) {
  switch (parameter.storage) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16:
      switch (parameter.layout) {
        case ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_DENSE:
          return id4_ideogram4_dit_program_dispatch_linear_packed_bf16_bf16(
              builder, name, token_count, token_capacity, input_size,
              output_size, input, parameter.weight, output);
        case ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_COMPACT_RHS_TILE:
          return id4_ideogram4_dit_program_dispatch_linear_packed_bf16_bf16_compact_rhs_tile(
              builder, name, token_count, token_capacity, input_size,
              output_size, input, parameter.weight, output);
        default:
          return iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "Ideogram4 DiT BF16 linear dispatch weight layout %" PRIu32
              " is not supported",
              (uint32_t)parameter.layout);
      }
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED:
      return id4_ideogram4_dit_program_dispatch_linear_packed_fp8_bf16(
          builder, name, token_count, token_capacity, input_size, output_size,
          input, parameter.weight, parameter.scale, output);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "Ideogram4 DiT linear dispatch storage %" PRIu32
                              " is not supported",
                              (uint32_t)parameter.storage);
  }
}

static iree_status_t
id4_ideogram4_dit_program_dispatch_linear_parameter_f32_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_ideogram4_dit_program_linear_parameter_t parameter,
    id4_pipeline_program_tensor_t output) {
  switch (parameter.storage) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16:
      if (parameter.layout !=
          ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_DENSE) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "Ideogram4 DiT BF16 output linear dispatch does not support "
            "weight layout %" PRIu32,
            (uint32_t)parameter.layout);
      }
      return id4_ideogram4_dit_program_dispatch_linear_bf16_bf16(
          builder, name, token_count, input_size, output_size, input,
          parameter.weight, output);
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED:
      return id4_ideogram4_dit_program_dispatch_linear_fp8_bf16(
          builder, name, token_count, token_capacity, input_size, output_size,
          input, parameter.weight, parameter.scale, output);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "Ideogram4 DiT linear dispatch storage %" PRIu32
                              " is not supported",
                              (uint32_t)parameter.storage);
  }
}

static uint32_t id4_ideogram4_dit_program_min_u32(uint32_t lhs, uint32_t rhs) {
  return lhs < rhs ? lhs : rhs;
}

static id4_ideogram4_dit_program_linear_weight_layout_t
id4_ideogram4_dit_program_select_bf16_resident_linear_weight_layout(
    id4_ideogram4_dit_activation_format_t activation_format,
    uint32_t token_capacity, uint32_t input_size, uint32_t output_size) {
  if (activation_format == ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL) {
    return ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_DENSE;
  }
  if ((token_capacity % ID4_IDEOGRAM4_DIT_BF16_TOKEN_CAPACITY_BLOCK) == 0 &&
      (input_size % ID4_IDEOGRAM4_DIT_COMPACT_RHS_TILE_INPUT_BLOCK) == 0 &&
      (output_size % ID4_IDEOGRAM4_DIT_COMPACT_RHS_TILE_OUTPUT_BLOCK) == 0) {
    return ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_COMPACT_RHS_TILE;
  }
  return ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_DENSE;
}

static iree_status_t id4_ideogram4_dit_program_format_blocked_attention_name(
    iree_string_view_t branch_name, uint32_t layer_ordinal,
    uint32_t query_block_offset, iree_string_view_t suffix, char* suffix_buffer,
    iree_host_size_t suffix_buffer_capacity, char* name_buffer,
    iree_host_size_t name_buffer_capacity, iree_string_view_t* out_name) {
  iree_string_view_t block_suffix = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      suffix_buffer, suffix_buffer_capacity, &block_suffix,
      "attention.blocked.q%" PRIu32 ".%.*s", query_block_offset,
      (int)suffix.size, suffix.data));
  return id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, block_suffix, name_buffer,
      name_buffer_capacity, out_name);
}

static iree_status_t id4_ideogram4_dit_program_format_online_attention_name(
    iree_string_view_t branch_name, uint32_t layer_ordinal,
    uint32_t query_block_offset, char* suffix_buffer,
    iree_host_size_t suffix_buffer_capacity, char* name_buffer,
    iree_host_size_t name_buffer_capacity, iree_string_view_t* out_name) {
  iree_string_view_t block_suffix = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      suffix_buffer, suffix_buffer_capacity, &block_suffix,
      "attention.online.q%" PRIu32, query_block_offset));
  return id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, block_suffix, name_buffer,
      name_buffer_capacity, out_name);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_blocked_wmma_attention(
    id4_pipeline_program_builder_t* builder, iree_string_view_t branch_name,
    uint32_t layer_ordinal, uint32_t total_token_count,
    uint32_t bf16_token_capacity, uint32_t attention_head_count,
    uint32_t head_size, id4_pipeline_program_tensor_t rotated_query,
    id4_pipeline_program_tensor_t rotated_key,
    id4_pipeline_program_tensor_t value,
    id4_pipeline_program_tensor_t attention_context) {
  for (uint32_t query_block_offset = 0;
       query_block_offset < bf16_token_capacity;
       query_block_offset +=
       ID4_IDEOGRAM4_DIT_BLOCKED_ATTENTION_QUERY_TOKEN_COUNT) {
    const uint32_t query_block_token_count = id4_ideogram4_dit_program_min_u32(
        ID4_IDEOGRAM4_DIT_BLOCKED_ATTENTION_QUERY_TOKEN_COUNT,
        bf16_token_capacity - query_block_offset);
    const id4_pipeline_program_shape_t attention_block_shape =
        id4_pipeline_program_make_shape_rank3(
            attention_head_count, query_block_token_count, bf16_token_capacity);

    char suffix_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    char scores_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    char probabilities_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    char qk_scores_dispatch_name_buffer
        [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    char softmax_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    char pv_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    char after_qk_scores_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    char after_softmax_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    char after_pv_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t scores_name = iree_string_view_empty();
    iree_string_view_t probabilities_name = iree_string_view_empty();
    iree_string_view_t qk_scores_dispatch_name = iree_string_view_empty();
    iree_string_view_t softmax_dispatch_name = iree_string_view_empty();
    iree_string_view_t pv_dispatch_name = iree_string_view_empty();
    iree_string_view_t after_qk_scores_name = iree_string_view_empty();
    iree_string_view_t after_softmax_name = iree_string_view_empty();
    iree_string_view_t after_pv_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_format_blocked_attention_name(
            branch_name, layer_ordinal, query_block_offset, IREE_SV("scores"),
            suffix_buffer, IREE_ARRAYSIZE(suffix_buffer), scores_name_buffer,
            IREE_ARRAYSIZE(scores_name_buffer), &scores_name));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_format_blocked_attention_name(
            branch_name, layer_ordinal, query_block_offset,
            IREE_SV("probabilities"), suffix_buffer,
            IREE_ARRAYSIZE(suffix_buffer), probabilities_name_buffer,
            IREE_ARRAYSIZE(probabilities_name_buffer), &probabilities_name));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_format_blocked_attention_name(
            branch_name, layer_ordinal, query_block_offset,
            IREE_SV("qk_scores"), suffix_buffer, IREE_ARRAYSIZE(suffix_buffer),
            qk_scores_dispatch_name_buffer,
            IREE_ARRAYSIZE(qk_scores_dispatch_name_buffer),
            &qk_scores_dispatch_name));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_format_blocked_attention_name(
            branch_name, layer_ordinal, query_block_offset, IREE_SV("softmax"),
            suffix_buffer, IREE_ARRAYSIZE(suffix_buffer),
            softmax_dispatch_name_buffer,
            IREE_ARRAYSIZE(softmax_dispatch_name_buffer),
            &softmax_dispatch_name));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_format_blocked_attention_name(
            branch_name, layer_ordinal, query_block_offset, IREE_SV("pv"),
            suffix_buffer, IREE_ARRAYSIZE(suffix_buffer),
            pv_dispatch_name_buffer, IREE_ARRAYSIZE(pv_dispatch_name_buffer),
            &pv_dispatch_name));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_format_blocked_attention_name(
            branch_name, layer_ordinal, query_block_offset,
            IREE_SV("after_qk_scores"), suffix_buffer,
            IREE_ARRAYSIZE(suffix_buffer), after_qk_scores_name_buffer,
            IREE_ARRAYSIZE(after_qk_scores_name_buffer),
            &after_qk_scores_name));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_format_blocked_attention_name(
            branch_name, layer_ordinal, query_block_offset,
            IREE_SV("after_softmax"), suffix_buffer,
            IREE_ARRAYSIZE(suffix_buffer), after_softmax_name_buffer,
            IREE_ARRAYSIZE(after_softmax_name_buffer), &after_softmax_name));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_format_blocked_attention_name(
            branch_name, layer_ordinal, query_block_offset, IREE_SV("after_pv"),
            suffix_buffer, IREE_ARRAYSIZE(suffix_buffer), after_pv_name_buffer,
            IREE_ARRAYSIZE(after_pv_name_buffer), &after_pv_name));

    id4_pipeline_program_tensor_t attention_scores =
        id4_pipeline_program_tensor_invalid();
    id4_pipeline_program_tensor_t attention_probabilities =
        id4_pipeline_program_tensor_invalid();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, scores_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
        attention_block_shape, &attention_scores));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, probabilities_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        attention_block_shape, &attention_probabilities));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_attention_blocked_qk_scores_bf16_f32_wmma(
            builder, qk_scores_dispatch_name, total_token_count,
            bf16_token_capacity, query_block_offset, query_block_token_count,
            attention_head_count, head_size, rotated_query, rotated_key,
            attention_scores));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_barrier(builder, after_qk_scores_name));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_attention_blocked_softmax_f32_bf16(
            builder, softmax_dispatch_name, total_token_count,
            bf16_token_capacity, query_block_offset, query_block_token_count,
            attention_head_count, head_size, attention_scores,
            attention_probabilities));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_barrier(builder, after_softmax_name));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_attention_blocked_pv_bf16_bf16_wmma(
            builder, pv_dispatch_name, total_token_count, bf16_token_capacity,
            query_block_offset, query_block_token_count, attention_head_count,
            head_size, attention_probabilities, value, attention_context));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_barrier(builder, after_pv_name));
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_dispatch_online_wmma_attention(
    id4_pipeline_program_builder_t* builder, iree_string_view_t branch_name,
    uint32_t layer_ordinal, uint32_t total_token_count,
    uint32_t bf16_token_capacity, uint32_t attention_head_count,
    uint32_t head_size, id4_pipeline_program_tensor_t rotated_query,
    id4_pipeline_program_tensor_t rotated_key,
    id4_pipeline_program_tensor_t value,
    id4_pipeline_program_tensor_t attention_context) {
  for (uint32_t query_block_offset = 0;
       query_block_offset < bf16_token_capacity;
       query_block_offset +=
       ID4_IDEOGRAM4_DIT_ONLINE_ATTENTION_QUERY_TOKEN_COUNT) {
    const uint32_t query_block_token_count = id4_ideogram4_dit_program_min_u32(
        ID4_IDEOGRAM4_DIT_ONLINE_ATTENTION_QUERY_TOKEN_COUNT,
        bf16_token_capacity - query_block_offset);
    char suffix_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    char dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t dispatch_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_online_attention_name(
        branch_name, layer_ordinal, query_block_offset, suffix_buffer,
        IREE_ARRAYSIZE(suffix_buffer), dispatch_name_buffer,
        IREE_ARRAYSIZE(dispatch_name_buffer), &dispatch_name));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_attention_online_bf16_bf16_wmma(
            builder, dispatch_name, total_token_count, bf16_token_capacity,
            query_block_offset, query_block_token_count, attention_head_count,
            head_size, rotated_query, rotated_key, value, attention_context));
  }
  return iree_ok_status();
}

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
  const id4_ideogram4_dit_parameter_sources_t parameter_sources =
      options->parameter_sources;
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
  const id4_ideogram4_dit_activation_format_t activation_format =
      options->activation_format;
  const id4_ideogram4_dit_weight_execution_format_t weight_execution_format =
      options->weight_execution_format;
  const id4_ideogram4_dit_attention_implementation_t attention_implementation =
      options->attention_implementation;
  const id4_ideogram4_dit_feed_forward_implementation_t
      feed_forward_implementation = options->feed_forward_implementation;
  const iree_string_view_list_t diagnostic_tap_names =
      options->diagnostic_tap_names;
  const uint32_t head_size = hidden_size / attention_head_count;
  const bool f32_canonical_activations =
      activation_format == ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL;
  const bool materialized_wmma_attention =
      attention_implementation ==
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA;
  const bool blocked_wmma_attention =
      attention_implementation ==
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_BLOCKED_WMMA;
  const bool online_wmma_attention =
      attention_implementation ==
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA;
  const bool wmma_attention = materialized_wmma_attention ||
                              blocked_wmma_attention || online_wmma_attention;
  const bool bf16_linear_input_attention = !f32_canonical_activations;
  const bool pytorch_parity_feed_forward =
      feed_forward_implementation ==
      ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY;
  const bool bf16_resident_weight_execution =
      weight_execution_format ==
      ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT;
  switch (attention_implementation) {
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING:
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA:
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_BLOCKED_WMMA:
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram4 DiT attention implementation %" PRIu32
                              " is invalid",
                              (uint32_t)attention_implementation);
  }
  switch (feed_forward_implementation) {
    case ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_FUSED_PRODUCT:
    case ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram4 DiT feed-forward implementation %" PRIu32 " is invalid",
          (uint32_t)feed_forward_implementation);
  }
  if (wmma_attention && f32_canonical_activations) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram4 DiT WMMA attention requires BF16 linear-input "
        "activations");
  }
  if (pytorch_parity_feed_forward && f32_canonical_activations) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram4 DiT PyTorch feed-forward parity requires BF16 linear-input "
        "activations");
  }
  uint32_t bf16_token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_calculate_bf16_token_capacity(
      total_token_count, &bf16_token_capacity));
  const id4_pipeline_program_shape_t canonical_hidden_shape =
      id4_pipeline_program_make_shape_rank2(hidden_size, total_token_count);
  const id4_pipeline_program_shape_t token_major_hidden_shape =
      id4_pipeline_program_make_shape_rank2(total_token_count, hidden_size);
  const id4_pipeline_program_shape_t resident_hidden_shape =
      f32_canonical_activations ? canonical_hidden_shape
                                : token_major_hidden_shape;
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
  char adaln_modulation_bf16_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char after_adaln_modulation_bf16_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char adaln_modulation_cast_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t adaln_modulation_bf16_name = iree_string_view_empty();
  iree_string_view_t after_adaln_modulation_bf16_name =
      iree_string_view_empty();
  iree_string_view_t adaln_modulation_cast_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      adaln_modulation_name, IREE_SV("bf16"), adaln_modulation_bf16_name_buffer,
      IREE_ARRAYSIZE(adaln_modulation_bf16_name_buffer),
      &adaln_modulation_bf16_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      adaln_modulation_name, IREE_SV("after_bf16"),
      after_adaln_modulation_bf16_name_buffer,
      IREE_ARRAYSIZE(after_adaln_modulation_bf16_name_buffer),
      &after_adaln_modulation_bf16_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      adaln_modulation_name, IREE_SV("canonical_cast"),
      adaln_modulation_cast_name_buffer,
      IREE_ARRAYSIZE(adaln_modulation_cast_name_buffer),
      &adaln_modulation_cast_name));
  char after_adaln_modulation_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_adaln_modulation_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_adaln_modulation"),
      after_adaln_modulation_name_buffer,
      IREE_ARRAYSIZE(after_adaln_modulation_name_buffer),
      &after_adaln_modulation_name));
  id4_pipeline_program_tensor_t raw_modulation =
      id4_pipeline_program_tensor_invalid();
  if (f32_canonical_activations) {
    const id4_ideogram4_dit_program_dense_options_t adaln_modulation_options = {
        .operation_name = adaln_modulation_name,
        .output_name = adaln_modulation_name,
        .parameter_sources = parameter_sources,
        .parameter_prefix = adaln_modulation_prefix,
        .weight_execution_format = weight_execution_format,
        .input = adaln_input,
        .input_size = adaln_size,
        .output_size = modulation_output_size,
        .activation_kind = ID4_IDEOGRAM4_DIT_ACTIVATION_IDENTITY,
    };
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dense_f32(
        builder, &adaln_modulation_options, &raw_modulation));
  } else {
    id4_pipeline_program_tensor_t raw_modulation_bf16 =
        id4_pipeline_program_tensor_invalid();
    const id4_ideogram4_dit_program_dense_options_t adaln_modulation_options = {
        .operation_name = adaln_modulation_name,
        .output_name = adaln_modulation_bf16_name,
        .parameter_sources = parameter_sources,
        .parameter_prefix = adaln_modulation_prefix,
        .weight_execution_format = weight_execution_format,
        .input = adaln_input,
        .input_size = adaln_size,
        .output_size = modulation_output_size,
        .activation_kind = ID4_IDEOGRAM4_DIT_ACTIVATION_IDENTITY,
    };
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dense_bf16(
        builder, &adaln_modulation_options, &raw_modulation_bf16));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_barrier(
        builder, after_adaln_modulation_bf16_name));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, adaln_modulation_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
        id4_pipeline_program_make_shape_rank1(modulation_output_size),
        &raw_modulation));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_cast_bf16_f32(
        builder, adaln_modulation_cast_name,
        id4_pipeline_program_make_shape_rank1(modulation_output_size),
        raw_modulation_bf16, raw_modulation));
  }
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
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter_bf16(
      builder, parameter_sources, layer_ordinal,
      IREE_SV("attention_norm1.weight"),
      id4_pipeline_program_make_shape_rank1(hidden_size),
      &attention_norm1_weight));
  id4_pipeline_program_tensor_t attention_input =
      id4_pipeline_program_tensor_invalid();
  const bool attention_input_as_f32 =
      activation_format == ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL;
  const bool tap_attention_input = id4_ideogram4_dit_program_has_diagnostic_tap(
      diagnostic_tap_names, attention_input_name);
  if (attention_input_as_f32) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, attention_input_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
        canonical_hidden_shape, &attention_input));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_modulated_rmsnorm(
        builder, attention_input_dispatch_name, total_token_count, hidden_size,
        hidden_input, attention_norm1_weight, scale_msa, attention_input));
  } else {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, attention_input_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        id4_pipeline_program_make_shape_rank2(bf16_token_capacity, hidden_size),
        &attention_input));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_modulated_rmsnorm_token_major_linear_input_bf16(
            builder, attention_input_dispatch_name, total_token_count,
            bf16_token_capacity, hidden_size, hidden_input,
            attention_norm1_weight, scale_msa, attention_input));
  }

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
  if (attention_input_as_f32) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
        builder, attention_input_name, attention_input));
  } else if (tap_attention_input) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap_linear_input_bf16_as_f32(
        builder, attention_input_name, total_token_count, bf16_token_capacity,
        hidden_size, attention_input));
  }

  uint64_t qkv_size64 = 0;
  if (!id4_ideogram4_dit_program_checked_mul_u64(hidden_size, 3, &qkv_size64) ||
      qkv_size64 > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 DiT QKV size overflow");
  }
  const uint32_t qkv_size = (uint32_t)qkv_size64;
  char qkv_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char qkv_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char qkv_projection_output_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t qkv_name = iree_string_view_empty();
  iree_string_view_t qkv_dispatch_name = iree_string_view_empty();
  iree_string_view_t qkv_projection_output_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.qkv"), qkv_name_buffer,
      IREE_ARRAYSIZE(qkv_name_buffer), &qkv_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.qkv_projection"),
      qkv_dispatch_name_buffer, IREE_ARRAYSIZE(qkv_dispatch_name_buffer),
      &qkv_dispatch_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.qkv_projection.output"),
      qkv_projection_output_name_buffer,
      IREE_ARRAYSIZE(qkv_projection_output_name_buffer),
      &qkv_projection_output_name));

  id4_ideogram4_dit_program_linear_parameter_t qkv_parameter;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_linear_parameter(
      builder, parameter_sources, layer_ordinal,
      IREE_SV("attention.qkv.weight"), hidden_size, qkv_size,
      id4_ideogram4_dit_program_select_bf16_resident_linear_weight_layout(
          activation_format, bf16_token_capacity, hidden_size, qkv_size),
      weight_execution_format, &qkv_parameter));
  id4_pipeline_program_tensor_t qkv = id4_pipeline_program_tensor_invalid();
  const id4_pipeline_program_dtype_t qkv_dtype =
      f32_canonical_activations ? ID4_PIPELINE_PROGRAM_DTYPE_F32
                                : ID4_PIPELINE_PROGRAM_DTYPE_BF16;
  const id4_pipeline_program_shape_t qkv_shape =
      f32_canonical_activations
          ? id4_pipeline_program_make_shape_rank2(qkv_size, total_token_count)
          : id4_pipeline_program_make_shape_rank2(bf16_token_capacity,
                                                  qkv_size);
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, qkv_name, qkv_dtype, qkv_shape, &qkv));
  if (f32_canonical_activations) {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_linear_parameter_f32(
            builder, qkv_dispatch_name, total_token_count, hidden_size,
            qkv_size, attention_input, qkv_parameter, qkv));
  } else {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_linear_parameter_bf16(
            builder, qkv_dispatch_name, total_token_count, bf16_token_capacity,
            hidden_size, qkv_size, attention_input, qkv_parameter, qkv));
  }

  char after_qkv_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_qkv_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_attention_qkv"),
      after_qkv_name_buffer, IREE_ARRAYSIZE(after_qkv_name_buffer),
      &after_qkv_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_qkv_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_tap(builder, qkv_projection_output_name, qkv));

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
  const bool capture_attention_context =
      id4_ideogram4_dit_program_has_diagnostic_tap(diagnostic_tap_names,
                                                   attention_context_name);

  char value_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char rotated_query_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char rotated_key_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char qkv_norm_rotary_dispatch_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t value_name = iree_string_view_empty();
  iree_string_view_t rotated_query_name = iree_string_view_empty();
  iree_string_view_t rotated_key_name = iree_string_view_empty();
  iree_string_view_t qkv_norm_rotary_dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.value"), value_name_buffer,
      IREE_ARRAYSIZE(value_name_buffer), &value_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.query_rotary"),
      rotated_query_name_buffer, IREE_ARRAYSIZE(rotated_query_name_buffer),
      &rotated_query_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.key_rotary"),
      rotated_key_name_buffer, IREE_ARRAYSIZE(rotated_key_name_buffer),
      &rotated_key_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("attention.qkv_norm_rotary"),
      qkv_norm_rotary_dispatch_name_buffer,
      IREE_ARRAYSIZE(qkv_norm_rotary_dispatch_name_buffer),
      &qkv_norm_rotary_dispatch_name));

  id4_pipeline_program_tensor_t norm_q_weight =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t norm_k_weight =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter_bf16(
      builder, parameter_sources, layer_ordinal,
      IREE_SV("attention.norm_q.weight"),
      id4_pipeline_program_make_shape_rank1(head_size), &norm_q_weight));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter_bf16(
      builder, parameter_sources, layer_ordinal,
      IREE_SV("attention.norm_k.weight"),
      id4_pipeline_program_make_shape_rank1(head_size), &norm_k_weight));

  id4_pipeline_program_tensor_t value = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t rotated_query =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t rotated_key =
      id4_pipeline_program_tensor_invalid();
  const id4_pipeline_program_shape_t attention_stream_shape =
      wmma_attention ? id4_pipeline_program_make_shape_rank2(
                           bf16_token_capacity, hidden_size)
                     : id4_pipeline_program_make_shape_rank2(total_token_count,
                                                             hidden_size);
  const id4_pipeline_program_dtype_t rotated_query_dtype =
      f32_canonical_activations ? ID4_PIPELINE_PROGRAM_DTYPE_F32
                                : ID4_PIPELINE_PROGRAM_DTYPE_BF16;
  const id4_pipeline_program_dtype_t attention_key_value_dtype =
      f32_canonical_activations ? ID4_PIPELINE_PROGRAM_DTYPE_F16
                                : ID4_PIPELINE_PROGRAM_DTYPE_BF16;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, value_name, attention_key_value_dtype, attention_stream_shape,
      &value));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, rotated_query_name, rotated_query_dtype, attention_stream_shape,
      &rotated_query));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, rotated_key_name, attention_key_value_dtype,
      attention_stream_shape, &rotated_key));
  if (wmma_attention) {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_qkv_norm_rotary_packed_value(
            builder, qkv_norm_rotary_dispatch_name, total_token_count,
            bf16_token_capacity, attention_head_count, head_size, qkv,
            norm_q_weight, norm_k_weight, position_embedding, rotated_query,
            rotated_key, value));
  } else {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_qkv_norm_rotary(
        builder, qkv_norm_rotary_dispatch_name, total_token_count,
        attention_head_count, head_size, activation_format, qkv, norm_q_weight,
        norm_k_weight, position_embedding, rotated_query, rotated_key, value));
  }

  char after_attention_qkv_norm_rotary_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_attention_qkv_norm_rotary_name =
      iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_attention_qkv_norm_rotary"),
      after_attention_qkv_norm_rotary_name_buffer,
      IREE_ARRAYSIZE(after_attention_qkv_norm_rotary_name_buffer),
      &after_attention_qkv_norm_rotary_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_barrier(
      builder, after_attention_qkv_norm_rotary_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(builder, qkv_name, qkv));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
      builder, rotated_query_name, rotated_query));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_tap(builder, rotated_key_name, rotated_key));

  id4_pipeline_program_tensor_t attention_context =
      id4_pipeline_program_tensor_invalid();
  if (bf16_linear_input_attention) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, attention_context_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        id4_pipeline_program_make_shape_rank2(bf16_token_capacity, hidden_size),
        &attention_context));
  }
  if (materialized_wmma_attention) {
    char attention_scores_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    char attention_probabilities_name_buffer
        [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    char qk_scores_dispatch_name_buffer
        [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    char softmax_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    char pv_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t attention_scores_name = iree_string_view_empty();
    iree_string_view_t attention_probabilities_name = iree_string_view_empty();
    iree_string_view_t qk_scores_dispatch_name = iree_string_view_empty();
    iree_string_view_t softmax_dispatch_name = iree_string_view_empty();
    iree_string_view_t pv_dispatch_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
        branch_name, layer_ordinal, IREE_SV("attention.scores"),
        attention_scores_name_buffer,
        IREE_ARRAYSIZE(attention_scores_name_buffer), &attention_scores_name));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
        branch_name, layer_ordinal, IREE_SV("attention.probabilities"),
        attention_probabilities_name_buffer,
        IREE_ARRAYSIZE(attention_probabilities_name_buffer),
        &attention_probabilities_name));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
        branch_name, layer_ordinal, IREE_SV("attention.qk_scores"),
        qk_scores_dispatch_name_buffer,
        IREE_ARRAYSIZE(qk_scores_dispatch_name_buffer),
        &qk_scores_dispatch_name));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
        branch_name, layer_ordinal, IREE_SV("attention.softmax"),
        softmax_dispatch_name_buffer,
        IREE_ARRAYSIZE(softmax_dispatch_name_buffer), &softmax_dispatch_name));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
        branch_name, layer_ordinal, IREE_SV("attention.pv"),
        pv_dispatch_name_buffer, IREE_ARRAYSIZE(pv_dispatch_name_buffer),
        &pv_dispatch_name));

    id4_pipeline_program_tensor_t attention_scores =
        id4_pipeline_program_tensor_invalid();
    id4_pipeline_program_tensor_t attention_probabilities =
        id4_pipeline_program_tensor_invalid();
    const id4_pipeline_program_shape_t attention_matrix_shape =
        id4_pipeline_program_make_shape_rank3(
            attention_head_count, bf16_token_capacity, bf16_token_capacity);
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, attention_scores_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
        attention_matrix_shape, &attention_scores));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, attention_probabilities_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        attention_matrix_shape, &attention_probabilities));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_attention_qk_scores_all_heads_bf16_f32_wmma(
            builder, qk_scores_dispatch_name, total_token_count,
            bf16_token_capacity, attention_head_count, head_size, rotated_query,
            rotated_key, attention_scores));

    char after_attention_qk_scores_name_buffer
        [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t after_attention_qk_scores_name =
        iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
        branch_name, layer_ordinal, IREE_SV("after_attention_qk_scores"),
        after_attention_qk_scores_name_buffer,
        IREE_ARRAYSIZE(after_attention_qk_scores_name_buffer),
        &after_attention_qk_scores_name));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_barrier(
        builder, after_attention_qk_scores_name));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_attention_softmax_all_heads_f32_bf16(
            builder, softmax_dispatch_name, total_token_count,
            bf16_token_capacity, attention_head_count, head_size,
            attention_scores, attention_probabilities));

    char after_attention_softmax_name_buffer
        [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t after_attention_softmax_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
        branch_name, layer_ordinal, IREE_SV("after_attention_softmax"),
        after_attention_softmax_name_buffer,
        IREE_ARRAYSIZE(after_attention_softmax_name_buffer),
        &after_attention_softmax_name));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_barrier(
        builder, after_attention_softmax_name));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_attention_pv_all_heads_bf16_bf16_wmma(
            builder, pv_dispatch_name, total_token_count, bf16_token_capacity,
            attention_head_count, head_size, attention_probabilities, value,
            attention_context));
  } else if (blocked_wmma_attention) {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_blocked_wmma_attention(
            builder, branch_name, layer_ordinal, total_token_count,
            bf16_token_capacity, attention_head_count, head_size, rotated_query,
            rotated_key, value, attention_context));
  } else if (online_wmma_attention) {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_online_wmma_attention(
            builder, branch_name, layer_ordinal, total_token_count,
            bf16_token_capacity, attention_head_count, head_size, rotated_query,
            rotated_key, value, attention_context));
  } else if (f32_canonical_activations) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, attention_context_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
        id4_pipeline_program_make_shape_rank2(hidden_size, total_token_count),
        &attention_context));
  }
  if (f32_canonical_activations) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_attention(
        builder, attention_dispatch_name, total_token_count,
        attention_head_count, head_size, rotated_query, rotated_key, value,
        attention_context));
  } else if (!wmma_attention) {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_attention_bf16_linear_input(
            builder, attention_dispatch_name, total_token_count,
            attention_head_count, head_size, rotated_query, rotated_key, value,
            attention_context));
  }

  char after_attention_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_attention_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_attention"),
      after_attention_name_buffer, IREE_ARRAYSIZE(after_attention_name_buffer),
      &after_attention_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_attention_name));
  if (capture_attention_context) {
    if (bf16_linear_input_attention) {
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_tap_linear_input_bf16_as_f32(
              builder, attention_context_name, total_token_count,
              bf16_token_capacity, hidden_size, attention_context));
    } else {
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
          builder, attention_context_name, attention_context));
    }
  }

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

  id4_ideogram4_dit_program_linear_parameter_t attention_output_parameter;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_linear_parameter(
      builder, parameter_sources, layer_ordinal, IREE_SV("attention.o.weight"),
      hidden_size, hidden_size,
      id4_ideogram4_dit_program_select_bf16_resident_linear_weight_layout(
          activation_format, bf16_token_capacity, hidden_size, hidden_size),
      weight_execution_format, &attention_output_parameter));
  id4_pipeline_program_tensor_t attention_output =
      id4_pipeline_program_tensor_invalid();
  const bool tap_attention_output =
      id4_ideogram4_dit_program_has_diagnostic_tap(diagnostic_tap_names,
                                                   attention_output_name);
  const bool attention_output_as_f32 = f32_canonical_activations;
  const id4_pipeline_program_dtype_t attention_output_dtype =
      attention_output_as_f32 ? ID4_PIPELINE_PROGRAM_DTYPE_F32
                              : ID4_PIPELINE_PROGRAM_DTYPE_BF16;
  const id4_pipeline_program_shape_t attention_output_shape =
      attention_output_as_f32 ? id4_pipeline_program_make_shape_rank2(
                                    hidden_size, total_token_count)
                              : id4_pipeline_program_make_shape_rank2(
                                    bf16_token_capacity, hidden_size);
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, attention_output_name, attention_output_dtype,
      attention_output_shape, &attention_output));
  if (attention_output_as_f32) {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_linear_parameter_f32(
            builder, attention_output_dispatch_name, total_token_count,
            hidden_size, hidden_size, attention_context,
            attention_output_parameter, attention_output));
  } else {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_linear_parameter_bf16(
            builder, attention_output_dispatch_name, total_token_count,
            bf16_token_capacity, hidden_size, hidden_size, attention_context,
            attention_output_parameter, attention_output));
  }

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
  if (attention_output_as_f32) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
        builder, attention_output_name, attention_output));
  } else if (tap_attention_output) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap_linear_input_bf16_as_f32(
        builder, attention_output_name, total_token_count, bf16_token_capacity,
        hidden_size, attention_output));
  }

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
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter_bf16(
      builder, parameter_sources, layer_ordinal,
      IREE_SV("attention_norm2.weight"),
      id4_pipeline_program_make_shape_rank1(hidden_size),
      &attention_norm2_weight));
  id4_pipeline_program_tensor_t post_attention_hidden =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, post_attention_hidden_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      resident_hidden_shape, &post_attention_hidden));
  if (attention_output_as_f32) {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_rmsnorm_gated_residual(
            builder, attention_residual_dispatch_name, total_token_count,
            hidden_size, attention_output, attention_norm2_weight, gate_msa,
            hidden_input, post_attention_hidden));
  } else {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_rmsnorm_gated_residual_input_bf16_token_major(
            builder, attention_residual_dispatch_name, total_token_count,
            hidden_size, attention_output, attention_norm2_weight, gate_msa,
            hidden_input, post_attention_hidden));
  }

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
  if (f32_canonical_activations) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
        builder, post_attention_hidden_name, post_attention_hidden));
  } else if (id4_ideogram4_dit_program_has_diagnostic_tap(
                 diagnostic_tap_names, post_attention_hidden_name)) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap_token_major_f32_as_f32(
        builder, post_attention_hidden_name, total_token_count, hidden_size,
        post_attention_hidden));
  }

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
  char mlp_hidden_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char
      mlp_hidden_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t mlp_hidden_name = iree_string_view_empty();
  iree_string_view_t mlp_hidden_dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.hidden"), mlp_hidden_name_buffer,
      IREE_ARRAYSIZE(mlp_hidden_name_buffer), &mlp_hidden_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.w1_w3_silu_product"),
      mlp_hidden_dispatch_name_buffer,
      IREE_ARRAYSIZE(mlp_hidden_dispatch_name_buffer),
      &mlp_hidden_dispatch_name));
  char mlp_gate_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char mlp_gate_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char mlp_up_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char mlp_up_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t mlp_gate_name = iree_string_view_empty();
  iree_string_view_t mlp_gate_dispatch_name = iree_string_view_empty();
  iree_string_view_t mlp_up_name = iree_string_view_empty();
  iree_string_view_t mlp_up_dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.w1_projection.output"),
      mlp_gate_name_buffer, IREE_ARRAYSIZE(mlp_gate_name_buffer),
      &mlp_gate_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.w1_projection"),
      mlp_gate_dispatch_name_buffer,
      IREE_ARRAYSIZE(mlp_gate_dispatch_name_buffer), &mlp_gate_dispatch_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.w3_projection.output"),
      mlp_up_name_buffer, IREE_ARRAYSIZE(mlp_up_name_buffer), &mlp_up_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("ffn.w3_projection"),
      mlp_up_dispatch_name_buffer, IREE_ARRAYSIZE(mlp_up_dispatch_name_buffer),
      &mlp_up_dispatch_name));
  const bool capture_mlp_hidden =
      activation_format == ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL ||
      id4_ideogram4_dit_program_has_diagnostic_tap(diagnostic_tap_names,
                                                   mlp_hidden_name);
  const bool capture_mlp_hidden_as_f32 =
      activation_format == ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL;
  const bool tap_ffn_input = id4_ideogram4_dit_program_has_diagnostic_tap(
      diagnostic_tap_names, ffn_input_name);
  const bool capture_ffn_input_as_f32 =
      activation_format == ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL;
  const bool tap_mlp_gate = id4_ideogram4_dit_program_has_diagnostic_tap(
      diagnostic_tap_names, mlp_gate_name);
  const bool tap_mlp_up = id4_ideogram4_dit_program_has_diagnostic_tap(
      diagnostic_tap_names, mlp_up_name);
  if (!pytorch_parity_feed_forward && (tap_mlp_gate || tap_mlp_up)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram4 DiT feed-forward projection taps require PyTorch parity "
        "feed-forward structure");
  }
  const bool materialize_feed_forward_projections = pytorch_parity_feed_forward;

  id4_pipeline_program_tensor_t ffn_norm1_weight =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter_bf16(
      builder, parameter_sources, layer_ordinal, IREE_SV("ffn_norm1.weight"),
      id4_pipeline_program_make_shape_rank1(hidden_size), &ffn_norm1_weight));
  id4_pipeline_program_tensor_t ffn_input =
      id4_pipeline_program_tensor_invalid();
  if (capture_ffn_input_as_f32) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, ffn_input_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
        canonical_hidden_shape, &ffn_input));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_modulated_rmsnorm(
        builder, ffn_input_dispatch_name, total_token_count, hidden_size,
        post_attention_hidden, ffn_norm1_weight, scale_mlp, ffn_input));
  } else {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, ffn_input_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        id4_pipeline_program_make_shape_rank2(bf16_token_capacity, hidden_size),
        &ffn_input));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_modulated_rmsnorm_token_major_linear_input_bf16(
            builder, ffn_input_dispatch_name, total_token_count,
            bf16_token_capacity, hidden_size, post_attention_hidden,
            ffn_norm1_weight, scale_mlp, ffn_input));
  }

  char after_ffn_input_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_ffn_input_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_ffn_input"),
      after_ffn_input_name_buffer, IREE_ARRAYSIZE(after_ffn_input_name_buffer),
      &after_ffn_input_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_ffn_input_name));
  if (capture_ffn_input_as_f32) {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_tap(builder, ffn_input_name, ffn_input));
  } else if (tap_ffn_input) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap_linear_input_bf16_as_f32(
        builder, ffn_input_name, total_token_count, bf16_token_capacity,
        hidden_size, ffn_input));
  }

  const id4_pipeline_program_shape_t intermediate_stream_shape =
      id4_pipeline_program_make_shape_rank2(intermediate_size,
                                            total_token_count);
  const id4_ideogram4_dit_program_linear_weight_layout_t
      feed_forward_w1_w3_weight_layout =
          bf16_resident_weight_execution
              ? id4_ideogram4_dit_program_select_bf16_resident_linear_weight_layout(
                    activation_format, bf16_token_capacity, hidden_size,
                    intermediate_size)
              : ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_DENSE;
  const id4_ideogram4_dit_program_linear_weight_layout_t
      feed_forward_w2_weight_layout =
          bf16_resident_weight_execution
              ? id4_ideogram4_dit_program_select_bf16_resident_linear_weight_layout(
                    activation_format, bf16_token_capacity, intermediate_size,
                    hidden_size)
              : ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_DENSE;
  id4_ideogram4_dit_program_linear_parameter_t feed_forward_w1_parameter;
  id4_ideogram4_dit_program_linear_parameter_t feed_forward_w3_parameter;
  id4_ideogram4_dit_program_linear_parameter_t feed_forward_w2_parameter;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_linear_parameter(
      builder, parameter_sources, layer_ordinal,
      IREE_SV("feed_forward.w1.weight"), hidden_size, intermediate_size,
      feed_forward_w1_w3_weight_layout, weight_execution_format,
      &feed_forward_w1_parameter));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_linear_parameter(
      builder, parameter_sources, layer_ordinal,
      IREE_SV("feed_forward.w3.weight"), hidden_size, intermediate_size,
      feed_forward_w1_w3_weight_layout, weight_execution_format,
      &feed_forward_w3_parameter));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_linear_parameter(
      builder, parameter_sources, layer_ordinal,
      IREE_SV("feed_forward.w2.weight"), intermediate_size, hidden_size,
      feed_forward_w2_weight_layout, weight_execution_format,
      &feed_forward_w2_parameter));

  id4_pipeline_program_tensor_t mlp_hidden =
      id4_pipeline_program_tensor_invalid();
  if (capture_mlp_hidden_as_f32) {
    if (feed_forward_w1_parameter.storage !=
            ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16 ||
        feed_forward_w3_parameter.storage !=
            ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Ideogram4 DiT FP8 feed-forward gate/up diagnostic capture "
          "requires a fused F32 output route");
    }
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, mlp_hidden_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
        intermediate_stream_shape, &mlp_hidden));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_mlp_gate_up_silu(
        builder, mlp_hidden_dispatch_name, total_token_count, hidden_size,
        intermediate_size, ffn_input, feed_forward_w1_parameter.weight,
        feed_forward_w3_parameter.weight, mlp_hidden));
  } else {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, mlp_hidden_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        id4_pipeline_program_make_shape_rank2(bf16_token_capacity,
                                              intermediate_size),
        &mlp_hidden));
    if (materialize_feed_forward_projections) {
      id4_pipeline_program_tensor_t mlp_gate =
          id4_pipeline_program_tensor_invalid();
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
          builder, mlp_gate_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
          id4_pipeline_program_make_shape_rank2(bf16_token_capacity,
                                                intermediate_size),
          &mlp_gate));
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_dispatch_linear_parameter_bf16(
              builder, mlp_gate_dispatch_name, total_token_count,
              bf16_token_capacity, hidden_size, intermediate_size, ffn_input,
              feed_forward_w1_parameter, mlp_gate));

      id4_pipeline_program_tensor_t mlp_up =
          id4_pipeline_program_tensor_invalid();
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
          builder, mlp_up_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
          id4_pipeline_program_make_shape_rank2(bf16_token_capacity,
                                                intermediate_size),
          &mlp_up));
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_dispatch_linear_parameter_bf16(
              builder, mlp_up_dispatch_name, total_token_count,
              bf16_token_capacity, hidden_size, intermediate_size, ffn_input,
              feed_forward_w3_parameter, mlp_up));

      char after_mlp_projection_name_buffer
          [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
      iree_string_view_t after_mlp_projection_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
          branch_name, layer_ordinal, IREE_SV("after_ffn_w1_w3_projection"),
          after_mlp_projection_name_buffer,
          IREE_ARRAYSIZE(after_mlp_projection_name_buffer),
          &after_mlp_projection_name));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_barrier(
          builder, after_mlp_projection_name));

      if (tap_mlp_gate) {
        IREE_RETURN_IF_ERROR(
            id4_ideogram4_dit_program_tap_linear_input_bf16_as_f32(
                builder, mlp_gate_name, total_token_count, bf16_token_capacity,
                intermediate_size, mlp_gate));
      }
      if (tap_mlp_up) {
        IREE_RETURN_IF_ERROR(
            id4_ideogram4_dit_program_tap_linear_input_bf16_as_f32(
                builder, mlp_up_name, total_token_count, bf16_token_capacity,
                intermediate_size, mlp_up));
      }
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_silu_product_bf16(
          builder, mlp_hidden_dispatch_name, total_token_count,
          bf16_token_capacity, intermediate_size, mlp_gate, mlp_up,
          mlp_hidden));
    } else if (feed_forward_w1_parameter.storage ==
                   ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16 &&
               feed_forward_w3_parameter.storage ==
                   ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16) {
      id4_pipeline_program_tensor_t mlp_gate =
          id4_pipeline_program_tensor_invalid();
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
          builder, mlp_gate_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
          id4_pipeline_program_make_shape_rank2(bf16_token_capacity,
                                                intermediate_size),
          &mlp_gate));
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_dispatch_linear_parameter_bf16(
              builder, mlp_gate_dispatch_name, total_token_count,
              bf16_token_capacity, hidden_size, intermediate_size, ffn_input,
              feed_forward_w1_parameter, mlp_gate));

      char after_mlp_gate_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
      iree_string_view_t after_mlp_gate_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
          branch_name, layer_ordinal, IREE_SV("after_ffn_w1_projection"),
          after_mlp_gate_name_buffer,
          IREE_ARRAYSIZE(after_mlp_gate_name_buffer), &after_mlp_gate_name));
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_barrier(builder, after_mlp_gate_name));

      switch (feed_forward_w3_parameter.layout) {
        case ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_DENSE: {
          IREE_RETURN_IF_ERROR(
              id4_ideogram4_dit_program_dispatch_mlp_up_silu_product_packed_bf16(
                  builder, mlp_hidden_dispatch_name, total_token_count,
                  bf16_token_capacity, hidden_size, intermediate_size,
                  ffn_input, feed_forward_w3_parameter.weight, mlp_gate,
                  mlp_hidden));
          break;
        }
        case ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_WEIGHT_LAYOUT_COMPACT_RHS_TILE: {
          IREE_RETURN_IF_ERROR(
              id4_ideogram4_dit_program_dispatch_mlp_up_silu_product_packed_bf16_compact_rhs_tile(
                  builder, mlp_hidden_dispatch_name, total_token_count,
                  bf16_token_capacity, hidden_size, intermediate_size,
                  ffn_input, feed_forward_w3_parameter.weight, mlp_gate,
                  mlp_hidden));
          break;
        }
        default:
          return iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "Ideogram4 DiT BF16 feed-forward up/product weight layout "
              "%" PRIu32 " is not supported",
              (uint32_t)feed_forward_w3_parameter.layout);
      }
    } else if (feed_forward_w1_parameter.storage ==
                   ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED &&
               feed_forward_w3_parameter.storage ==
                   ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED) {
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_dispatch_mlp_gate_up_silu_product_packed_fp8_bf16(
              builder, mlp_hidden_dispatch_name, total_token_count,
              bf16_token_capacity, hidden_size, intermediate_size, ffn_input,
              feed_forward_w1_parameter.weight, feed_forward_w1_parameter.scale,
              feed_forward_w3_parameter.weight, feed_forward_w3_parameter.scale,
              mlp_hidden));
    } else {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Ideogram4 DiT feed-forward gate/up storage pair %" PRIu32 "/%" PRIu32
          " is not supported",
          (uint32_t)feed_forward_w1_parameter.storage,
          (uint32_t)feed_forward_w3_parameter.storage);
    }
  }

  char after_mlp_hidden_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_mlp_hidden_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_ffn_hidden"),
      after_mlp_hidden_name_buffer,
      IREE_ARRAYSIZE(after_mlp_hidden_name_buffer), &after_mlp_hidden_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_mlp_hidden_name));
  if (capture_mlp_hidden) {
    if (capture_mlp_hidden_as_f32) {
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_tap(builder, mlp_hidden_name, mlp_hidden));
    } else {
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_tap_linear_input_bf16_as_f32(
              builder, mlp_hidden_name, total_token_count, bf16_token_capacity,
              intermediate_size, mlp_hidden));
    }
  }

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
  const bool tap_mlp_output = id4_ideogram4_dit_program_has_diagnostic_tap(
      diagnostic_tap_names, mlp_output_name);
  const bool mlp_output_as_f32 = f32_canonical_activations;
  const id4_pipeline_program_dtype_t mlp_output_dtype =
      mlp_output_as_f32 ? ID4_PIPELINE_PROGRAM_DTYPE_F32
                        : ID4_PIPELINE_PROGRAM_DTYPE_BF16;
  const id4_pipeline_program_shape_t mlp_output_shape =
      mlp_output_as_f32 ? id4_pipeline_program_make_shape_rank2(
                              hidden_size, total_token_count)
                        : id4_pipeline_program_make_shape_rank2(
                              bf16_token_capacity, hidden_size);
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, mlp_output_name, mlp_output_dtype, mlp_output_shape,
      &mlp_output));
  if (mlp_output_as_f32) {
    if (capture_mlp_hidden_as_f32) {
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_dispatch_linear_parameter_f32(
              builder, mlp_output_dispatch_name, total_token_count,
              intermediate_size, hidden_size, mlp_hidden,
              feed_forward_w2_parameter, mlp_output));
    } else {
      if (feed_forward_w2_parameter.storage !=
          ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "Ideogram4 DiT FP8 feed-forward output currently requires "
            "canonical F32 MLP hidden/output");
      }
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_dispatch_linear_packed_bf16_f32(
              builder, mlp_output_dispatch_name, total_token_count,
              total_token_count, intermediate_size, hidden_size, mlp_hidden,
              feed_forward_w2_parameter.weight, mlp_output));
    }
  } else {
    if (capture_mlp_hidden_as_f32) {
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_dispatch_linear_parameter_f32_bf16(
              builder, mlp_output_dispatch_name, total_token_count,
              bf16_token_capacity, intermediate_size, hidden_size, mlp_hidden,
              feed_forward_w2_parameter, mlp_output));
    } else {
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_dispatch_linear_parameter_bf16(
              builder, mlp_output_dispatch_name, total_token_count,
              bf16_token_capacity, intermediate_size, hidden_size, mlp_hidden,
              feed_forward_w2_parameter, mlp_output));
    }
  }

  char after_mlp_output_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_mlp_output_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_ffn_output"),
      after_mlp_output_name_buffer,
      IREE_ARRAYSIZE(after_mlp_output_name_buffer), &after_mlp_output_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_mlp_output_name));
  if (mlp_output_as_f32) {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_tap(builder, mlp_output_name, mlp_output));
  } else if (tap_mlp_output) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap_linear_input_bf16_as_f32(
        builder, mlp_output_name, total_token_count, bf16_token_capacity,
        hidden_size, mlp_output));
  }

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
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_layer_parameter_bf16(
      builder, parameter_sources, layer_ordinal, IREE_SV("ffn_norm2.weight"),
      id4_pipeline_program_make_shape_rank1(hidden_size), &ffn_norm2_weight));
  id4_pipeline_program_tensor_t layer_output =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, layer_output_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      resident_hidden_shape, &layer_output));
  if (mlp_output_as_f32) {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_rmsnorm_gated_residual(
            builder, mlp_residual_dispatch_name, total_token_count, hidden_size,
            mlp_output, ffn_norm2_weight, gate_mlp, post_attention_hidden,
            layer_output));
  } else {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_rmsnorm_gated_residual_input_bf16_token_major(
            builder, mlp_residual_dispatch_name, total_token_count, hidden_size,
            mlp_output, ffn_norm2_weight, gate_mlp, post_attention_hidden,
            layer_output));
  }

  char after_layer_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t after_layer_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_branch_layer_name(
      branch_name, layer_ordinal, IREE_SV("after_layer"),
      after_layer_name_buffer, IREE_ARRAYSIZE(after_layer_name_buffer),
      &after_layer_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_layer_name));
  if (f32_canonical_activations) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
        builder, layer_output_name, layer_output));
  } else if (id4_ideogram4_dit_program_has_diagnostic_tap(diagnostic_tap_names,
                                                          layer_output_name)) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap_token_major_f32_as_f32(
        builder, layer_output_name, total_token_count, hidden_size,
        layer_output));
  }
  *out_hidden = layer_output;
  return iree_ok_status();
}
