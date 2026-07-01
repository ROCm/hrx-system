// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_dit_program.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/stages/ideogram4_dit_program_block.h"

enum {
  ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK = 16,
  ID4_IDEOGRAM4_DIT_LINEAR_WMMA_TOKEN_BLOCK = 32,
  ID4_IDEOGRAM4_DIT_LINEAR_OUTPUT_ROW_BLOCK = 16,
  ID4_IDEOGRAM4_DIT_LINEAR_WMMA_OUTPUT_ROW_BLOCK = 32,
  ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_WIDE_OUTPUT_ROW_BLOCK = 64,
  ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_ONE_WAVE_TOKEN_BLOCK = 64,
  ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_ONE_WAVE_OUTPUT_ROW_BLOCK = 64,
  ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_ONE_WAVE_MIN_TOKEN_CAPACITY = 64,
  ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_TOKEN_BLOCK = 128,
  ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_OUTPUT_ROW_BLOCK = 64,
  ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_COMPACT_OUTPUT_ROW_BLOCK = 128,
  ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_MIN_TOKEN_CAPACITY = 128,
  ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_TAIL_TOKEN_LIMIT = 64,
  ID4_IDEOGRAM4_DIT_ELEMENTWISE_MAX_ELEMENT_COUNT = 268435456,
  ID4_IDEOGRAM4_DIT_LINEAR_INPUT_PACK_MAX_ELEMENT_COUNT = 268435456,
  ID4_IDEOGRAM4_DIT_ATTENTION_QUERY_BLOCK_SIZE = 8,
  ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY = 16,
  ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT = 7,
  ID4_IDEOGRAM4_DIT_LLM_HIDDEN_STATE_LAYER_COUNT = 13,
};

typedef struct id4_ideogram4_dit_program_config_value_t {
  // Loom config key selected by the dispatch authoring site.
  iree_string_view_t key;
  // Unsigned integer config value formatted for loomc.
  uint32_t value;
} id4_ideogram4_dit_program_config_value_t;

typedef struct id4_ideogram4_dit_program_linear_parameter_t {
  // Physical storage format used for the logical weight parameter.
  id4_ideogram4_dit_parameter_storage_t storage;
  // Weight tensor loaded from the selected parameter source.
  id4_pipeline_program_tensor_t weight;
  // Row-scale tensor for scaled FP8 weights; invalid for BF16 weights.
  id4_pipeline_program_tensor_t scale;
} id4_ideogram4_dit_program_linear_parameter_t;

typedef struct id4_ideogram4_dit_program_linear_body_t {
  // Token rows produced by each body dispatch workgroup.
  uint32_t token_block;
  // Output rows produced by each body dispatch workgroup.
  uint32_t output_row_block;
  // Loom module path selected for the body dispatch.
  iree_string_view_t module_path;
  // Exported Loom function selected for the body dispatch.
  iree_string_view_t function_name;
} id4_ideogram4_dit_program_linear_body_t;

typedef struct id4_ideogram4_dit_program_linear_tail_t {
  // Maximum token rows accepted by one tail dispatch.
  uint32_t dispatch_token_count_limit;
  // Loom module path selected for the tail dispatch.
  iree_string_view_t module_path;
  // Exported Loom kernel function selected for the tail dispatch.
  iree_string_view_t function_name;
} id4_ideogram4_dit_program_linear_tail_t;

typedef enum id4_ideogram4_dit_program_linear_output_layout_e {
  ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_OUTPUT_LAYOUT_F32_CANONICAL = 0,
  ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_OUTPUT_LAYOUT_BF16_PACKED = 1,
} id4_ideogram4_dit_program_linear_output_layout_t;

static const id4_ideogram4_dit_model_config_t
    id4_ideogram4_dit_program_ideogram4_model_config_value = {
        .layer_count = 34,
        .input_channel_count = 128,
        .hidden_size = 4608,
        .intermediate_size = 12288,
        .attention_head_count = 18,
        .adaln_size = 512,
        .llm_feature_count = 53248,
        .image_indicator_count = 2,
};

iree_status_t id4_ideogram4_dit_weight_execution_format_parse(
    iree_string_view_t value,
    id4_ideogram4_dit_weight_execution_format_t* out_format) {
  IREE_ASSERT_ARGUMENT(out_format);
  if (iree_string_view_equal(value, IREE_SV("bf16_resident"))) {
    *out_format = ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("fp8_direct"))) {
    *out_format = ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_DIRECT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(
          value, IREE_SV("fp8_direct_feed_forward_bf16_resident"))) {
    *out_format =
        ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_DIRECT_FEED_FORWARD_BF16_RESIDENT;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "DiT weight execution format must be bf16_resident, fp8_direct, or "
      "fp8_direct_feed_forward_bf16_resident");
}

iree_string_view_t id4_ideogram4_dit_weight_execution_format_name(
    id4_ideogram4_dit_weight_execution_format_t format) {
  switch (format) {
    case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT:
      return IREE_SV("bf16_resident");
    case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_DIRECT:
      return IREE_SV("fp8_direct");
    case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_DIRECT_FEED_FORWARD_BF16_RESIDENT:
      return IREE_SV("fp8_direct_feed_forward_bf16_resident");
    default:
      return IREE_SV("invalid");
  }
}

static id4_ideogram4_dit_weight_execution_format_t
id4_ideogram4_dit_program_generic_linear_weight_execution_format(
    id4_ideogram4_dit_weight_execution_format_t weight_execution_format) {
  switch (weight_execution_format) {
    case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT:
      return ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT;
    case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_DIRECT:
    case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_DIRECT_FEED_FORWARD_BF16_RESIDENT:
      return ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_DIRECT;
    default:
      return ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_INVALID;
  }
}

static iree_status_t id4_ideogram4_dit_program_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static id4_ideogram4_dit_program_linear_body_t
id4_ideogram4_dit_program_select_linear_bf16_bf16_body(uint32_t token_capacity,
                                                       uint32_t output_size) {
  const uint32_t two_wave_tail_token_count =
      token_capacity % ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_TOKEN_BLOCK;
  const bool use_two_wave_body =
      token_capacity >=
          ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_MIN_TOKEN_CAPACITY &&
      two_wave_tail_token_count <
          ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_TAIL_TOKEN_LIMIT &&
      (output_size %
       ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_OUTPUT_ROW_BLOCK) == 0;
  if (use_two_wave_body) {
    return (id4_ideogram4_dit_program_linear_body_t){
        .token_block = ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_TOKEN_BLOCK,
        .output_row_block =
            ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_OUTPUT_ROW_BLOCK,
        .module_path = IREE_SV("ideogram4/linear_bf16_bf16_wmma_m128n64_2wave"),
        .function_name =
            IREE_SV("id4_ideogram4_linear_bf16_bf16_wmma_m128n64_2wave"),
    };
  }
  const bool use_one_wave_body =
      token_capacity >=
          ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_ONE_WAVE_MIN_TOKEN_CAPACITY &&
      (token_capacity %
       ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_ONE_WAVE_TOKEN_BLOCK) == 0 &&
      (output_size %
       ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_ONE_WAVE_OUTPUT_ROW_BLOCK) == 0;
  if (use_one_wave_body) {
    return (id4_ideogram4_dit_program_linear_body_t){
        .token_block = ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_ONE_WAVE_TOKEN_BLOCK,
        .output_row_block =
            ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_ONE_WAVE_OUTPUT_ROW_BLOCK,
        .module_path = IREE_SV("ideogram4/linear_bf16_bf16_wmma_m64n64"),
        .function_name = IREE_SV("id4_ideogram4_linear_bf16_bf16_wmma_m64n64"),
    };
  }
  const bool use_wide_body =
      (output_size %
       ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_WIDE_OUTPUT_ROW_BLOCK) == 0;
  if (use_wide_body) {
    return (id4_ideogram4_dit_program_linear_body_t){
        .token_block = ID4_IDEOGRAM4_DIT_LINEAR_WMMA_TOKEN_BLOCK,
        .output_row_block =
            ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_WIDE_OUTPUT_ROW_BLOCK,
        .module_path = IREE_SV("ideogram4/linear_bf16_bf16_wmma_m32n64"),
        .function_name = IREE_SV("id4_ideogram4_linear_bf16_bf16_wmma_m32n64"),
    };
  }
  return (id4_ideogram4_dit_program_linear_body_t){
      .token_block = ID4_IDEOGRAM4_DIT_LINEAR_WMMA_TOKEN_BLOCK,
      .output_row_block = ID4_IDEOGRAM4_DIT_LINEAR_WMMA_OUTPUT_ROW_BLOCK,
      .module_path = IREE_SV("ideogram4/linear_bf16_bf16_wmma"),
      .function_name = IREE_SV("id4_ideogram4_linear_bf16_bf16_wmma"),
  };
}

static id4_ideogram4_dit_program_linear_body_t
id4_ideogram4_dit_program_select_linear_fp8_bf16_body(uint32_t token_capacity,
                                                      uint32_t output_size) {
  const bool use_two_wave_body =
      token_capacity >=
          ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_MIN_TOKEN_CAPACITY &&
      (output_size %
       ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_OUTPUT_ROW_BLOCK) == 0;
  if (use_two_wave_body) {
    return (id4_ideogram4_dit_program_linear_body_t){
        .token_block = ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_TOKEN_BLOCK,
        .output_row_block =
            ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_OUTPUT_ROW_BLOCK,
        .module_path = IREE_SV("ideogram4/linear_fp8_bf16_wmma_m128n64_2wave"),
        .function_name =
            IREE_SV("id4_ideogram4_linear_fp8_bf16_wmma_m128n64_2wave"),
    };
  }
  return (id4_ideogram4_dit_program_linear_body_t){
      .token_block = ID4_IDEOGRAM4_DIT_LINEAR_WMMA_TOKEN_BLOCK,
      .output_row_block = ID4_IDEOGRAM4_DIT_LINEAR_WMMA_OUTPUT_ROW_BLOCK,
      .module_path = IREE_SV("ideogram4/linear_fp8_bf16_wmma"),
      .function_name = IREE_SV("id4_ideogram4_linear_fp8_bf16_wmma"),
  };
}

iree_status_t id4_ideogram4_dit_program_format(char* buffer,
                                               iree_host_size_t buffer_capacity,
                                               iree_string_view_t* out_string,
                                               const char* format, ...) {
  va_list varargs;
  va_start(varargs, format);
  int length = vsnprintf(buffer, buffer_capacity, format, varargs);
  va_end(varargs);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format Ideogram4 DiT program string");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_format_u32(
    uint32_t value, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  return id4_ideogram4_dit_program_format(buffer, buffer_capacity, out_string,
                                          "%" PRIu32, value);
}

iree_status_t id4_ideogram4_dit_program_format_layer_parameter(
    uint32_t layer_ordinal, iree_string_view_t suffix, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string) {
  return id4_ideogram4_dit_program_format(
      buffer, buffer_capacity, out_string, "layers.%" PRIu32 ".%.*s",
      layer_ordinal, (int)suffix.size, suffix.data);
}

iree_status_t id4_ideogram4_dit_program_format_branch_layer_name(
    iree_string_view_t branch_name, uint32_t layer_ordinal,
    iree_string_view_t suffix, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  return id4_ideogram4_dit_program_format(
      buffer, buffer_capacity, out_string, "%.*s.layers.%" PRIu32 ".%.*s",
      (int)branch_name.size, branch_name.data, layer_ordinal, (int)suffix.size,
      suffix.data);
}

bool id4_ideogram4_dit_program_checked_mul_u64(uint64_t lhs, uint64_t rhs,
                                               uint64_t* out_result) {
  if (lhs != 0 && rhs > UINT64_MAX / lhs) return false;
  *out_result = lhs * rhs;
  return true;
}

static iree_status_t id4_ideogram4_dit_program_bf16_matrix_row_range(
    uint32_t row_offset, uint32_t row_count, uint32_t column_count,
    iree_device_size_t* out_offset, iree_device_size_t* out_length) {
  iree_device_size_t row_byte_length = 0;
  iree_device_size_t offset = 0;
  iree_device_size_t length = 0;
  if (!iree_device_size_checked_mul((iree_device_size_t)column_count,
                                    id4_pipeline_program_dtype_byte_length(
                                        ID4_PIPELINE_PROGRAM_DTYPE_BF16),
                                    &row_byte_length) ||
      !iree_device_size_checked_mul((iree_device_size_t)row_offset,
                                    row_byte_length, &offset) ||
      !iree_device_size_checked_mul((iree_device_size_t)row_count,
                                    row_byte_length, &length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 BF16 matrix row byte range overflow");
  }
  *out_offset = offset;
  *out_length = length;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_linear_output_binding(
    id4_ideogram4_dit_program_linear_output_layout_t output_layout,
    id4_pipeline_program_tensor_t output, uint32_t token_offset,
    uint32_t token_count, uint32_t output_size,
    id4_pipeline_program_dispatch_binding_t* out_binding) {
  switch (output_layout) {
    case ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_OUTPUT_LAYOUT_F32_CANONICAL:
      *out_binding = id4_pipeline_program_write(output);
      return iree_ok_status();
    case ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_OUTPUT_LAYOUT_BF16_PACKED: {
      iree_device_size_t output_offset = 0;
      iree_device_size_t output_length = 0;
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_bf16_matrix_row_range(
          token_offset, token_count, output_size, &output_offset,
          &output_length));
      *out_binding = id4_pipeline_program_write_range(output, output_offset,
                                                      output_length);
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported Ideogram4 linear output layout %u",
                              (uint32_t)output_layout);
  }
}

static iree_status_t id4_ideogram4_dit_program_shape_element_count_u32(
    id4_pipeline_program_shape_t shape, iree_string_view_t tensor_name,
    uint32_t* out_element_count) {
  uint64_t element_count = 0;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_shape_element_count(shape, &element_count));
  if (element_count == 0 ||
      element_count > ID4_IDEOGRAM4_DIT_ELEMENTWISE_MAX_ELEMENT_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 DiT tensor `%.*s` element count %" PRIu64
                            " is outside elementwise dispatch range",
                            (int)tensor_name.size, tensor_name.data,
                            element_count);
  }
  *out_element_count = (uint32_t)element_count;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_format_child_name(
    iree_string_view_t parent_name, iree_string_view_t child_suffix,
    char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  return id4_ideogram4_dit_program_format(
      buffer, buffer_capacity, out_string, "%.*s.%.*s", (int)parent_name.size,
      parent_name.data, (int)child_suffix.size, child_suffix.data);
}

static bool id4_ideogram4_dit_program_checked_add_u64(uint64_t lhs,
                                                      uint64_t rhs,
                                                      uint64_t* out_result) {
  if (rhs > UINT64_MAX - lhs) return false;
  *out_result = lhs + rhs;
  return true;
}

static uint32_t id4_ideogram4_dit_program_ceil_div_u32(uint32_t dividend,
                                                       uint32_t divisor) {
  return dividend / divisor + (dividend % divisor != 0 ? 1 : 0);
}

static iree_status_t id4_ideogram4_dit_program_validate_model_config(
    const id4_ideogram4_dit_model_config_t* model) {
  if (model->layer_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT layer count must be nonzero");
  }
  if (model->input_channel_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT input channel count must be "
                            "nonzero");
  }
  if ((model->input_channel_count % 4) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT input channel count must be a "
                            "multiple of 4");
  }
  if (model->hidden_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT hidden size must be nonzero");
  }
  if ((model->hidden_size % 4) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT hidden size must be a multiple of "
                            "4");
  }
  if (model->intermediate_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT intermediate size must be nonzero");
  }
  if (model->attention_head_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT attention head count must be "
                            "nonzero");
  }
  if ((model->hidden_size % model->attention_head_count) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT hidden size must divide evenly by "
                            "attention head count");
  }
  if (((model->hidden_size / model->attention_head_count) % 2) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT attention head size must be even");
  }
  if (model->adaln_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT AdaLN size must be nonzero");
  }
  if (model->llm_feature_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT LLM feature count must be nonzero");
  }
  if ((model->llm_feature_count % 4) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT LLM feature count must be a "
                            "multiple of 4");
  }
  if ((model->llm_feature_count %
       ID4_IDEOGRAM4_DIT_LLM_HIDDEN_STATE_LAYER_COUNT) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT LLM feature count must divide "
                            "evenly by Qwen hidden-state layer count");
  }
  if (model->image_indicator_count != 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT image indicator count must be 2");
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_validate_activation_format(
    id4_ideogram4_dit_activation_format_t activation_format) {
  switch (activation_format) {
    case ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL:
    case ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram4 DiT activation format %" PRIu32
                              " is invalid",
                              (uint32_t)activation_format);
  }
}

static iree_status_t id4_ideogram4_dit_program_validate_weight_execution_format(
    id4_ideogram4_dit_weight_execution_format_t weight_execution_format) {
  switch (weight_execution_format) {
    case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT:
    case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_DIRECT:
    case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_DIRECT_FEED_FORWARD_BF16_RESIDENT:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram4 DiT weight execution format %" PRIu32
                              " is invalid",
                              (uint32_t)weight_execution_format);
  }
}

static iree_status_t
id4_ideogram4_dit_program_validate_attention_implementation(
    id4_ideogram4_dit_attention_implementation_t attention_implementation) {
  switch (attention_implementation) {
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING:
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA:
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_BLOCKED_WMMA:
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram4 DiT attention implementation %" PRIu32
                              " is invalid",
                              (uint32_t)attention_implementation);
  }
}

static iree_status_t
id4_ideogram4_dit_program_validate_feed_forward_implementation(
    id4_ideogram4_dit_feed_forward_implementation_t
        feed_forward_implementation) {
  switch (feed_forward_implementation) {
    case ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_FUSED_PRODUCT:
    case ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY:
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram4 DiT feed-forward implementation %" PRIu32 " is invalid",
          (uint32_t)feed_forward_implementation);
  }
}

static iree_status_t id4_ideogram4_dit_program_validate_parameter_storage(
    id4_ideogram4_dit_parameter_storage_t storage) {
  switch (storage) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16:
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram4 DiT parameter storage %" PRIu32
                              " is invalid",
                              (uint32_t)storage);
  }
}

static iree_status_t id4_ideogram4_dit_program_validate_parameter_sources(
    id4_ideogram4_dit_parameter_sources_t sources) {
  if (sources.rule_count != 0 && !sources.rules) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT parameter source rules are required when rule count is "
        "nonzero");
  }
  for (iree_host_size_t i = 0; i < sources.rule_count; ++i) {
    const id4_ideogram4_dit_parameter_source_rule_t rule = sources.rules[i];
    if (iree_string_view_is_empty(rule.key)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram4 DiT parameter source rule %" PRIhsz " key is empty", i);
    }
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_validate_parameter_storage(rule.storage));
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (iree_string_view_equal(rule.key, sources.rules[j].key)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram4 DiT parameter source rule `%.*s` is duplicated",
            (int)rule.key.size, rule.key.data);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_validate_diagnostic_taps(
    iree_string_view_list_t diagnostic_tap_names) {
  if (diagnostic_tap_names.count == 0) {
    if (diagnostic_tap_names.values) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram4 DiT diagnostic tap values must be NULL when count is 0");
    }
    return iree_ok_status();
  }
  if (!diagnostic_tap_names.values) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT diagnostic tap values are required when count is "
        "nonzero");
  }
  for (iree_host_size_t i = 0; i < diagnostic_tap_names.count; ++i) {
    if (iree_string_view_is_empty(diagnostic_tap_names.values[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram4 DiT diagnostic tap name %" PRIhsz " is empty", i);
    }
  }
  return iree_ok_status();
}

bool id4_ideogram4_dit_program_has_diagnostic_tap(
    iree_string_view_list_t diagnostic_tap_names, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < diagnostic_tap_names.count; ++i) {
    if (iree_string_view_equal(diagnostic_tap_names.values[i], name)) {
      return true;
    }
  }
  return false;
}

iree_status_t id4_ideogram4_dit_program_image_token_count(
    id4_ideogram4_dit_model_config_t model,
    id4_pipeline_program_shape_t latent_shape, uint32_t* out_token_count) {
  if (!out_token_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT image token count output is "
                            "required");
  }
  *out_token_count = 0;
  if (latent_shape.rank != 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT latent shape rank must be 4");
  }
  if (latent_shape.dims[2] != model.input_channel_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT latent channel count %" PRIu64
                            " does not match model channel count %" PRIu32,
                            latent_shape.dims[2], model.input_channel_count);
  }
  if (latent_shape.dims[3] != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT latent batch dimension must be 1");
  }
  uint64_t token_count = 0;
  if (!id4_ideogram4_dit_program_checked_mul_u64(
          latent_shape.dims[0], latent_shape.dims[1], &token_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 DiT image token count overflow");
  }
  if (token_count == 0 ||
      token_count > ID4_IDEOGRAM4_DIT_PRELUDE_IMAGE_MAX_TOKEN_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram4 DiT image token count %" PRIu64 " exceeds max count %u",
        token_count, ID4_IDEOGRAM4_DIT_PRELUDE_IMAGE_MAX_TOKEN_COUNT);
  }
  *out_token_count = (uint32_t)token_count;
  return iree_ok_status();
}

iree_status_t id4_ideogram4_dit_program_calculate_bf16_token_capacity(
    uint32_t total_token_count, uint32_t* out_token_capacity) {
  if (!out_token_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT token capacity output is required");
  }
  *out_token_capacity = 0;
  if (total_token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT token count must be non-zero");
  }
  if (total_token_count >
      UINT32_MAX - (ID4_IDEOGRAM4_DIT_BF16_TOKEN_CAPACITY_BLOCK - 1)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 DiT token capacity overflow");
  }
  *out_token_capacity =
      ((total_token_count + ID4_IDEOGRAM4_DIT_BF16_TOKEN_CAPACITY_BLOCK - 1) /
       ID4_IDEOGRAM4_DIT_BF16_TOKEN_CAPACITY_BLOCK) *
      ID4_IDEOGRAM4_DIT_BF16_TOKEN_CAPACITY_BLOCK;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_token_counts(
    const id4_ideogram4_dit_program_options_t* options,
    uint32_t* out_image_token_count, uint32_t* out_total_token_count) {
  *out_image_token_count = 0;
  *out_total_token_count = 0;

  uint32_t image_token_count = 0;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_image_token_count(
      options->model, options->request.latent_shape, &image_token_count));

  switch (options->request.conditioning_mode) {
    case ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED:
      if (options->request.text_token_count != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram4 DiT unconditioned requests must not provide text "
            "tokens");
      }
      *out_image_token_count = image_token_count;
      *out_total_token_count = image_token_count;
      return iree_ok_status();
    case ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED: {
      if (options->request.text_token_count == 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram4 DiT conditioned requests require text tokens");
      }
      uint64_t total_token_count = 0;
      if (!id4_ideogram4_dit_program_checked_add_u64(
              options->request.text_token_count, image_token_count,
              &total_token_count)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "Ideogram4 DiT combined token count overflow");
      }
      if (total_token_count > ID4_IDEOGRAM4_DIT_PRELUDE_MAX_TOKEN_COUNT) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "Ideogram4 DiT combined token count %" PRIu64
                                " exceeds max count %u",
                                total_token_count,
                                ID4_IDEOGRAM4_DIT_PRELUDE_MAX_TOKEN_COUNT);
      }
      *out_image_token_count = image_token_count;
      *out_total_token_count = (uint32_t)total_token_count;
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram4 DiT conditioning mode %" PRIu32
                              " is invalid",
                              (uint32_t)options->request.conditioning_mode);
  }
}

static iree_status_t id4_ideogram4_dit_program_validate_options(
    const id4_ideogram4_dit_program_options_t* options,
    const id4_pipeline_program_builder_t* builder,
    uint32_t* out_image_token_count, uint32_t* out_total_token_count) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT program options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram4 DiT program")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram4 DiT program extension structures are not supported");
  }
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT program builder is required");
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_validate_model_config(&options->model));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_validate_parameter_sources(
      options->parameter_sources));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_validate_activation_format(
      options->activation_format));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_validate_weight_execution_format(
          options->weight_execution_format));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_validate_attention_implementation(
          options->attention_implementation));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_validate_feed_forward_implementation(
          options->feed_forward_implementation));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_validate_diagnostic_taps(
      options->diagnostic_tap_names));
  return id4_ideogram4_dit_program_token_counts(options, out_image_token_count,
                                                out_total_token_count);
}

static iree_status_t id4_ideogram4_dit_program_make_config_binding(
    iree_string_view_t key, uint32_t value, char* value_buffer,
    iree_host_size_t value_buffer_capacity,
    id4_pipeline_kernel_config_binding_t* out_binding) {
  iree_string_view_t value_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_u32(
      value, value_buffer, value_buffer_capacity, &value_string));
  *out_binding = id4_pipeline_make_kernel_config_binding(key, value_string);
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_make_config_bindings(
    iree_host_size_t config_value_count,
    const id4_ideogram4_dit_program_config_value_t* config_values,
    char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                      [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY],
    id4_pipeline_kernel_config_binding_t* out_bindings) {
  if (config_value_count > ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram4 DiT config binding count %" PRIhsz " exceeds max count %u",
        config_value_count, ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT);
  }
  for (iree_host_size_t i = 0; i < config_value_count; ++i) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_binding(
        config_values[i].key, config_values[i].value, value_buffers[i],
        ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY, &out_bindings[i]));
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_import_tensor(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_import_tensor_flags_t flags,
    id4_pipeline_program_dtype_t dtype, id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  id4_pipeline_program_import_tensor_options_t options = {
      .structure_size = sizeof(options),
      .flags = flags,
      .name = name,
      .dtype = dtype,
      .shape = shape,
  };
  return id4_pipeline_program_import_tensor(builder, &options, out_tensor);
}

iree_status_t id4_ideogram4_dit_program_parameter(
    id4_pipeline_program_builder_t* builder, iree_string_view_t source_scope,
    iree_string_view_t key, id4_pipeline_program_dtype_t dtype,
    id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  const id4_pipeline_program_parameter_source_t sources[] = {
      {
          // Provider scope containing the direct parameter tensor.
          .source_scope = source_scope,
          // Provider key for the direct parameter tensor.
          .key = key,
          // Provider dtype matching the execution tensor.
          .dtype = dtype,
          // Provider shape matching the execution tensor.
          .shape = shape,
      },
  };
  id4_pipeline_program_parameter_options_t options = {
      .structure_size = sizeof(options),
      .encoding = ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
      .source_count = IREE_ARRAYSIZE(sources),
      .sources = sources,
      .key = key,
      .dtype = dtype,
      .shape = shape,
  };
  return id4_pipeline_program_parameter(builder, &options, out_tensor);
}

iree_status_t id4_ideogram4_dit_program_parameter_source_resolve(
    id4_ideogram4_dit_parameter_sources_t sources, iree_string_view_t key,
    id4_ideogram4_dit_parameter_source_rule_t* out_source) {
  IREE_ASSERT_ARGUMENT(out_source);
  for (iree_host_size_t i = 0; i < sources.rule_count; ++i) {
    if (iree_string_view_equal(sources.rules[i].key, key)) {
      *out_source = sources.rules[i];
      return iree_ok_status();
    }
  }
  *out_source = (id4_ideogram4_dit_parameter_source_rule_t){
      .key = key,
      .source_scope = sources.default_scope,
      .storage = ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16,
  };
  return iree_ok_status();
}

iree_status_t id4_ideogram4_dit_program_format_parameter_scale_key(
    iree_string_view_t weight_key, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string) {
  return id4_ideogram4_dit_program_format(buffer, buffer_capacity, out_string,
                                          "%.*s_scale", (int)weight_key.size,
                                          weight_key.data);
}

iree_status_t id4_ideogram4_dit_program_parameter_fp8_e4m3_scaled_to_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t source_scope,
    iree_string_view_t weight_key, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t* out_tensor) {
  const id4_pipeline_program_shape_t weight_shape =
      id4_pipeline_program_make_shape_rank2(output_size, input_size);
  char scale_key_buffer[ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t scale_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_parameter_scale_key(
      weight_key, scale_key_buffer, IREE_ARRAYSIZE(scale_key_buffer),
      &scale_key));
  const id4_pipeline_program_parameter_source_t sources[] = {
      {
          // Provider scope containing the compact FP8 weight tensor.
          .source_scope = source_scope,
          // Provider key for the compact FP8 weight tensor.
          .key = weight_key,
          // Provider dtype for the compact weight tensor.
          .dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
          // Provider shape matching the BF16 execution tensor.
          .shape = weight_shape,
      },
      {
          // Provider scope containing the F32 row-scale tensor.
          .source_scope = source_scope,
          // Provider key for the F32 row-scale tensor.
          .key = scale_key,
          // Provider dtype for the row-scale tensor.
          .dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
          // Provider shape with one scale value per output row.
          .shape = id4_pipeline_program_make_shape_rank1(output_size),
      },
  };
  id4_pipeline_program_parameter_options_t options = {
      .structure_size = sizeof(options),
      .encoding =
          ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_SCALED_TO_BF16,
      .source_count = IREE_ARRAYSIZE(sources),
      .sources = sources,
      .key = weight_key,
      .dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      .shape = weight_shape,
  };
  return id4_pipeline_program_parameter(builder, &options, out_tensor);
}

iree_status_t id4_ideogram4_dit_program_parameter_bf16_linear_rhs_tile(
    id4_pipeline_program_builder_t* builder, iree_string_view_t source_scope,
    iree_string_view_t weight_key, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t* out_tensor) {
  const id4_pipeline_program_shape_t weight_shape =
      id4_pipeline_program_make_shape_rank2(output_size, input_size);
  const id4_pipeline_program_parameter_source_t sources[] = {
      {
          // Provider scope containing the logical row-major BF16 weight tensor.
          .source_scope = source_scope,
          // Provider key for the logical row-major BF16 weight tensor.
          .key = weight_key,
          // Provider dtype for the logical weight tensor.
          .dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
          // Provider shape matching the logical execution tensor.
          .shape = weight_shape,
      },
  };
  id4_pipeline_program_parameter_options_t options = {
      .structure_size = sizeof(options),
      .encoding = ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_BF16_LINEAR_RHS_TILE,
      .source_count = IREE_ARRAYSIZE(sources),
      .sources = sources,
      .key = weight_key,
      .dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      .shape = weight_shape,
  };
  return id4_pipeline_program_parameter(builder, &options, out_tensor);
}

iree_status_t
id4_ideogram4_dit_program_parameter_fp8_e4m3_scaled_to_bf16_linear_rhs_tile(
    id4_pipeline_program_builder_t* builder, iree_string_view_t source_scope,
    iree_string_view_t weight_key, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t* out_tensor) {
  const id4_pipeline_program_shape_t weight_shape =
      id4_pipeline_program_make_shape_rank2(output_size, input_size);
  char scale_key_buffer[ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t scale_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_parameter_scale_key(
      weight_key, scale_key_buffer, IREE_ARRAYSIZE(scale_key_buffer),
      &scale_key));
  const id4_pipeline_program_parameter_source_t sources[] = {
      {
          // Provider scope containing the compact FP8 weight tensor.
          .source_scope = source_scope,
          // Provider key for the compact FP8 weight tensor.
          .key = weight_key,
          // Provider dtype for the compact weight tensor.
          .dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
          // Provider shape matching the logical execution tensor.
          .shape = weight_shape,
      },
      {
          // Provider scope containing the F32 row-scale tensor.
          .source_scope = source_scope,
          // Provider key for the F32 row-scale tensor.
          .key = scale_key,
          // Provider dtype for the row-scale tensor.
          .dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
          // Provider shape with one scale value per output row.
          .shape = id4_pipeline_program_make_shape_rank1(output_size),
      },
  };
  id4_pipeline_program_parameter_options_t options = {
      .structure_size = sizeof(options),
      .encoding =
          ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_SCALED_TO_BF16_LINEAR_RHS_TILE,
      .source_count = IREE_ARRAYSIZE(sources),
      .sources = sources,
      .key = weight_key,
      .dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      .shape = weight_shape,
  };
  return id4_pipeline_program_parameter(builder, &options, out_tensor);
}

iree_status_t id4_ideogram4_dit_program_parameter_fp8_e4m3_scaled(
    id4_pipeline_program_builder_t* builder, iree_string_view_t source_scope,
    iree_string_view_t weight_key, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t* out_weight,
    id4_pipeline_program_tensor_t* out_scale) {
  const id4_pipeline_program_shape_t weight_shape =
      id4_pipeline_program_make_shape_rank2(output_size, input_size);
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter(
      builder, source_scope, weight_key, ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
      weight_shape, out_weight));

  char scale_key_buffer[ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t scale_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_parameter_scale_key(
      weight_key, scale_key_buffer, IREE_ARRAYSIZE(scale_key_buffer),
      &scale_key));
  return id4_ideogram4_dit_program_parameter(
      builder, source_scope, scale_key, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(output_size), out_scale);
}

static iree_status_t id4_ideogram4_dit_program_parameter_bf16(
    id4_pipeline_program_builder_t* builder,
    id4_ideogram4_dit_parameter_sources_t sources, iree_string_view_t key,
    id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  id4_ideogram4_dit_parameter_source_rule_t source;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter_source_resolve(
      sources, key, &source));
  if (source.storage != ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Ideogram4 DiT parameter `%.*s` storage %" PRIu32
                            " is not supported at this authoring site",
                            (int)key.size, key.data, (uint32_t)source.storage);
  }
  return id4_ideogram4_dit_program_parameter(builder, source.source_scope, key,
                                             ID4_PIPELINE_PROGRAM_DTYPE_BF16,
                                             shape, out_tensor);
}

static iree_status_t id4_ideogram4_dit_program_linear_parameter(
    id4_pipeline_program_builder_t* builder,
    id4_ideogram4_dit_parameter_sources_t sources, iree_string_view_t key,
    uint32_t input_size, uint32_t output_size,
    id4_ideogram4_dit_weight_execution_format_t weight_execution_format,
    id4_ideogram4_dit_program_linear_parameter_t* out_parameter) {
  const id4_ideogram4_dit_weight_execution_format_t
      linear_weight_execution_format =
          id4_ideogram4_dit_program_generic_linear_weight_execution_format(
              weight_execution_format);
  *out_parameter = (id4_ideogram4_dit_program_linear_parameter_t){
      .storage = ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_INVALID,
      .weight = id4_pipeline_program_tensor_invalid(),
      .scale = id4_pipeline_program_tensor_invalid(),
  };

  id4_ideogram4_dit_parameter_source_rule_t source;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter_source_resolve(
      sources, key, &source));
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
            (int)key.size, key.data, (uint32_t)weight_execution_format);
      }
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter(
          builder, source.source_scope, key, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
          weight_shape, &out_parameter->weight));
      out_parameter->storage = ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16;
      break;
    }
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED: {
      switch (linear_weight_execution_format) {
        case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT: {
          IREE_RETURN_IF_ERROR(
              id4_ideogram4_dit_program_parameter_fp8_e4m3_scaled_to_bf16(
                  builder, source.source_scope, key, input_size, output_size,
                  &out_parameter->weight));
          out_parameter->storage = ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16;
          break;
        }
        case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_DIRECT: {
          IREE_RETURN_IF_ERROR(
              id4_ideogram4_dit_program_parameter_fp8_e4m3_scaled(
                  builder, source.source_scope, key, input_size, output_size,
                  &out_parameter->weight, &out_parameter->scale));
          out_parameter->storage =
              ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED;
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
                              (int)key.size, key.data,
                              (uint32_t)source.storage);
  }
  return iree_ok_status();
}

iree_status_t id4_ideogram4_dit_program_acquire_tensor(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_dtype_t dtype, id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  id4_pipeline_program_acquire_tensor_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
      .dtype = dtype,
      .shape = shape,
  };
  return id4_pipeline_program_acquire_tensor(builder, &options, out_tensor);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_loom(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    iree_string_view_t module_path, iree_string_view_t function_name,
    iree_host_size_t config_binding_count,
    const id4_pipeline_kernel_config_binding_t* config_bindings,
    iree_host_size_t binding_count,
    const id4_pipeline_program_dispatch_binding_t* bindings) {
  id4_pipeline_program_dispatch_loom_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
      .kernel = id4_pipeline_make_kernel_ref(module_path, function_name),
      .config_binding_count = config_binding_count,
      .config_bindings = config_bindings,
      .binding_count = binding_count,
      .bindings = bindings,
  };
  return id4_pipeline_program_dispatch_loom(builder, &options);
}

iree_status_t id4_ideogram4_dit_program_dispatch_zero_tail_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t logical_element_count, uint32_t dispatch_element_count,
    id4_pipeline_program_tensor_t output) {
  if (logical_element_count >= dispatch_element_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT BF16 zero-tail logical element count %" PRIu32
        " must be less than dispatch element count %" PRIu32,
        logical_element_count, dispatch_element_count);
  }
  if (dispatch_element_count >
      ID4_IDEOGRAM4_DIT_ELEMENTWISE_MAX_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram4 DiT BF16 zero-tail dispatch element count %" PRIu32
        " exceeds max count %u",
        dispatch_element_count,
        ID4_IDEOGRAM4_DIT_ELEMENTWISE_MAX_ELEMENT_COUNT);
  }
  const uint32_t tail_element_count =
      dispatch_element_count - logical_element_count;
  iree_device_size_t output_offset = 0;
  iree_device_size_t output_length = 0;
  if (!iree_device_size_checked_mul((iree_device_size_t)logical_element_count,
                                    id4_pipeline_program_dtype_byte_length(
                                        ID4_PIPELINE_PROGRAM_DTYPE_BF16),
                                    &output_offset) ||
      !iree_device_size_checked_mul((iree_device_size_t)tail_element_count,
                                    id4_pipeline_program_dtype_byte_length(
                                        ID4_PIPELINE_PROGRAM_DTYPE_BF16),
                                    &output_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram4 DiT BF16 zero-tail output byte range overflow");
  }
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.tensor.zero_tail.element_count"), logical_element_count},
      {IREE_SV("id4.tensor.zero_tail.dispatch_element_count"),
       dispatch_element_count},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_write_range(output, output_offset, output_length),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("tensor/zero_tail_bf16"),
      IREE_SV("id4_tensor_zero_tail_bf16"), IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

iree_status_t id4_ideogram4_dit_program_barrier(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name) {
  id4_pipeline_program_barrier_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
  };
  return id4_pipeline_program_barrier(builder, &options);
}

iree_status_t id4_ideogram4_dit_program_tap(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_tensor_t tensor) {
  id4_pipeline_program_tap_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
      .tensor = tensor,
  };
  return id4_pipeline_program_tap(builder, &options);
}

static iree_status_t id4_ideogram4_dit_program_export(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_tensor_t tensor) {
  id4_pipeline_program_export_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
      .tensor = tensor,
  };
  return id4_pipeline_program_export(builder, &options);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_prelude_image(
    id4_pipeline_program_builder_t* builder, uint32_t image_token_count,
    uint32_t total_token_count, uint32_t output_token_offset,
    uint32_t image_width, uint32_t image_height, uint32_t input_channel_count,
    uint32_t hidden_size, id4_pipeline_program_tensor_t image_tokens,
    id4_ideogram4_dit_program_linear_parameter_t input_proj,
    id4_pipeline_program_tensor_t input_proj_bias,
    id4_pipeline_program_tensor_t image_indicator,
    id4_pipeline_program_tensor_t image_indicator_embedding,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.prelude_image.token_count"), image_token_count},
      {IREE_SV("id4.ideogram4.prelude_image.output_token_count"),
       total_token_count},
      {IREE_SV("id4.ideogram4.prelude_image.output_token_offset"),
       output_token_offset},
      {IREE_SV("id4.ideogram4.prelude_image.image_width"), image_width},
      {IREE_SV("id4.ideogram4.prelude_image.image_height"), image_height},
      {IREE_SV("id4.ideogram4.prelude_image.input_channel_count"),
       input_channel_count},
      {IREE_SV("id4.ideogram4.prelude_image.hidden_size"), hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  switch (input_proj.storage) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16: {
      id4_pipeline_program_dispatch_binding_t bindings[] = {
          id4_pipeline_program_read(image_tokens),
          id4_pipeline_program_read(input_proj.weight),
          id4_pipeline_program_read(input_proj_bias),
          id4_pipeline_program_read(image_indicator),
          id4_pipeline_program_read(image_indicator_embedding),
          id4_pipeline_program_write(output),
      };
      return id4_ideogram4_dit_program_dispatch_loom(
          builder, IREE_SV("ideogram4.prelude.image_projection"),
          IREE_SV("ideogram4/prelude_image_bf16_f32"),
          IREE_SV("id4_ideogram4_prelude_image_bf16_f32"),
          IREE_ARRAYSIZE(config_values), config_bindings,
          IREE_ARRAYSIZE(bindings), bindings);
    }
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED: {
      id4_pipeline_program_dispatch_binding_t bindings[] = {
          id4_pipeline_program_read(image_tokens),
          id4_pipeline_program_read(input_proj.weight),
          id4_pipeline_program_read(input_proj.scale),
          id4_pipeline_program_read(input_proj_bias),
          id4_pipeline_program_read(image_indicator),
          id4_pipeline_program_read(image_indicator_embedding),
          id4_pipeline_program_write(output),
      };
      return id4_ideogram4_dit_program_dispatch_loom(
          builder, IREE_SV("ideogram4.prelude.image_projection"),
          IREE_SV("ideogram4/prelude_image_fp8_f32"),
          IREE_SV("id4_ideogram4_prelude_image_fp8_f32"),
          IREE_ARRAYSIZE(config_values), config_bindings,
          IREE_ARRAYSIZE(bindings), bindings);
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "Ideogram4 image projection storage %" PRIu32
                              " is not supported",
                              (uint32_t)input_proj.storage);
  }
}

static iree_status_t id4_ideogram4_dit_program_dispatch_condition_rmsnorm(
    id4_pipeline_program_builder_t* builder, uint32_t text_token_count,
    uint32_t llm_feature_count, id4_pipeline_program_tensor_t condition,
    id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.condition_rmsnorm.token_count"),
       text_token_count},
      {IREE_SV("id4.ideogram4.condition_rmsnorm.feature_count"),
       llm_feature_count},
      {IREE_SV("id4.ideogram4.condition_rmsnorm.hidden_state_layer_count"),
       ID4_IDEOGRAM4_DIT_LLM_HIDDEN_STATE_LAYER_COUNT},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(condition),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, IREE_SV("ideogram4.cond.prelude.llm_cond_norm"),
      IREE_SV("ideogram4/condition_rmsnorm_f32"),
      IREE_SV("id4_ideogram4_condition_rmsnorm_f32"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_condition_rmsnorm_pack(
    id4_pipeline_program_builder_t* builder, uint32_t text_token_count,
    uint32_t llm_feature_count, id4_pipeline_program_tensor_t condition,
    id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t* out_packed_condition) {
  const uint32_t padded_text_token_count =
      id4_ideogram4_dit_program_ceil_div_u32(
          text_token_count, ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK) *
      ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, IREE_SV("ideogram4.cond.prelude.llm_cond_norm_packed"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(padded_text_token_count,
                                            llm_feature_count),
      out_packed_condition));
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.condition_rmsnorm_pack.text_token_count"),
       text_token_count},
      {IREE_SV("id4.ideogram4.condition_rmsnorm_pack.padded_token_count"),
       padded_text_token_count},
      {IREE_SV("id4.ideogram4.condition_rmsnorm_pack.feature_count"),
       llm_feature_count},
      {IREE_SV("id4.ideogram4.condition_rmsnorm_pack.hidden_state_layer_count"),
       ID4_IDEOGRAM4_DIT_LLM_HIDDEN_STATE_LAYER_COUNT},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(condition),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_write(*out_packed_condition),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, IREE_SV("ideogram4.cond.prelude.llm_cond_norm_pack"),
      IREE_SV("ideogram4/condition_rmsnorm_pack_f32_bf16"),
      IREE_SV("id4_ideogram4_condition_rmsnorm_pack_f32_bf16"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t
id4_ideogram4_dit_program_dispatch_condition_project_packed(
    id4_pipeline_program_builder_t* builder, uint32_t text_token_count,
    uint32_t total_token_count, uint32_t llm_feature_count,
    uint32_t hidden_size, iree_string_view_list_t diagnostic_tap_names,
    id4_pipeline_program_tensor_t packed_condition,
    id4_ideogram4_dit_program_linear_parameter_t projection,
    id4_pipeline_program_tensor_t bias,
    id4_pipeline_program_tensor_t image_indicator,
    id4_pipeline_program_tensor_t image_indicator_embedding,
    id4_pipeline_program_tensor_t output) {
  const uint32_t padded_text_token_count =
      id4_ideogram4_dit_program_ceil_div_u32(
          text_token_count, ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK) *
      ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK;
  uint64_t post_element_count64 = 0;
  if (!id4_ideogram4_dit_program_checked_mul_u64(text_token_count, hidden_size,
                                                 &post_element_count64) ||
      post_element_count64 >
          ID4_IDEOGRAM4_DIT_LINEAR_INPUT_PACK_MAX_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram4 condition projection post element count overflow");
  }
  const uint32_t post_element_count = (uint32_t)post_element_count64;

  const iree_string_view_t projected_condition_name =
      IREE_SV("ideogram4.cond.prelude.llm_cond_proj.raw");
  const bool tap_projected_condition =
      id4_ideogram4_dit_program_has_diagnostic_tap(diagnostic_tap_names,
                                                   projected_condition_name);
  id4_pipeline_program_tensor_t projected_condition =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, projected_condition_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(hidden_size,
                                            padded_text_token_count),
      &projected_condition));
  switch (projection.storage) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16: {
      const id4_ideogram4_dit_program_config_value_t linear_config_values[] = {
          {IREE_SV("id4.ideogram4.linear_wmma.token_count"),
           padded_text_token_count},
          {IREE_SV("id4.ideogram4.linear_wmma.dispatch_token_count"),
           padded_text_token_count},
          {IREE_SV("id4.ideogram4.linear_wmma.input_size"), llm_feature_count},
          {IREE_SV("id4.ideogram4.linear_wmma.output_size"), hidden_size},
      };
      char linear_value_buffers
          [ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
          [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
      id4_pipeline_kernel_config_binding_t linear_config_bindings
          [ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
          IREE_ARRAYSIZE(linear_config_values), linear_config_values,
          linear_value_buffers, linear_config_bindings));
      id4_pipeline_program_dispatch_binding_t linear_bindings[] = {
          id4_pipeline_program_read(packed_condition),
          id4_pipeline_program_read(projection.weight),
          id4_pipeline_program_write(projected_condition),
      };
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_loom(
          builder, IREE_SV("ideogram4.cond.prelude.llm_cond_proj.wmma"),
          IREE_SV("ideogram4/linear_bf16_f32_wmma"),
          IREE_SV("id4_ideogram4_linear_bf16_f32_wmma"),
          IREE_ARRAYSIZE(linear_config_values), linear_config_bindings,
          IREE_ARRAYSIZE(linear_bindings), linear_bindings));
      break;
    }
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED: {
      const id4_ideogram4_dit_program_config_value_t linear_config_values[] = {
          {IREE_SV("id4.ideogram4.linear_fp8_bf16_f32_wmma.token_count"),
           padded_text_token_count},
          {IREE_SV(
               "id4.ideogram4.linear_fp8_bf16_f32_wmma.dispatch_token_count"),
           padded_text_token_count},
          {IREE_SV("id4.ideogram4.linear_fp8_bf16_f32_wmma.input_size"),
           llm_feature_count},
          {IREE_SV("id4.ideogram4.linear_fp8_bf16_f32_wmma.output_size"),
           hidden_size},
      };
      char linear_value_buffers
          [ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
          [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
      id4_pipeline_kernel_config_binding_t linear_config_bindings
          [ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
          IREE_ARRAYSIZE(linear_config_values), linear_config_values,
          linear_value_buffers, linear_config_bindings));
      id4_pipeline_program_dispatch_binding_t linear_bindings[] = {
          id4_pipeline_program_read(packed_condition),
          id4_pipeline_program_read(projection.weight),
          id4_pipeline_program_read(projection.scale),
          id4_pipeline_program_write(projected_condition),
      };
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_loom(
          builder, IREE_SV("ideogram4.cond.prelude.llm_cond_proj.wmma"),
          IREE_SV("ideogram4/linear_fp8_bf16_f32_wmma"),
          IREE_SV("id4_ideogram4_linear_fp8_bf16_f32_wmma"),
          IREE_ARRAYSIZE(linear_config_values), linear_config_bindings,
          IREE_ARRAYSIZE(linear_bindings), linear_bindings));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "Ideogram4 condition projection storage %" PRIu32
                              " is not supported",
                              (uint32_t)projection.storage);
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_barrier(
      builder, IREE_SV("ideogram4.cond.prelude.after_llm_cond_proj_wmma")));
  if (tap_projected_condition) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
        builder, projected_condition_name, projected_condition));
  }

  const id4_ideogram4_dit_program_config_value_t post_config_values[] = {
      {IREE_SV("id4.ideogram4.condition_project_bias.text_token_count"),
       text_token_count},
      {IREE_SV("id4.ideogram4.condition_project_bias.padded_token_count"),
       padded_text_token_count},
      {IREE_SV("id4.ideogram4.condition_project_bias.total_token_count"),
       total_token_count},
      {IREE_SV("id4.ideogram4.condition_project_bias.hidden_size"),
       hidden_size},
      {IREE_SV("id4.ideogram4.condition_project_bias.element_count"),
       post_element_count},
  };
  char post_value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                         [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      post_config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(post_config_values), post_config_values,
      post_value_buffers, post_config_bindings));
  id4_pipeline_program_dispatch_binding_t post_bindings[] = {
      id4_pipeline_program_read(projected_condition),
      id4_pipeline_program_read(bias),
      id4_pipeline_program_read(image_indicator),
      id4_pipeline_program_read(image_indicator_embedding),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, IREE_SV("ideogram4.cond.prelude.llm_cond_proj.bias_indicator"),
      IREE_SV("ideogram4/condition_project_bias_indicator_f32"),
      IREE_SV("id4_ideogram4_condition_project_bias_indicator_f32"),
      IREE_ARRAYSIZE(post_config_values), post_config_bindings,
      IREE_ARRAYSIZE(post_bindings), post_bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_condition_project(
    id4_pipeline_program_builder_t* builder, uint32_t text_token_count,
    uint32_t total_token_count, uint32_t llm_feature_count,
    uint32_t hidden_size, iree_string_view_list_t diagnostic_tap_names,
    id4_pipeline_program_tensor_t condition,
    id4_ideogram4_dit_program_linear_parameter_t projection,
    id4_pipeline_program_tensor_t bias,
    id4_pipeline_program_tensor_t image_indicator,
    id4_pipeline_program_tensor_t image_indicator_embedding,
    id4_pipeline_program_tensor_t output) {
  const uint32_t padded_text_token_count =
      id4_ideogram4_dit_program_ceil_div_u32(
          text_token_count, ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK) *
      ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK;
  uint64_t pack_element_count64 = 0;
  if (!id4_ideogram4_dit_program_checked_mul_u64(
          padded_text_token_count, llm_feature_count, &pack_element_count64) ||
      pack_element_count64 >
          ID4_IDEOGRAM4_DIT_LINEAR_INPUT_PACK_MAX_ELEMENT_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 condition pack element count overflow");
  }
  const uint32_t pack_element_count = (uint32_t)pack_element_count64;

  id4_pipeline_program_tensor_t packed_condition =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, IREE_SV("ideogram4.cond.prelude.llm_cond_norm_packed"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(padded_text_token_count,
                                            llm_feature_count),
      &packed_condition));
  const id4_ideogram4_dit_program_config_value_t pack_config_values[] = {
      {IREE_SV("id4.ideogram4.condition_input_pack.text_token_count"),
       text_token_count},
      {IREE_SV("id4.ideogram4.condition_input_pack.padded_token_count"),
       padded_text_token_count},
      {IREE_SV("id4.ideogram4.condition_input_pack.feature_count"),
       llm_feature_count},
      {IREE_SV("id4.ideogram4.condition_input_pack.element_count"),
       pack_element_count},
  };
  char pack_value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                         [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      pack_config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(pack_config_values), pack_config_values,
      pack_value_buffers, pack_config_bindings));
  id4_pipeline_program_dispatch_binding_t pack_bindings[] = {
      id4_pipeline_program_read(condition),
      id4_pipeline_program_write(packed_condition),
  };
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_loom(
      builder, IREE_SV("ideogram4.cond.prelude.llm_cond_pack"),
      IREE_SV("ideogram4/condition_input_pack_f32_bf16"),
      IREE_SV("id4_ideogram4_condition_input_pack_f32_bf16"),
      IREE_ARRAYSIZE(pack_config_values), pack_config_bindings,
      IREE_ARRAYSIZE(pack_bindings), pack_bindings));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_barrier(
      builder, IREE_SV("ideogram4.cond.prelude.after_llm_cond_pack")));
  return id4_ideogram4_dit_program_dispatch_condition_project_packed(
      builder, text_token_count, total_token_count, llm_feature_count,
      hidden_size, diagnostic_tap_names, packed_condition, projection, bias,
      image_indicator, image_indicator_embedding, output);
}

iree_status_t id4_ideogram4_dit_program_dispatch_adaln_split(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t hidden_size, id4_pipeline_program_tensor_t raw_modulation,
    id4_pipeline_program_tensor_t scale_msa,
    id4_pipeline_program_tensor_t gate_msa,
    id4_pipeline_program_tensor_t scale_mlp,
    id4_pipeline_program_tensor_t gate_mlp) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.adaln_split.hidden_size"), hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(raw_modulation),
      id4_pipeline_program_write(scale_msa),
      id4_pipeline_program_write(gate_msa),
      id4_pipeline_program_write(scale_mlp),
      id4_pipeline_program_write(gate_mlp),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/adaln_split_f32"),
      IREE_SV("id4_ideogram4_adaln_split_f32"), IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_modulated_rmsnorm(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t hidden_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t scale, id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.modulated_rmsnorm.token_count"), token_count},
      {IREE_SV("id4.ideogram4.modulated_rmsnorm.hidden_size"), hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_read(scale),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/modulated_rmsnorm_f32"),
      IREE_SV("id4_ideogram4_modulated_rmsnorm_f32"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t
id4_ideogram4_dit_program_dispatch_modulated_rmsnorm_token_major_linear_input_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t hidden_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t scale, id4_pipeline_program_tensor_t output) {
  if (token_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 modulated RMSNorm token-major BF16 pack token count must "
        "be nonzero");
  }
  if (token_capacity < token_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 modulated RMSNorm token-major BF16 pack token capacity "
        "%" PRIu32 " is smaller than token count %" PRIu32,
        token_capacity, token_count);
  }
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.modulated_rmsnorm.token_count"), token_count},
      {IREE_SV("id4.ideogram4.modulated_rmsnorm.token_capacity"),
       token_capacity},
      {IREE_SV("id4.ideogram4.modulated_rmsnorm.hidden_size"), hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_read(scale),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/modulated_rmsnorm_f32"),
      IREE_SV("id4_ideogram4_modulated_rmsnorm_token_major_linear_input_bf16"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_rmsnorm_gated_residual(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t hidden_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t gate, id4_pipeline_program_tensor_t residual,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.rmsnorm_gated_residual.token_count"),
       token_count},
      {IREE_SV("id4.ideogram4.rmsnorm_gated_residual.hidden_size"),
       hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),   id4_pipeline_program_read(weight),
      id4_pipeline_program_read(gate),    id4_pipeline_program_read(residual),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/rmsnorm_gated_residual_f32"),
      IREE_SV("id4_ideogram4_rmsnorm_gated_residual_f32"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t
id4_ideogram4_dit_program_dispatch_rmsnorm_gated_residual_input_bf16_token_major(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t hidden_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t gate, id4_pipeline_program_tensor_t residual,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.rmsnorm_gated_residual.token_count"),
       token_count},
      {IREE_SV("id4.ideogram4.rmsnorm_gated_residual.hidden_size"),
       hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),   id4_pipeline_program_read(weight),
      id4_pipeline_program_read(gate),    id4_pipeline_program_read(residual),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/rmsnorm_gated_residual_f32"),
      IREE_SV("id4_ideogram4_rmsnorm_gated_residual_input_bf16_token_major"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_mlp_gate_up_silu(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size, uint32_t intermediate_size,
    id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t gate_weight,
    id4_pipeline_program_tensor_t up_weight,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.mlp_gate_up_silu.token_count"), token_count},
      {IREE_SV("id4.ideogram4.mlp_gate_up_silu.input_size"), input_size},
      {IREE_SV("id4.ideogram4.mlp_gate_up_silu.intermediate_size"),
       intermediate_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(gate_weight),
      id4_pipeline_program_read(up_weight),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/mlp_gate_up_silu_f32"),
      IREE_SV("id4_ideogram4_mlp_gate_up_silu_f32"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t
id4_ideogram4_dit_program_dispatch_mlp_gate_up_silu_linear_input_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size, uint32_t intermediate_size,
    id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t gate_weight,
    id4_pipeline_program_tensor_t up_weight,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.mlp_gate_up_silu.token_count"), token_count},
      {IREE_SV("id4.ideogram4.mlp_gate_up_silu.input_size"), input_size},
      {IREE_SV("id4.ideogram4.mlp_gate_up_silu.intermediate_size"),
       intermediate_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(gate_weight),
      id4_pipeline_program_read(up_weight),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/mlp_gate_up_silu_f32"),
      IREE_SV("id4_ideogram4_mlp_gate_up_silu_linear_input_bf16"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_silu_product_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t intermediate_size,
    id4_pipeline_program_tensor_t gate, id4_pipeline_program_tensor_t up,
    id4_pipeline_program_tensor_t output) {
  if (token_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 SiLU/product token count must be nonzero");
  }
  if (token_capacity < token_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 SiLU/product token capacity %" PRIu32
                            " is smaller than token count %" PRIu32,
                            token_capacity, token_count);
  }
  uint64_t element_count64 = 0;
  if (!id4_ideogram4_dit_program_checked_mul_u64(token_count, intermediate_size,
                                                 &element_count64) ||
      element_count64 > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 SiLU/product element count overflow");
  }
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.silu_product_bf16.token_count"), token_count},
      {IREE_SV("id4.ideogram4.silu_product_bf16.token_capacity"),
       token_capacity},
      {IREE_SV("id4.ideogram4.silu_product_bf16.intermediate_size"),
       intermediate_size},
      {IREE_SV("id4.ideogram4.silu_product_bf16.element_count"),
       (uint32_t)element_count64},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(gate),
      id4_pipeline_program_read(up),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/silu_product_bf16"),
      IREE_SV("id4_ideogram4_silu_product_bf16"), IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

iree_status_t
id4_ideogram4_dit_program_dispatch_mlp_gate_up_silu_product_fp8_bf16_wmma(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t intermediate_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t gate_weight,
    id4_pipeline_program_tensor_t gate_scale,
    id4_pipeline_program_tensor_t up_weight,
    id4_pipeline_program_tensor_t up_scale,
    id4_pipeline_program_tensor_t output) {
  if (token_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 FP8 MLP gate/up/product token count must be nonzero");
  }
  if (token_capacity < token_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 FP8 MLP gate/up/product token capacity %" PRIu32
        " is smaller than token count %" PRIu32,
        token_capacity, token_count);
  }
  if ((token_capacity %
       ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_TOKEN_BLOCK) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 FP8 MLP gate/up/product token capacity %" PRIu32
        " must be a multiple of %u",
        token_capacity,
        ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_TOKEN_BLOCK);
  }
  if ((input_size % ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 FP8 MLP gate/up/product input size %" PRIu32
        " must be a multiple of %u",
        input_size, ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK);
  }
  if ((intermediate_size %
       ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_COMPACT_OUTPUT_ROW_BLOCK) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 FP8 MLP gate/up/product intermediate size %" PRIu32
        " must be a multiple of %u",
        intermediate_size,
        ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_COMPACT_OUTPUT_ROW_BLOCK);
  }
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
               "token_capacity"),
       token_capacity},
      {IREE_SV("id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
               "dispatch_token_count"),
       token_capacity},
      {IREE_SV("id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma.input_size"),
       input_size},
      {IREE_SV("id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
               "intermediate_size"),
       intermediate_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(gate_weight),
      id4_pipeline_program_read(gate_scale),
      id4_pipeline_program_read(up_weight),
      id4_pipeline_program_read(up_scale),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name,
      IREE_SV("ideogram4/mlp_gate_up_silu_product_fp8_bf16_wmma"),
      IREE_SV("id4_ideogram4_mlp_gate_up_silu_product_fp8_bf16_wmma"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t
id4_ideogram4_dit_program_dispatch_mlp_up_silu_product_packed_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t intermediate_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t up_weight, id4_pipeline_program_tensor_t gate,
    id4_pipeline_program_tensor_t output) {
  if (token_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 MLP up/product token count must be nonzero");
  }
  if (token_capacity < token_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 MLP up/product token capacity %" PRIu32
                            " is smaller than token count %" PRIu32,
                            token_capacity, token_count);
  }
  if ((input_size % ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 MLP up/product input size %" PRIu32
                            " must be a multiple of %u",
                            input_size, ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK);
  }
  if ((intermediate_size % ID4_IDEOGRAM4_DIT_LINEAR_WMMA_OUTPUT_ROW_BLOCK) !=
      0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 MLP up/product intermediate size %" PRIu32
        " must be a multiple of %u",
        intermediate_size, ID4_IDEOGRAM4_DIT_LINEAR_WMMA_OUTPUT_ROW_BLOCK);
  }
  const uint32_t body_token_count =
      token_capacity -
      (token_capacity % ID4_IDEOGRAM4_DIT_LINEAR_WMMA_TOKEN_BLOCK);
  const uint32_t tail_token_count =
      token_count > body_token_count ? token_count - body_token_count : 0;

  if (body_token_count != 0) {
    char body_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t body_dispatch_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
        name, IREE_SV("wmma"), body_dispatch_name_buffer,
        IREE_ARRAYSIZE(body_dispatch_name_buffer), &body_dispatch_name));
    const id4_ideogram4_dit_program_config_value_t body_config_values[] = {
        {IREE_SV("id4.ideogram4.mlp_up_silu_product_wmma.token_count"),
         token_capacity},
        {IREE_SV("id4.ideogram4.mlp_up_silu_product_wmma.dispatch_token_count"),
         body_token_count},
        {IREE_SV("id4.ideogram4.mlp_up_silu_product_wmma.input_size"),
         input_size},
        {IREE_SV("id4.ideogram4.mlp_up_silu_product_wmma.intermediate_size"),
         intermediate_size},
    };
    char body_value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                           [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
    id4_pipeline_kernel_config_binding_t
        body_config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
        IREE_ARRAYSIZE(body_config_values), body_config_values,
        body_value_buffers, body_config_bindings));
    id4_pipeline_program_dispatch_binding_t body_bindings[] = {
        id4_pipeline_program_read(input),
        id4_pipeline_program_read(up_weight),
        id4_pipeline_program_read(gate),
        id4_pipeline_program_write(output),
    };
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_loom(
        builder, body_dispatch_name,
        IREE_SV("ideogram4/mlp_up_silu_product_bf16_wmma"),
        IREE_SV("id4_ideogram4_mlp_up_silu_product_bf16_wmma"),
        IREE_ARRAYSIZE(body_config_values), body_config_bindings,
        IREE_ARRAYSIZE(body_bindings), body_bindings));
  }

  if (tail_token_count != 0) {
    if (body_token_count != 0) {
      char after_body_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
      iree_string_view_t after_body_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
          name, IREE_SV("after_wmma"), after_body_name_buffer,
          IREE_ARRAYSIZE(after_body_name_buffer), &after_body_name));
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_barrier(builder, after_body_name));
    }
    char tail_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t tail_dispatch_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
        name, IREE_SV("tail"), tail_dispatch_name_buffer,
        IREE_ARRAYSIZE(tail_dispatch_name_buffer), &tail_dispatch_name));
    const id4_ideogram4_dit_program_config_value_t tail_config_values[] = {
        {IREE_SV("id4.ideogram4.mlp_up_silu_product_tail.token_count"),
         token_capacity},
        {IREE_SV("id4.ideogram4.mlp_up_silu_product_tail.token_offset"),
         body_token_count},
        {IREE_SV("id4.ideogram4.mlp_up_silu_product_tail.dispatch_token_count"),
         tail_token_count},
        {IREE_SV("id4.ideogram4.mlp_up_silu_product_tail.input_size"),
         input_size},
        {IREE_SV("id4.ideogram4.mlp_up_silu_product_tail.intermediate_size"),
         intermediate_size},
    };
    char tail_value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                           [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
    id4_pipeline_kernel_config_binding_t
        tail_config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
        IREE_ARRAYSIZE(tail_config_values), tail_config_values,
        tail_value_buffers, tail_config_bindings));
    id4_pipeline_program_dispatch_binding_t tail_bindings[] = {
        id4_pipeline_program_read(input),
        id4_pipeline_program_read(up_weight),
        id4_pipeline_program_read(gate),
        id4_pipeline_program_write(output),
    };
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_loom(
        builder, tail_dispatch_name,
        IREE_SV("ideogram4/mlp_up_silu_product_bf16_tail"),
        IREE_SV("id4_ideogram4_mlp_up_silu_product_bf16_tail"),
        IREE_ARRAYSIZE(tail_config_values), tail_config_bindings,
        IREE_ARRAYSIZE(tail_bindings), tail_bindings));
  }
  return iree_ok_status();
}

iree_status_t
id4_ideogram4_dit_program_dispatch_mlp_up_silu_product_packed_bf16_compact_rhs_tile(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t intermediate_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t up_weight, id4_pipeline_program_tensor_t gate,
    id4_pipeline_program_tensor_t output) {
  if (token_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 compact RHS MLP up/product token count must be nonzero");
  }
  if (token_capacity < token_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 compact RHS MLP up/product token capacity %" PRIu32
        " is smaller than token count %" PRIu32,
        token_capacity, token_count);
  }
  if ((token_capacity %
       ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_TOKEN_BLOCK) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 compact RHS MLP up/product token capacity %" PRIu32
        " must be a multiple of %u",
        token_capacity,
        ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_TOKEN_BLOCK);
  }
  if ((input_size % ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 compact RHS MLP up/product input size %" PRIu32
        " must be a multiple of %u",
        input_size, ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK);
  }
  if ((intermediate_size %
       ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_COMPACT_OUTPUT_ROW_BLOCK) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 compact RHS MLP up/product intermediate size %" PRIu32
        " must be a multiple of %u",
        intermediate_size,
        ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_COMPACT_OUTPUT_ROW_BLOCK);
  }
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.mlp_up_silu_product_compact_rhs_m128n128_wmma."
               "token_count"),
       token_capacity},
      {IREE_SV("id4.ideogram4.mlp_up_silu_product_compact_rhs_m128n128_wmma."
               "dispatch_token_count"),
       token_capacity},
      {IREE_SV("id4.ideogram4.mlp_up_silu_product_compact_rhs_m128n128_wmma."
               "input_size"),
       input_size},
      {IREE_SV("id4.ideogram4.mlp_up_silu_product_compact_rhs_m128n128_wmma."
               "intermediate_size"),
       intermediate_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(up_weight),
      id4_pipeline_program_read(gate),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name,
      IREE_SV("ideogram4/"
              "mlp_up_silu_product_bf16_wmma_compact_rhs_tile_m128n128_"
              "4wave_quadrant"),
      IREE_SV("id4_ideogram4_mlp_up_silu_product_bf16_wmma_compact_rhs_tile_"
              "m128n128_4wave_quadrant"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_linear_packed_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t weight, id4_pipeline_program_tensor_t output,
    id4_ideogram4_dit_program_linear_output_layout_t output_layout,
    id4_ideogram4_dit_program_linear_body_t body,
    id4_ideogram4_dit_program_linear_tail_t tail) {
  if (token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 linear token count must be nonzero");
  }
  if (token_capacity < token_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 linear token capacity %" PRIu32
                            " is smaller than token count %" PRIu32,
                            token_capacity, token_count);
  }
  if ((input_size % ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 linear input size %" PRIu32
                            " must be a multiple of %u",
                            input_size, ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK);
  }
  if (body.token_block < ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK ||
      (body.token_block % ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 linear body token block %" PRIu32
                            " must be a multiple of %u",
                            body.token_block,
                            ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK);
  }
  if ((output_size % body.output_row_block) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 linear output size %" PRIu32
                            " must be a multiple of %u",
                            output_size, body.output_row_block);
  }
  const uint32_t output_token_capacity =
      output_layout ==
              ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_OUTPUT_LAYOUT_BF16_PACKED
          ? token_capacity
          : token_count;
  const uint32_t body_token_count =
      output_token_capacity - (output_token_capacity % body.token_block);
  const uint32_t tail_token_count = output_token_capacity - body_token_count;
  if (tail.dispatch_token_count_limit == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 linear tail dispatch limit is zero");
  }

  if (body_token_count != 0) {
    id4_pipeline_program_dispatch_binding_t body_output_binding;
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_linear_output_binding(
        output_layout, output, /*token_offset=*/0, body_token_count,
        output_size, &body_output_binding));
    char body_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t body_dispatch_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
        name, IREE_SV("wmma"), body_dispatch_name_buffer,
        IREE_ARRAYSIZE(body_dispatch_name_buffer), &body_dispatch_name));
    const id4_ideogram4_dit_program_config_value_t body_config_values[] = {
        {IREE_SV("id4.ideogram4.linear_wmma.token_count"), token_capacity},
        {IREE_SV("id4.ideogram4.linear_wmma.dispatch_token_count"),
         body_token_count},
        {IREE_SV("id4.ideogram4.linear_wmma.input_size"), input_size},
        {IREE_SV("id4.ideogram4.linear_wmma.output_size"), output_size},
    };
    char body_value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                           [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
    id4_pipeline_kernel_config_binding_t
        body_config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
        IREE_ARRAYSIZE(body_config_values), body_config_values,
        body_value_buffers, body_config_bindings));
    id4_pipeline_program_dispatch_binding_t body_bindings[] = {
        id4_pipeline_program_read(input),
        id4_pipeline_program_read(weight),
        body_output_binding,
    };
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_loom(
        builder, body_dispatch_name, body.module_path, body.function_name,
        IREE_ARRAYSIZE(body_config_values), body_config_bindings,
        IREE_ARRAYSIZE(body_bindings), body_bindings));
  }

  if (tail_token_count != 0 && body_token_count != 0 &&
      output_layout ==
          ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_OUTPUT_LAYOUT_F32_CANONICAL) {
    char after_body_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t after_body_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
        name, IREE_SV("after_wmma"), after_body_name_buffer,
        IREE_ARRAYSIZE(after_body_name_buffer), &after_body_name));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_barrier(builder, after_body_name));
  }

  uint32_t remaining_tail_token_count = tail_token_count;
  uint32_t tail_token_offset = body_token_count;
  for (uint32_t tail_ordinal = 0; remaining_tail_token_count != 0;
       ++tail_ordinal) {
    const uint32_t tail_dispatch_token_count =
        remaining_tail_token_count > tail.dispatch_token_count_limit
            ? tail.dispatch_token_count_limit
            : remaining_tail_token_count;
    id4_pipeline_program_dispatch_binding_t tail_output_binding;
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_linear_output_binding(
        output_layout, output, tail_token_offset, tail_dispatch_token_count,
        output_size, &tail_output_binding));
    char tail_suffix_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t tail_suffix = IREE_SV("tail");
    if (tail_token_count > tail.dispatch_token_count_limit) {
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
          tail_suffix_buffer, IREE_ARRAYSIZE(tail_suffix_buffer), &tail_suffix,
          "tail%" PRIu32, tail_ordinal));
    }
    char tail_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t tail_dispatch_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
        name, tail_suffix, tail_dispatch_name_buffer,
        IREE_ARRAYSIZE(tail_dispatch_name_buffer), &tail_dispatch_name));
    const id4_ideogram4_dit_program_config_value_t tail_config_values[] = {
        {IREE_SV("id4.ideogram4.linear_tail.token_count"), token_capacity},
        {IREE_SV("id4.ideogram4.linear_tail.token_offset"), tail_token_offset},
        {IREE_SV("id4.ideogram4.linear_tail.dispatch_token_count"),
         tail_dispatch_token_count},
        {IREE_SV("id4.ideogram4.linear_tail.input_size"), input_size},
        {IREE_SV("id4.ideogram4.linear_tail.output_size"), output_size},
    };
    char tail_value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                           [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
    id4_pipeline_kernel_config_binding_t
        tail_config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
        IREE_ARRAYSIZE(tail_config_values), tail_config_values,
        tail_value_buffers, tail_config_bindings));
    id4_pipeline_program_dispatch_binding_t tail_bindings[] = {
        id4_pipeline_program_read(input),
        id4_pipeline_program_read(weight),
        tail_output_binding,
    };
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_loom(
        builder, tail_dispatch_name, tail.module_path, tail.function_name,
        IREE_ARRAYSIZE(tail_config_values), tail_config_bindings,
        IREE_ARRAYSIZE(tail_bindings), tail_bindings));
    tail_token_offset += tail_dispatch_token_count;
    remaining_tail_token_count -= tail_dispatch_token_count;
  }
  return iree_ok_status();
}

iree_status_t id4_ideogram4_dit_program_dispatch_linear_packed_bf16_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_linear_body_t body = {
      .token_block = ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK,
      .output_row_block = ID4_IDEOGRAM4_DIT_LINEAR_WMMA_OUTPUT_ROW_BLOCK,
      .module_path = IREE_SV("ideogram4/linear_bf16_f32_wmma"),
      .function_name = IREE_SV("id4_ideogram4_linear_bf16_f32_wmma"),
  };
  const id4_ideogram4_dit_program_linear_tail_t tail = {
      .dispatch_token_count_limit = 16,
      .module_path = IREE_SV("ideogram4/linear_bf16_f32_tail"),
      .function_name = IREE_SV("id4_ideogram4_linear_bf16_f32_tail"),
  };
  return id4_ideogram4_dit_program_dispatch_linear_packed_bf16(
      builder, name, token_count, token_capacity, input_size, output_size,
      input, weight, output,
      ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_OUTPUT_LAYOUT_F32_CANONICAL, body, tail);
}

iree_status_t id4_ideogram4_dit_program_dispatch_linear_packed_bf16_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_linear_body_t body =
      id4_ideogram4_dit_program_select_linear_bf16_bf16_body(token_capacity,
                                                             output_size);
  const id4_ideogram4_dit_program_linear_tail_t tail = {
      .dispatch_token_count_limit = 64,
      .module_path = IREE_SV("ideogram4/linear_bf16_bf16_tail"),
      .function_name = IREE_SV("id4_ideogram4_linear_bf16_bf16_tail"),
  };
  return id4_ideogram4_dit_program_dispatch_linear_packed_bf16(
      builder, name, token_count, token_capacity, input_size, output_size,
      input, weight, output,
      ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_OUTPUT_LAYOUT_BF16_PACKED, body, tail);
}

static iree_status_t
id4_ideogram4_dit_program_dispatch_linear_packed_bf16_bf16_compact_rhs_tile_body(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t weight, id4_pipeline_program_tensor_t output,
    id4_ideogram4_dit_program_linear_body_t body) {
  if ((token_capacity %
       ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_TOKEN_BLOCK) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 compact RHS linear token capacity %" PRIu32
        " must be a multiple of %u",
        token_capacity,
        ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_TOKEN_BLOCK);
  }
  if ((output_size %
       ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_COMPACT_OUTPUT_ROW_BLOCK) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 compact RHS linear output size %" PRIu32
        " must be a multiple of %u",
        output_size,
        ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_COMPACT_OUTPUT_ROW_BLOCK);
  }
  const id4_ideogram4_dit_program_linear_tail_t tail = {
      .dispatch_token_count_limit = 1,
      .module_path = iree_string_view_empty(),
      .function_name = iree_string_view_empty(),
  };
  return id4_ideogram4_dit_program_dispatch_linear_packed_bf16(
      builder, name, token_count, token_capacity, input_size, output_size,
      input, weight, output,
      ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_OUTPUT_LAYOUT_BF16_PACKED, body, tail);
}

iree_status_t
id4_ideogram4_dit_program_dispatch_linear_packed_bf16_bf16_compact_rhs_tile(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_linear_body_t body = {
      .token_block = ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_TOKEN_BLOCK,
      .output_row_block =
          ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_COMPACT_OUTPUT_ROW_BLOCK,
      .module_path = IREE_SV(
          "ideogram4/"
          "linear_bf16_bf16_wmma_compact_rhs_tile_m128n128_4wave_quadrant"),
      .function_name = IREE_SV("id4_ideogram4_linear_bf16_bf16_wmma_compact_"
                               "rhs_tile_m128n128_4wave_quadrant"),
  };
  return id4_ideogram4_dit_program_dispatch_linear_packed_bf16_bf16_compact_rhs_tile_body(
      builder, name, token_count, token_capacity, input_size, output_size,
      input, weight, output, body);
}

iree_status_t
id4_ideogram4_dit_program_dispatch_linear_packed_bf16_bf16_compact_rhs_tile_workgroup_staged(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_linear_body_t body = {
      .token_block = ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_TWO_WAVE_TOKEN_BLOCK,
      .output_row_block =
          ID4_IDEOGRAM4_DIT_LINEAR_BF16_BF16_COMPACT_OUTPUT_ROW_BLOCK,
      .module_path =
          IREE_SV("ideogram4/"
                  "linear_bf16_bf16_wmma_compact_rhs_tile_m128n128_4wave_"
                  "workgroup_staged"),
      .function_name =
          IREE_SV("id4_ideogram4_linear_bf16_bf16_wmma_compact_rhs_tile_"
                  "m128n128_4wave_workgroup_staged"),
  };
  return id4_ideogram4_dit_program_dispatch_linear_packed_bf16_bf16_compact_rhs_tile_body(
      builder, name, token_count, token_capacity, input_size, output_size,
      input, weight, output, body);
}

iree_status_t id4_ideogram4_dit_program_dispatch_linear_packed_fp8_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t weight, id4_pipeline_program_tensor_t scale,
    id4_pipeline_program_tensor_t output) {
  if (token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 FP8 linear token count must be nonzero");
  }
  if (token_capacity < token_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 FP8 linear token capacity %" PRIu32
                            " is smaller than token count %" PRIu32,
                            token_capacity, token_count);
  }
  if ((token_capacity % ID4_IDEOGRAM4_DIT_LINEAR_WMMA_TOKEN_BLOCK) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 FP8 linear token capacity %" PRIu32
                            " must be a multiple of %u",
                            token_capacity,
                            ID4_IDEOGRAM4_DIT_LINEAR_WMMA_TOKEN_BLOCK);
  }
  if ((input_size % ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 FP8 linear input size %" PRIu32
                            " must be a multiple of %u",
                            input_size, ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK);
  }
  const id4_ideogram4_dit_program_linear_body_t body =
      id4_ideogram4_dit_program_select_linear_fp8_bf16_body(token_capacity,
                                                            output_size);
  if (body.token_block < ID4_IDEOGRAM4_DIT_LINEAR_WMMA_TOKEN_BLOCK ||
      (body.token_block % ID4_IDEOGRAM4_DIT_LINEAR_WMMA_TOKEN_BLOCK) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 FP8 linear body token block %" PRIu32
                            " must be a multiple of %u",
                            body.token_block,
                            ID4_IDEOGRAM4_DIT_LINEAR_WMMA_TOKEN_BLOCK);
  }
  if ((output_size % body.output_row_block) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 FP8 linear output size %" PRIu32
                            " must be a multiple of %u",
                            output_size, body.output_row_block);
  }
  const uint32_t body_token_count =
      token_capacity - (token_capacity % body.token_block);
  const id4_ideogram4_dit_program_linear_tail_t tail = {
      .dispatch_token_count_limit = ID4_IDEOGRAM4_DIT_LINEAR_WMMA_TOKEN_BLOCK,
      .module_path = IREE_SV("ideogram4/linear_fp8_bf16_wmma"),
      .function_name = IREE_SV("id4_ideogram4_linear_fp8_bf16_wmma"),
  };

  if (body_token_count != 0) {
    id4_pipeline_program_dispatch_binding_t output_binding;
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_linear_output_binding(
        ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_OUTPUT_LAYOUT_BF16_PACKED, output,
        /*token_offset=*/0, body_token_count, output_size, &output_binding));
    char dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t dispatch_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
        name, IREE_SV("wmma"), dispatch_name_buffer,
        IREE_ARRAYSIZE(dispatch_name_buffer), &dispatch_name));
    const id4_ideogram4_dit_program_config_value_t config_values[] = {
        {IREE_SV("id4.ideogram4.linear_fp8_wmma.token_count"), token_capacity},
        {IREE_SV("id4.ideogram4.linear_fp8_wmma.token_offset"), 0},
        {IREE_SV("id4.ideogram4.linear_fp8_wmma.dispatch_token_count"),
         body_token_count},
        {IREE_SV("id4.ideogram4.linear_fp8_wmma.input_size"), input_size},
        {IREE_SV("id4.ideogram4.linear_fp8_wmma.output_size"), output_size},
    };
    char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                      [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
    id4_pipeline_kernel_config_binding_t
        config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
        IREE_ARRAYSIZE(config_values), config_values, value_buffers,
        config_bindings));
    id4_pipeline_program_dispatch_binding_t bindings[] = {
        id4_pipeline_program_read(input),
        id4_pipeline_program_read(weight),
        id4_pipeline_program_read(scale),
        output_binding,
    };
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_loom(
        builder, dispatch_name, body.module_path, body.function_name,
        IREE_ARRAYSIZE(config_values), config_bindings,
        IREE_ARRAYSIZE(bindings), bindings));
  }

  uint32_t remaining_tail_token_count = token_capacity - body_token_count;
  uint32_t tail_token_offset = body_token_count;
  for (uint32_t tail_ordinal = 0; remaining_tail_token_count != 0;
       ++tail_ordinal) {
    const uint32_t tail_dispatch_token_count =
        remaining_tail_token_count > tail.dispatch_token_count_limit
            ? tail.dispatch_token_count_limit
            : remaining_tail_token_count;
    id4_pipeline_program_dispatch_binding_t output_binding;
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_linear_output_binding(
        ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_OUTPUT_LAYOUT_BF16_PACKED, output,
        tail_token_offset, tail_dispatch_token_count, output_size,
        &output_binding));
    char tail_suffix_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t tail_suffix = IREE_SV("tail");
    if (remaining_tail_token_count > tail.dispatch_token_count_limit) {
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
          tail_suffix_buffer, IREE_ARRAYSIZE(tail_suffix_buffer), &tail_suffix,
          "tail%" PRIu32, tail_ordinal));
    }
    char dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t dispatch_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
        name, tail_suffix, dispatch_name_buffer,
        IREE_ARRAYSIZE(dispatch_name_buffer), &dispatch_name));
    const id4_ideogram4_dit_program_config_value_t config_values[] = {
        {IREE_SV("id4.ideogram4.linear_fp8_wmma.token_count"), token_capacity},
        {IREE_SV("id4.ideogram4.linear_fp8_wmma.token_offset"),
         tail_token_offset},
        {IREE_SV("id4.ideogram4.linear_fp8_wmma.dispatch_token_count"),
         tail_dispatch_token_count},
        {IREE_SV("id4.ideogram4.linear_fp8_wmma.input_size"), input_size},
        {IREE_SV("id4.ideogram4.linear_fp8_wmma.output_size"), output_size},
    };
    char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                      [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
    id4_pipeline_kernel_config_binding_t
        config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
        IREE_ARRAYSIZE(config_values), config_values, value_buffers,
        config_bindings));
    id4_pipeline_program_dispatch_binding_t bindings[] = {
        id4_pipeline_program_read(input),
        id4_pipeline_program_read(weight),
        id4_pipeline_program_read(scale),
        output_binding,
    };
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_loom(
        builder, dispatch_name, tail.module_path, tail.function_name,
        IREE_ARRAYSIZE(config_values), config_bindings,
        IREE_ARRAYSIZE(bindings), bindings));
    tail_token_offset += tail_dispatch_token_count;
    remaining_tail_token_count -= tail_dispatch_token_count;
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_pack_linear_input_f32_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t* out_packed_input) {
  if (token_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 linear input pack token count must be nonzero");
  }
  if (token_capacity < token_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 linear input pack token capacity %" PRIu32
        " is smaller than token count %" PRIu32,
        token_capacity, token_count);
  }
  uint64_t input_element_count_u64 = 0;
  if (!id4_ideogram4_dit_program_checked_mul_u64(token_capacity, input_size,
                                                 &input_element_count_u64)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 linear input element count overflow");
  }
  if (input_element_count_u64 >
      ID4_IDEOGRAM4_DIT_LINEAR_INPUT_PACK_MAX_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram4 linear input element count %" PRIu64 " exceeds max %u",
        input_element_count_u64,
        ID4_IDEOGRAM4_DIT_LINEAR_INPUT_PACK_MAX_ELEMENT_COUNT);
  }
  const uint32_t input_element_count = (uint32_t)input_element_count_u64;

  char packed_input_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char pack_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char after_pack_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t packed_input_name = iree_string_view_empty();
  iree_string_view_t pack_dispatch_name = iree_string_view_empty();
  iree_string_view_t after_pack_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      name, IREE_SV("input_bf16"), packed_input_name_buffer,
      IREE_ARRAYSIZE(packed_input_name_buffer), &packed_input_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      name, IREE_SV("input_pack"), pack_dispatch_name_buffer,
      IREE_ARRAYSIZE(pack_dispatch_name_buffer), &pack_dispatch_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      name, IREE_SV("after_input_pack"), after_pack_name_buffer,
      IREE_ARRAYSIZE(after_pack_name_buffer), &after_pack_name));

  id4_pipeline_program_tensor_t packed_input =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, packed_input_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_capacity, input_size),
      &packed_input));

  const id4_ideogram4_dit_program_config_value_t pack_config_values[] = {
      {IREE_SV("id4.ideogram4.linear_input_pack.token_count"), token_count},
      {IREE_SV("id4.ideogram4.linear_input_pack.token_capacity"),
       token_capacity},
      {IREE_SV("id4.ideogram4.linear_input_pack.input_size"), input_size},
      {IREE_SV("id4.ideogram4.linear_input_pack.element_count"),
       input_element_count},
  };
  char pack_value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                         [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      pack_config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(pack_config_values), pack_config_values,
      pack_value_buffers, pack_config_bindings));
  id4_pipeline_program_dispatch_binding_t pack_bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_write(packed_input),
  };
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_loom(
      builder, pack_dispatch_name,
      IREE_SV("ideogram4/linear_input_pack_f32_bf16"),
      IREE_SV("id4_ideogram4_linear_input_pack_f32_bf16"),
      IREE_ARRAYSIZE(pack_config_values), pack_config_bindings,
      IREE_ARRAYSIZE(pack_bindings), pack_bindings));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_pack_name));
  *out_packed_input = packed_input;
  return iree_ok_status();
}

iree_status_t id4_ideogram4_dit_program_dispatch_linear_input_unpack_bf16_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t output) {
  uint64_t element_count64 = 0;
  if (!id4_ideogram4_dit_program_checked_mul_u64(token_count, input_size,
                                                 &element_count64) ||
      element_count64 > ID4_IDEOGRAM4_DIT_LINEAR_INPUT_PACK_MAX_ELEMENT_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 DiT linear input unpack size overflow");
  }
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.linear_input_unpack.token_count"), token_count},
      {IREE_SV("id4.ideogram4.linear_input_unpack.token_capacity"),
       token_capacity},
      {IREE_SV("id4.ideogram4.linear_input_unpack.input_size"), input_size},
      {IREE_SV("id4.ideogram4.linear_input_unpack.element_count"),
       (uint32_t)element_count64},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/linear_input_unpack_bf16_f32"),
      IREE_SV("id4_ideogram4_linear_input_unpack_bf16_f32"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_linear_input_unpack_f32_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t output) {
  uint64_t element_count64 = 0;
  if (!id4_ideogram4_dit_program_checked_mul_u64(token_count, input_size,
                                                 &element_count64) ||
      element_count64 > ID4_IDEOGRAM4_DIT_LINEAR_INPUT_PACK_MAX_ELEMENT_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 DiT linear input unpack size overflow");
  }
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.linear_input_unpack.token_count"), token_count},
      {IREE_SV("id4.ideogram4.linear_input_unpack.token_capacity"),
       token_count},
      {IREE_SV("id4.ideogram4.linear_input_unpack.input_size"), input_size},
      {IREE_SV("id4.ideogram4.linear_input_unpack.element_count"),
       (uint32_t)element_count64},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/linear_input_unpack_bf16_f32"),
      IREE_SV("id4_ideogram4_linear_input_unpack_f32_f32"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_linear_bf16_from_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t output,
    id4_ideogram4_dit_program_linear_output_layout_t output_layout,
    id4_ideogram4_dit_program_linear_body_t body,
    id4_ideogram4_dit_program_linear_tail_t tail) {
  id4_pipeline_program_tensor_t packed_input =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_pack_linear_input_f32_bf16(
      builder, name, token_count, token_count, input_size, input,
      &packed_input));
  return id4_ideogram4_dit_program_dispatch_linear_packed_bf16(
      builder, name, token_count, token_count, input_size, output_size,
      packed_input, weight, output, output_layout, body, tail);
}

iree_status_t id4_ideogram4_dit_program_dispatch_linear_bf16_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_linear_body_t body = {
      .token_block = ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK,
      .output_row_block = ID4_IDEOGRAM4_DIT_LINEAR_WMMA_OUTPUT_ROW_BLOCK,
      .module_path = IREE_SV("ideogram4/linear_bf16_f32_wmma"),
      .function_name = IREE_SV("id4_ideogram4_linear_bf16_f32_wmma"),
  };
  const id4_ideogram4_dit_program_linear_tail_t tail = {
      .dispatch_token_count_limit = 16,
      .module_path = IREE_SV("ideogram4/linear_bf16_f32_tail"),
      .function_name = IREE_SV("id4_ideogram4_linear_bf16_f32_tail"),
  };
  return id4_ideogram4_dit_program_dispatch_linear_bf16_from_f32(
      builder, name, token_count, input_size, output_size, input, weight,
      output, ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_OUTPUT_LAYOUT_F32_CANONICAL,
      body, tail);
}

iree_status_t id4_ideogram4_dit_program_dispatch_linear_bf16_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_linear_body_t body =
      id4_ideogram4_dit_program_select_linear_bf16_bf16_body(token_count,
                                                             output_size);
  const id4_ideogram4_dit_program_linear_tail_t tail = {
      .dispatch_token_count_limit = 64,
      .module_path = IREE_SV("ideogram4/linear_bf16_bf16_tail"),
      .function_name = IREE_SV("id4_ideogram4_linear_bf16_bf16_tail"),
  };
  return id4_ideogram4_dit_program_dispatch_linear_bf16_from_f32(
      builder, name, token_count, input_size, output_size, input, weight,
      output, ID4_IDEOGRAM4_DIT_PROGRAM_LINEAR_OUTPUT_LAYOUT_BF16_PACKED, body,
      tail);
}

iree_status_t id4_ideogram4_dit_program_dispatch_linear_fp8_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t weight, id4_pipeline_program_tensor_t scale,
    id4_pipeline_program_tensor_t output) {
  id4_pipeline_program_tensor_t packed_input =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_pack_linear_input_f32_bf16(
      builder, name, token_count, token_capacity, input_size, input,
      &packed_input));
  return id4_ideogram4_dit_program_dispatch_linear_packed_fp8_bf16(
      builder, name, token_count, token_capacity, input_size, output_size,
      packed_input, weight, scale, output);
}

iree_status_t id4_ideogram4_dit_program_dispatch_linear_fp8_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t scale, id4_pipeline_program_tensor_t output) {
  if (token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 FP8 linear token count must be nonzero");
  }
  if (input_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 FP8 linear input size must be nonzero");
  }
  if (output_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 FP8 linear output size must be nonzero");
  }

  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.linear_fp8.token_count"), token_count},
      {IREE_SV("id4.ideogram4.linear_fp8.token_offset"), 0},
      {IREE_SV("id4.ideogram4.linear_fp8.dispatch_token_count"), token_count},
      {IREE_SV("id4.ideogram4.linear_fp8.input_size"), input_size},
      {IREE_SV("id4.ideogram4.linear_fp8.output_size"), output_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_read(scale),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/linear_fp8_f32"),
      IREE_SV("id4_ideogram4_linear_fp8_f32"), IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_silu(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t element_count, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.silu.element_count"), element_count},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/silu_f32"),
      IREE_SV("id4_ideogram4_silu_f32"), IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_silu_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t element_count, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.silu_bf16.element_count"), element_count},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/silu_bf16"),
      IREE_SV("id4_ideogram4_silu_bf16"), IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_modulated_layernorm(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t hidden_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t scale,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.modulated_layernorm.token_count"), token_count},
      {IREE_SV("id4.ideogram4.modulated_layernorm.hidden_size"), hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(scale),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/modulated_layernorm_f32"),
      IREE_SV("id4_ideogram4_modulated_layernorm_f32"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t
id4_ideogram4_dit_program_dispatch_modulated_layernorm_bf16_input_token_major(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t hidden_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t scale,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.modulated_layernorm_bf16.token_count"),
       token_count},
      {IREE_SV("id4.ideogram4.modulated_layernorm_bf16.hidden_size"),
       hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(scale),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/modulated_layernorm_bf16"),
      IREE_SV("id4_ideogram4_modulated_layernorm_bf16_input_token_major"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t
id4_ideogram4_dit_program_dispatch_modulated_layernorm_image_bf16_input_token_major(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_offset, uint32_t dispatch_token_count,
    uint32_t hidden_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t scale, id4_pipeline_program_tensor_t output) {
  if (token_offset > token_count ||
      dispatch_token_count > token_count - token_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram4 DiT BF16 token-major image layernorm dispatch token range "
        "overflow");
  }
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.modulated_layernorm_image_bf16.token_count"),
       token_count},
      {IREE_SV("id4.ideogram4.modulated_layernorm_image_bf16.token_offset"),
       token_offset},
      {IREE_SV("id4.ideogram4.modulated_layernorm_image_bf16."
               "dispatch_token_count"),
       dispatch_token_count},
      {IREE_SV("id4.ideogram4.modulated_layernorm_image_bf16.hidden_size"),
       hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(scale),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/modulated_layernorm_image_bf16"),
      IREE_SV("id4_ideogram4_modulated_layernorm_image_bf16_input_token_major"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_cast(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    iree_string_view_t config_key, iree_string_view_t module_path,
    iree_string_view_t function_name, id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t output) {
  uint32_t element_count = 0;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_shape_element_count_u32(
      shape, name, &element_count));
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {config_key, element_count},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, module_path, function_name, IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_cast_f32_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_shape_t shape, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t output) {
  return id4_ideogram4_dit_program_dispatch_cast(
      builder, name, IREE_SV("id4.elementwise.cast_f32_bf16.element_count"),
      IREE_SV("elementwise/cast_f32_bf16"),
      IREE_SV("id4_elementwise_cast_f32_bf16"), shape, input, output);
}

iree_status_t id4_ideogram4_dit_program_dispatch_cast_bf16_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_shape_t shape, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t output) {
  return id4_ideogram4_dit_program_dispatch_cast(
      builder, name, IREE_SV("id4.elementwise.cast_bf16_f32.element_count"),
      IREE_SV("elementwise/cast_bf16_f32"),
      IREE_SV("id4_elementwise_cast_bf16_f32"), shape, input, output);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_linear_bias_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_offset, uint32_t dispatch_token_count,
    uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t input,
    id4_ideogram4_dit_program_linear_parameter_t weight,
    id4_pipeline_program_tensor_t bias, id4_pipeline_program_tensor_t output) {
  if (token_offset > token_count ||
      dispatch_token_count > token_count - token_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram4 DiT linear-bias dispatch token range overflow");
  }
  switch (weight.storage) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16: {
      const id4_ideogram4_dit_program_config_value_t config_values[] = {
          {IREE_SV("id4.ideogram4.linear_bias.token_count"), token_count},
          {IREE_SV("id4.ideogram4.linear_bias.token_offset"), token_offset},
          {IREE_SV("id4.ideogram4.linear_bias.dispatch_token_count"),
           dispatch_token_count},
          {IREE_SV("id4.ideogram4.linear_bias.input_size"), input_size},
          {IREE_SV("id4.ideogram4.linear_bias.output_size"), output_size},
      };
      char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                        [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
      id4_pipeline_kernel_config_binding_t
          config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
          IREE_ARRAYSIZE(config_values), config_values, value_buffers,
          config_bindings));
      id4_pipeline_program_dispatch_binding_t bindings[] = {
          id4_pipeline_program_read(input),
          id4_pipeline_program_read(weight.weight),
          id4_pipeline_program_read(bias),
          id4_pipeline_program_write(output),
      };
      return id4_ideogram4_dit_program_dispatch_loom(
          builder, name, IREE_SV("ideogram4/linear_bias_bf16_f32"),
          IREE_SV("id4_ideogram4_linear_bias_bf16_f32"),
          IREE_ARRAYSIZE(config_values), config_bindings,
          IREE_ARRAYSIZE(bindings), bindings);
    }
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED: {
      const id4_ideogram4_dit_program_config_value_t config_values[] = {
          {IREE_SV("id4.ideogram4.linear_bias_fp8.token_count"), token_count},
          {IREE_SV("id4.ideogram4.linear_bias_fp8.token_offset"), token_offset},
          {IREE_SV("id4.ideogram4.linear_bias_fp8.dispatch_token_count"),
           dispatch_token_count},
          {IREE_SV("id4.ideogram4.linear_bias_fp8.input_size"), input_size},
          {IREE_SV("id4.ideogram4.linear_bias_fp8.output_size"), output_size},
      };
      char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                        [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
      id4_pipeline_kernel_config_binding_t
          config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
          IREE_ARRAYSIZE(config_values), config_values, value_buffers,
          config_bindings));
      id4_pipeline_program_dispatch_binding_t bindings[] = {
          id4_pipeline_program_read(input),
          id4_pipeline_program_read(weight.weight),
          id4_pipeline_program_read(weight.scale),
          id4_pipeline_program_read(bias),
          id4_pipeline_program_write(output),
      };
      return id4_ideogram4_dit_program_dispatch_loom(
          builder, name, IREE_SV("ideogram4/linear_bias_fp8_f32"),
          IREE_SV("id4_ideogram4_linear_bias_fp8_f32"),
          IREE_ARRAYSIZE(config_values), config_bindings,
          IREE_ARRAYSIZE(bindings), bindings);
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "Ideogram4 DiT linear-bias storage %" PRIu32
                              " is not supported",
                              (uint32_t)weight.storage);
  }
}

static iree_status_t id4_ideogram4_dit_program_dispatch_linear_bias_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_offset, uint32_t dispatch_token_count,
    uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t input,
    id4_ideogram4_dit_program_linear_parameter_t weight,
    id4_pipeline_program_tensor_t bias, id4_pipeline_program_tensor_t output) {
  if (token_offset > token_count ||
      dispatch_token_count > token_count - token_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram4 DiT BF16 linear-bias dispatch token range overflow");
  }
  switch (weight.storage) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16: {
      const id4_ideogram4_dit_program_config_value_t config_values[] = {
          {IREE_SV("id4.ideogram4.linear_bias_bf16.token_count"), token_count},
          {IREE_SV("id4.ideogram4.linear_bias_bf16.token_offset"),
           token_offset},
          {IREE_SV("id4.ideogram4.linear_bias_bf16.dispatch_token_count"),
           dispatch_token_count},
          {IREE_SV("id4.ideogram4.linear_bias_bf16.input_size"), input_size},
          {IREE_SV("id4.ideogram4.linear_bias_bf16.output_size"), output_size},
      };
      char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                        [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
      id4_pipeline_kernel_config_binding_t
          config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
          IREE_ARRAYSIZE(config_values), config_values, value_buffers,
          config_bindings));
      id4_pipeline_program_dispatch_binding_t bindings[] = {
          id4_pipeline_program_read(input),
          id4_pipeline_program_read(weight.weight),
          id4_pipeline_program_read(bias),
          id4_pipeline_program_write(output),
      };
      return id4_ideogram4_dit_program_dispatch_loom(
          builder, name, IREE_SV("ideogram4/linear_bias_bf16_bf16"),
          IREE_SV("id4_ideogram4_linear_bias_bf16_bf16"),
          IREE_ARRAYSIZE(config_values), config_bindings,
          IREE_ARRAYSIZE(bindings), bindings);
    }
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED: {
      char input_f32_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
      char input_cast_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
      char after_input_cast_name_buffer
          [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
      char output_f32_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
      char after_output_f32_name_buffer
          [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
      char output_cast_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
      iree_string_view_t input_f32_name = iree_string_view_empty();
      iree_string_view_t input_cast_name = iree_string_view_empty();
      iree_string_view_t after_input_cast_name = iree_string_view_empty();
      iree_string_view_t output_f32_name = iree_string_view_empty();
      iree_string_view_t after_output_f32_name = iree_string_view_empty();
      iree_string_view_t output_cast_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
          name, IREE_SV("input_f32"), input_f32_name_buffer,
          IREE_ARRAYSIZE(input_f32_name_buffer), &input_f32_name));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
          name, IREE_SV("cast_input_bf16_f32"), input_cast_name_buffer,
          IREE_ARRAYSIZE(input_cast_name_buffer), &input_cast_name));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
          name, IREE_SV("after_input_f32"), after_input_cast_name_buffer,
          IREE_ARRAYSIZE(after_input_cast_name_buffer),
          &after_input_cast_name));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
          name, IREE_SV("output_f32"), output_f32_name_buffer,
          IREE_ARRAYSIZE(output_f32_name_buffer), &output_f32_name));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
          name, IREE_SV("after_output_f32"), after_output_f32_name_buffer,
          IREE_ARRAYSIZE(after_output_f32_name_buffer),
          &after_output_f32_name));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
          name, IREE_SV("cast_output_f32_bf16"), output_cast_name_buffer,
          IREE_ARRAYSIZE(output_cast_name_buffer), &output_cast_name));

      id4_pipeline_program_tensor_t input_f32 =
          id4_pipeline_program_tensor_invalid();
      const id4_pipeline_program_shape_t input_shape =
          id4_pipeline_program_make_shape_rank2(input_size, token_count);
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
          builder, input_f32_name, ID4_PIPELINE_PROGRAM_DTYPE_F32, input_shape,
          &input_f32));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_cast_bf16_f32(
          builder, input_cast_name, input_shape, input, input_f32));
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_barrier(builder, after_input_cast_name));

      id4_pipeline_program_tensor_t output_f32 =
          id4_pipeline_program_tensor_invalid();
      const id4_pipeline_program_shape_t output_shape =
          id4_pipeline_program_make_shape_rank2(output_size,
                                                dispatch_token_count);
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
          builder, output_f32_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
          output_shape, &output_f32));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_linear_bias_f32(
          builder, name, token_count, token_offset, dispatch_token_count,
          input_size, output_size, input_f32, weight, bias, output_f32));
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_barrier(builder, after_output_f32_name));
      return id4_ideogram4_dit_program_dispatch_cast_f32_bf16(
          builder, output_cast_name, output_shape, output_f32, output);
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "Ideogram4 DiT BF16 linear-bias storage %" PRIu32
                              " is not supported",
                              (uint32_t)weight.storage);
  }
}

static iree_status_t
id4_ideogram4_dit_program_dispatch_linear_bias_packed_bf16_bf16_wmma(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t bias, id4_pipeline_program_tensor_t output) {
  if (token_count == 0 || token_count % 32 != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT BF16 linear-bias WMMA token count must be a "
        "nonzero multiple of 32");
  }
  if (input_size == 0 || input_size % 16 != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT BF16 linear-bias WMMA input size must be a "
        "nonzero multiple of 16");
  }
  if (output_size == 0 || output_size % 64 != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT BF16 linear-bias WMMA output size must be a "
        "nonzero multiple of 64");
  }
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.linear_bias_bf16_wmma.token_count"), token_count},
      {IREE_SV("id4.ideogram4.linear_bias_bf16_wmma.input_size"), input_size},
      {IREE_SV("id4.ideogram4.linear_bias_bf16_wmma.output_size"), output_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_read(bias),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/linear_bias_bf16_bf16_wmma"),
      IREE_SV("id4_ideogram4_linear_bias_bf16_bf16_wmma"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_unpatchify_scale(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t width, uint32_t height, uint32_t input_channel_count,
    uint32_t total_token_count, uint32_t output_token_offset,
    id4_pipeline_program_tensor_t projected,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.unpatchify_scale.width"), width},
      {IREE_SV("id4.ideogram4.unpatchify_scale.height"), height},
      {IREE_SV("id4.ideogram4.unpatchify_scale.input_channel_count"),
       input_channel_count},
      {IREE_SV("id4.ideogram4.unpatchify_scale.total_token_count"),
       total_token_count},
      {IREE_SV("id4.ideogram4.unpatchify_scale.output_token_offset"),
       output_token_offset},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(projected),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/unpatchify_scale_f32"),
      IREE_SV("id4_ideogram4_unpatchify_scale_f32"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t
id4_ideogram4_dit_program_dispatch_unpatchify_scale_packed_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t width, uint32_t height, uint32_t input_channel_count,
    id4_pipeline_program_tensor_t projected,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.unpatchify_scale_packed_bf16.width"), width},
      {IREE_SV("id4.ideogram4.unpatchify_scale_packed_bf16.height"), height},
      {IREE_SV("id4.ideogram4.unpatchify_scale_packed_bf16."
               "input_channel_count"),
       input_channel_count},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(projected),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/unpatchify_scale_packed_bf16"),
      IREE_SV("id4_ideogram4_unpatchify_scale_packed_bf16"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_qkv_split(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t hidden_size,
    id4_pipeline_program_tensor_t qkv, id4_pipeline_program_tensor_t query,
    id4_pipeline_program_tensor_t key, id4_pipeline_program_tensor_t value) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.qkv_split.token_count"), token_count},
      {IREE_SV("id4.ideogram4.qkv_split.hidden_size"), hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(qkv),
      id4_pipeline_program_write(query),
      id4_pipeline_program_write(key),
      id4_pipeline_program_write(value),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/qkv_split_f32"),
      IREE_SV("id4_ideogram4_qkv_split_f32"), IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

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
    id4_pipeline_program_tensor_t value) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.qkv_norm_rotary.token_count"), token_count},
      {IREE_SV("id4.ideogram4.qkv_norm_rotary.attention_head_count"),
       attention_head_count},
      {IREE_SV("id4.ideogram4.qkv_norm_rotary.head_size"), head_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(qkv),
      id4_pipeline_program_read(norm_q_weight),
      id4_pipeline_program_read(norm_k_weight),
      id4_pipeline_program_read(position_embedding),
      id4_pipeline_program_write(rotated_query),
      id4_pipeline_program_write(rotated_key),
      id4_pipeline_program_write(value),
  };
  iree_string_view_t module_path = iree_string_view_empty();
  iree_string_view_t function_name = iree_string_view_empty();
  switch (activation_format) {
    case ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL:
      module_path = IREE_SV("ideogram4/qkv_norm_rotary_f32");
      function_name = IREE_SV("id4_ideogram4_qkv_norm_rotary_f32");
      break;
    case ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT:
      module_path = IREE_SV("ideogram4/qkv_norm_rotary_bf16");
      function_name = IREE_SV("id4_ideogram4_qkv_norm_rotary_bf16");
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported activation format %u",
                              (uint32_t)activation_format);
  }
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, module_path, function_name, IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

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
    id4_pipeline_program_tensor_t value) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.qkv_norm_rotary.token_count"), token_count},
      {IREE_SV("id4.ideogram4.qkv_norm_rotary.attention_head_count"),
       attention_head_count},
      {IREE_SV("id4.ideogram4.qkv_norm_rotary.head_size"), head_size},
      {IREE_SV("id4.ideogram4.qkv_norm_rotary.value_token_stride"),
       token_capacity},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(qkv),
      id4_pipeline_program_read(norm_q_weight),
      id4_pipeline_program_read(norm_k_weight),
      id4_pipeline_program_read(position_embedding),
      id4_pipeline_program_write(rotated_query),
      id4_pipeline_program_write(rotated_key),
      id4_pipeline_program_write(value),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/qkv_norm_rotary_bf16"),
      IREE_SV("id4_ideogram4_qkv_norm_rotary_bf16_packed_value"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_head_rmsnorm(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.head_rmsnorm.token_count"), token_count},
      {IREE_SV("id4.ideogram4.head_rmsnorm.attention_head_count"),
       attention_head_count},
      {IREE_SV("id4.ideogram4.head_rmsnorm.head_size"), head_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/head_rmsnorm_f32"),
      IREE_SV("id4_ideogram4_head_rmsnorm_f32"), IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_rotary_apply(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t position_embedding,
    id4_pipeline_program_tensor_t rotated_query,
    id4_pipeline_program_tensor_t rotated_key) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.rotary_apply.token_count"), token_count},
      {IREE_SV("id4.ideogram4.rotary_apply.attention_head_count"),
       attention_head_count},
      {IREE_SV("id4.ideogram4.rotary_apply.head_size"), head_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(query),
      id4_pipeline_program_read(key),
      id4_pipeline_program_read(position_embedding),
      id4_pipeline_program_write(rotated_query),
      id4_pipeline_program_write(rotated_key),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/rotary_apply_f32"),
      IREE_SV("id4_ideogram4_rotary_apply_f32"), IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_attention_export(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    iree_string_view_t module_path, iree_string_view_t function_name,
    uint32_t token_count, uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t value, id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.attention_query_block8_rounded.token_count"),
       token_count},
      {IREE_SV(
           "id4.ideogram4.attention_query_block8_rounded.attention_head_count"),
       attention_head_count},
      {IREE_SV("id4.ideogram4.attention_query_block8_rounded.head_size"),
       head_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(query),
      id4_pipeline_program_read(key),
      id4_pipeline_program_read(value),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, module_path, function_name, IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_attention(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t value, id4_pipeline_program_tensor_t output) {
  return id4_ideogram4_dit_program_dispatch_attention_export(
      builder, name, IREE_SV("ideogram4/attention_query_block8_rounded_f32"),
      IREE_SV("id4_ideogram4_attention_query_block8_rounded_f32"), token_count,
      attention_head_count, head_size, query, key, value, output);
}

iree_status_t id4_ideogram4_dit_program_dispatch_attention_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t value, id4_pipeline_program_tensor_t output) {
  return id4_ideogram4_dit_program_dispatch_attention_export(
      builder, name, IREE_SV("ideogram4/attention_query_block8_rounded_bf16"),
      IREE_SV("id4_ideogram4_attention_query_block8_rounded_bf16_f32"),
      token_count, attention_head_count, head_size, query, key, value, output);
}

iree_status_t id4_ideogram4_dit_program_dispatch_attention_bf16_linear_input(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t value, id4_pipeline_program_tensor_t output) {
  return id4_ideogram4_dit_program_dispatch_attention_export(
      builder, name, IREE_SV("ideogram4/attention_query_block8_rounded_bf16"),
      IREE_SV("id4_ideogram4_attention_query_block8_rounded_bf16_linear_input"),
      token_count, attention_head_count, head_size, query, key, value, output);
}

static iree_status_t
id4_ideogram4_dit_program_dispatch_attention_materialized_export(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    iree_string_view_t function_name, uint32_t valid_token_count,
    uint32_t padded_token_count, uint32_t attention_head_count,
    uint32_t head_size, iree_host_size_t binding_count,
    id4_pipeline_program_dispatch_binding_t* bindings) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.attention_materialized_wmma.valid_token_count"),
       valid_token_count},
      {IREE_SV("id4.ideogram4.attention_materialized_wmma.padded_token_count"),
       padded_token_count},
      {IREE_SV(
           "id4.ideogram4.attention_materialized_wmma.attention_head_count"),
       attention_head_count},
      {IREE_SV("id4.ideogram4.attention_materialized_wmma.head_size"),
       head_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/attention_materialized_bf16_wmma"),
      function_name, IREE_ARRAYSIZE(config_values), config_bindings,
      binding_count, bindings);
}

iree_status_t
id4_ideogram4_dit_program_dispatch_attention_qk_scores_all_heads_bf16_f32_wmma(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t valid_token_count, uint32_t padded_token_count,
    uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t scores) {
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(query),
      id4_pipeline_program_read(key),
      id4_pipeline_program_write(scores),
  };
  return id4_ideogram4_dit_program_dispatch_attention_materialized_export(
      builder, name,
      IREE_SV("id4_ideogram4_attention_qk_scores_all_heads_bf16_f32_wmma"),
      valid_token_count, padded_token_count, attention_head_count, head_size,
      IREE_ARRAYSIZE(bindings), bindings);
}

iree_status_t
id4_ideogram4_dit_program_dispatch_attention_softmax_all_heads_f32_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t valid_token_count, uint32_t padded_token_count,
    uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t scores,
    id4_pipeline_program_tensor_t probabilities) {
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(scores),
      id4_pipeline_program_write(probabilities),
  };
  return id4_ideogram4_dit_program_dispatch_attention_materialized_export(
      builder, name,
      IREE_SV("id4_ideogram4_attention_softmax_all_heads_f32_bf16"),
      valid_token_count, padded_token_count, attention_head_count, head_size,
      IREE_ARRAYSIZE(bindings), bindings);
}

iree_status_t
id4_ideogram4_dit_program_dispatch_attention_pv_all_heads_bf16_bf16_wmma(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t valid_token_count, uint32_t padded_token_count,
    uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t probabilities,
    id4_pipeline_program_tensor_t value, id4_pipeline_program_tensor_t output) {
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(probabilities),
      id4_pipeline_program_read(value),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_attention_materialized_export(
      builder, name,
      IREE_SV("id4_ideogram4_attention_pv_all_heads_bf16_bf16_wmma"),
      valid_token_count, padded_token_count, attention_head_count, head_size,
      IREE_ARRAYSIZE(bindings), bindings);
}

static iree_status_t
id4_ideogram4_dit_program_dispatch_attention_blocked_export(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    iree_string_view_t function_name, uint32_t valid_token_count,
    uint32_t padded_token_count, uint32_t query_block_offset,
    uint32_t query_block_token_count, uint32_t attention_head_count,
    uint32_t head_size, iree_host_size_t binding_count,
    id4_pipeline_program_dispatch_binding_t* bindings) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.attention_blocked_wmma.valid_token_count"),
       valid_token_count},
      {IREE_SV("id4.ideogram4.attention_blocked_wmma.padded_token_count"),
       padded_token_count},
      {IREE_SV("id4.ideogram4.attention_blocked_wmma.query_block_offset"),
       query_block_offset},
      {IREE_SV("id4.ideogram4.attention_blocked_wmma.query_block_token_count"),
       query_block_token_count},
      {IREE_SV("id4.ideogram4.attention_blocked_wmma.attention_head_count"),
       attention_head_count},
      {IREE_SV("id4.ideogram4.attention_blocked_wmma.head_size"), head_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/attention_blocked_bf16_wmma"),
      function_name, IREE_ARRAYSIZE(config_values), config_bindings,
      binding_count, bindings);
}

iree_status_t
id4_ideogram4_dit_program_dispatch_attention_blocked_qk_scores_bf16_f32_wmma(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t valid_token_count, uint32_t padded_token_count,
    uint32_t query_block_offset, uint32_t query_block_token_count,
    uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t scores) {
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(query),
      id4_pipeline_program_read(key),
      id4_pipeline_program_write(scores),
  };
  return id4_ideogram4_dit_program_dispatch_attention_blocked_export(
      builder, name,
      IREE_SV("id4_ideogram4_attention_blocked_qk_scores_bf16_f32_wmma"),
      valid_token_count, padded_token_count, query_block_offset,
      query_block_token_count, attention_head_count, head_size,
      IREE_ARRAYSIZE(bindings), bindings);
}

iree_status_t
id4_ideogram4_dit_program_dispatch_attention_blocked_softmax_f32_bf16(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t valid_token_count, uint32_t padded_token_count,
    uint32_t query_block_offset, uint32_t query_block_token_count,
    uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t scores,
    id4_pipeline_program_tensor_t probabilities) {
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(scores),
      id4_pipeline_program_write(probabilities),
  };
  return id4_ideogram4_dit_program_dispatch_attention_blocked_export(
      builder, name,
      IREE_SV("id4_ideogram4_attention_blocked_softmax_f32_bf16"),
      valid_token_count, padded_token_count, query_block_offset,
      query_block_token_count, attention_head_count, head_size,
      IREE_ARRAYSIZE(bindings), bindings);
}

iree_status_t
id4_ideogram4_dit_program_dispatch_attention_blocked_pv_bf16_bf16_wmma(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t valid_token_count, uint32_t padded_token_count,
    uint32_t query_block_offset, uint32_t query_block_token_count,
    uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t probabilities,
    id4_pipeline_program_tensor_t value, id4_pipeline_program_tensor_t output) {
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(probabilities),
      id4_pipeline_program_read(value),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_attention_blocked_export(
      builder, name,
      IREE_SV("id4_ideogram4_attention_blocked_pv_bf16_bf16_wmma"),
      valid_token_count, padded_token_count, query_block_offset,
      query_block_token_count, attention_head_count, head_size,
      IREE_ARRAYSIZE(bindings), bindings);
}

iree_status_t
id4_ideogram4_dit_program_dispatch_attention_online_bf16_bf16_wmma(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t valid_token_count, uint32_t padded_token_count,
    uint32_t query_block_offset, uint32_t query_block_token_count,
    uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t value, id4_pipeline_program_tensor_t output) {
  if (head_size != 256) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Ideogram4 online WMMA attention head size %" PRIu32
                            " is not supported",
                            head_size);
  }
  iree_device_size_t hidden_size = 0;
  iree_device_size_t row_byte_length = 0;
  iree_device_size_t output_offset = 0;
  iree_device_size_t output_length = 0;
  if (!iree_device_size_checked_mul((iree_device_size_t)attention_head_count,
                                    (iree_device_size_t)head_size,
                                    &hidden_size) ||
      !iree_device_size_checked_mul(hidden_size,
                                    id4_pipeline_program_dtype_byte_length(
                                        ID4_PIPELINE_PROGRAM_DTYPE_BF16),
                                    &row_byte_length) ||
      !iree_device_size_checked_mul((iree_device_size_t)query_block_offset,
                                    row_byte_length, &output_offset) ||
      !iree_device_size_checked_mul((iree_device_size_t)query_block_token_count,
                                    row_byte_length, &output_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram4 online WMMA attention output write range overflow");
  }
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.attention_online_wmma.valid_token_count"),
       valid_token_count},
      {IREE_SV("id4.ideogram4.attention_online_wmma.padded_token_count"),
       padded_token_count},
      {IREE_SV("id4.ideogram4.attention_online_wmma.query_block_offset"),
       query_block_offset},
      {IREE_SV("id4.ideogram4.attention_online_wmma.query_block_token_count"),
       query_block_token_count},
      {IREE_SV("id4.ideogram4.attention_online_wmma.attention_head_count"),
       attention_head_count},
      {IREE_SV("id4.ideogram4.attention_online_wmma.head_size"), head_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(query),
      id4_pipeline_program_read(key),
      id4_pipeline_program_read(value),
      id4_pipeline_program_write_range(output, output_offset, output_length),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/attention_online_bf16_wmma"),
      IREE_SV("id4_ideogram4_attention_online_bf16_bf16_wmma"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_timestep_embedding(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t hidden_size, id4_pipeline_program_tensor_t timestep,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.timestep_embedding.embedding_size"), hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(timestep),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/timestep_embedding_f32"),
      IREE_SV("id4_ideogram4_timestep_embedding_f32"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_tap_bf16_as_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_shape_t shape, id4_pipeline_program_tensor_t input) {
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
      name, IREE_SV("canonical_cast"), dispatch_name_buffer,
      IREE_ARRAYSIZE(dispatch_name_buffer), &dispatch_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      name, IREE_SV("after_canonical_cast"), barrier_name_buffer,
      IREE_ARRAYSIZE(barrier_name_buffer), &barrier_name));

  id4_pipeline_program_tensor_t canonical_tensor =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, canonical_tensor_name, ID4_PIPELINE_PROGRAM_DTYPE_F32, shape,
      &canonical_tensor));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_cast_bf16_f32(
      builder, dispatch_name, shape, input, canonical_tensor));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, barrier_name));
  return id4_ideogram4_dit_program_tap(builder, name, canonical_tensor);
}

iree_status_t id4_ideogram4_dit_program_dense_f32(
    id4_pipeline_program_builder_t* builder,
    const id4_ideogram4_dit_program_dense_options_t* options,
    id4_pipeline_program_tensor_t* out_output) {
  if ((options->input_size % 4) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT dense input size must be a multiple of 4");
  }
  if (options->output_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT dense output size must be nonzero");
  }
  if (options->activation_kind != ID4_IDEOGRAM4_DIT_ACTIVATION_IDENTITY &&
      options->activation_kind != ID4_IDEOGRAM4_DIT_ACTIVATION_SILU) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT dense activation kind %" PRIu32
                            " is invalid",
                            options->activation_kind);
  }

  char weight_key_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char bias_key_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t weight_key = iree_string_view_empty();
  iree_string_view_t bias_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      weight_key_buffer, IREE_ARRAYSIZE(weight_key_buffer), &weight_key,
      "%.*s.weight", (int)options->parameter_prefix.size,
      options->parameter_prefix.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      bias_key_buffer, IREE_ARRAYSIZE(bias_key_buffer), &bias_key, "%.*s.bias",
      (int)options->parameter_prefix.size, options->parameter_prefix.data));

  id4_ideogram4_dit_program_linear_parameter_t weight = {
      .storage = ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_INVALID,
      .weight = id4_pipeline_program_tensor_invalid(),
      .scale = id4_pipeline_program_tensor_invalid(),
  };
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_linear_parameter(
      builder, options->parameter_sources, weight_key, options->input_size,
      options->output_size, options->weight_execution_format, &weight));
  id4_pipeline_program_tensor_t bias = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter_bf16(
      builder, options->parameter_sources, bias_key,
      id4_pipeline_program_make_shape_rank1(options->output_size), &bias));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, options->output_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(options->output_size), out_output));

  switch (weight.storage) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16: {
      const id4_ideogram4_dit_program_config_value_t config_values[] = {
          {IREE_SV("id4.ideogram4.dense_bias.input_size"), options->input_size},
          {IREE_SV("id4.ideogram4.dense_bias.output_size"),
           options->output_size},
          {IREE_SV("id4.ideogram4.dense_bias.activation_kind"),
           options->activation_kind},
      };
      char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                        [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
      id4_pipeline_kernel_config_binding_t
          config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
          IREE_ARRAYSIZE(config_values), config_values, value_buffers,
          config_bindings));
      id4_pipeline_program_dispatch_binding_t bindings[] = {
          id4_pipeline_program_read(options->input),
          id4_pipeline_program_read(weight.weight),
          id4_pipeline_program_read(bias),
          id4_pipeline_program_write(*out_output),
      };
      return id4_ideogram4_dit_program_dispatch_loom(
          builder, options->operation_name,
          IREE_SV("ideogram4/dense_bias_bf16_f32"),
          IREE_SV("id4_ideogram4_dense_bias_bf16_f32"),
          IREE_ARRAYSIZE(config_values), config_bindings,
          IREE_ARRAYSIZE(bindings), bindings);
    }
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED: {
      const id4_ideogram4_dit_program_config_value_t config_values[] = {
          {IREE_SV("id4.ideogram4.dense_bias_fp8.input_size"),
           options->input_size},
          {IREE_SV("id4.ideogram4.dense_bias_fp8.output_size"),
           options->output_size},
          {IREE_SV("id4.ideogram4.dense_bias_fp8.activation_kind"),
           options->activation_kind},
      };
      char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                        [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
      id4_pipeline_kernel_config_binding_t
          config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
          IREE_ARRAYSIZE(config_values), config_values, value_buffers,
          config_bindings));
      id4_pipeline_program_dispatch_binding_t bindings[] = {
          id4_pipeline_program_read(options->input),
          id4_pipeline_program_read(weight.weight),
          id4_pipeline_program_read(weight.scale),
          id4_pipeline_program_read(bias),
          id4_pipeline_program_write(*out_output),
      };
      return id4_ideogram4_dit_program_dispatch_loom(
          builder, options->operation_name,
          IREE_SV("ideogram4/dense_bias_fp8_f32"),
          IREE_SV("id4_ideogram4_dense_bias_fp8_f32"),
          IREE_ARRAYSIZE(config_values), config_bindings,
          IREE_ARRAYSIZE(bindings), bindings);
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "Ideogram4 DiT dense parameter `%.*s` storage "
                              "%" PRIu32 " is not supported",
                              (int)weight_key.size, weight_key.data,
                              (uint32_t)weight.storage);
  }
}

iree_status_t id4_ideogram4_dit_program_dense_bf16(
    id4_pipeline_program_builder_t* builder,
    const id4_ideogram4_dit_program_dense_options_t* options,
    id4_pipeline_program_tensor_t* out_output) {
  if ((options->input_size % 4) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT BF16 dense input size must be a multiple of 4");
  }
  if (options->output_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT BF16 dense output size must be "
                            "nonzero");
  }
  if (options->activation_kind != ID4_IDEOGRAM4_DIT_ACTIVATION_IDENTITY &&
      options->activation_kind != ID4_IDEOGRAM4_DIT_ACTIVATION_SILU) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT BF16 dense activation kind %" PRIu32
                            " is invalid",
                            options->activation_kind);
  }

  char weight_key_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char bias_key_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t weight_key = iree_string_view_empty();
  iree_string_view_t bias_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      weight_key_buffer, IREE_ARRAYSIZE(weight_key_buffer), &weight_key,
      "%.*s.weight", (int)options->parameter_prefix.size,
      options->parameter_prefix.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      bias_key_buffer, IREE_ARRAYSIZE(bias_key_buffer), &bias_key, "%.*s.bias",
      (int)options->parameter_prefix.size, options->parameter_prefix.data));

  id4_ideogram4_dit_program_linear_parameter_t weight = {
      .storage = ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_INVALID,
      .weight = id4_pipeline_program_tensor_invalid(),
      .scale = id4_pipeline_program_tensor_invalid(),
  };
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_linear_parameter(
      builder, options->parameter_sources, weight_key, options->input_size,
      options->output_size, options->weight_execution_format, &weight));
  id4_pipeline_program_tensor_t bias = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter_bf16(
      builder, options->parameter_sources, bias_key,
      id4_pipeline_program_make_shape_rank1(options->output_size), &bias));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, options->output_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(options->output_size), out_output));

  switch (weight.storage) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16: {
      const id4_ideogram4_dit_program_config_value_t config_values[] = {
          {IREE_SV("id4.ideogram4.dense_bias_bf16.input_size"),
           options->input_size},
          {IREE_SV("id4.ideogram4.dense_bias_bf16.output_size"),
           options->output_size},
          {IREE_SV("id4.ideogram4.dense_bias_bf16.activation_kind"),
           options->activation_kind},
      };
      char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                        [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
      id4_pipeline_kernel_config_binding_t
          config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
          IREE_ARRAYSIZE(config_values), config_values, value_buffers,
          config_bindings));
      id4_pipeline_program_dispatch_binding_t bindings[] = {
          id4_pipeline_program_read(options->input),
          id4_pipeline_program_read(weight.weight),
          id4_pipeline_program_read(bias),
          id4_pipeline_program_write(*out_output),
      };
      return id4_ideogram4_dit_program_dispatch_loom(
          builder, options->operation_name,
          IREE_SV("ideogram4/dense_bias_bf16_bf16"),
          IREE_SV("id4_ideogram4_dense_bias_bf16_bf16"),
          IREE_ARRAYSIZE(config_values), config_bindings,
          IREE_ARRAYSIZE(bindings), bindings);
    }
    case ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED: {
      char f32_input_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
      char input_cast_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
      char
          after_f32_input_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
      char f32_output_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
      char cast_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
      char after_f32_output_name_buffer
          [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
      iree_string_view_t f32_input_name = iree_string_view_empty();
      iree_string_view_t input_cast_name = iree_string_view_empty();
      iree_string_view_t after_f32_input_name = iree_string_view_empty();
      iree_string_view_t f32_output_name = iree_string_view_empty();
      iree_string_view_t cast_dispatch_name = iree_string_view_empty();
      iree_string_view_t after_f32_output_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
          options->operation_name, IREE_SV("input_f32"), f32_input_name_buffer,
          IREE_ARRAYSIZE(f32_input_name_buffer), &f32_input_name));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
          options->operation_name, IREE_SV("cast_input_bf16_f32"),
          input_cast_name_buffer, IREE_ARRAYSIZE(input_cast_name_buffer),
          &input_cast_name));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
          options->operation_name, IREE_SV("after_input_f32"),
          after_f32_input_name_buffer,
          IREE_ARRAYSIZE(after_f32_input_name_buffer), &after_f32_input_name));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
          options->output_name, IREE_SV("f32"), f32_output_name_buffer,
          IREE_ARRAYSIZE(f32_output_name_buffer), &f32_output_name));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
          options->operation_name, IREE_SV("cast_f32_bf16"),
          cast_dispatch_name_buffer, IREE_ARRAYSIZE(cast_dispatch_name_buffer),
          &cast_dispatch_name));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
          options->operation_name, IREE_SV("after_f32"),
          after_f32_output_name_buffer,
          IREE_ARRAYSIZE(after_f32_output_name_buffer),
          &after_f32_output_name));
      id4_pipeline_program_tensor_t f32_input =
          id4_pipeline_program_tensor_invalid();
      const id4_pipeline_program_shape_t input_shape =
          id4_pipeline_program_make_shape_rank1(options->input_size);
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
          builder, f32_input_name, ID4_PIPELINE_PROGRAM_DTYPE_F32, input_shape,
          &f32_input));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_cast_bf16_f32(
          builder, input_cast_name, input_shape, options->input, f32_input));
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_barrier(builder, after_f32_input_name));

      id4_pipeline_program_tensor_t f32_output =
          id4_pipeline_program_tensor_invalid();
      const id4_pipeline_program_shape_t output_shape =
          id4_pipeline_program_make_shape_rank1(options->output_size);
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
          builder, f32_output_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
          output_shape, &f32_output));

      const id4_ideogram4_dit_program_config_value_t config_values[] = {
          {IREE_SV("id4.ideogram4.dense_bias_fp8.input_size"),
           options->input_size},
          {IREE_SV("id4.ideogram4.dense_bias_fp8.output_size"),
           options->output_size},
          {IREE_SV("id4.ideogram4.dense_bias_fp8.activation_kind"),
           options->activation_kind},
      };
      char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                        [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
      id4_pipeline_kernel_config_binding_t
          config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
          IREE_ARRAYSIZE(config_values), config_values, value_buffers,
          config_bindings));
      id4_pipeline_program_dispatch_binding_t bindings[] = {
          id4_pipeline_program_read(f32_input),
          id4_pipeline_program_read(weight.weight),
          id4_pipeline_program_read(weight.scale),
          id4_pipeline_program_read(bias),
          id4_pipeline_program_write(f32_output),
      };
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_loom(
          builder, options->operation_name,
          IREE_SV("ideogram4/dense_bias_fp8_f32"),
          IREE_SV("id4_ideogram4_dense_bias_fp8_f32"),
          IREE_ARRAYSIZE(config_values), config_bindings,
          IREE_ARRAYSIZE(bindings), bindings));
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_barrier(builder, after_f32_output_name));
      return id4_ideogram4_dit_program_dispatch_cast_f32_bf16(
          builder, cast_dispatch_name, output_shape, f32_output, *out_output);
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "Ideogram4 DiT BF16 dense parameter `%.*s` "
                              "storage %" PRIu32 " is not supported",
                              (int)weight_key.size, weight_key.data,
                              (uint32_t)weight.storage);
  }
}

iree_status_t id4_ideogram4_dit_program_dense_bf16_rounded_f32(
    id4_pipeline_program_builder_t* builder,
    const id4_ideogram4_dit_program_dense_options_t* options,
    id4_pipeline_program_tensor_t* out_output) {
  if ((options->input_size % 4) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT rounded-F32 dense input size must be a multiple of 4");
  }
  if (options->output_size == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT rounded-F32 dense output size must be nonzero");
  }
  if (options->activation_kind != ID4_IDEOGRAM4_DIT_ACTIVATION_IDENTITY &&
      options->activation_kind != ID4_IDEOGRAM4_DIT_ACTIVATION_SILU) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT rounded-F32 dense activation kind %" PRIu32
        " is invalid",
        options->activation_kind);
  }

  char weight_key_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char bias_key_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t weight_key = iree_string_view_empty();
  iree_string_view_t bias_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      weight_key_buffer, IREE_ARRAYSIZE(weight_key_buffer), &weight_key,
      "%.*s.weight", (int)options->parameter_prefix.size,
      options->parameter_prefix.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      bias_key_buffer, IREE_ARRAYSIZE(bias_key_buffer), &bias_key, "%.*s.bias",
      (int)options->parameter_prefix.size, options->parameter_prefix.data));

  id4_ideogram4_dit_program_linear_parameter_t weight = {
      .storage = ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_INVALID,
      .weight = id4_pipeline_program_tensor_invalid(),
      .scale = id4_pipeline_program_tensor_invalid(),
  };
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_linear_parameter(
      builder, options->parameter_sources, weight_key, options->input_size,
      options->output_size, options->weight_execution_format, &weight));
  if (weight.storage != ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram4 DiT rounded-F32 dense parameter `%.*s` storage %" PRIu32
        " is not supported",
        (int)weight_key.size, weight_key.data, (uint32_t)weight.storage);
  }

  id4_pipeline_program_tensor_t bias = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter_bf16(
      builder, options->parameter_sources, bias_key,
      id4_pipeline_program_make_shape_rank1(options->output_size), &bias));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, options->output_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(options->output_size), out_output));

  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.dense_bias_bf16.input_size"),
       options->input_size},
      {IREE_SV("id4.ideogram4.dense_bias_bf16.output_size"),
       options->output_size},
      {IREE_SV("id4.ideogram4.dense_bias_bf16.activation_kind"),
       options->activation_kind},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(options->input),
      id4_pipeline_program_read(weight.weight),
      id4_pipeline_program_read(bias),
      id4_pipeline_program_write(*out_output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, options->operation_name,
      IREE_SV("ideogram4/dense_bias_bf16_bf16"),
      IREE_SV("id4_ideogram4_dense_bias_bf16_bf16_rounded_f32"),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_ideogram4_dit_program_author_final_output(
    const id4_ideogram4_dit_program_options_t* options,
    id4_pipeline_program_builder_t* builder, iree_string_view_t branch_name,
    uint32_t total_token_count, id4_pipeline_program_tensor_t hidden,
    id4_pipeline_program_tensor_t adaln_input) {
  const bool f32_canonical_activations =
      options->activation_format ==
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL;
  const uint32_t input_channel_count = options->model.input_channel_count;
  const uint32_t hidden_size = options->model.hidden_size;
  const uint32_t adaln_size = options->model.adaln_size;
  const uint32_t text_token_count = options->request.text_token_count;
  const uint32_t width = (uint32_t)options->request.latent_shape.dims[0];
  const uint32_t height = (uint32_t)options->request.latent_shape.dims[1];
  uint64_t image_token_count64 = 0;
  if (!id4_ideogram4_dit_program_checked_mul_u64(width, height,
                                                 &image_token_count64) ||
      image_token_count64 > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 DiT final image token count overflow");
  }
  const uint32_t image_token_count = (uint32_t)image_token_count64;

  char final_silu_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char after_final_silu_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t final_silu_name = iree_string_view_empty();
  iree_string_view_t after_final_silu_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      final_silu_name_buffer, IREE_ARRAYSIZE(final_silu_name_buffer),
      &final_silu_name, "%.*s.final.adaln_input_silu", (int)branch_name.size,
      branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      after_final_silu_name_buffer,
      IREE_ARRAYSIZE(after_final_silu_name_buffer), &after_final_silu_name,
      "%.*s.final.after_adaln_input_silu", (int)branch_name.size,
      branch_name.data));

  id4_pipeline_program_tensor_t final_silu =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, final_silu_name,
      f32_canonical_activations ? ID4_PIPELINE_PROGRAM_DTYPE_F32
                                : ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(adaln_size), &final_silu));
  if (f32_canonical_activations) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_silu(
        builder, final_silu_name, adaln_size, adaln_input, final_silu));
  } else {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_silu_bf16(
        builder, final_silu_name, adaln_size, adaln_input, final_silu));
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_final_silu_name));

  char final_scale_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char after_final_scale_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t final_scale_name = iree_string_view_empty();
  iree_string_view_t after_final_scale_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      final_scale_name_buffer, IREE_ARRAYSIZE(final_scale_name_buffer),
      &final_scale_name, "%.*s.final.scale", (int)branch_name.size,
      branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      after_final_scale_name_buffer,
      IREE_ARRAYSIZE(after_final_scale_name_buffer), &after_final_scale_name,
      "%.*s.final.after_scale", (int)branch_name.size, branch_name.data));
  const bool tap_final_scale = id4_ideogram4_dit_program_has_diagnostic_tap(
      options->diagnostic_tap_names, final_scale_name);

  id4_pipeline_program_tensor_t final_scale =
      id4_pipeline_program_tensor_invalid();
  const id4_ideogram4_dit_program_dense_options_t final_scale_options = {
      .operation_name = final_scale_name,
      .output_name = final_scale_name,
      .parameter_sources = options->parameter_sources,
      .parameter_prefix = IREE_SV("final_layer.adaln_modulation"),
      .weight_execution_format = options->weight_execution_format,
      .input = final_silu,
      .input_size = adaln_size,
      .output_size = hidden_size,
      .activation_kind = ID4_IDEOGRAM4_DIT_ACTIVATION_IDENTITY,
  };
  if (f32_canonical_activations) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dense_f32(
        builder, &final_scale_options, &final_scale));
  } else {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dense_bf16(
        builder, &final_scale_options, &final_scale));
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_final_scale_name));
  if (f32_canonical_activations) {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_tap(builder, final_scale_name, final_scale));
  } else if (tap_final_scale) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap_bf16_as_f32(
        builder, final_scale_name,
        id4_pipeline_program_make_shape_rank1(hidden_size), final_scale));
  }

  char final_norm_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char final_norm_packed_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char
      final_norm_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char final_norm_packed_dispatch_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char after_final_norm_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t final_norm_name = iree_string_view_empty();
  iree_string_view_t final_norm_packed_name = iree_string_view_empty();
  iree_string_view_t final_norm_dispatch_name = iree_string_view_empty();
  iree_string_view_t final_norm_packed_dispatch_name = iree_string_view_empty();
  iree_string_view_t after_final_norm_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      final_norm_name_buffer, IREE_ARRAYSIZE(final_norm_name_buffer),
      &final_norm_name, "%.*s.final.normalized", (int)branch_name.size,
      branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      final_norm_name, IREE_SV("image_packed"), final_norm_packed_name_buffer,
      IREE_ARRAYSIZE(final_norm_packed_name_buffer), &final_norm_packed_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      final_norm_dispatch_name_buffer,
      IREE_ARRAYSIZE(final_norm_dispatch_name_buffer),
      &final_norm_dispatch_name, "%.*s.final.modulated_layernorm",
      (int)branch_name.size, branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      final_norm_packed_dispatch_name_buffer,
      IREE_ARRAYSIZE(final_norm_packed_dispatch_name_buffer),
      &final_norm_packed_dispatch_name, "%.*s.final.modulated_layernorm_image",
      (int)branch_name.size, branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      after_final_norm_name_buffer,
      IREE_ARRAYSIZE(after_final_norm_name_buffer), &after_final_norm_name,
      "%.*s.final.after_layernorm", (int)branch_name.size, branch_name.data));
  const bool tap_final_norm = id4_ideogram4_dit_program_has_diagnostic_tap(
      options->diagnostic_tap_names, final_norm_name);

  id4_ideogram4_dit_program_linear_parameter_t final_linear_weight = {
      .storage = ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_INVALID,
      .weight = id4_pipeline_program_tensor_invalid(),
      .scale = id4_pipeline_program_tensor_invalid(),
  };
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_linear_parameter(
      builder, options->parameter_sources, IREE_SV("final_layer.linear.weight"),
      hidden_size, input_channel_count, options->weight_execution_format,
      &final_linear_weight));
  id4_pipeline_program_tensor_t final_linear_bias =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter_bf16(
      builder, options->parameter_sources, IREE_SV("final_layer.linear.bias"),
      id4_pipeline_program_make_shape_rank1(input_channel_count),
      &final_linear_bias));

  const bool use_packed_bf16_final_projection =
      !f32_canonical_activations &&
      final_linear_weight.storage == ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16 &&
      image_token_count % 32 == 0 && hidden_size % 16 == 0 &&
      input_channel_count % 64 == 0;
  const id4_pipeline_program_shape_t final_norm_shape =
      id4_pipeline_program_make_shape_rank2(hidden_size, total_token_count);
  const id4_pipeline_program_shape_t final_norm_packed_shape =
      id4_pipeline_program_make_shape_rank2(image_token_count, hidden_size);
  id4_pipeline_program_tensor_t final_norm =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t final_norm_packed =
      id4_pipeline_program_tensor_invalid();
  if (f32_canonical_activations || !use_packed_bf16_final_projection ||
      tap_final_norm) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, final_norm_name,
        f32_canonical_activations ? ID4_PIPELINE_PROGRAM_DTYPE_F32
                                  : ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        final_norm_shape, &final_norm));
    if (f32_canonical_activations) {
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_dispatch_modulated_layernorm(
              builder, final_norm_dispatch_name, total_token_count, hidden_size,
              hidden, final_scale, final_norm));
    } else {
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_dispatch_modulated_layernorm_bf16_input_token_major(
              builder, final_norm_dispatch_name, total_token_count, hidden_size,
              hidden, final_scale, final_norm));
    }
  }
  if (use_packed_bf16_final_projection) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, final_norm_packed_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        final_norm_packed_shape, &final_norm_packed));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_modulated_layernorm_image_bf16_input_token_major(
            builder, final_norm_packed_dispatch_name, total_token_count,
            text_token_count, image_token_count, hidden_size, hidden,
            final_scale, final_norm_packed));
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_final_norm_name));
  if (f32_canonical_activations) {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_tap(builder, final_norm_name, final_norm));
  } else if (tap_final_norm) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap_bf16_as_f32(
        builder, final_norm_name,
        id4_pipeline_program_make_shape_rank2(hidden_size, total_token_count),
        final_norm));
  }

  char projected_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char projected_packed_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char projected_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char after_projected_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char projected_f32_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char projected_cast_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char after_projected_cast_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t projected_name = iree_string_view_empty();
  iree_string_view_t projected_packed_name = iree_string_view_empty();
  iree_string_view_t projected_dispatch_name = iree_string_view_empty();
  iree_string_view_t after_projected_name = iree_string_view_empty();
  iree_string_view_t projected_f32_name = iree_string_view_empty();
  iree_string_view_t projected_cast_name = iree_string_view_empty();
  iree_string_view_t after_projected_cast_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      projected_name_buffer, IREE_ARRAYSIZE(projected_name_buffer),
      &projected_name, "%.*s.final.projected", (int)branch_name.size,
      branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      projected_name, IREE_SV("packed"), projected_packed_name_buffer,
      IREE_ARRAYSIZE(projected_packed_name_buffer), &projected_packed_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      projected_dispatch_name_buffer,
      IREE_ARRAYSIZE(projected_dispatch_name_buffer), &projected_dispatch_name,
      "%.*s.final.linear", (int)branch_name.size, branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      after_projected_name_buffer, IREE_ARRAYSIZE(after_projected_name_buffer),
      &after_projected_name, "%.*s.final.after_linear", (int)branch_name.size,
      branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      projected_name, IREE_SV("f32"), projected_f32_name_buffer,
      IREE_ARRAYSIZE(projected_f32_name_buffer), &projected_f32_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      projected_name, IREE_SV("cast_f32"), projected_cast_name_buffer,
      IREE_ARRAYSIZE(projected_cast_name_buffer), &projected_cast_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      projected_name, IREE_SV("after_cast_f32"),
      after_projected_cast_name_buffer,
      IREE_ARRAYSIZE(after_projected_cast_name_buffer),
      &after_projected_cast_name));
  const bool tap_projected = id4_ideogram4_dit_program_has_diagnostic_tap(
      options->diagnostic_tap_names, projected_name);

  id4_pipeline_program_tensor_t projected =
      id4_pipeline_program_tensor_invalid();
  const id4_pipeline_program_shape_t projected_shape =
      id4_pipeline_program_make_shape_rank2(input_channel_count,
                                            image_token_count);
  const id4_pipeline_program_shape_t projected_packed_shape =
      id4_pipeline_program_make_shape_rank2(image_token_count,
                                            input_channel_count);
  id4_pipeline_program_tensor_t projected_f32 =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t projected_packed =
      id4_pipeline_program_tensor_invalid();
  if (use_packed_bf16_final_projection) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, projected_packed_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        projected_packed_shape, &projected_packed));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_linear_bias_packed_bf16_bf16_wmma(
            builder, projected_dispatch_name, image_token_count, hidden_size,
            input_channel_count, final_norm_packed, final_linear_weight.weight,
            final_linear_bias, projected_packed));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_barrier(builder, after_projected_name));
    if (tap_projected) {
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
          builder, projected_f32_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
          projected_shape, &projected_f32));
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_dispatch_linear_input_unpack_bf16_f32(
              builder, projected_cast_name, image_token_count,
              image_token_count, input_channel_count, projected_packed,
              projected_f32));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_barrier(
          builder, after_projected_cast_name));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
          builder, projected_name, projected_f32));
    }
  } else {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, projected_name,
        f32_canonical_activations ? ID4_PIPELINE_PROGRAM_DTYPE_F32
                                  : ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        projected_shape, &projected));
    if (f32_canonical_activations) {
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_linear_bias_f32(
          builder, projected_dispatch_name, total_token_count, text_token_count,
          image_token_count, hidden_size, input_channel_count, final_norm,
          final_linear_weight, final_linear_bias, projected));
    } else {
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_linear_bias_bf16(
          builder, projected_dispatch_name, total_token_count, text_token_count,
          image_token_count, hidden_size, input_channel_count, final_norm,
          final_linear_weight, final_linear_bias, projected));
    }
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_barrier(builder, after_projected_name));
    if (f32_canonical_activations) {
      projected_f32 = projected;
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_tap(builder, projected_name, projected));
    } else {
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
          builder, projected_f32_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
          projected_shape, &projected_f32));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_cast_bf16_f32(
          builder, projected_cast_name, projected_shape, projected,
          projected_f32));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_barrier(
          builder, after_projected_cast_name));
      if (tap_projected) {
        IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
            builder, projected_name, projected_f32));
      }
    }
  }

  id4_pipeline_program_tensor_t velocity =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_import_tensor(
      builder, IREE_SV("velocity"), 0, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      options->request.latent_shape, &velocity));

  char velocity_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char after_velocity_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char velocity_capture_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t velocity_dispatch_name = iree_string_view_empty();
  iree_string_view_t after_velocity_name = iree_string_view_empty();
  iree_string_view_t velocity_capture_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      velocity_dispatch_name_buffer,
      IREE_ARRAYSIZE(velocity_dispatch_name_buffer), &velocity_dispatch_name,
      "%.*s.final.unpatchify_scale", (int)branch_name.size, branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      after_velocity_name_buffer, IREE_ARRAYSIZE(after_velocity_name_buffer),
      &after_velocity_name, "%.*s.final.after_velocity", (int)branch_name.size,
      branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      velocity_capture_name_buffer,
      IREE_ARRAYSIZE(velocity_capture_name_buffer), &velocity_capture_name,
      "%.*s.output.velocity", (int)branch_name.size, branch_name.data));

  if (use_packed_bf16_final_projection) {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_dispatch_unpatchify_scale_packed_bf16(
            builder, velocity_dispatch_name, width, height, input_channel_count,
            projected_packed, velocity));
  } else {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_unpatchify_scale(
        builder, velocity_dispatch_name, width, height, input_channel_count,
        image_token_count, 0, projected_f32, velocity));
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_velocity_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_tap(builder, velocity_capture_name, velocity));
  return id4_ideogram4_dit_program_export(builder, IREE_SV("velocity"),
                                          velocity);
}

iree_status_t id4_ideogram4_dit_program_author_forward(
    const id4_ideogram4_dit_program_options_t* options,
    id4_pipeline_program_builder_t* builder) {
  uint32_t image_token_count = 0;
  uint32_t total_token_count = 0;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_validate_options(
      options, builder, &image_token_count, &total_token_count));
  const uint32_t input_channel_count = options->model.input_channel_count;
  const uint32_t hidden_size = options->model.hidden_size;
  const uint32_t intermediate_size = options->model.intermediate_size;
  const uint32_t attention_head_count = options->model.attention_head_count;
  const uint32_t head_size = hidden_size / attention_head_count;
  const uint32_t adaln_size = options->model.adaln_size;
  const uint32_t llm_feature_count = options->model.llm_feature_count;
  const uint32_t text_token_count = options->request.text_token_count;
  const bool is_conditioned = options->request.conditioning_mode ==
                              ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED;
  const bool f32_canonical_activations =
      options->activation_format ==
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL;
  const iree_string_view_t branch_name =
      is_conditioned ? IREE_SV("ideogram4.cond") : IREE_SV("ideogram4.uncond");
  const iree_string_view_t prelude_hidden_name =
      is_conditioned ? IREE_SV("ideogram4.cond.prelude.hidden")
                     : IREE_SV("ideogram4.uncond.prelude.hidden");
  const iree_string_view_t timestep_embedding_name =
      is_conditioned ? IREE_SV("ideogram4.cond.prelude.timestep_embedding")
                     : IREE_SV("ideogram4.uncond.prelude.timestep_embedding");
  const iree_string_view_t timestep_mlp_in_name =
      is_conditioned ? IREE_SV("ideogram4.cond.prelude.t_embedding.mlp_in")
                     : IREE_SV("ideogram4.uncond.prelude.t_embedding.mlp_in");
  const iree_string_view_t timestep_mlp_out_name =
      is_conditioned ? IREE_SV("ideogram4.cond.prelude.t_embedding.mlp_out")
                     : IREE_SV("ideogram4.uncond.prelude.t_embedding.mlp_out");
  const iree_string_view_t adaln_input_name =
      is_conditioned ? IREE_SV("ideogram4.cond.prelude.adaln_input")
                     : IREE_SV("ideogram4.uncond.prelude.adaln_input");
  const iree_string_view_t after_inputs_name =
      is_conditioned ? IREE_SV("ideogram4.cond.prelude.after_inputs")
                     : IREE_SV("ideogram4.uncond.prelude.after_inputs");
  const iree_string_view_t after_timestep_mlp_in_name =
      is_conditioned
          ? IREE_SV("ideogram4.cond.prelude.after_t_embedding_mlp_in")
          : IREE_SV("ideogram4.uncond.prelude.after_t_embedding_mlp_in");
  const iree_string_view_t after_timestep_mlp_out_name =
      is_conditioned
          ? IREE_SV("ideogram4.cond.prelude.after_t_embedding_mlp_out")
          : IREE_SV("ideogram4.uncond.prelude.after_t_embedding_mlp_out");
  const iree_string_view_t after_adaln_name =
      is_conditioned ? IREE_SV("ideogram4.cond.prelude.after_adaln_proj")
                     : IREE_SV("ideogram4.uncond.prelude.after_adaln_proj");
  const id4_pipeline_program_shape_t hidden_shape =
      id4_pipeline_program_make_shape_rank2(hidden_size, total_token_count);
  const id4_pipeline_program_shape_t token_major_hidden_shape =
      id4_pipeline_program_make_shape_rank2(total_token_count, hidden_size);
  const id4_pipeline_program_shape_t hidden_vector_shape =
      id4_pipeline_program_make_shape_rank1(hidden_size);
  const id4_pipeline_program_shape_t adaln_vector_shape =
      id4_pipeline_program_make_shape_rank1(adaln_size);
  char prelude_hidden_raw_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char prelude_hidden_cast_f32_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char after_prelude_hidden_round_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char
      timestep_mlp_in_f32_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char timestep_mlp_in_cast_name_buffer
      [ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char adaln_input_bf16_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char adaln_input_cast_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t prelude_hidden_raw_name = iree_string_view_empty();
  iree_string_view_t prelude_hidden_cast_f32_name = iree_string_view_empty();
  iree_string_view_t after_prelude_hidden_round_name = iree_string_view_empty();
  iree_string_view_t timestep_mlp_in_f32_name = iree_string_view_empty();
  iree_string_view_t timestep_mlp_in_cast_name = iree_string_view_empty();
  iree_string_view_t adaln_input_bf16_name = iree_string_view_empty();
  iree_string_view_t adaln_input_cast_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      prelude_hidden_name, IREE_SV("raw"), prelude_hidden_raw_name_buffer,
      IREE_ARRAYSIZE(prelude_hidden_raw_name_buffer),
      &prelude_hidden_raw_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      prelude_hidden_name, IREE_SV("cast_f32"),
      prelude_hidden_cast_f32_name_buffer,
      IREE_ARRAYSIZE(prelude_hidden_cast_f32_name_buffer),
      &prelude_hidden_cast_f32_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      prelude_hidden_name, IREE_SV("after_round"),
      after_prelude_hidden_round_name_buffer,
      IREE_ARRAYSIZE(after_prelude_hidden_round_name_buffer),
      &after_prelude_hidden_round_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      timestep_mlp_in_name, IREE_SV("f32"), timestep_mlp_in_f32_name_buffer,
      IREE_ARRAYSIZE(timestep_mlp_in_f32_name_buffer),
      &timestep_mlp_in_f32_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      timestep_mlp_in_name, IREE_SV("cast"), timestep_mlp_in_cast_name_buffer,
      IREE_ARRAYSIZE(timestep_mlp_in_cast_name_buffer),
      &timestep_mlp_in_cast_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      adaln_input_name, IREE_SV("bf16"), adaln_input_bf16_name_buffer,
      IREE_ARRAYSIZE(adaln_input_bf16_name_buffer), &adaln_input_bf16_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      adaln_input_name, IREE_SV("cast"), adaln_input_cast_name_buffer,
      IREE_ARRAYSIZE(adaln_input_cast_name_buffer), &adaln_input_cast_name));
  const bool tap_adaln_input = id4_ideogram4_dit_program_has_diagnostic_tap(
      options->diagnostic_tap_names, adaln_input_name);
  const bool tap_timestep_mlp_in = id4_ideogram4_dit_program_has_diagnostic_tap(
      options->diagnostic_tap_names, timestep_mlp_in_name);
  const bool tap_timestep_mlp_out =
      id4_ideogram4_dit_program_has_diagnostic_tap(
          options->diagnostic_tap_names, timestep_mlp_out_name);

  id4_pipeline_program_tensor_t x = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_import_tensor(
      builder, IREE_SV("x"),
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      ID4_PIPELINE_PROGRAM_DTYPE_F32, options->request.latent_shape, &x));
  id4_pipeline_program_tensor_t timestep =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_import_tensor(
      builder, IREE_SV("timestep"),
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      ID4_PIPELINE_PROGRAM_DTYPE_F32, id4_pipeline_program_make_shape_rank1(1),
      &timestep));
  id4_pipeline_program_tensor_t image_indicator =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_import_tensor(
      builder, IREE_SV("image_indicator"),
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      ID4_PIPELINE_PROGRAM_DTYPE_I32,
      id4_pipeline_program_make_shape_rank2(total_token_count, 1),
      &image_indicator));
  id4_pipeline_program_tensor_t position_embedding =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_import_tensor(
      builder, IREE_SV("position_embedding"),
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank4(2, 2, head_size / 2,
                                            total_token_count),
      &position_embedding));
  id4_pipeline_program_tensor_t condition =
      id4_pipeline_program_tensor_invalid();
  if (is_conditioned) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_import_tensor(
        builder, IREE_SV("condition"),
        ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
        ID4_PIPELINE_PROGRAM_DTYPE_F32,
        id4_pipeline_program_make_shape_rank2(llm_feature_count,
                                              text_token_count),
        &condition));
  }

  id4_ideogram4_dit_program_linear_parameter_t input_proj = {
      .storage = ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_INVALID,
      .weight = id4_pipeline_program_tensor_invalid(),
      .scale = id4_pipeline_program_tensor_invalid(),
  };
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_linear_parameter(
      builder, options->parameter_sources, IREE_SV("input_proj.weight"),
      input_channel_count, hidden_size, options->weight_execution_format,
      &input_proj));
  id4_pipeline_program_tensor_t input_proj_bias =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter_bf16(
      builder, options->parameter_sources, IREE_SV("input_proj.bias"),
      id4_pipeline_program_make_shape_rank1(hidden_size), &input_proj_bias));
  id4_pipeline_program_tensor_t image_indicator_embedding =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter_bf16(
      builder, options->parameter_sources,
      IREE_SV("embed_image_indicator.weight"),
      id4_pipeline_program_make_shape_rank2(
          options->model.image_indicator_count, hidden_size),
      &image_indicator_embedding));
  id4_pipeline_program_tensor_t llm_cond_norm_weight =
      id4_pipeline_program_tensor_invalid();
  id4_ideogram4_dit_program_linear_parameter_t llm_cond_proj = {
      .storage = ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_INVALID,
      .weight = id4_pipeline_program_tensor_invalid(),
      .scale = id4_pipeline_program_tensor_invalid(),
  };
  id4_pipeline_program_tensor_t llm_cond_proj_bias =
      id4_pipeline_program_tensor_invalid();
  if (is_conditioned) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter_bf16(
        builder, options->parameter_sources, IREE_SV("llm_cond_norm.weight"),
        id4_pipeline_program_make_shape_rank1(llm_feature_count),
        &llm_cond_norm_weight));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_linear_parameter(
        builder, options->parameter_sources, IREE_SV("llm_cond_proj.weight"),
        llm_feature_count, hidden_size, options->weight_execution_format,
        &llm_cond_proj));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter_bf16(
        builder, options->parameter_sources, IREE_SV("llm_cond_proj.bias"),
        id4_pipeline_program_make_shape_rank1(hidden_size),
        &llm_cond_proj_bias));
  }

  id4_pipeline_program_tensor_t prelude_hidden =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t prelude_hidden_producer_output =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder,
      f32_canonical_activations ? prelude_hidden_name : prelude_hidden_raw_name,
      ID4_PIPELINE_PROGRAM_DTYPE_F32, hidden_shape,
      &prelude_hidden_producer_output));
  if (f32_canonical_activations) {
    prelude_hidden = prelude_hidden_producer_output;
  }
  id4_pipeline_program_tensor_t timestep_embedding =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, timestep_embedding_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      hidden_vector_shape, &timestep_embedding));

  if (is_conditioned) {
    const bool tap_condition_norm =
        id4_ideogram4_dit_program_has_diagnostic_tap(
            options->diagnostic_tap_names,
            IREE_SV("ideogram4.cond.prelude.llm_cond_norm"));
    const bool needs_canonical_condition_norm = f32_canonical_activations;
    if (needs_canonical_condition_norm) {
      id4_pipeline_program_tensor_t condition_norm =
          id4_pipeline_program_tensor_invalid();
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
          builder, IREE_SV("ideogram4.cond.prelude.llm_cond_norm"),
          ID4_PIPELINE_PROGRAM_DTYPE_F32,
          id4_pipeline_program_make_shape_rank2(llm_feature_count,
                                                text_token_count),
          &condition_norm));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_condition_rmsnorm(
          builder, text_token_count, llm_feature_count, condition,
          llm_cond_norm_weight, condition_norm));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_barrier(
          builder, IREE_SV("ideogram4.cond.prelude.after_llm_cond_norm")));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
          builder, IREE_SV("ideogram4.cond.prelude.llm_cond_norm"),
          condition_norm));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_condition_project(
          builder, text_token_count, total_token_count, llm_feature_count,
          hidden_size, options->diagnostic_tap_names, condition_norm,
          llm_cond_proj, llm_cond_proj_bias, image_indicator,
          image_indicator_embedding, prelude_hidden_producer_output));
    } else {
      id4_pipeline_program_tensor_t packed_condition =
          id4_pipeline_program_tensor_invalid();
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_dispatch_condition_rmsnorm_pack(
              builder, text_token_count, llm_feature_count, condition,
              llm_cond_norm_weight, &packed_condition));
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_barrier(
          builder, IREE_SV("ideogram4.cond.prelude.after_llm_cond_norm_pack")));
      if (tap_condition_norm) {
        const uint32_t padded_text_token_count =
            id4_ideogram4_dit_program_ceil_div_u32(
                text_token_count, ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK) *
            ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK;
        IREE_RETURN_IF_ERROR(
            id4_ideogram4_dit_program_tap_linear_input_bf16_as_f32(
                builder, IREE_SV("ideogram4.cond.prelude.llm_cond_norm"),
                text_token_count, padded_text_token_count, llm_feature_count,
                packed_condition));
      }
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_dispatch_condition_project_packed(
              builder, text_token_count, total_token_count, llm_feature_count,
              hidden_size, options->diagnostic_tap_names, packed_condition,
              llm_cond_proj, llm_cond_proj_bias, image_indicator,
              image_indicator_embedding, prelude_hidden_producer_output));
    }
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_barrier(
        builder, IREE_SV("ideogram4.cond.prelude.after_llm_cond_proj")));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
        builder, IREE_SV("ideogram4.cond.prelude.llm_cond_proj"),
        prelude_hidden_producer_output));
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_prelude_image(
      builder, image_token_count, total_token_count, text_token_count,
      (uint32_t)options->request.latent_shape.dims[0],
      (uint32_t)options->request.latent_shape.dims[1], input_channel_count,
      hidden_size, x, input_proj, input_proj_bias, image_indicator,
      image_indicator_embedding, prelude_hidden_producer_output));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_timestep_embedding(
      builder, timestep_embedding_name, hidden_size, timestep,
      timestep_embedding));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
      builder, timestep_embedding_name, timestep_embedding));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_inputs_name));

  if (!f32_canonical_activations) {
    id4_pipeline_program_tensor_t prelude_hidden_bf16 =
        id4_pipeline_program_tensor_invalid();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_pack_linear_input_f32_bf16(
        builder, prelude_hidden_name, total_token_count, total_token_count,
        hidden_size, prelude_hidden_producer_output, &prelude_hidden_bf16));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, prelude_hidden_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
        token_major_hidden_shape, &prelude_hidden));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_cast_bf16_f32(
        builder, prelude_hidden_cast_f32_name, token_major_hidden_shape,
        prelude_hidden_bf16, prelude_hidden));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_barrier(
        builder, after_prelude_hidden_round_name));
  }

  id4_pipeline_program_tensor_t timestep_mlp_in_f32 =
      id4_pipeline_program_tensor_invalid();
  const id4_ideogram4_dit_program_dense_options_t mlp_in_options = {
      .operation_name = timestep_mlp_in_name,
      .output_name = f32_canonical_activations ? timestep_mlp_in_name
                                               : timestep_mlp_in_f32_name,
      .parameter_sources = options->parameter_sources,
      .parameter_prefix = IREE_SV("t_embedding.mlp_in"),
      .weight_execution_format = options->weight_execution_format,
      .input = timestep_embedding,
      .input_size = hidden_size,
      .output_size = hidden_size,
      .activation_kind = ID4_IDEOGRAM4_DIT_ACTIVATION_SILU,
  };
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dense_f32(
      builder, &mlp_in_options, &timestep_mlp_in_f32));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_timestep_mlp_in_name));
  id4_pipeline_program_tensor_t timestep_mlp_in = timestep_mlp_in_f32;
  if (!f32_canonical_activations) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, timestep_mlp_in_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        hidden_vector_shape, &timestep_mlp_in));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_cast_f32_bf16(
        builder, timestep_mlp_in_cast_name, hidden_vector_shape,
        timestep_mlp_in_f32, timestep_mlp_in));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_barrier(builder, timestep_mlp_in_cast_name));
    if (tap_timestep_mlp_in) {
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap_bf16_as_f32(
          builder, timestep_mlp_in_name, hidden_vector_shape, timestep_mlp_in));
    }
  }

  id4_pipeline_program_tensor_t timestep_mlp_out =
      id4_pipeline_program_tensor_invalid();
  const id4_ideogram4_dit_program_dense_options_t mlp_out_options = {
      .operation_name = timestep_mlp_out_name,
      .output_name = timestep_mlp_out_name,
      .parameter_sources = options->parameter_sources,
      .parameter_prefix = IREE_SV("t_embedding.mlp_out"),
      .weight_execution_format = options->weight_execution_format,
      .input = timestep_mlp_in,
      .input_size = hidden_size,
      .output_size = hidden_size,
      .activation_kind = ID4_IDEOGRAM4_DIT_ACTIVATION_IDENTITY,
  };
  if (f32_canonical_activations) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dense_f32(
        builder, &mlp_out_options, &timestep_mlp_out));
  } else {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dense_bf16(
        builder, &mlp_out_options, &timestep_mlp_out));
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_timestep_mlp_out_name));
  if (!f32_canonical_activations && tap_timestep_mlp_out) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap_bf16_as_f32(
        builder, timestep_mlp_out_name, hidden_vector_shape, timestep_mlp_out));
  }

  id4_pipeline_program_tensor_t adaln_input_stage_output =
      id4_pipeline_program_tensor_invalid();
  const id4_ideogram4_dit_program_dense_options_t adaln_options = {
      .operation_name = IREE_SV("ideogram4.prelude.adaln_proj"),
      .output_name =
          f32_canonical_activations ? adaln_input_name : adaln_input_bf16_name,
      .parameter_sources = options->parameter_sources,
      .parameter_prefix = IREE_SV("adaln_proj"),
      .weight_execution_format = options->weight_execution_format,
      .input = timestep_mlp_out,
      .input_size = hidden_size,
      .output_size = adaln_size,
      .activation_kind = ID4_IDEOGRAM4_DIT_ACTIVATION_SILU,
  };
  if (f32_canonical_activations) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dense_f32(
        builder, &adaln_options, &adaln_input_stage_output));
  } else {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dense_bf16(
        builder, &adaln_options, &adaln_input_stage_output));
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_adaln_name));
  id4_pipeline_program_tensor_t adaln_input = adaln_input_stage_output;
  if (!f32_canonical_activations && tap_adaln_input) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, adaln_input_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
        adaln_vector_shape, &adaln_input));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_cast_bf16_f32(
        builder, adaln_input_cast_name, adaln_vector_shape,
        adaln_input_stage_output, adaln_input));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_barrier(builder, adaln_input_cast_name));
  }

  if (f32_canonical_activations) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
        builder, prelude_hidden_name, prelude_hidden));
  } else if (id4_ideogram4_dit_program_has_diagnostic_tap(
                 options->diagnostic_tap_names, prelude_hidden_name)) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap_token_major_f32_as_f32(
        builder, prelude_hidden_name, total_token_count, hidden_size,
        prelude_hidden));
  }
  if (f32_canonical_activations || tap_adaln_input) {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_dit_program_tap(builder, adaln_input_name, adaln_input));
  }

  id4_pipeline_program_tensor_t layer_input = prelude_hidden;
  for (uint32_t layer_ordinal = 0; layer_ordinal < options->model.layer_count;
       ++layer_ordinal) {
    id4_pipeline_program_tensor_t layer_output =
        id4_pipeline_program_tensor_invalid();
    id4_ideogram4_dit_program_block_options_t block_options = {
        .builder = builder,
        .branch_name = branch_name,
        .parameter_sources = options->parameter_sources,
        .layer_ordinal = layer_ordinal,
        .adaln_size = adaln_size,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
        .attention_head_count = attention_head_count,
        .total_token_count = total_token_count,
        .hidden_input = layer_input,
        .adaln_input =
            f32_canonical_activations ? adaln_input : adaln_input_stage_output,
        .position_embedding = position_embedding,
        .activation_format = options->activation_format,
        .weight_execution_format = options->weight_execution_format,
        .attention_implementation = options->attention_implementation,
        .feed_forward_implementation = options->feed_forward_implementation,
        .diagnostic_tap_names = options->diagnostic_tap_names,
    };
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_author_transformer_block(
        &block_options, &layer_output));
    layer_input = layer_output;
  }
  return id4_ideogram4_dit_program_author_final_output(
      options, builder, branch_name, total_token_count, layer_input,
      f32_canonical_activations ? adaln_input : adaln_input_stage_output);
}

const id4_ideogram4_dit_model_config_t*
id4_ideogram4_dit_program_ideogram4_model_config(void) {
  return &id4_ideogram4_dit_program_ideogram4_model_config_value;
}
