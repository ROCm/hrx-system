// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/qwen3_vl_program.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/pipeline/program_matrix.h"

#define ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY 192

enum {
  ID4_QWEN3_VL_LINEAR_SCALAR_INPUT_BLOCK = 4,
  ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M16 = 16,
  ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M32 = 32,
  ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M48 = 48,
  ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M64 = 64,
  ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M128 = 128,
  ID4_QWEN3_VL_LINEAR_WMMA_INPUT_BLOCK = 16,
  ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK_M32 = 32,
  ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK_M64 = 64,
  ID4_QWEN3_VL_LINEAR_WMMA_M32N64_MIN_TOKEN_COUNT = 1024,
  ID4_QWEN3_VL_LINEAR_WMMA_M32N64_MAX_OUTPUT_SIZE = 4096,
  ID4_QWEN3_VL_LINEAR_WMMA_M48N32_MIN_DOWN_INPUT_SIZE = 8192,
  ID4_QWEN3_VL_LINEAR_WMMA_M64N64_MIN_TOKEN_COUNT = 512,
  ID4_QWEN3_VL_LINEAR_WMMA_M64N64_MIN_OUTPUT_SIZE = 4096,
  ID4_QWEN3_VL_LINEAR_WMMA_M128N64_FP8_MIN_OUTPUT_SIZE = 1024,
  ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE = 128,
  ID4_QWEN3_VL_CONDITION_BLOCK_ELEMENT_COUNT = 2048,
  ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY = 16,
  ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT = 8,
};

static const uint32_t id4_qwen3_vl_program_ideogram4_selected_layers[] = {
    0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35,
};

static const id4_qwen3_vl_model_config_t
    id4_qwen3_vl_program_ideogram4_model_config_value = {
        // Number of decoder layers.
        .layer_count = 36,
        // Vocabulary row count in the token embedding table.
        .vocab_size = 151936,
        // Hidden-state channel count.
        .hidden_size = 4096,
        // MLP intermediate channel count.
        .intermediate_size = 12288,
        // Number of query attention heads.
        .attention_head_count = 32,
        // Number of key/value attention heads.
        .key_value_head_count = 8,
        // Channel count per attention head.
        .head_size = 128,
        // Maximum prompt token positions accepted by the runner.
        .max_token_count = 4096,
        // Number of post-layer hidden states used by Ideogram 4 conditioning.
        .selected_layer_count =
            IREE_ARRAYSIZE(id4_qwen3_vl_program_ideogram4_selected_layers),
        // Zero-based selected layer ordinals.
        .selected_layer_ordinals =
            id4_qwen3_vl_program_ideogram4_selected_layers,
};

iree_status_t id4_qwen3_vl_parameter_format_parse(
    iree_string_view_t value, id4_qwen3_vl_parameter_format_t* out_format) {
  IREE_ASSERT_ARGUMENT(out_format);
  if (iree_string_view_equal(value, IREE_SV("bf16"))) {
    *out_format = ID4_QWEN3_VL_PARAMETER_FORMAT_BF16;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("fp8_e4m3_block_scaled"))) {
    *out_format = ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "Qwen3-VL parameter format must be bf16 or "
                          "fp8_e4m3_block_scaled");
}

iree_string_view_t id4_qwen3_vl_parameter_format_name(
    id4_qwen3_vl_parameter_format_t format) {
  switch (format) {
    case ID4_QWEN3_VL_PARAMETER_FORMAT_BF16:
      return IREE_SV("bf16");
    case ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED:
      return IREE_SV("fp8_e4m3_block_scaled");
    default:
      return IREE_SV("invalid");
  }
}

iree_status_t id4_qwen3_vl_weight_execution_strategy_parse(
    iree_string_view_t value,
    id4_qwen3_vl_weight_execution_strategy_t* out_strategy) {
  IREE_ASSERT_ARGUMENT(out_strategy);
  if (iree_string_view_equal(value, IREE_SV("row_major"))) {
    *out_strategy = ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("compact_rhs"))) {
    *out_strategy = ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_COMPACT_RHS;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("hybrid_compact_rhs"))) {
    *out_strategy = ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("streaming_compact_rhs"))) {
    *out_strategy =
        ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_STREAMING_COMPACT_RHS;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "Qwen3-VL weight execution strategy must be row_major, compact_rhs, "
      "hybrid_compact_rhs, or streaming_compact_rhs");
}

iree_string_view_t id4_qwen3_vl_weight_execution_strategy_name(
    id4_qwen3_vl_weight_execution_strategy_t strategy) {
  switch (strategy) {
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR:
      return IREE_SV("row_major");
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_COMPACT_RHS:
      return IREE_SV("compact_rhs");
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS:
      return IREE_SV("hybrid_compact_rhs");
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_STREAMING_COMPACT_RHS:
      return IREE_SV("streaming_compact_rhs");
    default:
      return IREE_SV("invalid");
  }
}

iree_status_t id4_qwen3_vl_attention_implementation_parse(
    iree_string_view_t value,
    id4_qwen3_vl_attention_implementation_t* out_implementation) {
  IREE_ASSERT_ARGUMENT(out_implementation);
  if (iree_string_view_equal(value, IREE_SV("auto"))) {
    *out_implementation = ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_AUTO;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("materialized"))) {
    *out_implementation = ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_MATERIALIZED;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("wmma"))) {
    *out_implementation = ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_WMMA;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "Qwen3-VL attention implementation must be auto, materialized, or wmma");
}

iree_string_view_t id4_qwen3_vl_attention_implementation_name(
    id4_qwen3_vl_attention_implementation_t implementation) {
  switch (implementation) {
    case ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_AUTO:
      return IREE_SV("auto");
    case ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_MATERIALIZED:
      return IREE_SV("materialized");
    case ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_WMMA:
      return IREE_SV("wmma");
    default:
      return IREE_SV("invalid");
  }
}

static iree_status_t id4_qwen3_vl_program_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_qwen3_vl_program_format(
    char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string, const char* format, ...) {
  va_list varargs;
  va_start(varargs, format);
  int length = vsnprintf(buffer, buffer_capacity, format, varargs);
  va_end(varargs);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format Qwen3-VL program string");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_program_format_u32(
    uint32_t value, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  return id4_qwen3_vl_program_format(buffer, buffer_capacity, out_string,
                                     "%" PRIu32, value);
}

static iree_status_t id4_qwen3_vl_program_format_child_name(
    iree_string_view_t parent_name, iree_string_view_t child_name, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string) {
  return id4_qwen3_vl_program_format(
      buffer, buffer_capacity, out_string, "%.*s.%.*s", (int)parent_name.size,
      parent_name.data, (int)child_name.size, child_name.data);
}

static bool id4_qwen3_vl_program_checked_mul_u32(uint32_t lhs, uint32_t rhs,
                                                 uint32_t* out_result) {
  if (lhs != 0 && rhs > UINT32_MAX / lhs) return false;
  *out_result = lhs * rhs;
  return true;
}

typedef struct id4_qwen3_vl_program_bf16_coverage_t {
  // Number of BF16 elements that are semantically initialized.
  uint32_t logical_element_count;
  // Number of BF16 elements covered by the padded tensor allocation.
  uint32_t dispatch_element_count;
  // Number of bytes that are semantically initialized.
  iree_device_size_t logical_byte_length;
  // Number of bytes covered by the padded tensor allocation.
  iree_device_size_t dispatch_byte_length;
} id4_qwen3_vl_program_bf16_coverage_t;

static iree_status_t id4_qwen3_vl_program_bf16_byte_length(
    uint32_t element_count, iree_device_size_t* out_byte_length) {
  if (!iree_device_size_checked_mul(element_count, sizeof(uint16_t),
                                    out_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL BF16 byte length overflow");
  }
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_program_bf16_coverage(
    const char* name, uint32_t logical_outer_count,
    uint32_t logical_inner_count, uint32_t dispatch_outer_count,
    uint32_t dispatch_inner_count,
    id4_qwen3_vl_program_bf16_coverage_t* out_coverage) {
  uint32_t logical_element_count = 0;
  if (!id4_qwen3_vl_program_checked_mul_u32(
          logical_outer_count, logical_inner_count, &logical_element_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL %s logical element count overflow", name);
  }
  uint32_t dispatch_element_count = 0;
  if (!id4_qwen3_vl_program_checked_mul_u32(dispatch_outer_count,
                                            dispatch_inner_count,
                                            &dispatch_element_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL %s dispatch element count overflow",
                            name);
  }
  iree_device_size_t logical_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_bf16_byte_length(
      logical_element_count, &logical_byte_length));
  iree_device_size_t dispatch_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_bf16_byte_length(
      dispatch_element_count, &dispatch_byte_length));
  out_coverage->logical_element_count = logical_element_count;
  out_coverage->dispatch_element_count = dispatch_element_count;
  out_coverage->logical_byte_length = logical_byte_length;
  out_coverage->dispatch_byte_length = dispatch_byte_length;
  return iree_ok_status();
}

static id4_pipeline_program_dispatch_binding_t
id4_qwen3_vl_program_bf16_logical_write(
    id4_pipeline_program_tensor_t tensor,
    const id4_qwen3_vl_program_bf16_coverage_t* coverage) {
  return id4_pipeline_program_write_range(tensor, 0,
                                          coverage->logical_byte_length);
}

static uint32_t id4_qwen3_vl_program_ceil_div_u32(uint32_t numerator,
                                                  uint32_t denominator) {
  return numerator / denominator + (numerator % denominator != 0);
}

static iree_status_t id4_qwen3_vl_program_make_condition_partial_count(
    uint32_t hidden_row_count, uint32_t token_count,
    uint32_t* out_partial_count) {
  uint32_t element_count = 0;
  if (!id4_qwen3_vl_program_checked_mul_u32(hidden_row_count, token_count,
                                            &element_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL condition element count overflow");
  }
  *out_partial_count = id4_qwen3_vl_program_ceil_div_u32(
      element_count, ID4_QWEN3_VL_CONDITION_BLOCK_ELEMENT_COUNT);
  return iree_ok_status();
}

static uint32_t id4_qwen3_vl_program_query_width(
    const id4_qwen3_vl_model_config_t* config) {
  return config->attention_head_count * config->head_size;
}

static uint32_t id4_qwen3_vl_program_key_value_width(
    const id4_qwen3_vl_model_config_t* config) {
  return config->key_value_head_count * config->head_size;
}

static uint32_t id4_qwen3_vl_program_selected_hidden_row_count(
    const id4_qwen3_vl_model_config_t* config) {
  return config->selected_layer_count * config->hidden_size;
}

iree_status_t id4_qwen3_vl_program_calculate_token_capacity(
    id4_qwen3_vl_parameter_format_t parameter_format, uint32_t token_count,
    uint32_t* out_token_capacity) {
  if (!out_token_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL token capacity output is required");
  }
  *out_token_capacity = 0;
  if (token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL token count must be non-zero");
  }
  iree_host_size_t alignment = 0;
  switch (parameter_format) {
    case ID4_QWEN3_VL_PARAMETER_FORMAT_BF16:
      alignment = token_count > ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M16
                      ? ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M32
                      : ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M16;
      break;
    case ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED:
      alignment = ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M64;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen3-VL parameter format %" PRIu32
                              " is invalid",
                              (uint32_t)parameter_format);
  }
  iree_host_size_t token_capacity = 0;
  if (!iree_host_size_checked_align(token_count, alignment, &token_capacity) ||
      token_capacity > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL token capacity overflow");
  }
  *out_token_capacity = (uint32_t)token_capacity;
  return iree_ok_status();
}

iree_status_t id4_qwen3_vl_program_calculate_bf16_token_capacity(
    uint32_t token_count, uint32_t* out_token_capacity) {
  return id4_qwen3_vl_program_calculate_token_capacity(
      ID4_QWEN3_VL_PARAMETER_FORMAT_BF16, token_count, out_token_capacity);
}

static iree_status_t id4_qwen3_vl_program_calculate_options_token_capacity(
    const id4_qwen3_vl_program_options_t* options,
    uint32_t* out_token_capacity) {
  return id4_qwen3_vl_program_calculate_token_capacity(
      options->parameter_format, options->request.token_count,
      out_token_capacity);
}

static iree_status_t id4_qwen3_vl_program_validate_model_config(
    const id4_qwen3_vl_model_config_t* config) {
  if (config->layer_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL layer count must be nonzero");
  }
  if (config->vocab_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL vocab size must be nonzero");
  }
  if (config->hidden_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL hidden size must be nonzero");
  }
  if (config->intermediate_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL intermediate size must be nonzero");
  }
  if ((config->hidden_size % ID4_QWEN3_VL_LINEAR_WMMA_INPUT_BLOCK) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL hidden size must be a multiple of %u",
                            ID4_QWEN3_VL_LINEAR_WMMA_INPUT_BLOCK);
  }
  if ((config->intermediate_size %
       ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK_M32) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL intermediate size must be a multiple of %u",
        ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK_M32);
  }
  if (config->attention_head_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL attention head count must be nonzero");
  }
  if (config->key_value_head_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL key/value attention head count must be nonzero");
  }
  if (config->head_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL head size must be nonzero");
  }
  if ((config->head_size & 1u) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL head size must be even");
  }
  if (config->max_token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL max token count must be nonzero");
  }
  if (config->selected_layer_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL selected layer count must be nonzero");
  }
  if (!config->selected_layer_ordinals) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL selected layer ordinal array is "
                            "required");
  }
  uint32_t selected_hidden_row_count = 0;
  if (!id4_qwen3_vl_program_checked_mul_u32(config->selected_layer_count,
                                            config->hidden_size,
                                            &selected_hidden_row_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL selected hidden row count overflow");
  }
  if (selected_hidden_row_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL selected hidden row count must be "
                            "nonzero");
  }
  for (uint32_t i = 0; i < config->selected_layer_count; ++i) {
    const uint32_t layer_ordinal = config->selected_layer_ordinals[i];
    if (layer_ordinal >= config->layer_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen3-VL selected layer ordinal %" PRIu32
                              " exceeds layer count %" PRIu32,
                              layer_ordinal, config->layer_count);
    }
    if (i > 0 && layer_ordinal <= config->selected_layer_ordinals[i - 1]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen3-VL selected layer ordinals must be "
                              "strictly increasing");
    }
  }
  uint32_t query_width = 0;
  if (!id4_qwen3_vl_program_checked_mul_u32(config->attention_head_count,
                                            config->head_size, &query_width)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL query projection width overflow");
  }
  if (query_width != config->hidden_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL query head width must match hidden size");
  }
  uint32_t key_value_width = 0;
  if (!id4_qwen3_vl_program_checked_mul_u32(
          config->key_value_head_count, config->head_size, &key_value_width)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL key/value projection width overflow");
  }
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_program_validate_options(
    const id4_qwen3_vl_program_options_t* options,
    const id4_pipeline_program_builder_t* builder) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL program options are required");
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("Qwen3-VL program")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Qwen3-VL program extension structures are not "
                            "supported");
  }
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL program builder is required");
  }
  if (options->diagnostic_tap_names.count != 0 &&
      !options->diagnostic_tap_names.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL diagnostic tap names are required when "
                            "the tap count is nonzero");
  }
  if (iree_allocator_is_null(options->host_allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL host allocator is required");
  }
  for (iree_host_size_t i = 0; i < options->diagnostic_tap_names.count; ++i) {
    if (iree_string_view_is_empty(options->diagnostic_tap_names.values[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen3-VL diagnostic tap names must be nonempty");
    }
  }
  if (options->request.token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL token count must be nonzero");
  }
  if (!options->request.token_ids) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL token ids are required");
  }
  if (options->request.token_count > options->model.max_token_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Qwen3-VL token count %" PRIu32 " exceeds model maximum %" PRIu32,
        options->request.token_count, options->model.max_token_count);
  }
  for (uint32_t i = 0; i < options->request.token_count; ++i) {
    if (options->request.token_ids[i] < 0 ||
        (uint32_t)options->request.token_ids[i] >= options->model.vocab_size) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Qwen3-VL token id %" PRIi32 " at position %" PRIu32
          " exceeds vocabulary size %" PRIu32,
          options->request.token_ids[i], i, options->model.vocab_size);
    }
  }
  switch (options->parameter_format) {
    case ID4_QWEN3_VL_PARAMETER_FORMAT_BF16:
    case ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen3-VL parameter format %" PRIu32
                              " is invalid",
                              (uint32_t)options->parameter_format);
  }
  switch (options->weight_execution_strategy) {
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR:
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_COMPACT_RHS:
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS:
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_STREAMING_COMPACT_RHS:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen3-VL weight execution strategy %" PRIu32
                              " is invalid",
                              (uint32_t)options->weight_execution_strategy);
  }
  switch (options->attention_implementation) {
    case ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_AUTO:
    case ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_MATERIALIZED:
    case ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_WMMA:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen3-VL attention implementation %" PRIu32
                              " is invalid",
                              (uint32_t)options->attention_implementation);
  }
  return id4_qwen3_vl_program_validate_model_config(&options->model);
}

typedef enum id4_qwen3_vl_parameter_kind_e {
  // Token embedding table.
  ID4_QWEN3_VL_PARAMETER_TOKEN_EMBEDDING = 0,
  // Per-layer input RMSNorm weight.
  ID4_QWEN3_VL_PARAMETER_LAYER_INPUT_NORM = 1,
  // Per-layer query projection weight.
  ID4_QWEN3_VL_PARAMETER_LAYER_Q_PROJECTION = 2,
  // Per-layer key projection weight.
  ID4_QWEN3_VL_PARAMETER_LAYER_K_PROJECTION = 3,
  // Per-layer query head RMSNorm weight.
  ID4_QWEN3_VL_PARAMETER_LAYER_Q_NORM = 4,
  // Per-layer key head RMSNorm weight.
  ID4_QWEN3_VL_PARAMETER_LAYER_K_NORM = 5,
  // Per-layer value projection weight.
  ID4_QWEN3_VL_PARAMETER_LAYER_V_PROJECTION = 6,
  // Per-layer attention output projection weight.
  ID4_QWEN3_VL_PARAMETER_LAYER_O_PROJECTION = 7,
  // Per-layer post-attention RMSNorm weight.
  ID4_QWEN3_VL_PARAMETER_LAYER_POST_ATTENTION_NORM = 8,
  // Per-layer gated MLP gate projection weight.
  ID4_QWEN3_VL_PARAMETER_LAYER_GATE_PROJECTION = 9,
  // Per-layer gated MLP up projection weight.
  ID4_QWEN3_VL_PARAMETER_LAYER_UP_PROJECTION = 10,
  // Per-layer gated MLP down projection weight.
  ID4_QWEN3_VL_PARAMETER_LAYER_DOWN_PROJECTION = 11,
  // Final decoder RMSNorm weight.
  ID4_QWEN3_VL_PARAMETER_FINAL_NORM = 12,
} id4_qwen3_vl_parameter_kind_t;

typedef enum id4_qwen3_vl_tensor_kind_e {
  // Imported additive attention mask matrix.
  ID4_QWEN3_VL_TENSOR_ATTENTION_MASK = 0,
  // Imported token weight vector.
  ID4_QWEN3_VL_TENSOR_TOKEN_WEIGHTS = 1,
  // Hidden states produced by token embedding.
  ID4_QWEN3_VL_TENSOR_EMBEDDED_HIDDEN_STATES = 2,
  // Per-layer input RMSNorm output.
  ID4_QWEN3_VL_TENSOR_LAYER_INPUT_NORM = 3,
  // Per-layer query projection output.
  ID4_QWEN3_VL_TENSOR_LAYER_Q = 4,
  // Per-layer key projection output.
  ID4_QWEN3_VL_TENSOR_LAYER_K = 5,
  // Per-layer value projection output.
  ID4_QWEN3_VL_TENSOR_LAYER_V = 6,
  // Per-layer packed value tensor for attention PV WMMA.
  ID4_QWEN3_VL_TENSOR_LAYER_V_PACKED = 7,
  // Per-layer query head RMSNorm output.
  ID4_QWEN3_VL_TENSOR_LAYER_Q_NORM = 8,
  // Per-layer key head RMSNorm output.
  ID4_QWEN3_VL_TENSOR_LAYER_K_NORM = 9,
  // Per-layer rotary-applied query tensor.
  ID4_QWEN3_VL_TENSOR_LAYER_Q_ROTARY = 10,
  // Per-layer rotary-applied key tensor.
  ID4_QWEN3_VL_TENSOR_LAYER_K_ROTARY = 11,
  // Per-layer materialized attention score tensor.
  ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_SCORES = 12,
  // Per-layer materialized attention probability tensor.
  ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_PROBABILITIES = 13,
  // Per-layer attention context tensor.
  ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_CONTEXT = 14,
  // Per-layer attention output projection tensor.
  ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_OUTPUT = 15,
  // Per-layer post-attention residual output.
  ID4_QWEN3_VL_TENSOR_LAYER_AFTER_ATTENTION = 16,
  // Per-layer post-attention RMSNorm output.
  ID4_QWEN3_VL_TENSOR_LAYER_POST_ATTENTION_NORM = 17,
  // Per-layer MLP gate projection output.
  ID4_QWEN3_VL_TENSOR_LAYER_MLP_GATE = 18,
  // Per-layer MLP up projection output.
  ID4_QWEN3_VL_TENSOR_LAYER_MLP_UP = 19,
  // Per-layer SiLU-gated activation output.
  ID4_QWEN3_VL_TENSOR_LAYER_MLP_ACTIVATION = 20,
  // Per-layer MLP down projection output.
  ID4_QWEN3_VL_TENSOR_LAYER_MLP_DOWN = 21,
  // Per-layer decoder output after MLP residual.
  ID4_QWEN3_VL_TENSOR_LAYER_OUTPUT = 22,
  // Concatenated selected post-layer hidden states.
  ID4_QWEN3_VL_TENSOR_SELECTED_HIDDEN_STATES = 23,
  // Token-weighted condition output.
  ID4_QWEN3_VL_TENSOR_CONDITION = 24,
  // Partial condition normalization statistics buffer.
  ID4_QWEN3_VL_TENSOR_CONDITION_STATS = 25,
  // Final decoder RMSNorm output.
  ID4_QWEN3_VL_TENSOR_FINAL_NORM = 26,
  // Prompt-local token embedding rows gathered from the model embedding table.
  ID4_QWEN3_VL_TENSOR_TOKEN_EMBEDDING_ROWS = 27,
  // Exported condition tensor.
  ID4_QWEN3_VL_TENSOR_OUTPUT = 28,
} id4_qwen3_vl_tensor_kind_t;

typedef enum id4_qwen3_vl_operation_kind_e {
  // Token embedding lookup.
  ID4_QWEN3_VL_OPERATION_TOKEN_EMBEDDING = 0,
  // RMSNorm operation.
  ID4_QWEN3_VL_OPERATION_RMSNORM = 1,
  // Linear projection operation.
  ID4_QWEN3_VL_OPERATION_LINEAR = 2,
  // Rotary embedding application.
  ID4_QWEN3_VL_OPERATION_ROTARY = 3,
  // Attention score/probability/value operation.
  ID4_QWEN3_VL_OPERATION_ATTENTION = 4,
  // SiLU and gate multiplication operation.
  ID4_QWEN3_VL_OPERATION_SILU_GATE = 5,
  // Residual add operation.
  ID4_QWEN3_VL_OPERATION_RESIDUAL_ADD = 6,
  // Selected hidden-state packing operation.
  ID4_QWEN3_VL_OPERATION_SELECTED_HIDDEN_PACK = 7,
  // Condition token-weight application operation.
  ID4_QWEN3_VL_OPERATION_CONDITION_APPLY_TOKEN_WEIGHTS = 8,
  // Condition token-weight statistics reduction operation.
  ID4_QWEN3_VL_OPERATION_CONDITION_REDUCE_TOKEN_WEIGHT_STATS = 9,
  // Condition token-weight normalization operation.
  ID4_QWEN3_VL_OPERATION_CONDITION_NORMALIZE_TOKEN_WEIGHTS = 10,
  // Execution barrier operation.
  ID4_QWEN3_VL_OPERATION_BARRIER = 11,
  // Diagnostic tap operation.
  ID4_QWEN3_VL_OPERATION_TAP = 12,
} id4_qwen3_vl_operation_kind_t;

typedef enum id4_qwen3_vl_operation_site_e {
  // Global token embedding site.
  ID4_QWEN3_VL_OPERATION_SITE_TOKEN_EMBEDDING = 0,
  // Barrier after token embedding.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_TOKEN_EMBEDDING = 1,
  // Per-layer input RMSNorm site.
  ID4_QWEN3_VL_OPERATION_SITE_INPUT_LAYERNORM = 2,
  // Barrier after per-layer input RMSNorm.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_INPUT_LAYERNORM = 3,
  // Per-layer query projection site.
  ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_Q_PROJECTION = 4,
  // Per-layer key projection site.
  ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_K_PROJECTION = 5,
  // Per-layer value projection site.
  ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_V_PROJECTION = 6,
  // Barrier after per-layer QKV projections.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_QKV_PROJECTION = 7,
  // Per-layer query head RMSNorm site.
  ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_Q_NORM = 8,
  // Per-layer key head RMSNorm site.
  ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_K_NORM = 9,
  // Barrier after per-layer Q/K head RMSNorm.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_QK_NORM = 10,
  // Per-layer query rotary site.
  ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_Q = 11,
  // Per-layer key rotary site.
  ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_K = 12,
  // Barrier after per-layer rotary embedding.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_ROTARY_EMBEDDING = 13,
  // Per-layer self-attention score site.
  ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_SCORES = 14,
  // Barrier after per-layer attention scores.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION_SCORES = 15,
  // Per-layer self-attention softmax site.
  ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_SOFTMAX = 16,
  // Barrier after per-layer attention softmax.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION_SOFTMAX = 17,
  // Per-layer self-attention probability/value site.
  ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION = 18,
  // Barrier after per-layer attention.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION = 19,
  // Per-layer attention output projection site.
  ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_O_PROJECTION = 20,
  // Barrier after per-layer attention output projection.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION_OUTPUT_PROJECTION = 21,
  // Barrier after per-layer attention residual.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION_RESIDUAL = 22,
  // Per-layer post-attention RMSNorm site.
  ID4_QWEN3_VL_OPERATION_SITE_POST_ATTENTION_LAYERNORM = 23,
  // Barrier after per-layer post-attention RMSNorm.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_POST_ATTENTION_LAYERNORM = 24,
  // Per-layer MLP gate projection site.
  ID4_QWEN3_VL_OPERATION_SITE_MLP_GATE_PROJECTION = 25,
  // Per-layer MLP up projection site.
  ID4_QWEN3_VL_OPERATION_SITE_MLP_UP_PROJECTION = 26,
  // Barrier after per-layer MLP up/gate projections.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_MLP_UP_GATE_PROJECTION = 27,
  // Per-layer MLP activation site.
  ID4_QWEN3_VL_OPERATION_SITE_MLP = 28,
  // Barrier after per-layer MLP activation.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_MLP_ACTIVATION = 29,
  // Per-layer MLP down projection site.
  ID4_QWEN3_VL_OPERATION_SITE_MLP_DOWN_PROJECTION = 30,
  // Barrier after per-layer MLP down projection.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_MLP_DOWN_PROJECTION = 31,
  // Barrier after per-layer decoder output.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_LAYER_OUTPUT = 32,
  // Per-layer selected hidden-state pack site.
  ID4_QWEN3_VL_OPERATION_SITE_SELECTED_HIDDEN_PACK = 33,
  // Barrier after all selected hidden states are packed.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_SELECTED_HIDDEN_PACK = 34,
  // Global condition token-weight application site.
  ID4_QWEN3_VL_OPERATION_SITE_CONDITION_APPLY_TOKEN_WEIGHTS = 35,
  // Barrier after condition token-weight application.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_CONDITION_APPLY_TOKEN_WEIGHTS = 36,
  // Global condition token-weight statistics reduction site.
  ID4_QWEN3_VL_OPERATION_SITE_CONDITION_REDUCE_TOKEN_WEIGHT_STATS = 37,
  // Barrier after condition token-weight statistics reduction.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_CONDITION_REDUCE_TOKEN_WEIGHT_STATS = 38,
  // Global condition token-weight normalization site.
  ID4_QWEN3_VL_OPERATION_SITE_CONDITION_NORMALIZE_TOKEN_WEIGHTS = 39,
  // Barrier after the exported condition tensor is complete.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_CONDITION = 40,
  // Final decoder RMSNorm site.
  ID4_QWEN3_VL_OPERATION_SITE_FINAL_NORM = 41,
  // Barrier after final decoder RMSNorm.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_FINAL_NORM = 42,
  // Barrier after per-layer value projection packing.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_VALUE_PACK = 43,
  // Forward output diagnostic site.
  ID4_QWEN3_VL_OPERATION_SITE_OUTPUT = 44,
} id4_qwen3_vl_operation_site_t;

typedef enum id4_qwen3_vl_kernel_kind_e {
  // Token embedding kernel.
  ID4_QWEN3_VL_KERNEL_TOKEN_EMBEDDING = 0,
  // RMSNorm kernel.
  ID4_QWEN3_VL_KERNEL_RMSNORM = 1,
  // Dense BF16 activation/weight linear kernel.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16 = 2,
  // Rotary embedding kernel.
  ID4_QWEN3_VL_KERNEL_ROTARY = 3,
  // Materialized attention score kernel.
  ID4_QWEN3_VL_KERNEL_ATTENTION_SCORES = 4,
  // Materialized attention softmax kernel.
  ID4_QWEN3_VL_KERNEL_ATTENTION_SOFTMAX = 5,
  // Materialized attention probability/value kernel.
  ID4_QWEN3_VL_KERNEL_ATTENTION_PV = 6,
  // SiLU gate kernel.
  ID4_QWEN3_VL_KERNEL_SILU_GATE = 7,
  // Residual add kernel.
  ID4_QWEN3_VL_KERNEL_RESIDUAL_ADD = 8,
  // Selected hidden-state packing kernel.
  ID4_QWEN3_VL_KERNEL_SELECTED_HIDDEN_PACK = 9,
  // Condition token-weight application kernel.
  ID4_QWEN3_VL_KERNEL_CONDITION_APPLY_TOKEN_WEIGHTS = 10,
  // Condition token-weight statistics reduction kernel.
  ID4_QWEN3_VL_KERNEL_CONDITION_REDUCE_TOKEN_WEIGHT_STATS = 11,
  // Condition token-weight normalization kernel.
  ID4_QWEN3_VL_KERNEL_CONDITION_NORMALIZE_TOKEN_WEIGHTS = 12,
  // Dense BF16 activation/weight WMMA linear kernel for 16-token tiles.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M16N64 = 13,
  // Dense BF16 activation/weight 16x64 WMMA linear kernel with packed output.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M16N64_TRANSPOSE_OUTPUT = 14,
  // Dense BF16 activation/weight WMMA linear kernel for 32x32 tiles.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32 = 15,
  // Dense BF16 activation/weight 32x32 WMMA linear kernel with packed output.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32_TRANSPOSE_OUTPUT = 16,
  // Dense BF16 activation/weight WMMA linear kernel for 32x64 tiles.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N64 = 17,
  // Compact RHS BF16 activation/weight WMMA linear kernel for 32x64 tiles.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N64_COMPACT_RHS = 18,
  // Dense BF16 activation/weight WMMA linear kernel for 64x64 tiles.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M64N64 = 19,
  // Compact RHS BF16 activation/weight WMMA linear kernel for 64x64 tiles.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M64N64_COMPACT_RHS = 20,
  // Dense BF16 activation/weight WMMA linear kernel for 48x32 tiles.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M48N32 = 21,
  // Dense BF16 activation/weight 32x64 WMMA linear kernel with packed output.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N64_TRANSPOSE_OUTPUT = 22,
  // Compact RHS BF16 activation/weight 32x64 WMMA linear kernel with packed
  // output.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N64_COMPACT_RHS_TRANSPOSE_OUTPUT =
      23,
  // Dense BF16 activation/weight scalar linear tail kernel.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_TAIL = 24,
  // BF16 activation transpose pack kernel.
  ID4_QWEN3_VL_KERNEL_LINEAR_INPUT_PACK_TRANSPOSE_BF16_BF16 = 25,
  // Materialized QK score WMMA kernel.
  ID4_QWEN3_VL_KERNEL_ATTENTION_QK_SCORES_WMMA = 26,
  // Masked BF16 softmax kernel for WMMA attention.
  ID4_QWEN3_VL_KERNEL_ATTENTION_SOFTMAX_MASK_BF16 = 27,
  // Materialized PV WMMA kernel.
  ID4_QWEN3_VL_KERNEL_ATTENTION_PV_WMMA = 28,
  // Fused MLP gate/up/SwiGLU WMMA kernel for 16-token tiles.
  ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_BF16_WMMA_M16N32 = 29,
  // Fused MLP gate/up/SwiGLU WMMA kernel for 32-token tiles.
  ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_BF16_WMMA_M32N32 = 30,
  // BF16 tensor tail zeroing kernel.
  ID4_QWEN3_VL_KERNEL_ZERO_TAIL_BF16 = 31,
  // Fused residual add followed by RMSNorm kernel.
  ID4_QWEN3_VL_KERNEL_RESIDUAL_ADD_RMSNORM = 32,
  // Fused BF16 WMMA linear projection followed by residual add.
  ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M32N32 = 33,
  // Fused BF16 WMMA linear projection followed by residual add.
  ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M64N64 = 34,
  // Fused head RMSNorm followed by rotary embedding.
  ID4_QWEN3_VL_KERNEL_RMSNORM_ROTARY = 35,
  // Compact RHS fused MLP gate/up/SwiGLU WMMA kernel for 32-token tiles.
  ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_BF16_WMMA_M32N32_COMPACT_RHS =
      36,
  // Compact RHS fused BF16 WMMA linear projection followed by residual add.
  ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M64N64_COMPACT_RHS = 37,
  // Compact RHS fused BF16 WMMA linear projection followed by residual add.
  ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M32N32_COMPACT_RHS = 38,
  // Compact RHS BF16 activation/weight WMMA linear kernel for 48x32 tiles.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M48N32_COMPACT_RHS = 39,
  // Compact RHS FP8 block-scaled weight WMMA linear kernel for 64x64 tiles.
  ID4_QWEN3_VL_KERNEL_LINEAR_FP8_BLOCK_SCALED_BF16_WMMA_M64N64_COMPACT_RHS = 40,
  // Compact RHS block-scaled FP8 fused MLP gate/up/SwiGLU WMMA kernel.
  ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_FP8_BLOCK_SCALED_BF16_WMMA_M32N32_COMPACT_RHS =
      41,
  // Compact RHS block-scaled FP8 fused MLP down/residual WMMA kernel.
  ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_FP8_BLOCK_SCALED_BF16_WMMA_M64N64_COMPACT_RHS =
      42,
  // Online attention kernel that avoids materialized scores/probabilities.
  ID4_QWEN3_VL_KERNEL_ATTENTION_ONLINE_BF16 = 43,
  // Online BF16 WMMA attention kernel that avoids materialized
  // scores/probabilities.
  ID4_QWEN3_VL_KERNEL_ATTENTION_ONLINE_BF16_WMMA = 44,
  // Compact RHS BF16 activation/weight two-wave WMMA linear kernel for 128x64
  // tiles.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M128N64_2WAVE_COMPACT_RHS = 45,
  // Compact RHS fused BF16 WMMA linear projection followed by residual add for
  // 128x64 two-wave tiles.
  ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M128N64_2WAVE_COMPACT_RHS =
      46,
  // Compact RHS fused MLP gate/up/SwiGLU BF16 WMMA kernel for 128x32 two-wave
  // tiles.
  ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_BF16_WMMA_M128N32_2WAVE_COMPACT_RHS =
      47,
  // Compact RHS FP8 block-scaled weight WMMA linear kernel for 128x64 two-wave
  // tiles.
  ID4_QWEN3_VL_KERNEL_LINEAR_FP8_BLOCK_SCALED_BF16_WMMA_M128N64_2WAVE_COMPACT_RHS =
      48,
  // Compact RHS block-scaled FP8 fused MLP down/residual WMMA kernel for
  // 128x64 two-wave tiles.
  ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_FP8_BLOCK_SCALED_BF16_WMMA_M128N64_2WAVE_COMPACT_RHS =
      49,
  // Compact RHS block-scaled FP8 fused MLP gate/up/SwiGLU WMMA kernel for
  // 128x32 two-wave tiles.
  ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_FP8_BLOCK_SCALED_BF16_WMMA_M128N32_2WAVE_COMPACT_RHS =
      50,
  // Compact RHS BF16 activation/weight WMMA linear kernel for 32x32 tiles.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32_COMPACT_RHS = 51,
  // Compact RHS BF16 activation/weight 32x32 WMMA linear kernel with packed
  // output.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32_COMPACT_RHS_TRANSPOSE_OUTPUT =
      52,
} id4_qwen3_vl_kernel_kind_t;

typedef enum id4_qwen3_vl_config_key_e {
  // Token count config key.
  ID4_QWEN3_VL_CONFIG_TOKEN_COUNT = 0,
  // Hidden channel count config key.
  ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE = 1,
  // Linear input channel count config key.
  ID4_QWEN3_VL_CONFIG_INPUT_SIZE = 2,
  // Linear output channel count config key.
  ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE = 3,
  // MLP intermediate channel count config key.
  ID4_QWEN3_VL_CONFIG_INTERMEDIATE_SIZE = 4,
  // Query attention head count config key.
  ID4_QWEN3_VL_CONFIG_ATTENTION_HEAD_COUNT = 5,
  // Key/value attention head count config key.
  ID4_QWEN3_VL_CONFIG_KEY_VALUE_HEAD_COUNT = 6,
  // Generic attention head count config key for head-local kernels.
  ID4_QWEN3_VL_CONFIG_HEAD_COUNT = 7,
  // Per-head channel count config key.
  ID4_QWEN3_VL_CONFIG_HEAD_SIZE = 8,
  // Vocabulary row count config key.
  ID4_QWEN3_VL_CONFIG_VOCAB_SIZE = 9,
  // Selected hidden-state row count config key.
  ID4_QWEN3_VL_CONFIG_HIDDEN_ROW_COUNT = 10,
  // Selected-layer output slot config key.
  ID4_QWEN3_VL_CONFIG_SELECTED_LAYER_INDEX = 11,
  // Number of token rows handled by one dispatch.
  ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT = 12,
  // First logical token row handled by a partial dispatch.
  ID4_QWEN3_VL_CONFIG_TOKEN_START = 13,
  // Number of valid elements in an input tensor.
  ID4_QWEN3_VL_CONFIG_ELEMENT_COUNT = 14,
  // Total number of elements covered by a padded dispatch tensor.
  ID4_QWEN3_VL_CONFIG_DISPATCH_ELEMENT_COUNT = 15,
} id4_qwen3_vl_config_key_t;

enum {
  ID4_QWEN3_VL_PARAMETER_KIND_COUNT = ID4_QWEN3_VL_PARAMETER_FINAL_NORM + 1,
  ID4_QWEN3_VL_TENSOR_KIND_COUNT = ID4_QWEN3_VL_TENSOR_OUTPUT + 1,
  ID4_QWEN3_VL_OPERATION_KIND_COUNT = ID4_QWEN3_VL_OPERATION_TAP + 1,
  ID4_QWEN3_VL_OPERATION_SITE_COUNT = ID4_QWEN3_VL_OPERATION_SITE_OUTPUT + 1,
  ID4_QWEN3_VL_KERNEL_KIND_COUNT =
      ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32_COMPACT_RHS_TRANSPOSE_OUTPUT +
      1,
  ID4_QWEN3_VL_CONFIG_KEY_COUNT =
      ID4_QWEN3_VL_CONFIG_DISPATCH_ELEMENT_COUNT + 1,
};

typedef enum id4_qwen3_vl_program_operation_name_pattern_kind_e {
  // Pattern for token embedding, which has no appended operation kind.
  ID4_QWEN3_VL_PROGRAM_OPERATION_NAME_TOKEN_EMBEDDING = 0,
  // Pattern for per-layer operations with an appended operation kind.
  ID4_QWEN3_VL_PROGRAM_OPERATION_NAME_LAYER_OPERATION = 1,
  // Pattern for global operations with an appended operation kind.
  ID4_QWEN3_VL_PROGRAM_OPERATION_NAME_GLOBAL_OPERATION = 2,
  // Total number of operation name pattern rows.
  ID4_QWEN3_VL_PROGRAM_OPERATION_NAME_PATTERN_COUNT = 3,
} id4_qwen3_vl_program_operation_name_pattern_kind_t;

typedef struct id4_qwen3_vl_program_format_pattern_t {
  // True when the pattern requires a layer ordinal argument.
  bool uses_layer_ordinal;
  // Static printf-compatible format pattern.
  iree_string_view_t pattern;
} id4_qwen3_vl_program_format_pattern_t;

typedef struct id4_qwen3_vl_program_operation_kind_entry_t {
  // True when the operation kind appends its name after the site.
  bool appends_operation_name;
  // Static operation kind name.
  iree_string_view_t operation_name;
} id4_qwen3_vl_program_operation_kind_entry_t;

typedef struct id4_qwen3_vl_program_operation_site_entry_t {
  // True when the site is per-layer.
  bool uses_layer_ordinal;
  // Static operation site name.
  iree_string_view_t site_name;
} id4_qwen3_vl_program_operation_site_entry_t;

typedef struct id4_qwen3_vl_program_config_value_t {
  // Loom config key selected from the kernel-specific config table.
  id4_qwen3_vl_config_key_t key;
  // Unsigned integer config value formatted for loomc.
  uint32_t value;
} id4_qwen3_vl_program_config_value_t;

typedef struct id4_qwen3_vl_program_linear_wmma_kernel_set_t {
  // Kernel consuming direct row-major RHS weights and writing token-major rows.
  id4_qwen3_vl_kernel_kind_t direct_output_kernel_kind;
  // Kernel consuming compact RHS tiles and writing token-major rows.
  id4_qwen3_vl_kernel_kind_t compact_rhs_output_kernel_kind;
  // Kernel consuming direct row-major RHS weights and writing packed rows.
  id4_qwen3_vl_kernel_kind_t direct_transposed_output_kernel_kind;
  // Kernel consuming compact RHS tiles and writing packed rows.
  id4_qwen3_vl_kernel_kind_t compact_rhs_transposed_output_kernel_kind;
} id4_qwen3_vl_program_linear_wmma_kernel_set_t;

typedef struct id4_qwen3_vl_program_linear_wmma_tile_t {
  // Number of token rows covered by one WMMA dispatch tile.
  uint32_t token_block;
  // Number of output rows covered by one WMMA dispatch tile.
  uint32_t output_row_block;
  // Kernel set implementing this tile shape.
  id4_qwen3_vl_program_linear_wmma_kernel_set_t kernels;
} id4_qwen3_vl_program_linear_wmma_tile_t;

typedef struct id4_qwen3_vl_program_mlp_gate_up_silu_product_wmma_tile_t {
  // Number of token rows covered by one WMMA dispatch tile.
  uint32_t token_block;
  // Kernel consuming direct row-major gate and up weights.
  id4_qwen3_vl_kernel_kind_t direct_kernel_kind;
  // Kernel consuming compact RHS gate and up weight tiles.
  id4_qwen3_vl_kernel_kind_t compact_rhs_kernel_kind;
  // Kernel consuming compact RHS block-scaled FP8 gate and up weight tiles.
  id4_qwen3_vl_kernel_kind_t fp8_compact_rhs_kernel_kind;
  // Minimum token capacity required before compact RHS is selected.
  uint32_t compact_rhs_min_token_count;
} id4_qwen3_vl_program_mlp_gate_up_silu_product_wmma_tile_t;

typedef struct id4_qwen3_vl_program_linear_kernel_selection_t {
  // Kernel kind selected for the dispatch.
  id4_qwen3_vl_kernel_kind_t kernel_kind;
  // Parameter encoding required by the selected kernel.
  id4_pipeline_program_parameter_encoding_t weight_encoding;
} id4_qwen3_vl_program_linear_kernel_selection_t;

typedef struct id4_qwen3_vl_program_linear_weight_t {
  // Packed or direct execution weight tensor read by the linear dispatch.
  id4_pipeline_program_tensor_t weight;
  // Optional F32 block scale tensor read by FP8 block-scaled dispatches.
  id4_pipeline_program_tensor_t scale;
  // True when the selected source format requires a separate scale tensor.
  bool has_scale;
  // True when prior dispatches must be made visible before reading weight.
  bool requires_ready_barrier;
} id4_qwen3_vl_program_linear_weight_t;

typedef enum id4_qwen3_vl_program_linear_dispatch_flag_bits_e {
  // Inserts the selected weight's ready barrier before the consumer dispatch.
  ID4_QWEN3_VL_PROGRAM_LINEAR_DISPATCH_FLAG_INSERT_WEIGHT_READY_BARRIER = 1u
                                                                          << 0,
} id4_qwen3_vl_program_linear_dispatch_flag_bits_t;

typedef uint32_t id4_qwen3_vl_program_linear_dispatch_flags_t;

static bool id4_qwen3_vl_program_kernel_kind_is_valid(
    id4_qwen3_vl_kernel_kind_t kernel_kind) {
  return kernel_kind !=
         (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT;
}

static iree_status_t id4_qwen3_vl_program_select_linear_kernel(
    id4_qwen3_vl_weight_execution_strategy_t strategy,
    id4_qwen3_vl_kernel_kind_t direct_kernel_kind,
    id4_qwen3_vl_kernel_kind_t compact_rhs_kernel_kind,
    id4_qwen3_vl_program_linear_kernel_selection_t* out_selection) {
  switch (strategy) {
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR:
      if (!id4_qwen3_vl_program_kernel_kind_is_valid(direct_kernel_kind)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Qwen3-VL row-major weight execution strategy is not supported by "
            "the selected linear tile");
      }
      out_selection->kernel_kind = direct_kernel_kind;
      out_selection->weight_encoding =
          ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT;
      return iree_ok_status();
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_COMPACT_RHS:
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_STREAMING_COMPACT_RHS:
      if (!id4_qwen3_vl_program_kernel_kind_is_valid(compact_rhs_kernel_kind)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "Qwen3-VL compact-RHS weight execution "
                                "strategy is not supported by "
                                "the selected linear tile");
      }
      out_selection->kernel_kind = compact_rhs_kernel_kind;
      out_selection->weight_encoding =
          ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_BF16_LINEAR_RHS_TILE;
      return iree_ok_status();
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS:
      if (id4_qwen3_vl_program_kernel_kind_is_valid(compact_rhs_kernel_kind)) {
        out_selection->kernel_kind = compact_rhs_kernel_kind;
        out_selection->weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_BF16_LINEAR_RHS_TILE;
        return iree_ok_status();
      }
      if (id4_qwen3_vl_program_kernel_kind_is_valid(direct_kernel_kind)) {
        out_selection->kernel_kind = direct_kernel_kind;
        out_selection->weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT;
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Qwen3-VL hybrid compact-RHS weight execution strategy is not "
          "supported by the selected linear tile");
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen3-VL weight execution strategy %" PRIu32
                              " is invalid",
                              (uint32_t)strategy);
  }
}

static iree_status_t id4_qwen3_vl_program_select_linear_kernel_for_source(
    id4_qwen3_vl_parameter_format_t parameter_format,
    id4_qwen3_vl_weight_execution_strategy_t strategy,
    const id4_qwen3_vl_program_linear_wmma_tile_t* wmma_tile,
    uint32_t wmma_token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size,
    id4_qwen3_vl_program_linear_kernel_selection_t* out_selection) {
  switch (parameter_format) {
    case ID4_QWEN3_VL_PARAMETER_FORMAT_BF16:
      if (wmma_token_count == 0) return iree_ok_status();
      return id4_qwen3_vl_program_select_linear_kernel(
          strategy, wmma_tile->kernels.direct_output_kernel_kind,
          wmma_tile->kernels.compact_rhs_output_kernel_kind, out_selection);
    case ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED:
      if (strategy == ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Qwen3-VL block-scaled FP8 source weights require compact-RHS "
            "weight execution");
      }
      if (strategy ==
          ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_STREAMING_COMPACT_RHS) {
        if (wmma_token_count == 0 || wmma_token_count != token_capacity) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "Qwen3-VL streaming compact-RHS weight execution requires full "
              "WMMA coverage");
        }
        if (!id4_qwen3_vl_program_kernel_kind_is_valid(
                wmma_tile->kernels.compact_rhs_output_kernel_kind)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "Qwen3-VL streaming compact-RHS weight execution is not "
              "supported by the selected linear tile");
        }
        out_selection->kernel_kind =
            wmma_tile->kernels.compact_rhs_output_kernel_kind;
        out_selection->weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_BLOCK_SCALED_TO_BF16_LINEAR_RHS_TILE;
        return iree_ok_status();
      }
      if (wmma_token_count == 0 || wmma_token_count != token_capacity ||
          (input_size % ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE) != 0 ||
          (output_size % ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE) != 0) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "Qwen3-VL block-scaled FP8 source weights currently require "
            "full WMMA coverage with 128-aligned input and output sizes");
      }
      if (wmma_tile->token_block == ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M128 &&
          wmma_tile->output_row_block ==
              ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK_M64) {
        out_selection->kernel_kind =
            ID4_QWEN3_VL_KERNEL_LINEAR_FP8_BLOCK_SCALED_BF16_WMMA_M128N64_2WAVE_COMPACT_RHS;
        out_selection->weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE;
        return iree_ok_status();
      }
      if ((token_capacity % ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M64) != 0) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "Qwen3-VL block-scaled FP8 source weights currently require "
            "token capacity compatible with the m64n64 WMMA tile");
      }
      out_selection->kernel_kind =
          ID4_QWEN3_VL_KERNEL_LINEAR_FP8_BLOCK_SCALED_BF16_WMMA_M64N64_COMPACT_RHS;
      out_selection->weight_encoding =
          ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen3-VL parameter format %" PRIu32
                              " is invalid",
                              (uint32_t)parameter_format);
  }
}

static bool id4_qwen3_vl_program_selection_uses_compact_rhs(
    const id4_qwen3_vl_program_linear_kernel_selection_t* selection) {
  return selection->weight_encoding ==
             ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_BF16_LINEAR_RHS_TILE ||
         selection->weight_encoding ==
             ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE ||
         selection->weight_encoding ==
             ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_BLOCK_SCALED_TO_BF16_LINEAR_RHS_TILE;
}

static const id4_qwen3_vl_program_linear_wmma_tile_t
    id4_qwen3_vl_program_linear_wmma_m32n32_tile = {
        // Number of token rows covered by one WMMA dispatch tile.
        .token_block = ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M32,
        // Number of output rows covered by one WMMA dispatch tile.
        .output_row_block = ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK_M32,
        // Kernel set implementing this tile shape.
        .kernels =
            {
                .direct_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32,
                .compact_rhs_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32_COMPACT_RHS,
                .direct_transposed_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32_TRANSPOSE_OUTPUT,
                .compact_rhs_transposed_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32_COMPACT_RHS_TRANSPOSE_OUTPUT,
            },
};

static const id4_qwen3_vl_program_linear_wmma_tile_t
    id4_qwen3_vl_program_linear_wmma_m32n64_tile = {
        // Number of token rows covered by one WMMA dispatch tile.
        .token_block = ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M32,
        // Number of output rows covered by one WMMA dispatch tile.
        .output_row_block = ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK_M64,
        // Kernel set implementing this tile shape.
        .kernels =
            {
                .direct_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N64,
                .compact_rhs_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N64_COMPACT_RHS,
                .direct_transposed_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N64_TRANSPOSE_OUTPUT,
                .compact_rhs_transposed_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N64_COMPACT_RHS_TRANSPOSE_OUTPUT,
            },
};

static const id4_qwen3_vl_program_linear_wmma_tile_t
    id4_qwen3_vl_program_linear_wmma_m48n32_tile = {
        // Number of token rows covered by one WMMA dispatch tile.
        .token_block = ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M48,
        // Number of output rows covered by one WMMA dispatch tile.
        .output_row_block = ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK_M32,
        // Kernel set implementing this tile shape.
        .kernels =
            {
                .direct_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M48N32,
                .compact_rhs_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M48N32_COMPACT_RHS,
                .direct_transposed_output_kernel_kind =
                    (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
                .compact_rhs_transposed_output_kernel_kind =
                    (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
            },
};

static const id4_qwen3_vl_program_linear_wmma_tile_t
    id4_qwen3_vl_program_linear_wmma_m64n64_tile = {
        // Number of token rows covered by one WMMA dispatch tile.
        .token_block = ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M64,
        // Number of output rows covered by one WMMA dispatch tile.
        .output_row_block = ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK_M64,
        // Kernel set implementing this tile shape.
        .kernels =
            {
                .direct_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M64N64,
                .compact_rhs_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M64N64_COMPACT_RHS,
                .direct_transposed_output_kernel_kind =
                    (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
                .compact_rhs_transposed_output_kernel_kind =
                    (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
            },
};

static const id4_qwen3_vl_program_linear_wmma_tile_t
    id4_qwen3_vl_program_linear_wmma_m128n64_2wave_tile = {
        // Number of token rows covered by one WMMA dispatch tile.
        .token_block = ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M128,
        // Number of output rows covered by one WMMA dispatch tile.
        .output_row_block = ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK_M64,
        // Kernel set implementing this tile shape.
        .kernels =
            {
                .direct_output_kernel_kind =
                    (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
                .compact_rhs_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M128N64_2WAVE_COMPACT_RHS,
                .direct_transposed_output_kernel_kind =
                    (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
                .compact_rhs_transposed_output_kernel_kind =
                    (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
            },
};

static const id4_qwen3_vl_program_linear_wmma_tile_t
    id4_qwen3_vl_program_linear_wmma_m16n64_tile = {
        // Number of token rows covered by one WMMA dispatch tile.
        .token_block = ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M16,
        // Number of output rows covered by one WMMA dispatch tile.
        .output_row_block = ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK_M64,
        // Kernel set implementing this tile shape.
        .kernels =
            {
                .direct_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M16N64,
                .compact_rhs_output_kernel_kind =
                    (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
                .direct_transposed_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M16N64_TRANSPOSE_OUTPUT,
                .compact_rhs_transposed_output_kernel_kind =
                    (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
            },
};

static const id4_qwen3_vl_program_mlp_gate_up_silu_product_wmma_tile_t
    id4_qwen3_vl_program_mlp_gate_up_silu_product_wmma_m128n32_2wave_tile = {
        // Number of token rows covered by one WMMA dispatch tile.
        .token_block = ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M128,
        // No row-major consumer is selected for this tile shape.
        .direct_kernel_kind =
            (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
        // Kernel consuming compact RHS gate and up weight tiles.
        .compact_rhs_kernel_kind =
            ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_BF16_WMMA_M128N32_2WAVE_COMPACT_RHS,
        // Kernel consuming compact RHS block-scaled FP8 gate and up weight
        // tiles.
        .fp8_compact_rhs_kernel_kind =
            ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_FP8_BLOCK_SCALED_BF16_WMMA_M128N32_2WAVE_COMPACT_RHS,
        // Minimum token capacity required before compact RHS is selected.
        .compact_rhs_min_token_count =
            ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M128,
};

static const id4_qwen3_vl_program_mlp_gate_up_silu_product_wmma_tile_t
    id4_qwen3_vl_program_mlp_gate_up_silu_product_wmma_m32n32_tile = {
        // Number of token rows covered by one WMMA dispatch tile.
        .token_block = ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M32,
        // Kernel consuming direct row-major gate and up weights.
        .direct_kernel_kind =
            ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_BF16_WMMA_M32N32,
        // Kernel consuming compact RHS gate and up weight tiles.
        .compact_rhs_kernel_kind =
            ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_BF16_WMMA_M32N32_COMPACT_RHS,
        // Kernel consuming compact RHS block-scaled FP8 gate and up weight
        // tiles.
        .fp8_compact_rhs_kernel_kind =
            ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_FP8_BLOCK_SCALED_BF16_WMMA_M32N32_COMPACT_RHS,
        // Minimum token capacity required before compact RHS is selected.
        .compact_rhs_min_token_count = ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M32,
};

static const id4_qwen3_vl_program_mlp_gate_up_silu_product_wmma_tile_t
    id4_qwen3_vl_program_mlp_gate_up_silu_product_wmma_m16n32_tile = {
        // Number of token rows covered by one WMMA dispatch tile.
        .token_block = ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M16,
        // Kernel consuming direct row-major gate and up weights.
        .direct_kernel_kind =
            ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_BF16_WMMA_M16N32,
        // No compact RHS consumer is selected for this tile shape.
        .compact_rhs_kernel_kind =
            (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
        // No compact RHS block-scaled FP8 consumer is selected for this tile
        // shape.
        .fp8_compact_rhs_kernel_kind =
            (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
        // Minimum token capacity required before compact RHS is selected.
        .compact_rhs_min_token_count = UINT32_MAX,
};

static const id4_qwen3_vl_program_linear_wmma_tile_t
    id4_qwen3_vl_program_linear_residual_wmma_m32n32_tile = {
        // Number of token rows covered by one WMMA dispatch tile.
        .token_block = ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M32,
        // Number of output rows covered by one WMMA dispatch tile.
        .output_row_block = ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK_M32,
        // Kernel set implementing this tile shape.
        .kernels =
            {
                .direct_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M32N32,
                .compact_rhs_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M32N32_COMPACT_RHS,
                .direct_transposed_output_kernel_kind =
                    (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
                .compact_rhs_transposed_output_kernel_kind =
                    (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
            },
};

static const id4_qwen3_vl_program_linear_wmma_tile_t
    id4_qwen3_vl_program_linear_residual_wmma_m64n64_tile = {
        // Number of token rows covered by one WMMA dispatch tile.
        .token_block = ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M64,
        // Number of output rows covered by one WMMA dispatch tile.
        .output_row_block = ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK_M64,
        // Kernel set implementing this tile shape.
        .kernels =
            {
                .direct_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M64N64,
                .compact_rhs_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M64N64_COMPACT_RHS,
                .direct_transposed_output_kernel_kind =
                    (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
                .compact_rhs_transposed_output_kernel_kind =
                    (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
            },
};

static const id4_qwen3_vl_program_linear_wmma_tile_t
    id4_qwen3_vl_program_linear_residual_wmma_m128n64_2wave_tile = {
        // Number of token rows covered by one WMMA dispatch tile.
        .token_block = ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M128,
        // Number of output rows covered by one WMMA dispatch tile.
        .output_row_block = ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK_M64,
        // Kernel set implementing this tile shape.
        .kernels =
            {
                .direct_output_kernel_kind =
                    (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
                .compact_rhs_output_kernel_kind =
                    ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M128N64_2WAVE_COMPACT_RHS,
                .direct_transposed_output_kernel_kind =
                    (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
                .compact_rhs_transposed_output_kernel_kind =
                    (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
            },
};

static const id4_qwen3_vl_program_linear_wmma_tile_t*
id4_qwen3_vl_program_select_linear_wmma_tile(
    id4_qwen3_vl_parameter_format_t parameter_format,
    id4_qwen3_vl_weight_execution_strategy_t strategy, uint32_t token_count,
    uint32_t input_size, uint32_t output_size) {
  if ((input_size % ID4_QWEN3_VL_LINEAR_WMMA_INPUT_BLOCK) != 0) {
    return NULL;
  }
  const bool supports_m128_compact_rhs =
      (parameter_format == ID4_QWEN3_VL_PARAMETER_FORMAT_BF16 ||
       (parameter_format ==
            ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED &&
        output_size >= ID4_QWEN3_VL_LINEAR_WMMA_M128N64_FP8_MIN_OUTPUT_SIZE &&
        (input_size % ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE) == 0 &&
        (output_size % ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE) == 0));
  const uint32_t m128_min_output_size =
      parameter_format == ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED
          ? ID4_QWEN3_VL_LINEAR_WMMA_M128N64_FP8_MIN_OUTPUT_SIZE
          : ID4_QWEN3_VL_LINEAR_WMMA_M64N64_MIN_OUTPUT_SIZE;
  if (supports_m128_compact_rhs &&
      strategy != ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR &&
      token_count >= ID4_QWEN3_VL_LINEAR_WMMA_M64N64_MIN_TOKEN_COUNT &&
      output_size >= m128_min_output_size &&
      (token_count %
       id4_qwen3_vl_program_linear_wmma_m128n64_2wave_tile.token_block) == 0 &&
      (output_size %
       id4_qwen3_vl_program_linear_wmma_m128n64_2wave_tile.output_row_block) ==
          0) {
    return &id4_qwen3_vl_program_linear_wmma_m128n64_2wave_tile;
  }
  if (token_count >= ID4_QWEN3_VL_LINEAR_WMMA_M64N64_MIN_TOKEN_COUNT &&
      output_size >= ID4_QWEN3_VL_LINEAR_WMMA_M64N64_MIN_OUTPUT_SIZE &&
      (token_count %
       id4_qwen3_vl_program_linear_wmma_m64n64_tile.token_block) == 0 &&
      (output_size %
       id4_qwen3_vl_program_linear_wmma_m64n64_tile.output_row_block) == 0) {
    return &id4_qwen3_vl_program_linear_wmma_m64n64_tile;
  }
  if (token_count >= ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M48 &&
      (token_count %
       id4_qwen3_vl_program_linear_wmma_m48n32_tile.token_block) == 0 &&
      (output_size %
       id4_qwen3_vl_program_linear_wmma_m48n32_tile.output_row_block) == 0) {
    return &id4_qwen3_vl_program_linear_wmma_m48n32_tile;
  }
  const uint32_t m32_remainder =
      token_count % ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M32;
  if (token_count >= ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M32 &&
      m32_remainder < ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M16) {
    // The wider output tile has higher register pressure; keep it for long
    // token spans where lower dispatch count dominates.
    if (token_count >= ID4_QWEN3_VL_LINEAR_WMMA_M32N64_MIN_TOKEN_COUNT &&
        output_size <= ID4_QWEN3_VL_LINEAR_WMMA_M32N64_MAX_OUTPUT_SIZE &&
        (output_size %
         id4_qwen3_vl_program_linear_wmma_m32n64_tile.output_row_block) == 0) {
      return &id4_qwen3_vl_program_linear_wmma_m32n64_tile;
    }
    if ((output_size %
         id4_qwen3_vl_program_linear_wmma_m32n32_tile.output_row_block) != 0) {
      return NULL;
    }
    return &id4_qwen3_vl_program_linear_wmma_m32n32_tile;
  }
  if (token_count >= ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M16 &&
      (output_size %
       id4_qwen3_vl_program_linear_wmma_m16n64_tile.output_row_block) == 0) {
    return &id4_qwen3_vl_program_linear_wmma_m16n64_tile;
  }
  return NULL;
}

static bool id4_qwen3_vl_program_linear_wmma_tile_has_transposed_output(
    const id4_qwen3_vl_program_linear_wmma_tile_t* wmma_tile) {
  return id4_qwen3_vl_program_kernel_kind_is_valid(
             wmma_tile->kernels.direct_transposed_output_kernel_kind) ||
         id4_qwen3_vl_program_kernel_kind_is_valid(
             wmma_tile->kernels.compact_rhs_transposed_output_kernel_kind);
}

static const id4_qwen3_vl_program_linear_wmma_tile_t*
id4_qwen3_vl_program_select_linear_wmma_transposed_output_tile(
    id4_qwen3_vl_parameter_format_t parameter_format,
    id4_qwen3_vl_weight_execution_strategy_t strategy, uint32_t token_count,
    uint32_t input_size, uint32_t output_size) {
  const id4_qwen3_vl_program_linear_wmma_tile_t* wmma_tile =
      id4_qwen3_vl_program_select_linear_wmma_tile(
          parameter_format, strategy, token_count, input_size, output_size);
  if (wmma_tile &&
      id4_qwen3_vl_program_linear_wmma_tile_has_transposed_output(wmma_tile)) {
    return wmma_tile;
  }
  const uint32_t m32_remainder =
      token_count % ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M32;
  if (token_count >= ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M32 &&
      m32_remainder < ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M16 &&
      (output_size %
       id4_qwen3_vl_program_linear_wmma_m32n32_tile.output_row_block) == 0) {
    return &id4_qwen3_vl_program_linear_wmma_m32n32_tile;
  }
  if (token_count >= ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M16 &&
      (output_size %
       id4_qwen3_vl_program_linear_wmma_m16n64_tile.output_row_block) == 0) {
    return &id4_qwen3_vl_program_linear_wmma_m16n64_tile;
  }
  return NULL;
}

static const id4_qwen3_vl_program_mlp_gate_up_silu_product_wmma_tile_t*
id4_qwen3_vl_program_select_mlp_gate_up_silu_product_wmma_tile(
    id4_qwen3_vl_parameter_format_t parameter_format,
    id4_qwen3_vl_weight_execution_strategy_t strategy, uint32_t token_capacity,
    uint32_t hidden_size, uint32_t intermediate_size) {
  const bool supports_m128_compact_rhs =
      parameter_format == ID4_QWEN3_VL_PARAMETER_FORMAT_BF16 ||
      (parameter_format ==
           ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED &&
       (hidden_size % ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE) == 0 &&
       (intermediate_size % ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE) == 0);
  if (supports_m128_compact_rhs &&
      strategy != ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR &&
      token_capacity >= ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M128 &&
      (token_capacity % ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M128) == 0 &&
      (hidden_size % ID4_QWEN3_VL_LINEAR_WMMA_INPUT_BLOCK) == 0 &&
      (intermediate_size % ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK_M32) ==
          0) {
    return &id4_qwen3_vl_program_mlp_gate_up_silu_product_wmma_m128n32_2wave_tile;
  }
  if (token_capacity >= ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M32) {
    return &id4_qwen3_vl_program_mlp_gate_up_silu_product_wmma_m32n32_tile;
  }
  return &id4_qwen3_vl_program_mlp_gate_up_silu_product_wmma_m16n32_tile;
}

static iree_status_t
id4_qwen3_vl_program_select_mlp_gate_up_silu_product_kernel(
    id4_qwen3_vl_parameter_format_t parameter_format,
    id4_qwen3_vl_weight_execution_strategy_t strategy,
    const id4_qwen3_vl_program_mlp_gate_up_silu_product_wmma_tile_t* wmma_tile,
    uint32_t token_capacity, uint32_t hidden_size, uint32_t intermediate_size,
    id4_qwen3_vl_program_linear_kernel_selection_t* out_selection) {
  switch (parameter_format) {
    case ID4_QWEN3_VL_PARAMETER_FORMAT_BF16:
      return id4_qwen3_vl_program_select_linear_kernel(
          strategy, wmma_tile->direct_kernel_kind,
          wmma_tile->compact_rhs_kernel_kind, out_selection);
    case ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED:
      if (strategy == ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Qwen3-VL block-scaled FP8 source weights require compact-RHS "
            "MLP weight execution");
      }
      if (strategy ==
          ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_STREAMING_COMPACT_RHS) {
        if (!id4_qwen3_vl_program_kernel_kind_is_valid(
                wmma_tile->compact_rhs_kernel_kind)) {
          return iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "Qwen3-VL streaming compact-RHS fused MLP does not support the "
              "selected WMMA tile");
        }
        out_selection->kernel_kind = wmma_tile->compact_rhs_kernel_kind;
        out_selection->weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_BLOCK_SCALED_TO_BF16_LINEAR_RHS_TILE;
        return iree_ok_status();
      }
      if (!id4_qwen3_vl_program_kernel_kind_is_valid(
              wmma_tile->fp8_compact_rhs_kernel_kind)) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "Qwen3-VL block-scaled FP8 fused MLP does not support the "
            "selected WMMA tile");
      }
      if ((token_capacity % wmma_tile->token_block) != 0 ||
          (hidden_size % ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE) != 0 ||
          (intermediate_size % ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE) != 0) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "Qwen3-VL block-scaled FP8 fused MLP requires token capacity "
            "compatible with the selected tile and 128-aligned input and "
            "intermediate sizes");
      }
      out_selection->kernel_kind = wmma_tile->fp8_compact_rhs_kernel_kind;
      out_selection->weight_encoding =
          ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen3-VL parameter format %" PRIu32
                              " is invalid",
                              (uint32_t)parameter_format);
  }
}

static const id4_qwen3_vl_program_linear_wmma_tile_t*
id4_qwen3_vl_program_select_mlp_down_residual_wmma_tile(
    const id4_qwen3_vl_program_options_t* options, uint32_t token_capacity) {
  if ((options->model.intermediate_size %
       ID4_QWEN3_VL_LINEAR_WMMA_INPUT_BLOCK) != 0) {
    return NULL;
  }
  const bool supports_m128_compact_rhs =
      options->parameter_format == ID4_QWEN3_VL_PARAMETER_FORMAT_BF16 ||
      (options->parameter_format ==
           ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED &&
       (options->model.intermediate_size % ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE) ==
           0 &&
       (options->model.hidden_size % ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE) == 0);
  if (supports_m128_compact_rhs &&
      options->weight_execution_strategy !=
          ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR &&
      token_capacity >= ID4_QWEN3_VL_LINEAR_WMMA_M64N64_MIN_TOKEN_COUNT &&
      (token_capacity %
       id4_qwen3_vl_program_linear_residual_wmma_m128n64_2wave_tile
           .token_block) == 0 &&
      (options->model.hidden_size %
       id4_qwen3_vl_program_linear_residual_wmma_m128n64_2wave_tile
           .output_row_block) == 0) {
    return &id4_qwen3_vl_program_linear_residual_wmma_m128n64_2wave_tile;
  }
  if (token_capacity >= ID4_QWEN3_VL_LINEAR_WMMA_M64N64_MIN_TOKEN_COUNT &&
      (token_capacity %
       id4_qwen3_vl_program_linear_residual_wmma_m64n64_tile.token_block) ==
          0 &&
      (options->model.hidden_size %
       id4_qwen3_vl_program_linear_residual_wmma_m64n64_tile
           .output_row_block) == 0) {
    return &id4_qwen3_vl_program_linear_residual_wmma_m64n64_tile;
  }
  if (token_capacity >= ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK_M32 &&
      (token_capacity %
       id4_qwen3_vl_program_linear_residual_wmma_m32n32_tile.token_block) ==
          0 &&
      (options->model.hidden_size %
       id4_qwen3_vl_program_linear_residual_wmma_m32n32_tile
           .output_row_block) == 0) {
    return &id4_qwen3_vl_program_linear_residual_wmma_m32n32_tile;
  }
  return NULL;
}

static iree_status_t id4_qwen3_vl_program_select_mlp_down_residual_kernel(
    id4_qwen3_vl_parameter_format_t parameter_format,
    id4_qwen3_vl_weight_execution_strategy_t strategy,
    const id4_qwen3_vl_program_linear_wmma_tile_t* wmma_tile,
    uint32_t token_capacity, uint32_t input_size, uint32_t output_size,
    id4_qwen3_vl_program_linear_kernel_selection_t* out_selection) {
  switch (parameter_format) {
    case ID4_QWEN3_VL_PARAMETER_FORMAT_BF16:
      return id4_qwen3_vl_program_select_linear_kernel(
          strategy, wmma_tile->kernels.direct_output_kernel_kind,
          wmma_tile->kernels.compact_rhs_output_kernel_kind, out_selection);
    case ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED:
      if (strategy == ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Qwen3-VL block-scaled FP8 source weights require compact-RHS MLP "
            "down/residual weight execution");
      }
      if (strategy ==
          ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_STREAMING_COMPACT_RHS) {
        if (!id4_qwen3_vl_program_kernel_kind_is_valid(
                wmma_tile->kernels.compact_rhs_output_kernel_kind)) {
          return iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "Qwen3-VL streaming compact-RHS MLP down/residual does not "
              "support the selected WMMA tile");
        }
        out_selection->kernel_kind =
            wmma_tile->kernels.compact_rhs_output_kernel_kind;
        out_selection->weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_BLOCK_SCALED_TO_BF16_LINEAR_RHS_TILE;
        return iree_ok_status();
      }
      if (wmma_tile ==
          &id4_qwen3_vl_program_linear_residual_wmma_m128n64_2wave_tile) {
        out_selection->kernel_kind =
            ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_FP8_BLOCK_SCALED_BF16_WMMA_M128N64_2WAVE_COMPACT_RHS;
      } else if (wmma_tile ==
                 &id4_qwen3_vl_program_linear_residual_wmma_m64n64_tile) {
        out_selection->kernel_kind =
            ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_FP8_BLOCK_SCALED_BF16_WMMA_M64N64_COMPACT_RHS;
      } else {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "Qwen3-VL block-scaled FP8 fused MLP down/residual requires the "
            "m64n64 or m128n64 two-wave WMMA tile");
      }
      if ((token_capacity % wmma_tile->token_block) != 0 ||
          (input_size % ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE) != 0 ||
          (output_size % ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE) != 0) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "Qwen3-VL block-scaled FP8 fused MLP down/residual requires token "
            "capacity compatible with the selected tile and 128-aligned input "
            "and output sizes");
      }
      out_selection->weight_encoding =
          ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen3-VL parameter format %" PRIu32
                              " is invalid",
                              (uint32_t)parameter_format);
  }
}

static const char* const id4_qwen3_vl_program_operation_name_patterns
    [ID4_QWEN3_VL_PROGRAM_OPERATION_NAME_PATTERN_COUNT] = {
        [ID4_QWEN3_VL_PROGRAM_OPERATION_NAME_TOKEN_EMBEDDING] = "qwen3_vl.%.*s",
        [ID4_QWEN3_VL_PROGRAM_OPERATION_NAME_LAYER_OPERATION] =
            "qwen3_vl.layers.%" PRIu32 ".%.*s.%.*s",
        [ID4_QWEN3_VL_PROGRAM_OPERATION_NAME_GLOBAL_OPERATION] =
            "qwen3_vl.%.*s.%.*s",
};

static const id4_qwen3_vl_program_format_pattern_t
    id4_qwen3_vl_program_parameter_patterns[ID4_QWEN3_VL_PARAMETER_KIND_COUNT] =
        {
            [ID4_QWEN3_VL_PARAMETER_TOKEN_EMBEDDING] =
                {false, IREE_SVL("model.language_model.embed_tokens.weight")},
            [ID4_QWEN3_VL_PARAMETER_LAYER_INPUT_NORM] =
                {true, IREE_SVL("model.language_model.layers.%" PRIu32
                                ".input_layernorm.weight")},
            [ID4_QWEN3_VL_PARAMETER_LAYER_Q_PROJECTION] =
                {true, IREE_SVL("model.language_model.layers.%" PRIu32
                                ".self_attn.q_proj.weight")},
            [ID4_QWEN3_VL_PARAMETER_LAYER_K_PROJECTION] =
                {true, IREE_SVL("model.language_model.layers.%" PRIu32
                                ".self_attn.k_proj.weight")},
            [ID4_QWEN3_VL_PARAMETER_LAYER_Q_NORM] =
                {true, IREE_SVL("model.language_model.layers.%" PRIu32
                                ".self_attn.q_norm.weight")},
            [ID4_QWEN3_VL_PARAMETER_LAYER_K_NORM] =
                {true, IREE_SVL("model.language_model.layers.%" PRIu32
                                ".self_attn.k_norm.weight")},
            [ID4_QWEN3_VL_PARAMETER_LAYER_V_PROJECTION] =
                {true, IREE_SVL("model.language_model.layers.%" PRIu32
                                ".self_attn.v_proj.weight")},
            [ID4_QWEN3_VL_PARAMETER_LAYER_O_PROJECTION] =
                {true, IREE_SVL("model.language_model.layers.%" PRIu32
                                ".self_attn.o_proj.weight")},
            [ID4_QWEN3_VL_PARAMETER_LAYER_POST_ATTENTION_NORM] =
                {true, IREE_SVL("model.language_model.layers.%" PRIu32
                                ".post_attention_layernorm.weight")},
            [ID4_QWEN3_VL_PARAMETER_LAYER_GATE_PROJECTION] =
                {true, IREE_SVL("model.language_model.layers.%" PRIu32
                                ".mlp.gate_proj.weight")},
            [ID4_QWEN3_VL_PARAMETER_LAYER_UP_PROJECTION] =
                {true, IREE_SVL("model.language_model.layers.%" PRIu32
                                ".mlp.up_proj.weight")},
            [ID4_QWEN3_VL_PARAMETER_LAYER_DOWN_PROJECTION] =
                {true, IREE_SVL("model.language_model.layers.%" PRIu32
                                ".mlp.down_proj.weight")},
            [ID4_QWEN3_VL_PARAMETER_FINAL_NORM] =
                {false, IREE_SVL("model.language_model.norm.weight")},
};

static const id4_qwen3_vl_program_format_pattern_t
    id4_qwen3_vl_program_tensor_patterns[ID4_QWEN3_VL_TENSOR_KIND_COUNT] = {
        [ID4_QWEN3_VL_TENSOR_ATTENTION_MASK] = {false,
                                                IREE_SVL("attention_mask")},
        [ID4_QWEN3_VL_TENSOR_TOKEN_WEIGHTS] = {false,
                                               IREE_SVL("token_weights")},
        [ID4_QWEN3_VL_TENSOR_EMBEDDED_HIDDEN_STATES] =
            {false, IREE_SVL("qwen3_vl.embedded_hidden_states")},
        [ID4_QWEN3_VL_TENSOR_LAYER_INPUT_NORM] =
            {true, IREE_SVL("qwen3_vl.layers.%" PRIu32 ".input_layernorm")},
        [ID4_QWEN3_VL_TENSOR_LAYER_Q] = {true,
                                         IREE_SVL("qwen3_vl.layers.%" PRIu32
                                                  ".self_attn.q")},
        [ID4_QWEN3_VL_TENSOR_LAYER_K] = {true,
                                         IREE_SVL("qwen3_vl.layers.%" PRIu32
                                                  ".self_attn.k")},
        [ID4_QWEN3_VL_TENSOR_LAYER_V] = {true,
                                         IREE_SVL("qwen3_vl.layers.%" PRIu32
                                                  ".self_attn.v")},
        [ID4_QWEN3_VL_TENSOR_LAYER_V_PACKED] =
            {true, IREE_SVL("qwen3_vl.layers.%" PRIu32 ".self_attn.v_packed")},
        [ID4_QWEN3_VL_TENSOR_LAYER_Q_NORM] =
            {true, IREE_SVL("qwen3_vl.layers.%" PRIu32 ".self_attn.q_norm")},
        [ID4_QWEN3_VL_TENSOR_LAYER_K_NORM] =
            {true, IREE_SVL("qwen3_vl.layers.%" PRIu32 ".self_attn.k_norm")},
        [ID4_QWEN3_VL_TENSOR_LAYER_Q_ROTARY] =
            {true, IREE_SVL("qwen3_vl.layers.%" PRIu32 ".self_attn.q_rotary")},
        [ID4_QWEN3_VL_TENSOR_LAYER_K_ROTARY] =
            {true, IREE_SVL("qwen3_vl.layers.%" PRIu32 ".self_attn.k_rotary")},
        [ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_SCORES] =
            {true, IREE_SVL("qwen3_vl.layers.%" PRIu32 ".self_attn.scores")},
        [ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_PROBABILITIES] =
            {true,
             IREE_SVL("qwen3_vl.layers.%" PRIu32 ".self_attn.probabilities")},
        [ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_CONTEXT] =
            {true, IREE_SVL("qwen3_vl.layers.%" PRIu32 ".self_attn.context")},
        [ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_OUTPUT] =
            {true, IREE_SVL("qwen3_vl.layers.%" PRIu32 ".self_attn.output")},
        [ID4_QWEN3_VL_TENSOR_LAYER_AFTER_ATTENTION] =
            {true, IREE_SVL("qwen3_vl.layers.%" PRIu32 ".after_attention")},
        [ID4_QWEN3_VL_TENSOR_LAYER_POST_ATTENTION_NORM] =
            {true,
             IREE_SVL("qwen3_vl.layers.%" PRIu32 ".post_attention_layernorm")},
        [ID4_QWEN3_VL_TENSOR_LAYER_MLP_GATE] =
            {true, IREE_SVL("qwen3_vl.layers.%" PRIu32 ".mlp.gate")},
        [ID4_QWEN3_VL_TENSOR_LAYER_MLP_UP] =
            {true, IREE_SVL("qwen3_vl.layers.%" PRIu32 ".mlp.up")},
        [ID4_QWEN3_VL_TENSOR_LAYER_MLP_ACTIVATION] =
            {true, IREE_SVL("qwen3_vl.layers.%" PRIu32 ".mlp.activation")},
        [ID4_QWEN3_VL_TENSOR_LAYER_MLP_DOWN] =
            {true, IREE_SVL("qwen3_vl.layers.%" PRIu32 ".mlp.down")},
        [ID4_QWEN3_VL_TENSOR_LAYER_OUTPUT] =
            {true, IREE_SVL("qwen3_vl.layers.%" PRIu32 ".output")},
        [ID4_QWEN3_VL_TENSOR_SELECTED_HIDDEN_STATES] =
            {false, IREE_SVL("selected_hidden_states")},
        [ID4_QWEN3_VL_TENSOR_CONDITION] = {false,
                                           IREE_SVL("qwen3_vl.condition")},
        [ID4_QWEN3_VL_TENSOR_CONDITION_STATS] =
            {false, IREE_SVL("qwen3_vl.condition_stats")},
        [ID4_QWEN3_VL_TENSOR_FINAL_NORM] = {false,
                                            IREE_SVL("qwen3_vl.final_norm")},
        [ID4_QWEN3_VL_TENSOR_TOKEN_EMBEDDING_ROWS] =
            {false, IREE_SVL("qwen3_vl.token_embedding.rows")},
        [ID4_QWEN3_VL_TENSOR_OUTPUT] = {false, IREE_SVL("condition")},
};

static const id4_qwen3_vl_program_operation_site_entry_t
    id4_qwen3_vl_program_operation_sites[ID4_QWEN3_VL_OPERATION_SITE_COUNT] = {
        [ID4_QWEN3_VL_OPERATION_SITE_TOKEN_EMBEDDING] =
            {false, IREE_SVL("token_embedding")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_TOKEN_EMBEDDING] =
            {false, IREE_SVL("after_token_embedding")},
        [ID4_QWEN3_VL_OPERATION_SITE_INPUT_LAYERNORM] =
            {true, IREE_SVL("input_layernorm")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_INPUT_LAYERNORM] =
            {true, IREE_SVL("after_input_layernorm")},
        [ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_Q_PROJECTION] =
            {true, IREE_SVL("self_attn.q_proj")},
        [ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_K_PROJECTION] =
            {true, IREE_SVL("self_attn.k_proj")},
        [ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_V_PROJECTION] =
            {true, IREE_SVL("self_attn.v_proj")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_QKV_PROJECTION] =
            {true, IREE_SVL("after_qkv_projection")},
        [ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_Q_NORM] =
            {true, IREE_SVL("self_attn.q_norm")},
        [ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_K_NORM] =
            {true, IREE_SVL("self_attn.k_norm")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_QK_NORM] = {true,
                                                       IREE_SVL(
                                                           "after_qk_norm")},
        [ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_Q] = {true,
                                                          IREE_SVL(
                                                              "self_attn.q")},
        [ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_K] = {true,
                                                          IREE_SVL(
                                                              "self_attn.k")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_ROTARY_EMBEDDING] =
            {true, IREE_SVL("after_rotary_embedding")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_VALUE_PACK] =
            {true, IREE_SVL("after_value_pack")},
        [ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_SCORES] =
            {true, IREE_SVL("self_attn.scores")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION_SCORES] =
            {true, IREE_SVL("after_attention_scores")},
        [ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_SOFTMAX] =
            {true, IREE_SVL("self_attn.softmax")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION_SOFTMAX] =
            {true, IREE_SVL("after_attention_softmax")},
        [ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION] = {true,
                                                        IREE_SVL(
                                                            "self_attn.pv")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION] =
            {true, IREE_SVL("after_attention")},
        [ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_O_PROJECTION] =
            {true, IREE_SVL("self_attn.o_proj")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION_OUTPUT_PROJECTION] =
            {true, IREE_SVL("after_attention_output_projection")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION_RESIDUAL] =
            {true, IREE_SVL("after_attention_residual")},
        [ID4_QWEN3_VL_OPERATION_SITE_POST_ATTENTION_LAYERNORM] =
            {true, IREE_SVL("post_attention_layernorm")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_POST_ATTENTION_LAYERNORM] =
            {true, IREE_SVL("after_post_attention_layernorm")},
        [ID4_QWEN3_VL_OPERATION_SITE_MLP_GATE_PROJECTION] =
            {true, IREE_SVL("mlp.gate_proj")},
        [ID4_QWEN3_VL_OPERATION_SITE_MLP_UP_PROJECTION] = {true,
                                                           IREE_SVL(
                                                               "mlp.up_proj")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_MLP_UP_GATE_PROJECTION] =
            {true, IREE_SVL("after_mlp_up_gate_projection")},
        [ID4_QWEN3_VL_OPERATION_SITE_MLP] = {true, IREE_SVL("mlp")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_MLP_ACTIVATION] =
            {true, IREE_SVL("after_mlp_activation")},
        [ID4_QWEN3_VL_OPERATION_SITE_MLP_DOWN_PROJECTION] =
            {true, IREE_SVL("mlp.down_proj")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_MLP_DOWN_PROJECTION] =
            {true, IREE_SVL("after_mlp_down_projection")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_LAYER_OUTPUT] =
            {true, IREE_SVL("after_layer_output")},
        [ID4_QWEN3_VL_OPERATION_SITE_SELECTED_HIDDEN_PACK] =
            {true, IREE_SVL("selected_hidden_pack")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_SELECTED_HIDDEN_PACK] =
            {false, IREE_SVL("after_selected_hidden_pack")},
        [ID4_QWEN3_VL_OPERATION_SITE_CONDITION_APPLY_TOKEN_WEIGHTS] =
            {false, IREE_SVL("condition.apply_token_weights")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_CONDITION_APPLY_TOKEN_WEIGHTS] =
            {false, IREE_SVL("after_condition_apply_token_weights")},
        [ID4_QWEN3_VL_OPERATION_SITE_CONDITION_REDUCE_TOKEN_WEIGHT_STATS] =
            {false, IREE_SVL("condition.reduce_token_weight_stats")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_CONDITION_REDUCE_TOKEN_WEIGHT_STATS] =
            {false, IREE_SVL("after_condition_reduce_token_weight_stats")},
        [ID4_QWEN3_VL_OPERATION_SITE_CONDITION_NORMALIZE_TOKEN_WEIGHTS] =
            {false, IREE_SVL("condition.normalize_token_weights")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_CONDITION] =
            {false, IREE_SVL("after_condition")},
        [ID4_QWEN3_VL_OPERATION_SITE_FINAL_NORM] = {false,
                                                    IREE_SVL("final_norm")},
        [ID4_QWEN3_VL_OPERATION_SITE_AFTER_FINAL_NORM] =
            {false, IREE_SVL("after_final_norm")},
        [ID4_QWEN3_VL_OPERATION_SITE_OUTPUT] = {false, IREE_SVL("output")},
};

static const id4_qwen3_vl_program_operation_kind_entry_t
    id4_qwen3_vl_program_operation_kinds[ID4_QWEN3_VL_OPERATION_KIND_COUNT] = {
        [ID4_QWEN3_VL_OPERATION_TOKEN_EMBEDDING] = {false, IREE_SVL("")},
        [ID4_QWEN3_VL_OPERATION_RMSNORM] = {true, IREE_SVL("rmsnorm")},
        [ID4_QWEN3_VL_OPERATION_LINEAR] = {true, IREE_SVL("linear")},
        [ID4_QWEN3_VL_OPERATION_ROTARY] = {true, IREE_SVL("rotary")},
        [ID4_QWEN3_VL_OPERATION_ATTENTION] = {true, IREE_SVL("attention")},
        [ID4_QWEN3_VL_OPERATION_SILU_GATE] = {true, IREE_SVL("silu_gate")},
        [ID4_QWEN3_VL_OPERATION_RESIDUAL_ADD] = {true,
                                                 IREE_SVL("residual_add")},
        [ID4_QWEN3_VL_OPERATION_SELECTED_HIDDEN_PACK] = {true,
                                                         IREE_SVL("pack")},
        [ID4_QWEN3_VL_OPERATION_CONDITION_APPLY_TOKEN_WEIGHTS] =
            {true, IREE_SVL("apply_token_weights")},
        [ID4_QWEN3_VL_OPERATION_CONDITION_REDUCE_TOKEN_WEIGHT_STATS] =
            {true, IREE_SVL("reduce_token_weight_stats")},
        [ID4_QWEN3_VL_OPERATION_CONDITION_NORMALIZE_TOKEN_WEIGHTS] =
            {true, IREE_SVL("normalize_token_weights")},
        [ID4_QWEN3_VL_OPERATION_BARRIER] = {true, IREE_SVL("barrier")},
        [ID4_QWEN3_VL_OPERATION_TAP] = {true, IREE_SVL("tap")},
};

static const id4_pipeline_kernel_ref_t id4_qwen3_vl_program_kernel_refs[ID4_QWEN3_VL_KERNEL_KIND_COUNT] = {
    [ID4_QWEN3_VL_KERNEL_TOKEN_EMBEDDING] =
        {IREE_SVL("qwen3_vl/token_embedding"),
         IREE_SVL("id4_qwen3_vl_token_embedding_rows_bf16_bf16")},
    [ID4_QWEN3_VL_KERNEL_RMSNORM] = {IREE_SVL("qwen3_vl/rmsnorm"),
                                     IREE_SVL("id4_qwen3_vl_rmsnorm_bf16")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16] =
        {IREE_SVL("qwen3_vl/linear_bf16_f32"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16")},
    [ID4_QWEN3_VL_KERNEL_ROTARY] = {IREE_SVL("qwen3_vl/rotary_embedding"),
                                    IREE_SVL(
                                        "id4_qwen3_vl_rotary_embedding_bf16")},
    [ID4_QWEN3_VL_KERNEL_ATTENTION_SCORES] =
        {IREE_SVL("qwen3_vl/attention_materialized_f32"),
         IREE_SVL("id4_qwen3_vl_attention_scores_bf16_f32")},
    [ID4_QWEN3_VL_KERNEL_ATTENTION_SOFTMAX] =
        {IREE_SVL("qwen3_vl/attention_materialized_f32"),
         IREE_SVL("id4_qwen3_vl_attention_softmax_bf16")},
    [ID4_QWEN3_VL_KERNEL_ATTENTION_PV] =
        {IREE_SVL("qwen3_vl/attention_materialized_f32"),
         IREE_SVL("id4_qwen3_vl_attention_pv_bf16")},
    [ID4_QWEN3_VL_KERNEL_SILU_GATE] = {IREE_SVL("qwen3_vl/silu_gate_f32"),
                                       IREE_SVL("id4_qwen3_vl_silu_gate_bf16")},
    [ID4_QWEN3_VL_KERNEL_RESIDUAL_ADD] =
        {IREE_SVL("qwen3_vl/residual_add_f32"),
         IREE_SVL("id4_qwen3_vl_residual_add_bf16")},
    [ID4_QWEN3_VL_KERNEL_SELECTED_HIDDEN_PACK] =
        {IREE_SVL("qwen3_vl/selected_hidden_pack"),
         IREE_SVL("id4_qwen3_vl_selected_hidden_pack_bf16_bf16")},
    [ID4_QWEN3_VL_KERNEL_CONDITION_APPLY_TOKEN_WEIGHTS] =
        {IREE_SVL("qwen3_vl/condition"),
         IREE_SVL("id4_qwen3_vl_condition_apply_token_weights_bf16_f32")},
    [ID4_QWEN3_VL_KERNEL_CONDITION_REDUCE_TOKEN_WEIGHT_STATS] =
        {IREE_SVL("qwen3_vl/condition"),
         IREE_SVL("id4_qwen3_vl_condition_reduce_token_weight_stats_f32")},
    [ID4_QWEN3_VL_KERNEL_CONDITION_NORMALIZE_TOKEN_WEIGHTS] =
        {IREE_SVL("qwen3_vl/condition"),
         IREE_SVL("id4_qwen3_vl_condition_normalize_token_weights_f32")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M16N64] =
        {IREE_SVL("qwen3_vl/linear_bf16_bf16_wmma_m16n64"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_wmma_m16n64")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M16N64_TRANSPOSE_OUTPUT] =
        {IREE_SVL("qwen3_vl/linear_bf16_bf16_wmma_m16n64"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_wmma_m16n64_"
                  "transpose_output")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32] =
        {IREE_SVL("qwen3_vl/linear_bf16_bf16_wmma_m32n32"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_wmma_m32n32")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32_TRANSPOSE_OUTPUT] =
        {IREE_SVL("qwen3_vl/linear_bf16_bf16_wmma_m32n32"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_wmma_m32n32_"
                  "transpose_output")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32_COMPACT_RHS] =
        {IREE_SVL("qwen3_vl/linear_bf16_bf16_wmma_m32n32_compact_rhs"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_wmma_m32n32_compact_rhs")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32_COMPACT_RHS_TRANSPOSE_OUTPUT] =
        {IREE_SVL("qwen3_vl/linear_bf16_bf16_wmma_m32n32_compact_rhs"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_wmma_m32n32_compact_rhs_"
                  "transpose_output")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N64] =
        {IREE_SVL("qwen3_vl/linear_bf16_bf16_wmma_m32n64"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_wmma_m32n64")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N64_COMPACT_RHS] =
        {IREE_SVL("qwen3_vl/linear_bf16_bf16_wmma_m32n64_compact_rhs"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_wmma_m32n64_compact_rhs")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M64N64] =
        {IREE_SVL("qwen3_vl/linear_bf16_bf16_wmma_m64n64"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_wmma_m64n64")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M64N64_COMPACT_RHS] =
        {IREE_SVL("qwen3_vl/linear_bf16_bf16_wmma_m64n64_compact_rhs"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_wmma_m64n64_compact_rhs")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M128N64_2WAVE_COMPACT_RHS] =
        {IREE_SVL("qwen3_vl/"
                  "linear_bf16_bf16_wmma_m128n64_2wave_compact_rhs"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_wmma_m128n64_"
                  "2wave_compact_rhs")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M48N32] =
        {IREE_SVL("qwen3_vl/linear_bf16_bf16_wmma_m48n32"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_wmma_m48n32")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M48N32_COMPACT_RHS] =
        {IREE_SVL("qwen3_vl/linear_bf16_bf16_wmma_m48n32_compact_rhs"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_wmma_m48n32_compact_rhs")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N64_TRANSPOSE_OUTPUT] =
        {IREE_SVL("qwen3_vl/linear_bf16_bf16_wmma_m32n64"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_wmma_m32n64_"
                  "transpose_output")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N64_COMPACT_RHS_TRANSPOSE_OUTPUT] =
        {IREE_SVL("qwen3_vl/linear_bf16_bf16_wmma_m32n64_compact_rhs"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_wmma_m32n64_compact_rhs_"
                  "transpose_output")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_TAIL] =
        {IREE_SVL("qwen3_vl/linear_bf16_bf16_tail"),
         IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_tail")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_INPUT_PACK_TRANSPOSE_BF16_BF16] =
        {IREE_SVL("qwen3_vl/linear_input_pack_f32_bf16"),
         IREE_SVL("id4_qwen3_vl_linear_input_pack_transpose_bf16_bf16")},
    [ID4_QWEN3_VL_KERNEL_ATTENTION_QK_SCORES_WMMA] =
        {IREE_SVL("qwen3_vl/attention_wmma"),
         IREE_SVL("id4_qwen3_vl_attention_qk_scores_bf16_f32_wmma")},
    [ID4_QWEN3_VL_KERNEL_ATTENTION_SOFTMAX_MASK_BF16] =
        {IREE_SVL("qwen3_vl/attention_wmma"),
         IREE_SVL("id4_qwen3_vl_attention_softmax_mask_bf16")},
    [ID4_QWEN3_VL_KERNEL_ATTENTION_PV_WMMA] =
        {IREE_SVL("qwen3_vl/attention_wmma"),
         IREE_SVL("id4_qwen3_vl_attention_pv_bf16_bf16_wmma")},
    [ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_BF16_WMMA_M16N32] =
        {IREE_SVL("qwen3_vl/mlp_gate_up_silu_product_bf16_wmma"),
         IREE_SVL("id4_qwen3_vl_mlp_gate_up_silu_product_bf16_wmma")},
    [ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_BF16_WMMA_M32N32] =
        {IREE_SVL("qwen3_vl/mlp_gate_up_silu_product_bf16_wmma"),
         IREE_SVL("id4_qwen3_vl_mlp_gate_up_silu_product_bf16_wmma_m32n32")},
    [ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_BF16_WMMA_M32N32_COMPACT_RHS] =
        {IREE_SVL("qwen3_vl/mlp_gate_up_silu_product_bf16_wmma_compact_rhs"),
         IREE_SVL("id4_qwen3_vl_mlp_gate_up_silu_product_bf16_wmma_"
                  "m32n32_compact_rhs")},
    [ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_BF16_WMMA_M128N32_2WAVE_COMPACT_RHS] =
        {IREE_SVL("qwen3_vl/mlp_gate_up_silu_product_bf16_wmma_"
                  "m128n32_2wave_compact_rhs"),
         IREE_SVL("id4_qwen3_vl_mlp_gate_up_silu_product_bf16_wmma_"
                  "m128n32_2wave_compact_rhs")},
    [ID4_QWEN3_VL_KERNEL_ZERO_TAIL_BF16] = {IREE_SVL("tensor/zero_tail_bf16"),
                                            IREE_SVL(
                                                "id4_tensor_zero_tail_bf16")},
    [ID4_QWEN3_VL_KERNEL_RESIDUAL_ADD_RMSNORM] =
        {IREE_SVL("qwen3_vl/residual_add_f32"),
         IREE_SVL("id4_qwen3_vl_residual_add_rmsnorm_bf16")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M32N32] =
        {IREE_SVL("qwen3_vl/linear_residual_bf16_wmma"),
         IREE_SVL("id4_qwen3_vl_linear_residual_bf16_bf16_wmma_m32n32")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M64N64] =
        {IREE_SVL("qwen3_vl/linear_residual_bf16_wmma"),
         IREE_SVL("id4_qwen3_vl_linear_residual_bf16_bf16_wmma_m64n64")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M64N64_COMPACT_RHS] =
        {IREE_SVL("qwen3_vl/linear_residual_bf16_wmma_compact_rhs"),
         IREE_SVL("id4_qwen3_vl_linear_residual_bf16_bf16_wmma_m64n64_"
                  "compact_rhs")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M32N32_COMPACT_RHS] =
        {IREE_SVL("qwen3_vl/linear_residual_bf16_wmma_m32n32_compact_rhs"),
         IREE_SVL("id4_qwen3_vl_linear_residual_bf16_bf16_wmma_m32n32_"
                  "compact_rhs")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M128N64_2WAVE_COMPACT_RHS] =
        {IREE_SVL("qwen3_vl/"
                  "linear_residual_bf16_wmma_m128n64_2wave_compact_rhs"),
         IREE_SVL("id4_qwen3_vl_linear_residual_bf16_bf16_wmma_"
                  "m128n64_2wave_compact_rhs")},
    [ID4_QWEN3_VL_KERNEL_RMSNORM_ROTARY] =
        {IREE_SVL("qwen3_vl/rmsnorm_rotary"),
         IREE_SVL("id4_qwen3_vl_rmsnorm_rotary_bf16")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_FP8_BLOCK_SCALED_BF16_WMMA_M64N64_COMPACT_RHS] =
        {IREE_SVL("qwen3_vl/linear_fp8_block_scaled_bf16_wmma_m64n64_"
                  "compact_rhs"),
         IREE_SVL("id4_qwen3_vl_linear_fp8_block_scaled_bf16_wmma_m64n64_"
                  "compact_rhs")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_FP8_BLOCK_SCALED_BF16_WMMA_M128N64_2WAVE_COMPACT_RHS] =
        {IREE_SVL("qwen3_vl/"
                  "linear_fp8_block_scaled_bf16_wmma_m128n64_2wave_"
                  "compact_rhs"),
         IREE_SVL("id4_qwen3_vl_linear_fp8_block_scaled_bf16_wmma_"
                  "m128n64_2wave_compact_rhs")},
    [ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_FP8_BLOCK_SCALED_BF16_WMMA_M32N32_COMPACT_RHS] =
        {IREE_SVL("qwen3_vl/mlp_gate_up_silu_product_fp8_block_scaled_"
                  "bf16_wmma_compact_rhs"),
         IREE_SVL("id4_qwen3_vl_mlp_gate_up_silu_product_fp8_block_scaled_"
                  "bf16_wmma_m32n32_compact_rhs")},
    [ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_FP8_BLOCK_SCALED_BF16_WMMA_M128N32_2WAVE_COMPACT_RHS] =
        {IREE_SVL("qwen3_vl/mlp_gate_up_silu_product_fp8_block_scaled_"
                  "bf16_wmma_m128n32_2wave_compact_rhs"),
         IREE_SVL("id4_qwen3_vl_mlp_gate_up_silu_product_fp8_block_scaled_"
                  "bf16_wmma_m128n32_2wave_compact_rhs")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_FP8_BLOCK_SCALED_BF16_WMMA_M64N64_COMPACT_RHS] =
        {IREE_SVL("qwen3_vl/linear_residual_fp8_block_scaled_bf16_wmma_"
                  "compact_rhs"),
         IREE_SVL("id4_qwen3_vl_linear_residual_fp8_block_scaled_bf16_"
                  "wmma_m64n64_compact_rhs")},
    [ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_FP8_BLOCK_SCALED_BF16_WMMA_M128N64_2WAVE_COMPACT_RHS] =
        {IREE_SVL("qwen3_vl/linear_residual_fp8_block_scaled_bf16_wmma_"
                  "m128n64_2wave_compact_rhs"),
         IREE_SVL("id4_qwen3_vl_linear_residual_fp8_block_scaled_bf16_"
                  "wmma_m128n64_2wave_compact_rhs")},
    [ID4_QWEN3_VL_KERNEL_ATTENTION_ONLINE_BF16] =
        {IREE_SVL("qwen3_vl/attention_online_bf16"),
         IREE_SVL("id4_qwen3_vl_attention_online_bf16")},
    [ID4_QWEN3_VL_KERNEL_ATTENTION_ONLINE_BF16_WMMA] =
        {IREE_SVL("qwen3_vl/attention_online_bf16_wmma"),
         IREE_SVL("id4_qwen3_vl_attention_online_bf16_bf16_wmma")},
};

static const iree_string_view_t id4_qwen3_vl_program_config_keys
    [ID4_QWEN3_VL_KERNEL_KIND_COUNT][ID4_QWEN3_VL_CONFIG_KEY_COUNT] = {
        [ID4_QWEN3_VL_KERNEL_TOKEN_EMBEDDING] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.token_embedding.token_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.token_embedding.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE] =
                    IREE_SVL("id4.qwen3_vl.token_embedding.hidden_size"),
                [ID4_QWEN3_VL_CONFIG_VOCAB_SIZE] =
                    IREE_SVL("id4.qwen3_vl.token_embedding.vocab_size"),
            },
        [ID4_QWEN3_VL_KERNEL_RMSNORM] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.rmsnorm.token_count"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE] =
                    IREE_SVL("id4.qwen3_vl.rmsnorm.hidden_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear.token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_ROTARY] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.rotary.token_count"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.rotary.output_size"),
                [ID4_QWEN3_VL_CONFIG_HEAD_SIZE] =
                    IREE_SVL("id4.qwen3_vl.rotary.head_size"),
            },
        [ID4_QWEN3_VL_KERNEL_ATTENTION_SCORES] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.attention.token_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.attention.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE] =
                    IREE_SVL("id4.qwen3_vl.attention.hidden_size"),
                [ID4_QWEN3_VL_CONFIG_ATTENTION_HEAD_COUNT] =
                    IREE_SVL("id4.qwen3_vl.attention.attention_head_count"),
                [ID4_QWEN3_VL_CONFIG_KEY_VALUE_HEAD_COUNT] =
                    IREE_SVL("id4.qwen3_vl.attention.key_value_head_count"),
                [ID4_QWEN3_VL_CONFIG_HEAD_SIZE] =
                    IREE_SVL("id4.qwen3_vl.attention.head_size"),
            },
        [ID4_QWEN3_VL_KERNEL_ATTENTION_SOFTMAX] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.attention.token_count"),
                [ID4_QWEN3_VL_CONFIG_ATTENTION_HEAD_COUNT] =
                    IREE_SVL("id4.qwen3_vl.attention.attention_head_count"),
            },
        [ID4_QWEN3_VL_KERNEL_ATTENTION_PV] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.attention.token_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.attention.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE] =
                    IREE_SVL("id4.qwen3_vl.attention.hidden_size"),
                [ID4_QWEN3_VL_CONFIG_ATTENTION_HEAD_COUNT] =
                    IREE_SVL("id4.qwen3_vl.attention.attention_head_count"),
                [ID4_QWEN3_VL_CONFIG_KEY_VALUE_HEAD_COUNT] =
                    IREE_SVL("id4.qwen3_vl.attention.key_value_head_count"),
                [ID4_QWEN3_VL_CONFIG_HEAD_SIZE] =
                    IREE_SVL("id4.qwen3_vl.attention.head_size"),
            },
        [ID4_QWEN3_VL_KERNEL_SILU_GATE] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.silu_gate.token_count"),
                [ID4_QWEN3_VL_CONFIG_INTERMEDIATE_SIZE] =
                    IREE_SVL("id4.qwen3_vl.silu_gate.intermediate_size"),
            },
        [ID4_QWEN3_VL_KERNEL_RESIDUAL_ADD] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.residual_add.token_count"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE] =
                    IREE_SVL("id4.qwen3_vl.residual_add.hidden_size"),
            },
        [ID4_QWEN3_VL_KERNEL_SELECTED_HIDDEN_PACK] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.selected_hidden_pack.token_count"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE] =
                    IREE_SVL("id4.qwen3_vl.selected_hidden_pack.hidden_size"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_ROW_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.selected_hidden_pack.hidden_row_count"),
                [ID4_QWEN3_VL_CONFIG_SELECTED_LAYER_INDEX] =
                    IREE_SVL("id4.qwen3_vl.selected_hidden_pack.selected_layer_"
                             "index"),
            },
        [ID4_QWEN3_VL_KERNEL_CONDITION_APPLY_TOKEN_WEIGHTS] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.condition.token_count"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_ROW_COUNT] =
                    IREE_SVL("id4.qwen3_vl.condition.hidden_row_count"),
            },
        [ID4_QWEN3_VL_KERNEL_CONDITION_REDUCE_TOKEN_WEIGHT_STATS] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.condition.token_count"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_ROW_COUNT] =
                    IREE_SVL("id4.qwen3_vl.condition.hidden_row_count"),
            },
        [ID4_QWEN3_VL_KERNEL_CONDITION_NORMALIZE_TOKEN_WEIGHTS] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.condition.token_count"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_ROW_COUNT] =
                    IREE_SVL("id4.qwen3_vl.condition.hidden_row_count"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M16N64] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M16N64_TRANSPOSE_OUTPUT] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32_TRANSPOSE_OUTPUT] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32_COMPACT_RHS] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N32_COMPACT_RHS_TRANSPOSE_OUTPUT] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N64] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N64_COMPACT_RHS] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M64N64] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M64N64_COMPACT_RHS] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M128N64_2WAVE_COMPACT_RHS] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_FP8_BLOCK_SCALED_BF16_WMMA_M64N64_COMPACT_RHS] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_fp8_block_scaled_wmma."
                             "dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_fp8_block_scaled_wmma."
                             "input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_fp8_block_scaled_wmma."
                             "output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_FP8_BLOCK_SCALED_BF16_WMMA_M128N64_2WAVE_COMPACT_RHS] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_fp8_block_scaled_wmma."
                             "dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_fp8_block_scaled_wmma."
                             "input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_fp8_block_scaled_wmma."
                             "output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M48N32] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M48N32_COMPACT_RHS] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N64_TRANSPOSE_OUTPUT] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA_M32N64_COMPACT_RHS_TRANSPOSE_OUTPUT] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_TAIL] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_tail.token_count"),
                [ID4_QWEN3_VL_CONFIG_TOKEN_START] =
                    IREE_SVL("id4.qwen3_vl.linear_tail.token_start"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_tail.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_tail.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_tail.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_INPUT_PACK_TRANSPOSE_BF16_BF16] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_input_pack.token_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.linear_input_pack.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_input_pack.input_size"),
                [ID4_QWEN3_VL_CONFIG_ELEMENT_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_input_pack.element_count"),
            },
        [ID4_QWEN3_VL_KERNEL_ATTENTION_QK_SCORES_WMMA] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.attention_wmma.token_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.attention_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE] =
                    IREE_SVL("id4.qwen3_vl.attention_wmma.hidden_size"),
                [ID4_QWEN3_VL_CONFIG_ATTENTION_HEAD_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.attention_wmma.attention_head_count"),
                [ID4_QWEN3_VL_CONFIG_KEY_VALUE_HEAD_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.attention_wmma.key_value_head_count"),
                [ID4_QWEN3_VL_CONFIG_HEAD_SIZE] =
                    IREE_SVL("id4.qwen3_vl.attention_wmma.head_size"),
            },
        [ID4_QWEN3_VL_KERNEL_ATTENTION_SOFTMAX_MASK_BF16] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.attention_wmma.token_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.attention_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE] =
                    IREE_SVL("id4.qwen3_vl.attention_wmma.hidden_size"),
                [ID4_QWEN3_VL_CONFIG_ATTENTION_HEAD_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.attention_wmma.attention_head_count"),
                [ID4_QWEN3_VL_CONFIG_KEY_VALUE_HEAD_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.attention_wmma.key_value_head_count"),
                [ID4_QWEN3_VL_CONFIG_HEAD_SIZE] =
                    IREE_SVL("id4.qwen3_vl.attention_wmma.head_size"),
            },
        [ID4_QWEN3_VL_KERNEL_ATTENTION_PV_WMMA] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.attention_wmma.token_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.attention_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE] =
                    IREE_SVL("id4.qwen3_vl.attention_wmma.hidden_size"),
                [ID4_QWEN3_VL_CONFIG_ATTENTION_HEAD_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.attention_wmma.attention_head_count"),
                [ID4_QWEN3_VL_CONFIG_KEY_VALUE_HEAD_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.attention_wmma.key_value_head_count"),
                [ID4_QWEN3_VL_CONFIG_HEAD_SIZE] =
                    IREE_SVL("id4.qwen3_vl.attention_wmma.head_size"),
            },
        [ID4_QWEN3_VL_KERNEL_ATTENTION_ONLINE_BF16] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.attention_online.token_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.attention_online.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE] =
                    IREE_SVL("id4.qwen3_vl.attention_online.hidden_size"),
                [ID4_QWEN3_VL_CONFIG_ATTENTION_HEAD_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.attention_online.attention_head_count"),
                [ID4_QWEN3_VL_CONFIG_KEY_VALUE_HEAD_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.attention_online.key_value_head_count"),
                [ID4_QWEN3_VL_CONFIG_HEAD_SIZE] =
                    IREE_SVL("id4.qwen3_vl.attention_online.head_size"),
            },
        [ID4_QWEN3_VL_KERNEL_ATTENTION_ONLINE_BF16_WMMA] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.attention_online_wmma.token_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.attention_online_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE] =
                    IREE_SVL("id4.qwen3_vl.attention_online_wmma.hidden_size"),
                [ID4_QWEN3_VL_CONFIG_ATTENTION_HEAD_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.attention_online_wmma.attention_head_count"),
                [ID4_QWEN3_VL_CONFIG_KEY_VALUE_HEAD_COUNT] =
                    IREE_SVL("id4.qwen3_vl.attention_online_wmma."
                             "key_value_head_count"),
                [ID4_QWEN3_VL_CONFIG_HEAD_SIZE] =
                    IREE_SVL("id4.qwen3_vl.attention_online_wmma.head_size"),
            },
        [ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_BF16_WMMA_M16N32] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "input_size"),
                [ID4_QWEN3_VL_CONFIG_INTERMEDIATE_SIZE] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "intermediate_size"),
            },
        [ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_BF16_WMMA_M32N32] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "input_size"),
                [ID4_QWEN3_VL_CONFIG_INTERMEDIATE_SIZE] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "intermediate_size"),
            },
        [ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_BF16_WMMA_M32N32_COMPACT_RHS] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "input_size"),
                [ID4_QWEN3_VL_CONFIG_INTERMEDIATE_SIZE] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "intermediate_size"),
            },
        [ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_BF16_WMMA_M128N32_2WAVE_COMPACT_RHS] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "input_size"),
                [ID4_QWEN3_VL_CONFIG_INTERMEDIATE_SIZE] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "intermediate_size"),
            },
        [ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_FP8_BLOCK_SCALED_BF16_WMMA_M32N32_COMPACT_RHS] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "input_size"),
                [ID4_QWEN3_VL_CONFIG_INTERMEDIATE_SIZE] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "intermediate_size"),
            },
        [ID4_QWEN3_VL_KERNEL_MLP_GATE_UP_SILU_PRODUCT_FP8_BLOCK_SCALED_BF16_WMMA_M128N32_2WAVE_COMPACT_RHS] =
            {
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "input_size"),
                [ID4_QWEN3_VL_CONFIG_INTERMEDIATE_SIZE] =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "intermediate_size"),
            },
        [ID4_QWEN3_VL_KERNEL_ZERO_TAIL_BF16] =
            {
                [ID4_QWEN3_VL_CONFIG_ELEMENT_COUNT] =
                    IREE_SVL("id4.tensor.zero_tail.element_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_ELEMENT_COUNT] =
                    IREE_SVL("id4.tensor.zero_tail.dispatch_element_count"),
            },
        [ID4_QWEN3_VL_KERNEL_RESIDUAL_ADD_RMSNORM] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.residual_add.token_count"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE] =
                    IREE_SVL("id4.qwen3_vl.residual_add.hidden_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M32N32] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.token_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.dispatch_token_"
                             "count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M64N64] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.token_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.dispatch_token_"
                             "count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M64N64_COMPACT_RHS] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.token_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.linear_residual_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M32N32_COMPACT_RHS] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.token_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.linear_residual_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_BF16_BF16_WMMA_M128N64_2WAVE_COMPACT_RHS] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.token_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.linear_residual_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_FP8_BLOCK_SCALED_BF16_WMMA_M64N64_COMPACT_RHS] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.token_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.linear_residual_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_RESIDUAL_FP8_BLOCK_SCALED_BF16_WMMA_M128N64_2WAVE_COMPACT_RHS] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.token_count"),
                [ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT] = IREE_SVL(
                    "id4.qwen3_vl.linear_residual_wmma.dispatch_token_count"),
                [ID4_QWEN3_VL_CONFIG_INPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.input_size"),
                [ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE] =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.output_size"),
            },
        [ID4_QWEN3_VL_KERNEL_RMSNORM_ROTARY] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.rmsnorm_rotary.token_count"),
                [ID4_QWEN3_VL_CONFIG_HEAD_COUNT] =
                    IREE_SVL("id4.qwen3_vl.rmsnorm_rotary.head_count"),
                [ID4_QWEN3_VL_CONFIG_HEAD_SIZE] =
                    IREE_SVL("id4.qwen3_vl.rmsnorm_rotary.head_size"),
            },
};

static iree_status_t id4_qwen3_vl_program_copy_static_string(
    iree_string_view_t value, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  if (!buffer || buffer_capacity == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL output string storage is required");
  }
  if (!out_string) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL output string view is required");
  }
  if (value.size >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to copy Qwen3-VL program string");
  }
  memcpy(buffer, value.data, value.size);
  buffer[value.size] = 0;
  *out_string = iree_make_string_view(buffer, value.size);
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_program_lookup_format_pattern(
    const id4_qwen3_vl_program_format_pattern_t* patterns,
    iree_host_size_t pattern_count, int pattern_ordinal,
    iree_string_view_t pattern_kind_name,
    const id4_qwen3_vl_program_format_pattern_t** out_pattern) {
  *out_pattern = NULL;
  if (pattern_ordinal < 0 ||
      (iree_host_size_t)pattern_ordinal >= pattern_count ||
      iree_string_view_is_empty(
          patterns[(iree_host_size_t)pattern_ordinal].pattern)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "Qwen3-VL %.*s kind %d is invalid",
        (int)pattern_kind_name.size, pattern_kind_name.data, pattern_ordinal);
  }
  *out_pattern = &patterns[(iree_host_size_t)pattern_ordinal];
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_program_format_pattern(
    const id4_qwen3_vl_program_format_pattern_t* pattern,
    uint32_t layer_ordinal, iree_string_view_t pattern_kind_name, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string) {
  if (pattern->uses_layer_ordinal) {
    if (layer_ordinal == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen3-VL %.*s requires a layer ordinal",
                              (int)pattern_kind_name.size,
                              pattern_kind_name.data);
    }
    return id4_qwen3_vl_program_format(buffer, buffer_capacity, out_string,
                                       pattern->pattern.data, layer_ordinal);
  }
  if (layer_ordinal != UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL %.*s does not accept a layer ordinal",
                            (int)pattern_kind_name.size,
                            pattern_kind_name.data);
  }
  return id4_qwen3_vl_program_copy_static_string(pattern->pattern, buffer,
                                                 buffer_capacity, out_string);
}

static iree_status_t id4_qwen3_vl_program_format_parameter_key(
    id4_qwen3_vl_parameter_kind_t kind, uint32_t layer_ordinal, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string) {
  const id4_qwen3_vl_program_format_pattern_t* pattern = NULL;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_lookup_format_pattern(
      id4_qwen3_vl_program_parameter_patterns,
      IREE_ARRAYSIZE(id4_qwen3_vl_program_parameter_patterns), (int)kind,
      IREE_SV("parameter"), &pattern));
  return id4_qwen3_vl_program_format_pattern(pattern, layer_ordinal,
                                             IREE_SV("parameter"), buffer,
                                             buffer_capacity, out_string);
}

static iree_status_t id4_qwen3_vl_program_format_tensor_name(
    id4_qwen3_vl_tensor_kind_t kind, uint32_t layer_ordinal, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string) {
  const id4_qwen3_vl_program_format_pattern_t* pattern = NULL;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_lookup_format_pattern(
      id4_qwen3_vl_program_tensor_patterns,
      IREE_ARRAYSIZE(id4_qwen3_vl_program_tensor_patterns), (int)kind,
      IREE_SV("tensor"), &pattern));
  return id4_qwen3_vl_program_format_pattern(pattern, layer_ordinal,
                                             IREE_SV("tensor"), buffer,
                                             buffer_capacity, out_string);
}

static iree_status_t id4_qwen3_vl_program_has_diagnostic_tap(
    const id4_qwen3_vl_program_options_t* options,
    id4_qwen3_vl_tensor_kind_t kind, uint32_t layer_ordinal,
    bool* out_has_tap) {
  *out_has_tap = false;
  if (options->diagnostic_tap_names.count == 0) return iree_ok_status();
  char name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_tensor_name(
      kind, layer_ordinal, name_buffer, IREE_ARRAYSIZE(name_buffer), &name));
  for (iree_host_size_t i = 0; i < options->diagnostic_tap_names.count; ++i) {
    if (iree_string_view_equal(options->diagnostic_tap_names.values[i], name)) {
      *out_has_tap = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_program_format_operation_name(
    id4_qwen3_vl_operation_kind_t kind, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string) {
  if ((int)kind < 0 ||
      (iree_host_size_t)kind >=
          IREE_ARRAYSIZE(id4_qwen3_vl_program_operation_kinds)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL operation kind %d is invalid", (int)kind);
  }
  if ((int)site < 0 ||
      (iree_host_size_t)site >=
          IREE_ARRAYSIZE(id4_qwen3_vl_program_operation_sites)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL operation site %d is invalid", (int)site);
  }
  const iree_host_size_t kind_index = (iree_host_size_t)kind;
  const iree_host_size_t site_index = (iree_host_size_t)site;
  const id4_qwen3_vl_program_operation_kind_entry_t* operation_kind =
      &id4_qwen3_vl_program_operation_kinds[kind_index];
  const id4_qwen3_vl_program_operation_site_entry_t* operation_site =
      &id4_qwen3_vl_program_operation_sites[site_index];
  if (iree_string_view_is_empty(operation_site->site_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL operation site %d is invalid", (int)site);
  }
  if (operation_site->uses_layer_ordinal) {
    if (layer_ordinal == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen3-VL operation site %d requires a layer "
                              "ordinal",
                              (int)site);
    }
  } else if (layer_ordinal != UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL operation site %d does not accept a "
                            "layer ordinal",
                            (int)site);
  }
  if (!operation_kind->appends_operation_name) {
    if (site != ID4_QWEN3_VL_OPERATION_SITE_TOKEN_EMBEDDING) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen3-VL token embedding operation site %d is "
                              "invalid",
                              (int)site);
    }
    return id4_qwen3_vl_program_format(
        buffer, buffer_capacity, out_string,
        id4_qwen3_vl_program_operation_name_patterns
            [ID4_QWEN3_VL_PROGRAM_OPERATION_NAME_TOKEN_EMBEDDING],
        (int)operation_site->site_name.size, operation_site->site_name.data);
  }
  if (operation_site->uses_layer_ordinal) {
    return id4_qwen3_vl_program_format(
        buffer, buffer_capacity, out_string,
        id4_qwen3_vl_program_operation_name_patterns
            [ID4_QWEN3_VL_PROGRAM_OPERATION_NAME_LAYER_OPERATION],
        layer_ordinal, (int)operation_site->site_name.size,
        operation_site->site_name.data,
        (int)operation_kind->operation_name.size,
        operation_kind->operation_name.data);
  }
  return id4_qwen3_vl_program_format(
      buffer, buffer_capacity, out_string,
      id4_qwen3_vl_program_operation_name_patterns
          [ID4_QWEN3_VL_PROGRAM_OPERATION_NAME_GLOBAL_OPERATION],
      (int)operation_site->site_name.size, operation_site->site_name.data,
      (int)operation_kind->operation_name.size,
      operation_kind->operation_name.data);
}

static iree_status_t id4_qwen3_vl_program_kernel_ref(
    id4_qwen3_vl_kernel_kind_t kind, id4_pipeline_kernel_ref_t* out_ref) {
  if (!out_ref) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL kernel ref output is required");
  }
  if ((int)kind < 0 ||
      (iree_host_size_t)kind >=
          IREE_ARRAYSIZE(id4_qwen3_vl_program_kernel_refs) ||
      iree_string_view_is_empty(
          id4_qwen3_vl_program_kernel_refs[(iree_host_size_t)kind]
              .module_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL kernel kind %d is invalid", (int)kind);
  }
  *out_ref = id4_qwen3_vl_program_kernel_refs[(iree_host_size_t)kind];
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_program_unsupported_config_key(
    id4_qwen3_vl_kernel_kind_t kernel_kind, id4_qwen3_vl_config_key_t key) {
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "Qwen3-VL kernel kind %d does not accept config "
                          "key %d",
                          (int)kernel_kind, (int)key);
}

static iree_status_t id4_qwen3_vl_program_config_key(
    id4_qwen3_vl_kernel_kind_t kernel_kind, id4_qwen3_vl_config_key_t key,
    iree_string_view_t* out_key) {
  if (!out_key) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL config key output is required");
  }
  if ((int)kernel_kind < 0 ||
      (iree_host_size_t)kernel_kind >=
          IREE_ARRAYSIZE(id4_qwen3_vl_program_config_keys)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL kernel kind %d is invalid",
                            (int)kernel_kind);
  }
  if ((int)key < 0 || (iree_host_size_t)key >=
                          IREE_ARRAYSIZE(id4_qwen3_vl_program_config_keys[0])) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL config key %d is invalid", (int)key);
  }
  const iree_host_size_t kernel_index = (iree_host_size_t)kernel_kind;
  const iree_host_size_t key_index = (iree_host_size_t)key;
  iree_string_view_t config_key =
      id4_qwen3_vl_program_config_keys[kernel_index][key_index];
  if (iree_string_view_is_empty(config_key)) {
    return id4_qwen3_vl_program_unsupported_config_key(kernel_kind, key);
  }
  *out_key = config_key;
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_program_make_config_binding(
    id4_qwen3_vl_kernel_kind_t kernel_kind, id4_qwen3_vl_config_key_t key,
    uint32_t value, char* value_buffer, iree_host_size_t value_buffer_capacity,
    id4_pipeline_kernel_config_binding_t* out_binding) {
  iree_string_view_t value_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_u32(
      value, value_buffer, value_buffer_capacity, &value_string));
  iree_string_view_t key_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_program_config_key(kernel_kind, key, &key_string));
  *out_binding =
      id4_pipeline_make_kernel_config_binding(key_string, value_string);
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_program_make_config_bindings(
    id4_qwen3_vl_kernel_kind_t kernel_kind, iree_host_size_t config_value_count,
    const id4_qwen3_vl_program_config_value_t* config_values,
    char value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                      [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY],
    id4_pipeline_kernel_config_binding_t* out_bindings) {
  if (config_value_count != 0 && !config_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL config value array is required");
  }
  if (config_value_count > ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Qwen3-VL config binding count %" PRIhsz " exceeds max count %u",
        config_value_count, ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT);
  }
  if (config_value_count != 0 && !out_bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL config binding output is required");
  }
  for (iree_host_size_t i = 0; i < config_value_count; ++i) {
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_binding(
        kernel_kind, config_values[i].key, config_values[i].value,
        value_buffers[i], ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY,
        &out_bindings[i]));
  }
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_program_import_tensor(
    id4_pipeline_program_builder_t* builder, id4_qwen3_vl_tensor_kind_t kind,
    id4_pipeline_program_import_tensor_flags_t flags,
    id4_pipeline_program_dtype_t dtype, id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  char name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_tensor_name(
      kind, UINT32_MAX, name_buffer, IREE_ARRAYSIZE(name_buffer), &name));
  id4_pipeline_program_import_tensor_options_t options = {
      .structure_size = sizeof(options),
      .flags = flags,
      .name = name,
      .dtype = dtype,
      .shape = shape,
  };
  return id4_pipeline_program_import_tensor(builder, &options, out_tensor);
}

static iree_status_t id4_qwen3_vl_program_parameter(
    id4_pipeline_program_builder_t* builder, iree_string_view_t source_scope,
    id4_qwen3_vl_parameter_kind_t kind, uint32_t layer_ordinal,
    id4_pipeline_program_dtype_t dtype, id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  char key_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_parameter_key(
      kind, layer_ordinal, key_buffer, IREE_ARRAYSIZE(key_buffer), &key));
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

static iree_status_t id4_qwen3_vl_program_direct_parameter_key(
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

static iree_status_t id4_qwen3_vl_program_format_linear_scale_key(
    iree_string_view_t weight_key, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_key) {
  return id4_qwen3_vl_program_format(buffer, buffer_capacity, out_key,
                                     "%.*s_scale_inv", (int)weight_key.size,
                                     weight_key.data);
}

static iree_status_t id4_qwen3_vl_program_parameter_token_embedding_rows(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder,
    id4_pipeline_program_tensor_t* out_tensor) {
  char source_key_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t source_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_parameter_key(
      ID4_QWEN3_VL_PARAMETER_TOKEN_EMBEDDING, UINT32_MAX, source_key_buffer,
      IREE_ARRAYSIZE(source_key_buffer), &source_key));
  char execution_key_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t execution_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_tensor_name(
      ID4_QWEN3_VL_TENSOR_TOKEN_EMBEDDING_ROWS, UINT32_MAX,
      execution_key_buffer, IREE_ARRAYSIZE(execution_key_buffer),
      &execution_key));

  iree_device_size_t row_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_bf16_byte_length(
      options->model.hidden_size, &row_byte_length));

  id4_pipeline_program_parameter_source_span_t* source_spans = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      options->host_allocator, options->request.token_count,
      sizeof(source_spans[0]), (void**)&source_spans);
  for (uint32_t i = 0;
       i < options->request.token_count && iree_status_is_ok(status); ++i) {
    iree_device_size_t source_offset = 0;
    if (!iree_device_size_checked_mul(
            (iree_device_size_t)options->request.token_ids[i], row_byte_length,
            &source_offset)) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "Qwen3-VL token embedding source offset overflows");
      break;
    }
    iree_device_size_t target_offset = 0;
    if (!iree_device_size_checked_mul((iree_device_size_t)i, row_byte_length,
                                      &target_offset)) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "Qwen3-VL token embedding target offset overflows");
      break;
    }
    source_spans[i] = (id4_pipeline_program_parameter_source_span_t){
        // Byte offset in the provider embedding table.
        .source_offset = source_offset,
        // Byte offset in the compact execution embedding rows.
        .target_offset = target_offset,
        // One dense embedding row.
        .length = row_byte_length,
    };
  }

  const id4_pipeline_program_parameter_source_t sources[] = {
      {
          // Provider scope containing the full token embedding table.
          .source_scope = options->parameter_scope,
          // Provider key for the full token embedding table.
          .key = source_key,
          // Provider dtype matching the compact execution rows.
          .dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
          // Provider shape for the full token embedding table.
          .shape = id4_pipeline_program_make_shape_rank2(
              options->model.vocab_size, options->model.hidden_size),
      },
  };
  if (iree_status_is_ok(status)) {
    id4_pipeline_program_parameter_options_t parameter_options = {
        // Size of this structure for versioning.
        .structure_size = sizeof(parameter_options),
        // Direct gather into compact execution rows.
        .encoding = ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
        // Number of provider source descriptors.
        .source_count = IREE_ARRAYSIZE(sources),
        // Provider source descriptors.
        .sources = sources,
        // Number of prompt-local row spans.
        .source_span_count = options->request.token_count,
        // Prompt-local row spans.
        .source_spans = source_spans,
        // Execution tensor key and diagnostic name.
        .key = execution_key,
        // Execution tensor dtype.
        .dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        // Execution tensor shape.
        .shape = id4_pipeline_program_make_shape_rank2(
            options->request.token_count, options->model.hidden_size),
    };
    status =
        id4_pipeline_program_parameter(builder, &parameter_options, out_tensor);
  }
  iree_allocator_free(options->host_allocator, source_spans);
  return status;
}

static iree_status_t id4_qwen3_vl_program_parameter_bf16_linear_rhs_tile(
    id4_pipeline_program_builder_t* builder, iree_string_view_t source_scope,
    id4_qwen3_vl_parameter_kind_t kind, uint32_t layer_ordinal,
    uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t* out_tensor) {
  char key_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_parameter_key(
      kind, layer_ordinal, key_buffer, IREE_ARRAYSIZE(key_buffer), &key));
  const id4_pipeline_program_shape_t shape =
      id4_pipeline_program_make_shape_rank2(output_size, input_size);
  const id4_pipeline_program_parameter_source_t sources[] = {
      {
          // Provider scope containing the logical row-major BF16 weight tensor.
          .source_scope = source_scope,
          // Provider key for the logical row-major BF16 weight tensor.
          .key = key,
          // Provider dtype for the logical weight tensor.
          .dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
          // Provider shape matching the logical execution tensor.
          .shape = shape,
      },
  };
  id4_pipeline_program_parameter_options_t options = {
      .structure_size = sizeof(options),
      .encoding = ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_BF16_LINEAR_RHS_TILE,
      .source_count = IREE_ARRAYSIZE(sources),
      .sources = sources,
      .key = key,
      .dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      .shape = shape,
  };
  return id4_pipeline_program_parameter(builder, &options, out_tensor);
}

static iree_status_t id4_qwen3_vl_program_parameter_fp8_linear_rhs_tile(
    id4_pipeline_program_builder_t* builder, iree_string_view_t source_scope,
    uint32_t input_size, uint32_t output_size, iree_string_view_t key,
    id4_pipeline_program_tensor_t* out_tensor) {
  const id4_pipeline_program_shape_t shape =
      id4_pipeline_program_make_shape_rank2(output_size, input_size);
  const id4_pipeline_program_parameter_source_t sources[] = {
      {
          // Provider scope containing the logical row-major FP8 weight tensor.
          .source_scope = source_scope,
          // Provider key for the logical row-major FP8 weight tensor.
          .key = key,
          // Provider dtype for the logical weight tensor.
          .dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
          // Provider shape matching the logical execution tensor.
          .shape = shape,
      },
  };
  id4_pipeline_program_parameter_options_t options = {
      .structure_size = sizeof(options),
      .encoding =
          ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
      .source_count = IREE_ARRAYSIZE(sources),
      .sources = sources,
      .key = key,
      .dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
      .shape = shape,
  };
  return id4_pipeline_program_parameter(builder, &options, out_tensor);
}

static iree_status_t id4_qwen3_vl_program_acquire_named_tensor(
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

static iree_status_t id4_qwen3_vl_program_barrier_named(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name) {
  id4_pipeline_program_barrier_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
  };
  return id4_pipeline_program_barrier(builder, &options);
}

static iree_status_t id4_qwen3_vl_program_make_parameter_encoder_config_binding(
    iree_string_view_t key, uint32_t value, char* value_buffer,
    id4_pipeline_kernel_config_binding_t* out_binding) {
  iree_string_view_t value_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_u32(
      value, value_buffer, ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY,
      &value_string));
  *out_binding = id4_pipeline_make_kernel_config_binding(key, value_string);
  return iree_ok_status();
}

static iree_status_t
id4_qwen3_vl_program_encode_fp8_block_scaled_linear_rhs_tile(
    id4_pipeline_program_builder_t* builder, iree_string_view_t weight_key,
    uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t source_weight,
    id4_pipeline_program_tensor_t source_scale,
    id4_pipeline_program_tensor_t* out_tensor) {
  char compact_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t compact_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format(
      compact_name_buffer, IREE_ARRAYSIZE(compact_name_buffer), &compact_name,
      "%.*s.compact_rhs_bf16", (int)weight_key.size, weight_key.data));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_named_tensor(
      builder, compact_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(output_size, input_size),
      out_tensor));

  char dispatch_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format(
      dispatch_name_buffer, IREE_ARRAYSIZE(dispatch_name_buffer),
      &dispatch_name, "%.*s.encode", (int)compact_name.size,
      compact_name.data));

  char value_buffers[2][ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t config_bindings[2];
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_program_make_parameter_encoder_config_binding(
          IREE_SV("id4.parameter.fp8_e4m3_block_scaled_to_bf16_linear_rhs_"
                  "tile.output_size"),
          output_size, value_buffers[0], &config_bindings[0]));
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_program_make_parameter_encoder_config_binding(
          IREE_SV("id4.parameter.fp8_e4m3_block_scaled_to_bf16_linear_rhs_"
                  "tile.input_size"),
          input_size, value_buffers[1], &config_bindings[1]));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(source_weight),
      id4_pipeline_program_read(source_scale),
      id4_pipeline_program_write(*out_tensor),
  };
  id4_pipeline_program_dispatch_loom_options_t options = {
      .structure_size = sizeof(options),
      .name = dispatch_name,
      .kernel = id4_pipeline_make_kernel_ref(
          IREE_SV("parameter/"
                  "fp8_e4m3_block_scaled_to_bf16_linear_rhs_tile"),
          IREE_SV("id4_parameter_fp8_e4m3_block_scaled_to_bf16_linear_rhs_"
                  "tile")),
      .config_binding_count = IREE_ARRAYSIZE(config_bindings),
      .config_bindings = config_bindings,
      .binding_count = IREE_ARRAYSIZE(bindings),
      .bindings = bindings,
  };
  return id4_pipeline_program_dispatch_loom(builder, &options);
}

static iree_status_t id4_qwen3_vl_program_parameter_linear_weight(
    id4_pipeline_program_builder_t* builder, iree_string_view_t source_scope,
    id4_qwen3_vl_parameter_format_t parameter_format,
    id4_qwen3_vl_parameter_kind_t kind, uint32_t layer_ordinal,
    uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_parameter_encoding_t encoding,
    id4_qwen3_vl_program_linear_weight_t* out_weight) {
  out_weight->weight = id4_pipeline_program_tensor_invalid();
  out_weight->scale = id4_pipeline_program_tensor_invalid();
  out_weight->has_scale = false;
  out_weight->requires_ready_barrier = false;
  switch (parameter_format) {
    case ID4_QWEN3_VL_PARAMETER_FORMAT_BF16:
      switch (encoding) {
        case ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT:
          return id4_qwen3_vl_program_parameter(
              builder, source_scope, kind, layer_ordinal,
              ID4_PIPELINE_PROGRAM_DTYPE_BF16,
              id4_pipeline_program_make_shape_rank2(output_size, input_size),
              &out_weight->weight);
        case ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_BF16_LINEAR_RHS_TILE:
          return id4_qwen3_vl_program_parameter_bf16_linear_rhs_tile(
              builder, source_scope, kind, layer_ordinal, input_size,
              output_size, &out_weight->weight);
        default:
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "Qwen3-VL BF16 linear weight parameter does "
                                  "not support encoding %u",
                                  (uint32_t)encoding);
      }
    case ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED: {
      char weight_key_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
      iree_string_view_t weight_key = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_parameter_key(
          kind, layer_ordinal, weight_key_buffer,
          IREE_ARRAYSIZE(weight_key_buffer), &weight_key));
      char scale_key_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
      iree_string_view_t scale_key = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_linear_scale_key(
          weight_key, scale_key_buffer, IREE_ARRAYSIZE(scale_key_buffer),
          &scale_key));
      const id4_pipeline_program_shape_t weight_shape =
          id4_pipeline_program_make_shape_rank2(output_size, input_size);
      const id4_pipeline_program_shape_t scale_shape =
          id4_pipeline_program_make_shape_rank2(
              id4_qwen3_vl_program_ceil_div_u32(
                  output_size, ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE),
              id4_qwen3_vl_program_ceil_div_u32(
                  input_size, ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE));
      switch (encoding) {
        case ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE: {
          IREE_RETURN_IF_ERROR(
              id4_qwen3_vl_program_parameter_fp8_linear_rhs_tile(
                  builder, source_scope, input_size, output_size, weight_key,
                  &out_weight->weight));
          IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_direct_parameter_key(
              builder, source_scope, scale_key, ID4_PIPELINE_PROGRAM_DTYPE_F32,
              scale_shape, &out_weight->scale));
          out_weight->has_scale = true;
          return iree_ok_status();
        }
        case ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_BLOCK_SCALED_TO_BF16_LINEAR_RHS_TILE: {
          id4_pipeline_program_tensor_t source_weight =
              id4_pipeline_program_tensor_invalid();
          IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_direct_parameter_key(
              builder, source_scope, weight_key,
              ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3, weight_shape,
              &source_weight));
          id4_pipeline_program_tensor_t source_scale =
              id4_pipeline_program_tensor_invalid();
          IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_direct_parameter_key(
              builder, source_scope, scale_key, ID4_PIPELINE_PROGRAM_DTYPE_F32,
              scale_shape, &source_scale));
          IREE_RETURN_IF_ERROR(
              id4_qwen3_vl_program_encode_fp8_block_scaled_linear_rhs_tile(
                  builder, weight_key, input_size, output_size, source_weight,
                  source_scale, &out_weight->weight));
          out_weight->requires_ready_barrier = true;
          return iree_ok_status();
        }
        default:
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "Qwen3-VL block-scaled FP8 linear weight parameter does not "
              "support encoding %u",
              (uint32_t)encoding);
      }
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen3-VL parameter format %" PRIu32
                              " is invalid",
                              (uint32_t)parameter_format);
  }
}

static iree_status_t id4_qwen3_vl_program_acquire_tensor(
    id4_pipeline_program_builder_t* builder, id4_qwen3_vl_tensor_kind_t kind,
    uint32_t layer_ordinal, id4_pipeline_program_dtype_t dtype,
    id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  char name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_tensor_name(
      kind, layer_ordinal, name_buffer, IREE_ARRAYSIZE(name_buffer), &name));
  id4_pipeline_program_acquire_tensor_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
      .dtype = dtype,
      .shape = shape,
  };
  return id4_pipeline_program_acquire_tensor(builder, &options, out_tensor);
}

static iree_status_t id4_qwen3_vl_program_dispatch_named(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_qwen3_vl_kernel_kind_t kernel_kind,
    iree_host_size_t config_binding_count,
    const id4_pipeline_kernel_config_binding_t* config_bindings,
    iree_host_size_t binding_count,
    const id4_pipeline_program_dispatch_binding_t* bindings) {
  id4_pipeline_kernel_ref_t kernel_ref = {};
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_program_kernel_ref(kernel_kind, &kernel_ref));
  id4_pipeline_program_dispatch_loom_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
      .kernel = kernel_ref,
      .config_binding_count = config_binding_count,
      .config_bindings = config_bindings,
      .binding_count = binding_count,
      .bindings = bindings,
  };
  return id4_pipeline_program_dispatch_loom(builder, &options);
}

static iree_status_t id4_qwen3_vl_program_dispatch(
    id4_pipeline_program_builder_t* builder,
    id4_qwen3_vl_operation_kind_t operation_kind, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site, id4_qwen3_vl_kernel_kind_t kernel_kind,
    iree_host_size_t config_binding_count,
    const id4_pipeline_kernel_config_binding_t* config_bindings,
    iree_host_size_t binding_count,
    const id4_pipeline_program_dispatch_binding_t* bindings) {
  char name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_operation_name(
      operation_kind, layer_ordinal, site, name_buffer,
      IREE_ARRAYSIZE(name_buffer), &name));
  return id4_qwen3_vl_program_dispatch_named(
      builder, name, kernel_kind, config_binding_count, config_bindings,
      binding_count, bindings);
}

static iree_status_t id4_qwen3_vl_program_barrier(
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site) {
  char name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_operation_name(
      ID4_QWEN3_VL_OPERATION_BARRIER, layer_ordinal, site, name_buffer,
      IREE_ARRAYSIZE(name_buffer), &name));
  id4_pipeline_program_barrier_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
  };
  return id4_pipeline_program_barrier(builder, &options);
}

static iree_status_t id4_qwen3_vl_program_tap(
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site, id4_pipeline_program_tensor_t tensor) {
  char name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_operation_name(
      ID4_QWEN3_VL_OPERATION_TAP, layer_ordinal, site, name_buffer,
      IREE_ARRAYSIZE(name_buffer), &name));
  id4_pipeline_program_tap_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
      .tensor = tensor,
  };
  return id4_pipeline_program_tap(builder, &options);
}

static iree_status_t id4_qwen3_vl_program_tap_tensor(
    id4_pipeline_program_builder_t* builder, id4_qwen3_vl_tensor_kind_t kind,
    uint32_t layer_ordinal, id4_pipeline_program_tensor_t tensor) {
  char name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_tensor_name(
      kind, layer_ordinal, name_buffer, IREE_ARRAYSIZE(name_buffer), &name));
  id4_pipeline_program_tap_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
      .tensor = tensor,
  };
  return id4_pipeline_program_tap(builder, &options);
}

static iree_status_t id4_qwen3_vl_program_zero_bf16_tail(
    id4_pipeline_program_builder_t* builder, id4_qwen3_vl_tensor_kind_t kind,
    uint32_t layer_ordinal, id4_pipeline_program_tensor_t tensor,
    const id4_qwen3_vl_program_bf16_coverage_t* coverage) {
  if (coverage->logical_element_count == coverage->dispatch_element_count) {
    return iree_ok_status();
  }
  if (coverage->logical_element_count > coverage->dispatch_element_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL tensor tail element count %" PRIu32
                            " exceeds dispatch element count %" PRIu32,
                            coverage->logical_element_count,
                            coverage->dispatch_element_count);
  }
  const iree_device_size_t tail_byte_length =
      coverage->dispatch_byte_length - coverage->logical_byte_length;

  char tensor_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t tensor_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_tensor_name(
      kind, layer_ordinal, tensor_name_buffer,
      IREE_ARRAYSIZE(tensor_name_buffer), &tensor_name));
  char operation_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t operation_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_child_name(
      tensor_name, IREE_SV("zero_tail"), operation_name_buffer,
      IREE_ARRAYSIZE(operation_name_buffer), &operation_name));

  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_ELEMENT_COUNT, coverage->logical_element_count},
      {ID4_QWEN3_VL_CONFIG_DISPATCH_ELEMENT_COUNT,
       coverage->dispatch_element_count},
  };
  char value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      ID4_QWEN3_VL_KERNEL_ZERO_TAIL_BF16, IREE_ARRAYSIZE(config_values),
      config_values, value_buffers, config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_write_range(tensor, coverage->logical_byte_length,
                                       tail_byte_length),
  };
  return id4_qwen3_vl_program_dispatch_named(
      builder, operation_name, ID4_QWEN3_VL_KERNEL_ZERO_TAIL_BF16,
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_qwen3_vl_program_author_token_embedding(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder,
    id4_pipeline_program_tensor_t* out_hidden_states,
    id4_pipeline_program_tensor_t* out_attention_mask,
    id4_pipeline_program_tensor_t* out_token_weights) {
  const uint32_t token_count = options->request.token_count;
  const uint32_t hidden_size = options->model.hidden_size;
  uint32_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, &token_capacity));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_import_tensor(
      builder, ID4_QWEN3_VL_TENSOR_ATTENTION_MASK,
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(token_capacity, token_capacity),
      out_attention_mask));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_import_tensor(
      builder, ID4_QWEN3_VL_TENSOR_TOKEN_WEIGHTS,
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(token_count), out_token_weights));
  id4_pipeline_program_tensor_t embedding_rows =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_parameter_token_embedding_rows(
      options, builder, &embedding_rows));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, ID4_QWEN3_VL_TENSOR_EMBEDDED_HIDDEN_STATES, UINT32_MAX,
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_capacity, hidden_size),
      out_hidden_states));

  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
      {ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT, token_capacity},
      {ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE, hidden_size},
      {ID4_QWEN3_VL_CONFIG_VOCAB_SIZE, options->model.vocab_size},
  };
  char value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      ID4_QWEN3_VL_KERNEL_TOKEN_EMBEDDING, IREE_ARRAYSIZE(config_values),
      config_values, value_buffers, config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(embedding_rows),
      id4_pipeline_program_write(*out_hidden_states),
  };
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_TOKEN_EMBEDDING, UINT32_MAX,
      ID4_QWEN3_VL_OPERATION_SITE_TOKEN_EMBEDDING,
      ID4_QWEN3_VL_KERNEL_TOKEN_EMBEDDING, IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings));
  return id4_qwen3_vl_program_barrier(
      builder, UINT32_MAX, ID4_QWEN3_VL_OPERATION_SITE_AFTER_TOKEN_EMBEDDING);
}

static iree_status_t id4_qwen3_vl_program_author_rmsnorm_rows(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site,
    id4_qwen3_vl_parameter_kind_t parameter_kind,
    id4_qwen3_vl_tensor_kind_t output_kind, id4_pipeline_program_tensor_t input,
    uint32_t row_count, uint32_t channel_count, uint32_t output_width,
    id4_pipeline_program_tensor_t* out_output) {
  uint32_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, &token_capacity));
  id4_pipeline_program_tensor_t weight = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_parameter(
      builder, options->parameter_scope, parameter_kind, layer_ordinal,
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(channel_count), &weight));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, output_kind, layer_ordinal, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_capacity, output_width),
      out_output));
  id4_qwen3_vl_program_bf16_coverage_t output_coverage;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_bf16_coverage(
      "RMSNorm output", row_count, channel_count, token_capacity, output_width,
      &output_coverage));
  if ((output_coverage.dispatch_element_count % channel_count) != 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL RMSNorm dispatch row count overflow");
  }
  const uint32_t dispatch_row_count =
      output_coverage.dispatch_element_count / channel_count;

  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, dispatch_row_count},
      {ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE, channel_count},
  };
  char value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      ID4_QWEN3_VL_KERNEL_RMSNORM, IREE_ARRAYSIZE(config_values), config_values,
      value_buffers, config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_write(*out_output),
  };
  return id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_RMSNORM, layer_ordinal, site,
      ID4_QWEN3_VL_KERNEL_RMSNORM, IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

static iree_status_t id4_qwen3_vl_program_author_rmsnorm(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site,
    id4_qwen3_vl_parameter_kind_t parameter_kind,
    id4_qwen3_vl_tensor_kind_t output_kind, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t* out_output) {
  return id4_qwen3_vl_program_author_rmsnorm_rows(
      options, builder, layer_ordinal, site, parameter_kind, output_kind, input,
      options->request.token_count, options->model.hidden_size,
      options->model.hidden_size, out_output);
}

static bool id4_qwen3_vl_program_uses_resident_fp8_matrix_family(
    const id4_qwen3_vl_program_options_t* options) {
  return options->parameter_format ==
             ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED &&
         (options->weight_execution_strategy ==
              ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_COMPACT_RHS ||
          options->weight_execution_strategy ==
              ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS);
}

typedef struct id4_qwen3_vl_program_matrix_parameter_t {
  // Storage for the formatted provider weight key.
  char weight_key_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  // Storage for the formatted provider scale key.
  char scale_key_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  // Provider-side matrix parameter contract borrowing the key buffers.
  id4_pipeline_program_matrix_parameter_t value;
} id4_qwen3_vl_program_matrix_parameter_t;

static id4_pipeline_program_matrix_problem_t
id4_qwen3_vl_program_make_matrix_problem(
    const id4_qwen3_vl_program_options_t* options, uint32_t token_capacity,
    uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_matrix_epilogue_t epilogue) {
  return (id4_pipeline_program_matrix_problem_t){
      .valid_m = options->request.token_count,
      .m_capacity = token_capacity,
      .n = output_size,
      .k = input_size,
      .input_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      .input_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
      .accumulator_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
      .epilogue = epilogue,
      .output_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      .output_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
  };
}

static iree_status_t id4_qwen3_vl_program_make_matrix_parameter(
    const id4_qwen3_vl_program_options_t* options, uint32_t layer_ordinal,
    id4_qwen3_vl_parameter_kind_t parameter_kind, uint32_t input_size,
    uint32_t output_size,
    id4_qwen3_vl_program_matrix_parameter_t* out_parameter) {
  *out_parameter = (id4_qwen3_vl_program_matrix_parameter_t){0};
  iree_string_view_t weight_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_parameter_key(
      parameter_kind, layer_ordinal, out_parameter->weight_key_buffer,
      IREE_ARRAYSIZE(out_parameter->weight_key_buffer), &weight_key));
  iree_string_view_t scale_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_linear_scale_key(
      weight_key, out_parameter->scale_key_buffer,
      IREE_ARRAYSIZE(out_parameter->scale_key_buffer), &scale_key));
  out_parameter->value = (id4_pipeline_program_matrix_parameter_t){
      .weight =
          {
              .source_scope = options->parameter_scope,
              .key = weight_key,
              .dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
              .shape = id4_pipeline_program_make_shape_rank2(output_size,
                                                             input_size),
          },
      .weight_layout =
          ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
      .scale =
          {
              .source_scope = options->parameter_scope,
              .key = scale_key,
              .dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
              .shape = id4_pipeline_program_make_shape_rank2(
                  id4_qwen3_vl_program_ceil_div_u32(
                      output_size, ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE),
                  id4_qwen3_vl_program_ceil_div_u32(
                      input_size, ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE)),
          },
      .scale_layout =
          ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
  };
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_program_prepare_linear_weight(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_parameter_kind_t parameter_kind, uint32_t input_size,
    uint32_t output_size, uint32_t* out_token_capacity,
    uint32_t* out_wmma_token_count,
    id4_qwen3_vl_program_linear_kernel_selection_t* out_selection,
    id4_qwen3_vl_program_linear_weight_t* out_weight) {
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, out_token_capacity));
  if ((input_size % ID4_QWEN3_VL_LINEAR_SCALAR_INPUT_BLOCK) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL scalar linear input size %" PRIu32
                            " must be a multiple of %u",
                            input_size, ID4_QWEN3_VL_LINEAR_SCALAR_INPUT_BLOCK);
  }
  if (id4_qwen3_vl_program_uses_resident_fp8_matrix_family(options)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen3-VL resident FP8 parameters must use semantic matrix authoring");
  }
  const id4_qwen3_vl_program_linear_wmma_tile_t* wmma_tile =
      id4_qwen3_vl_program_select_linear_wmma_tile(
          options->parameter_format, options->weight_execution_strategy,
          *out_token_capacity, input_size, output_size);
  *out_wmma_token_count =
      wmma_tile
          ? *out_token_capacity - (*out_token_capacity % wmma_tile->token_block)
          : 0;
  *out_selection = (id4_qwen3_vl_program_linear_kernel_selection_t){
      // Kernel kind selected when a WMMA tile covers at least one row.
      .kernel_kind = (id4_qwen3_vl_kernel_kind_t)ID4_QWEN3_VL_KERNEL_KIND_COUNT,
      // Parameter encoding used by row-major scalar or WMMA paths.
      .weight_encoding = ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
  };
  if (options->weight_execution_strategy ==
          ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_COMPACT_RHS ||
      options->weight_execution_strategy ==
          ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_STREAMING_COMPACT_RHS) {
    if (!wmma_tile || *out_wmma_token_count != *out_token_capacity) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Qwen3-VL compact-RHS linear weight execution requires full WMMA "
          "coverage");
    }
  }
  id4_qwen3_vl_weight_execution_strategy_t wmma_strategy =
      options->weight_execution_strategy;
  if (wmma_strategy ==
          ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS &&
      options->parameter_format == ID4_QWEN3_VL_PARAMETER_FORMAT_BF16 &&
      (!wmma_tile || *out_wmma_token_count != *out_token_capacity)) {
    wmma_strategy = ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR;
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_select_linear_kernel_for_source(
      options->parameter_format, wmma_strategy, wmma_tile,
      *out_wmma_token_count, *out_token_capacity, input_size, output_size,
      out_selection));

  return id4_qwen3_vl_program_parameter_linear_weight(
      builder, options->parameter_scope, options->parameter_format,
      parameter_kind, layer_ordinal, input_size, output_size,
      out_selection->weight_encoding, out_weight);
}

static iree_status_t id4_qwen3_vl_program_author_parameter_matrix(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site,
    id4_qwen3_vl_parameter_kind_t parameter_kind,
    id4_qwen3_vl_tensor_kind_t output_kind, id4_pipeline_program_tensor_t input,
    uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t* out_output) {
  uint32_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, &token_capacity));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, output_kind, layer_ordinal, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_capacity, output_size),
      out_output));

  char operation_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t operation_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_operation_name(
      ID4_QWEN3_VL_OPERATION_LINEAR, layer_ordinal, site, operation_name_buffer,
      IREE_ARRAYSIZE(operation_name_buffer), &operation_name));
  char dispatch_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t dispatch_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_child_name(
      operation_name, IREE_SV("matrix"), dispatch_name_buffer,
      IREE_ARRAYSIZE(dispatch_name_buffer), &dispatch_name));

  id4_qwen3_vl_program_matrix_parameter_t parameter;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_matrix_parameter(
      options, layer_ordinal, parameter_kind, input_size, output_size,
      &parameter));
  const id4_pipeline_program_matrix_options_t matrix_options = {
      .structure_size = sizeof(matrix_options),
      .name = dispatch_name,
      .problem = id4_qwen3_vl_program_make_matrix_problem(
          options, token_capacity, input_size, output_size,
          ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE),
      .operands =
          {
              .input = input,
              .parameter = parameter.value,
              .addend = id4_pipeline_program_tensor_invalid(),
              .output = *out_output,
          },
  };
  return id4_pipeline_program_matrix(builder, &matrix_options);
}

static iree_status_t id4_qwen3_vl_program_author_linear_prepared(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site, id4_qwen3_vl_tensor_kind_t output_kind,
    id4_pipeline_program_tensor_t input, uint32_t input_size,
    uint32_t output_size, uint32_t token_capacity, uint32_t wmma_token_count,
    const id4_qwen3_vl_program_linear_kernel_selection_t* wmma_selection,
    const id4_qwen3_vl_program_linear_weight_t* weight,
    id4_qwen3_vl_program_linear_dispatch_flags_t flags,
    id4_pipeline_program_tensor_t* out_output) {
  const uint32_t token_count = options->request.token_count;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, output_kind, layer_ordinal, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_capacity, output_size),
      out_output));
  id4_qwen3_vl_program_bf16_coverage_t output_coverage;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_bf16_coverage(
      "linear output", token_count, output_size, token_capacity, output_size,
      &output_coverage));

  if (wmma_token_count != 0) {
    iree_device_size_t row_byte_length = 0;
    if (!iree_device_size_checked_mul(output_size, sizeof(uint16_t),
                                      &row_byte_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Qwen3-VL linear row byte length overflow");
    }
    iree_device_size_t wmma_byte_length = 0;
    if (!iree_device_size_checked_mul(wmma_token_count, row_byte_length,
                                      &wmma_byte_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Qwen3-VL linear WMMA byte coverage overflow");
    }

    char operation_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t operation_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_operation_name(
        ID4_QWEN3_VL_OPERATION_LINEAR, layer_ordinal, site,
        operation_name_buffer, IREE_ARRAYSIZE(operation_name_buffer),
        &operation_name));

    char wmma_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t wmma_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_child_name(
        operation_name,
        id4_qwen3_vl_program_selection_uses_compact_rhs(wmma_selection)
            ? IREE_SV("wmma_compact_rhs")
            : IREE_SV("wmma"),
        wmma_name_buffer, IREE_ARRAYSIZE(wmma_name_buffer), &wmma_name));
    if (iree_all_bits_set(
            flags,
            ID4_QWEN3_VL_PROGRAM_LINEAR_DISPATCH_FLAG_INSERT_WEIGHT_READY_BARRIER) &&
        weight->requires_ready_barrier) {
      char ready_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
      iree_string_view_t ready_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_child_name(
          operation_name, IREE_SV("weight_ready"), ready_name_buffer,
          IREE_ARRAYSIZE(ready_name_buffer), &ready_name));
      IREE_RETURN_IF_ERROR(
          id4_qwen3_vl_program_barrier_named(builder, ready_name));
    }
    const id4_qwen3_vl_program_config_value_t wmma_config_values[] = {
        {ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT, wmma_token_count},
        {ID4_QWEN3_VL_CONFIG_INPUT_SIZE, input_size},
        {ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE, output_size},
    };
    char wmma_value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                           [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
    id4_pipeline_kernel_config_binding_t
        wmma_config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
        wmma_selection->kernel_kind, IREE_ARRAYSIZE(wmma_config_values),
        wmma_config_values, wmma_value_buffers, wmma_config_bindings));
    id4_pipeline_program_dispatch_binding_t wmma_bindings[4];
    iree_host_size_t wmma_binding_count = 0;
    wmma_bindings[wmma_binding_count++] = id4_pipeline_program_read(input);
    wmma_bindings[wmma_binding_count++] =
        id4_pipeline_program_read(weight->weight);
    if (weight->has_scale) {
      wmma_bindings[wmma_binding_count++] =
          id4_pipeline_program_read(weight->scale);
    }
    wmma_bindings[wmma_binding_count++] =
        id4_pipeline_program_write_range(*out_output, 0, wmma_byte_length);
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch_named(
        builder, wmma_name, wmma_selection->kernel_kind,
        IREE_ARRAYSIZE(wmma_config_values), wmma_config_bindings,
        wmma_binding_count, wmma_bindings));

    const uint32_t tail_token_count =
        token_count > wmma_token_count ? token_count - wmma_token_count : 0;
    if (tail_token_count == 0) return iree_ok_status();
    iree_device_size_t tail_byte_length = 0;
    if (!iree_device_size_checked_mul(tail_token_count, row_byte_length,
                                      &tail_byte_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Qwen3-VL linear tail byte coverage overflow");
    }

    char tail_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t tail_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_child_name(
        operation_name, IREE_SV("tail"), tail_name_buffer,
        IREE_ARRAYSIZE(tail_name_buffer), &tail_name));
    const id4_qwen3_vl_program_config_value_t tail_config_values[] = {
        {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
        {ID4_QWEN3_VL_CONFIG_TOKEN_START, wmma_token_count},
        {ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT, tail_token_count},
        {ID4_QWEN3_VL_CONFIG_INPUT_SIZE, input_size},
        {ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE, output_size},
    };
    char tail_value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                           [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
    id4_pipeline_kernel_config_binding_t
        tail_config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
        ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_TAIL,
        IREE_ARRAYSIZE(tail_config_values), tail_config_values,
        tail_value_buffers, tail_config_bindings));
    id4_pipeline_program_dispatch_binding_t tail_bindings[] = {
        id4_pipeline_program_read(input),
        id4_pipeline_program_read(weight->weight),
        id4_pipeline_program_write_range(*out_output, wmma_byte_length,
                                         tail_byte_length),
    };
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch_named(
        builder, tail_name, ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_TAIL,
        IREE_ARRAYSIZE(tail_config_values), tail_config_bindings,
        IREE_ARRAYSIZE(tail_bindings), tail_bindings));
    return id4_qwen3_vl_program_zero_bf16_tail(
        builder, output_kind, layer_ordinal, *out_output, &output_coverage);
  }

  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
      {ID4_QWEN3_VL_CONFIG_INPUT_SIZE, input_size},
      {ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE, output_size},
  };
  char value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16, IREE_ARRAYSIZE(config_values),
      config_values, value_buffers, config_bindings));
  if (iree_all_bits_set(
          flags,
          ID4_QWEN3_VL_PROGRAM_LINEAR_DISPATCH_FLAG_INSERT_WEIGHT_READY_BARRIER) &&
      weight->requires_ready_barrier) {
    char operation_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t operation_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_operation_name(
        ID4_QWEN3_VL_OPERATION_LINEAR, layer_ordinal, site,
        operation_name_buffer, IREE_ARRAYSIZE(operation_name_buffer),
        &operation_name));
    char ready_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t ready_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_child_name(
        operation_name, IREE_SV("weight_ready"), ready_name_buffer,
        IREE_ARRAYSIZE(ready_name_buffer), &ready_name));
    IREE_RETURN_IF_ERROR(
        id4_qwen3_vl_program_barrier_named(builder, ready_name));
  }
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight->weight),
      id4_qwen3_vl_program_bf16_logical_write(*out_output, &output_coverage),
  };
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_LINEAR, layer_ordinal, site,
      ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16, IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings));
  return id4_qwen3_vl_program_zero_bf16_tail(
      builder, output_kind, layer_ordinal, *out_output, &output_coverage);
}

static iree_status_t id4_qwen3_vl_program_author_linear(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site,
    id4_qwen3_vl_parameter_kind_t parameter_kind,
    id4_qwen3_vl_tensor_kind_t output_kind, id4_pipeline_program_tensor_t input,
    uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t* out_output) {
  if (id4_qwen3_vl_program_uses_resident_fp8_matrix_family(options)) {
    return id4_qwen3_vl_program_author_parameter_matrix(
        options, builder, layer_ordinal, site, parameter_kind, output_kind,
        input, input_size, output_size, out_output);
  }
  uint32_t token_capacity = 0;
  uint32_t wmma_token_count = 0;
  id4_qwen3_vl_program_linear_kernel_selection_t selection;
  id4_qwen3_vl_program_linear_weight_t weight;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_prepare_linear_weight(
      options, builder, layer_ordinal, parameter_kind, input_size, output_size,
      &token_capacity, &wmma_token_count, &selection, &weight));
  return id4_qwen3_vl_program_author_linear_prepared(
      options, builder, layer_ordinal, site, output_kind, input, input_size,
      output_size, token_capacity, wmma_token_count, &selection, &weight,
      ID4_QWEN3_VL_PROGRAM_LINEAR_DISPATCH_FLAG_INSERT_WEIGHT_READY_BARRIER,
      out_output);
}

static bool id4_qwen3_vl_program_select_value_projection_packed_kernel(
    const id4_qwen3_vl_program_options_t* options, uint32_t token_capacity,
    id4_qwen3_vl_program_linear_kernel_selection_t* out_selection) {
  if (options->parameter_format ==
          ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED &&
      options->weight_execution_strategy !=
          ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_STREAMING_COMPACT_RHS) {
    return false;
  }
  const id4_qwen3_vl_program_linear_wmma_tile_t* wmma_tile =
      id4_qwen3_vl_program_select_linear_wmma_transposed_output_tile(
          options->parameter_format, options->weight_execution_strategy,
          token_capacity, options->model.hidden_size,
          id4_qwen3_vl_program_key_value_width(&options->model));
  if (!wmma_tile || (token_capacity % wmma_tile->token_block) != 0) {
    return false;
  }
  switch (options->weight_execution_strategy) {
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR:
      if (!id4_qwen3_vl_program_kernel_kind_is_valid(
              wmma_tile->kernels.direct_transposed_output_kernel_kind)) {
        return false;
      }
      out_selection->kernel_kind =
          wmma_tile->kernels.direct_transposed_output_kernel_kind;
      out_selection->weight_encoding =
          ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT;
      return true;
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_COMPACT_RHS:
      if (!id4_qwen3_vl_program_kernel_kind_is_valid(
              wmma_tile->kernels.compact_rhs_transposed_output_kernel_kind)) {
        return false;
      }
      out_selection->kernel_kind =
          wmma_tile->kernels.compact_rhs_transposed_output_kernel_kind;
      out_selection->weight_encoding =
          ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_BF16_LINEAR_RHS_TILE;
      return true;
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS:
      if (id4_qwen3_vl_program_kernel_kind_is_valid(
              wmma_tile->kernels.compact_rhs_transposed_output_kernel_kind)) {
        out_selection->kernel_kind =
            wmma_tile->kernels.compact_rhs_transposed_output_kernel_kind;
        out_selection->weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_BF16_LINEAR_RHS_TILE;
        return true;
      }
      if (id4_qwen3_vl_program_kernel_kind_is_valid(
              wmma_tile->kernels.direct_transposed_output_kernel_kind)) {
        out_selection->kernel_kind =
            wmma_tile->kernels.direct_transposed_output_kernel_kind;
        out_selection->weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT;
        return true;
      }
      return false;
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_STREAMING_COMPACT_RHS:
      if (!id4_qwen3_vl_program_kernel_kind_is_valid(
              wmma_tile->kernels.compact_rhs_transposed_output_kernel_kind)) {
        return false;
      }
      out_selection->kernel_kind =
          wmma_tile->kernels.compact_rhs_transposed_output_kernel_kind;
      out_selection->weight_encoding =
          options->parameter_format == ID4_QWEN3_VL_PARAMETER_FORMAT_BF16
              ? ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_BF16_LINEAR_RHS_TILE
              : ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_BLOCK_SCALED_TO_BF16_LINEAR_RHS_TILE;
      return true;
    default:
      return false;
  }
}

static bool id4_qwen3_vl_program_can_author_value_projection_packed(
    const id4_qwen3_vl_program_options_t* options, uint32_t token_capacity) {
  id4_qwen3_vl_program_linear_kernel_selection_t selection;
  memset(&selection, 0, sizeof(selection));
  return id4_qwen3_vl_program_select_value_projection_packed_kernel(
      options, token_capacity, &selection);
}

static iree_status_t
id4_qwen3_vl_program_prepare_value_projection_packed_weight(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    uint32_t* out_token_capacity,
    id4_qwen3_vl_program_linear_kernel_selection_t* out_selection,
    id4_qwen3_vl_program_linear_weight_t* out_weight) {
  const uint32_t hidden_size = options->model.hidden_size;
  const uint32_t key_value_width =
      id4_qwen3_vl_program_key_value_width(&options->model);
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, out_token_capacity));
  memset(out_selection, 0, sizeof(*out_selection));
  if (!id4_qwen3_vl_program_select_value_projection_packed_kernel(
          options, *out_token_capacity, out_selection)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL packed value projection requires a transpose-output WMMA "
        "tile");
  }

  return id4_qwen3_vl_program_parameter_linear_weight(
      builder, options->parameter_scope, options->parameter_format,
      ID4_QWEN3_VL_PARAMETER_LAYER_V_PROJECTION, layer_ordinal, hidden_size,
      key_value_width, out_selection->weight_encoding, out_weight);
}

static iree_status_t
id4_qwen3_vl_program_author_value_projection_packed_prepared(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_pipeline_program_tensor_t input, uint32_t token_capacity,
    const id4_qwen3_vl_program_linear_kernel_selection_t* packed_selection,
    const id4_qwen3_vl_program_linear_weight_t* weight,
    id4_qwen3_vl_program_linear_dispatch_flags_t flags,
    id4_pipeline_program_tensor_t* out_packed_value) {
  const uint32_t hidden_size = options->model.hidden_size;
  const uint32_t key_value_width =
      id4_qwen3_vl_program_key_value_width(&options->model);
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_V_PACKED, layer_ordinal,
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(key_value_width, token_capacity),
      out_packed_value));

  char operation_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t operation_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_operation_name(
      ID4_QWEN3_VL_OPERATION_LINEAR, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_V_PROJECTION,
      operation_name_buffer, IREE_ARRAYSIZE(operation_name_buffer),
      &operation_name));
  char packed_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t packed_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_child_name(
      operation_name,
      id4_qwen3_vl_program_selection_uses_compact_rhs(packed_selection)
          ? IREE_SV("wmma_packed_compact_rhs")
          : IREE_SV("wmma_packed"),
      packed_name_buffer, IREE_ARRAYSIZE(packed_name_buffer), &packed_name));
  if (iree_all_bits_set(
          flags,
          ID4_QWEN3_VL_PROGRAM_LINEAR_DISPATCH_FLAG_INSERT_WEIGHT_READY_BARRIER) &&
      weight->requires_ready_barrier) {
    char ready_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t ready_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_child_name(
        operation_name, IREE_SV("weight_ready"), ready_name_buffer,
        IREE_ARRAYSIZE(ready_name_buffer), &ready_name));
    IREE_RETURN_IF_ERROR(
        id4_qwen3_vl_program_barrier_named(builder, ready_name));
  }

  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT, token_capacity},
      {ID4_QWEN3_VL_CONFIG_INPUT_SIZE, hidden_size},
      {ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE, key_value_width},
  };
  char value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      packed_selection->kernel_kind, IREE_ARRAYSIZE(config_values),
      config_values, value_buffers, config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight->weight),
      id4_pipeline_program_write(*out_packed_value),
  };
  return id4_qwen3_vl_program_dispatch_named(
      builder, packed_name, packed_selection->kernel_kind,
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_qwen3_vl_program_author_value_projection_packed(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t* out_packed_value) {
  uint32_t token_capacity = 0;
  id4_qwen3_vl_program_linear_kernel_selection_t selection;
  id4_qwen3_vl_program_linear_weight_t weight;
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_program_prepare_value_projection_packed_weight(
          options, builder, layer_ordinal, &token_capacity, &selection,
          &weight));
  return id4_qwen3_vl_program_author_value_projection_packed_prepared(
      options, builder, layer_ordinal, input, token_capacity, &selection,
      &weight,
      ID4_QWEN3_VL_PROGRAM_LINEAR_DISPATCH_FLAG_INSERT_WEIGHT_READY_BARRIER,
      out_packed_value);
}

static iree_status_t id4_qwen3_vl_program_author_rotary(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site, id4_pipeline_program_tensor_t input,
    uint32_t channel_count, id4_qwen3_vl_tensor_kind_t output_kind,
    id4_pipeline_program_tensor_t* out_output) {
  uint32_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, &token_capacity));
  if ((channel_count % 2) != 0 || (options->model.head_size % 2) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL rotary dimensions must be even");
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, output_kind, layer_ordinal, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_capacity, channel_count),
      out_output));
  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_capacity},
      {ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE, channel_count},
      {ID4_QWEN3_VL_CONFIG_HEAD_SIZE, options->model.head_size},
  };
  char value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      ID4_QWEN3_VL_KERNEL_ROTARY, IREE_ARRAYSIZE(config_values), config_values,
      value_buffers, config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_write(*out_output),
  };
  return id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_ROTARY, layer_ordinal, site,
      ID4_QWEN3_VL_KERNEL_ROTARY, IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

static iree_status_t id4_qwen3_vl_program_author_rmsnorm_rotary(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site,
    id4_qwen3_vl_parameter_kind_t parameter_kind,
    id4_qwen3_vl_tensor_kind_t output_kind, id4_pipeline_program_tensor_t input,
    uint32_t head_count, id4_pipeline_program_tensor_t* out_output) {
  uint32_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, &token_capacity));
  if ((options->model.head_size % 2) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL fused RMSNorm rotary head size must be "
                            "even");
  }
  uint32_t output_width = 0;
  if (!id4_qwen3_vl_program_checked_mul_u32(
          head_count, options->model.head_size, &output_width)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Qwen3-VL fused RMSNorm rotary output width overflow");
  }

  id4_pipeline_program_tensor_t weight = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_parameter(
      builder, options->parameter_scope, parameter_kind, layer_ordinal,
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(options->model.head_size),
      &weight));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, output_kind, layer_ordinal, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_capacity, output_width),
      out_output));

  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_capacity},
      {ID4_QWEN3_VL_CONFIG_HEAD_COUNT, head_count},
      {ID4_QWEN3_VL_CONFIG_HEAD_SIZE, options->model.head_size},
  };
  char value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      ID4_QWEN3_VL_KERNEL_RMSNORM_ROTARY, IREE_ARRAYSIZE(config_values),
      config_values, value_buffers, config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_write(*out_output),
  };
  return id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_ROTARY, layer_ordinal, site,
      ID4_QWEN3_VL_KERNEL_RMSNORM_ROTARY, IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

static iree_status_t id4_qwen3_vl_program_dispatch_attention(
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site, id4_qwen3_vl_kernel_kind_t kernel_kind,
    iree_host_size_t config_value_count,
    const id4_qwen3_vl_program_config_value_t* config_values,
    iree_host_size_t binding_count,
    const id4_pipeline_program_dispatch_binding_t* bindings) {
  char value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      kernel_kind, config_value_count, config_values, value_buffers,
      config_bindings));
  return id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_ATTENTION, layer_ordinal, site,
      kernel_kind, config_value_count, config_bindings, binding_count,
      bindings);
}

static iree_status_t id4_qwen3_vl_program_author_attention_materialized(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t value,
    id4_pipeline_program_tensor_t attention_mask,
    id4_pipeline_program_tensor_t* out_context) {
  const uint32_t token_count = options->request.token_count;
  const uint32_t hidden_size = options->model.hidden_size;
  const uint32_t attention_head_count = options->model.attention_head_count;
  uint32_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, &token_capacity));
  id4_pipeline_program_tensor_t scores = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_SCORES, layer_ordinal,
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank3(attention_head_count, token_count,
                                            token_count),
      &scores));
  id4_pipeline_program_tensor_t probabilities =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_PROBABILITIES, layer_ordinal,
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank3(attention_head_count, token_count,
                                            token_count),
      &probabilities));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_CONTEXT, layer_ordinal,
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_capacity, hidden_size),
      out_context));

  const id4_pipeline_program_dispatch_binding_t score_bindings[] = {
      id4_pipeline_program_read(query),
      id4_pipeline_program_read(key),
      id4_pipeline_program_read(attention_mask),
      id4_pipeline_program_write(scores),
  };
  const id4_qwen3_vl_program_config_value_t score_config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
      {ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT, token_capacity},
      {ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE, hidden_size},
      {ID4_QWEN3_VL_CONFIG_ATTENTION_HEAD_COUNT, attention_head_count},
      {ID4_QWEN3_VL_CONFIG_KEY_VALUE_HEAD_COUNT,
       options->model.key_value_head_count},
      {ID4_QWEN3_VL_CONFIG_HEAD_SIZE, options->model.head_size},
  };
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch_attention(
      builder, layer_ordinal, ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_SCORES,
      ID4_QWEN3_VL_KERNEL_ATTENTION_SCORES, IREE_ARRAYSIZE(score_config_values),
      score_config_values, IREE_ARRAYSIZE(score_bindings), score_bindings));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION_SCORES));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_SCORES, layer_ordinal,
      scores));

  const id4_pipeline_program_dispatch_binding_t softmax_bindings[] = {
      id4_pipeline_program_read(scores),
      id4_pipeline_program_write(probabilities),
  };
  const id4_qwen3_vl_program_config_value_t softmax_config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
      {ID4_QWEN3_VL_CONFIG_ATTENTION_HEAD_COUNT, attention_head_count},
  };
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch_attention(
      builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_SOFTMAX,
      ID4_QWEN3_VL_KERNEL_ATTENTION_SOFTMAX,
      IREE_ARRAYSIZE(softmax_config_values), softmax_config_values,
      IREE_ARRAYSIZE(softmax_bindings), softmax_bindings));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION_SOFTMAX));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_PROBABILITIES, layer_ordinal,
      probabilities));

  id4_pipeline_program_dispatch_binding_t pv_bindings[] = {
      id4_pipeline_program_read(probabilities),
      id4_pipeline_program_read(value),
      id4_pipeline_program_write(*out_context),
  };
  const id4_qwen3_vl_program_config_value_t pv_config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
      {ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT, token_capacity},
      {ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE, hidden_size},
      {ID4_QWEN3_VL_CONFIG_ATTENTION_HEAD_COUNT, attention_head_count},
      {ID4_QWEN3_VL_CONFIG_KEY_VALUE_HEAD_COUNT,
       options->model.key_value_head_count},
      {ID4_QWEN3_VL_CONFIG_HEAD_SIZE, options->model.head_size},
  };
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch_attention(
      builder, layer_ordinal, ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION,
      ID4_QWEN3_VL_KERNEL_ATTENTION_PV, IREE_ARRAYSIZE(pv_config_values),
      pv_config_values, IREE_ARRAYSIZE(pv_bindings), pv_bindings));
  return id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_CONTEXT, layer_ordinal,
      *out_context);
}

static iree_status_t id4_qwen3_vl_program_dispatch_value_pack(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_pipeline_program_tensor_t value,
    id4_pipeline_program_tensor_t packed_value) {
  const uint32_t token_count = options->request.token_count;
  const uint32_t key_value_width =
      id4_qwen3_vl_program_key_value_width(&options->model);
  uint32_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, &token_capacity));
  uint32_t element_count = 0;
  if (!id4_qwen3_vl_program_checked_mul_u32(token_count, key_value_width,
                                            &element_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL value pack element count overflow");
  }

  char tensor_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t tensor_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_tensor_name(
      ID4_QWEN3_VL_TENSOR_LAYER_V_PACKED, layer_ordinal, tensor_name_buffer,
      IREE_ARRAYSIZE(tensor_name_buffer), &tensor_name));
  char operation_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t operation_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_child_name(
      tensor_name, IREE_SV("pack"), operation_name_buffer,
      IREE_ARRAYSIZE(operation_name_buffer), &operation_name));

  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
      {ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT, token_capacity},
      {ID4_QWEN3_VL_CONFIG_INPUT_SIZE, key_value_width},
      {ID4_QWEN3_VL_CONFIG_ELEMENT_COUNT, element_count},
  };
  char value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      ID4_QWEN3_VL_KERNEL_LINEAR_INPUT_PACK_TRANSPOSE_BF16_BF16,
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(value),
      id4_pipeline_program_write(packed_value),
  };
  return id4_qwen3_vl_program_dispatch_named(
      builder, operation_name,
      ID4_QWEN3_VL_KERNEL_LINEAR_INPUT_PACK_TRANSPOSE_BF16_BF16,
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_qwen3_vl_program_author_attention_online_packed_value(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t packed_value,
    id4_pipeline_program_tensor_t attention_mask,
    id4_pipeline_program_tensor_t* out_context) {
  const uint32_t token_count = options->request.token_count;
  const uint32_t hidden_size = options->model.hidden_size;
  const uint32_t attention_head_count = options->model.attention_head_count;
  uint32_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, &token_capacity));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_CONTEXT, layer_ordinal,
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_capacity, hidden_size),
      out_context));
  id4_qwen3_vl_program_bf16_coverage_t context_coverage;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_bf16_coverage(
      "attention context", token_count, hidden_size, token_capacity,
      hidden_size, &context_coverage));

  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
      {ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT, token_capacity},
      {ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE, hidden_size},
      {ID4_QWEN3_VL_CONFIG_ATTENTION_HEAD_COUNT, attention_head_count},
      {ID4_QWEN3_VL_CONFIG_KEY_VALUE_HEAD_COUNT,
       options->model.key_value_head_count},
      {ID4_QWEN3_VL_CONFIG_HEAD_SIZE, options->model.head_size},
  };
  const id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(query),
      id4_pipeline_program_read(key),
      id4_pipeline_program_read(packed_value),
      id4_pipeline_program_read(attention_mask),
      id4_qwen3_vl_program_bf16_logical_write(*out_context, &context_coverage),
  };
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch_attention(
      builder, layer_ordinal, ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION,
      ID4_QWEN3_VL_KERNEL_ATTENTION_ONLINE_BF16_WMMA,
      IREE_ARRAYSIZE(config_values), config_values, IREE_ARRAYSIZE(bindings),
      bindings));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_zero_bf16_tail(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_CONTEXT, layer_ordinal,
      *out_context, &context_coverage));
  return id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_CONTEXT, layer_ordinal,
      *out_context);
}

static iree_status_t id4_qwen3_vl_program_author_attention_wmma(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t value,
    id4_pipeline_program_tensor_t attention_mask,
    id4_pipeline_program_tensor_t* out_context) {
  const uint32_t key_value_width =
      id4_qwen3_vl_program_key_value_width(&options->model);
  uint32_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, &token_capacity));
  id4_pipeline_program_tensor_t packed_value =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_V_PACKED, layer_ordinal,
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(key_value_width, token_capacity),
      &packed_value));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch_value_pack(
      options, builder, layer_ordinal, value, packed_value));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal, ID4_QWEN3_VL_OPERATION_SITE_AFTER_VALUE_PACK));
  return id4_qwen3_vl_program_author_attention_online_packed_value(
      options, builder, layer_ordinal, query, key, packed_value, attention_mask,
      out_context);
}

static iree_status_t id4_qwen3_vl_program_attention_uses_materialized(
    const id4_qwen3_vl_program_options_t* options, uint32_t layer_ordinal,
    bool* out_uses_materialized_attention) {
  *out_uses_materialized_attention = false;
  bool has_scores_tap = false;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_has_diagnostic_tap(
      options, ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_SCORES, layer_ordinal,
      &has_scores_tap));
  bool has_probabilities_tap = false;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_has_diagnostic_tap(
      options, ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_PROBABILITIES, layer_ordinal,
      &has_probabilities_tap));
  switch (options->attention_implementation) {
    case ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_AUTO:
      *out_uses_materialized_attention =
          has_scores_tap || has_probabilities_tap;
      return iree_ok_status();
    case ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_MATERIALIZED:
      *out_uses_materialized_attention = true;
      return iree_ok_status();
    case ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_WMMA:
      if (has_scores_tap || has_probabilities_tap) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Qwen3-VL WMMA attention cannot expose materialized attention "
            "score or probability taps");
      }
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen3-VL attention implementation %" PRIu32
                              " is invalid",
                              (uint32_t)options->attention_implementation);
  }
}

static iree_status_t
id4_qwen3_vl_program_mlp_requires_materialized_projection_taps(
    const id4_qwen3_vl_program_options_t* options, uint32_t layer_ordinal,
    bool* out_requires_materialized_projection_taps) {
  bool has_gate_tap = false;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_has_diagnostic_tap(
      options, ID4_QWEN3_VL_TENSOR_LAYER_MLP_GATE, layer_ordinal,
      &has_gate_tap));
  bool has_up_tap = false;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_has_diagnostic_tap(
      options, ID4_QWEN3_VL_TENSOR_LAYER_MLP_UP, layer_ordinal, &has_up_tap));
  *out_requires_materialized_projection_taps = has_gate_tap || has_up_tap;
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_program_author_attention(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t value,
    id4_pipeline_program_tensor_t attention_mask,
    id4_pipeline_program_tensor_t* out_context) {
  bool uses_materialized_attention = false;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_attention_uses_materialized(
      options, layer_ordinal, &uses_materialized_attention));
  if (uses_materialized_attention) {
    return id4_qwen3_vl_program_author_attention_materialized(
        options, builder, layer_ordinal, query, key, value, attention_mask,
        out_context);
  }
  return id4_qwen3_vl_program_author_attention_wmma(
      options, builder, layer_ordinal, query, key, value, attention_mask,
      out_context);
}

static iree_status_t id4_qwen3_vl_program_author_mlp_gate_up_silu_product(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t* out_activation) {
  const uint32_t hidden_size = options->model.hidden_size;
  const uint32_t intermediate_size = options->model.intermediate_size;
  uint32_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, &token_capacity));
  if (id4_qwen3_vl_program_uses_resident_fp8_matrix_family(options)) {
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
        builder, ID4_QWEN3_VL_TENSOR_LAYER_MLP_ACTIVATION, layer_ordinal,
        ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        id4_pipeline_program_make_shape_rank2(token_capacity,
                                              intermediate_size),
        out_activation));
    char operation_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t operation_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_operation_name(
        ID4_QWEN3_VL_OPERATION_SILU_GATE, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_MLP, operation_name_buffer,
        IREE_ARRAYSIZE(operation_name_buffer), &operation_name));
    id4_qwen3_vl_program_matrix_parameter_t gate_parameter;
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_matrix_parameter(
        options, layer_ordinal, ID4_QWEN3_VL_PARAMETER_LAYER_GATE_PROJECTION,
        hidden_size, intermediate_size, &gate_parameter));
    id4_qwen3_vl_program_matrix_parameter_t up_parameter;
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_matrix_parameter(
        options, layer_ordinal, ID4_QWEN3_VL_PARAMETER_LAYER_UP_PROJECTION,
        hidden_size, intermediate_size, &up_parameter));
    const id4_pipeline_program_swiglu_options_t swiglu_options = {
        .structure_size = sizeof(swiglu_options),
        .name = operation_name,
        .projection = id4_qwen3_vl_program_make_matrix_problem(
            options, token_capacity, hidden_size, intermediate_size,
            ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE),
        .operands =
            {
                .input = input,
                .gate_parameter = gate_parameter.value,
                .up_parameter = up_parameter.value,
                .output = *out_activation,
            },
    };
    return id4_pipeline_program_swiglu(builder, &swiglu_options);
  }

  id4_qwen3_vl_program_linear_kernel_selection_t selection;
  const id4_qwen3_vl_program_mlp_gate_up_silu_product_wmma_tile_t* wmma_tile =
      id4_qwen3_vl_program_select_mlp_gate_up_silu_product_wmma_tile(
          options->parameter_format, options->weight_execution_strategy,
          token_capacity, hidden_size, intermediate_size);
  if ((token_capacity % wmma_tile->token_block) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL MLP token capacity %" PRIu32
                            " must be a multiple of tile size %" PRIu32,
                            token_capacity, wmma_tile->token_block);
  }
  if ((options->weight_execution_strategy ==
           ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_COMPACT_RHS ||
       options->weight_execution_strategy ==
           ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_STREAMING_COMPACT_RHS) &&
      token_capacity < wmma_tile->compact_rhs_min_token_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL compact-RHS fused MLP weight execution requires token "
        "capacity at least %" PRIu32,
        wmma_tile->compact_rhs_min_token_count);
  }
  id4_qwen3_vl_weight_execution_strategy_t strategy =
      options->weight_execution_strategy;
  if (strategy == ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS &&
      token_capacity < wmma_tile->compact_rhs_min_token_count) {
    strategy = ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR;
  }
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_program_select_mlp_gate_up_silu_product_kernel(
          options->parameter_format, strategy, wmma_tile, token_capacity,
          hidden_size, intermediate_size, &selection));

  id4_qwen3_vl_program_linear_weight_t gate_weight;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_parameter_linear_weight(
      builder, options->parameter_scope, options->parameter_format,
      ID4_QWEN3_VL_PARAMETER_LAYER_GATE_PROJECTION, layer_ordinal, hidden_size,
      intermediate_size, selection.weight_encoding, &gate_weight));
  id4_qwen3_vl_program_linear_weight_t up_weight;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_parameter_linear_weight(
      builder, options->parameter_scope, options->parameter_format,
      ID4_QWEN3_VL_PARAMETER_LAYER_UP_PROJECTION, layer_ordinal, hidden_size,
      intermediate_size, selection.weight_encoding, &up_weight));
  if (gate_weight.requires_ready_barrier || up_weight.requires_ready_barrier) {
    char operation_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t operation_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_operation_name(
        ID4_QWEN3_VL_OPERATION_SILU_GATE, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_MLP, operation_name_buffer,
        IREE_ARRAYSIZE(operation_name_buffer), &operation_name));
    char ready_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t ready_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_child_name(
        operation_name, IREE_SV("weights_ready"), ready_name_buffer,
        IREE_ARRAYSIZE(ready_name_buffer), &ready_name));
    IREE_RETURN_IF_ERROR(
        id4_qwen3_vl_program_barrier_named(builder, ready_name));
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_MLP_ACTIVATION, layer_ordinal,
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_capacity, intermediate_size),
      out_activation));

  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT, token_capacity},
      {ID4_QWEN3_VL_CONFIG_INPUT_SIZE, hidden_size},
      {ID4_QWEN3_VL_CONFIG_INTERMEDIATE_SIZE, intermediate_size},
  };
  char value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      selection.kernel_kind, IREE_ARRAYSIZE(config_values), config_values,
      value_buffers, config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[6];
  iree_host_size_t binding_count = 0;
  bindings[binding_count++] = id4_pipeline_program_read(input);
  bindings[binding_count++] = id4_pipeline_program_read(gate_weight.weight);
  if (gate_weight.has_scale) {
    bindings[binding_count++] = id4_pipeline_program_read(gate_weight.scale);
  }
  bindings[binding_count++] = id4_pipeline_program_read(up_weight.weight);
  if (up_weight.has_scale) {
    bindings[binding_count++] = id4_pipeline_program_read(up_weight.scale);
  }
  bindings[binding_count++] = id4_pipeline_program_write(*out_activation);
  return id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_SILU_GATE, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_MLP, selection.kernel_kind,
      IREE_ARRAYSIZE(config_values), config_bindings, binding_count, bindings);
}

static iree_status_t id4_qwen3_vl_program_author_silu_gate(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_pipeline_program_tensor_t gate, id4_pipeline_program_tensor_t up,
    id4_pipeline_program_tensor_t* out_activation) {
  const uint32_t intermediate_size = options->model.intermediate_size;
  uint32_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, &token_capacity));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_MLP_ACTIVATION, layer_ordinal,
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_capacity, intermediate_size),
      out_activation));
  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_capacity},
      {ID4_QWEN3_VL_CONFIG_INTERMEDIATE_SIZE, intermediate_size},
  };
  char value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      ID4_QWEN3_VL_KERNEL_SILU_GATE, IREE_ARRAYSIZE(config_values),
      config_values, value_buffers, config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(gate),
      id4_pipeline_program_read(up),
      id4_pipeline_program_write(*out_activation),
  };
  return id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_SILU_GATE, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_MLP, ID4_QWEN3_VL_KERNEL_SILU_GATE,
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static bool id4_qwen3_vl_program_can_author_mlp_down_residual(
    const id4_qwen3_vl_program_options_t* options, uint32_t token_capacity) {
  const id4_qwen3_vl_program_linear_wmma_tile_t* wmma_tile =
      id4_qwen3_vl_program_select_mlp_down_residual_wmma_tile(options,
                                                              token_capacity);
  if (!wmma_tile) return false;
  switch (options->parameter_format) {
    case ID4_QWEN3_VL_PARAMETER_FORMAT_BF16:
      switch (options->weight_execution_strategy) {
        case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR:
          return id4_qwen3_vl_program_kernel_kind_is_valid(
              wmma_tile->kernels.direct_output_kernel_kind);
        case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_COMPACT_RHS:
        case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_STREAMING_COMPACT_RHS:
          return id4_qwen3_vl_program_kernel_kind_is_valid(
              wmma_tile->kernels.compact_rhs_output_kernel_kind);
        case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS:
          return id4_qwen3_vl_program_kernel_kind_is_valid(
                     wmma_tile->kernels.compact_rhs_output_kernel_kind) ||
                 id4_qwen3_vl_program_kernel_kind_is_valid(
                     wmma_tile->kernels.direct_output_kernel_kind);
        default:
          return false;
      }
    case ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED:
      if (options->weight_execution_strategy ==
          ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR) {
        return false;
      }
      if (options->weight_execution_strategy ==
          ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_STREAMING_COMPACT_RHS) {
        return id4_qwen3_vl_program_kernel_kind_is_valid(
                   wmma_tile->kernels.compact_rhs_output_kernel_kind) &&
               (token_capacity % wmma_tile->token_block) == 0 &&
               (options->model.intermediate_size %
                ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE) == 0 &&
               (options->model.hidden_size %
                ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE) == 0;
      }
      const bool fp8_tile_is_supported =
          wmma_tile ==
              &id4_qwen3_vl_program_linear_residual_wmma_m128n64_2wave_tile ||
          wmma_tile == &id4_qwen3_vl_program_linear_residual_wmma_m64n64_tile;
      return fp8_tile_is_supported &&
             (token_capacity % wmma_tile->token_block) == 0 &&
             (options->model.intermediate_size %
              ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE) == 0 &&
             (options->model.hidden_size % ID4_QWEN3_VL_FP8_SCALE_BLOCK_SIZE) ==
                 0;
    default:
      return false;
  }
}

static iree_status_t id4_qwen3_vl_program_author_mlp_down_residual(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_pipeline_program_tensor_t activation,
    id4_pipeline_program_tensor_t residual,
    id4_pipeline_program_tensor_t* out_output) {
  const uint32_t hidden_size = options->model.hidden_size;
  const uint32_t intermediate_size = options->model.intermediate_size;
  uint32_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, &token_capacity));
  if (id4_qwen3_vl_program_uses_resident_fp8_matrix_family(options)) {
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
        builder, ID4_QWEN3_VL_TENSOR_LAYER_OUTPUT, layer_ordinal,
        ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        id4_pipeline_program_make_shape_rank2(token_capacity, hidden_size),
        out_output));
    char operation_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t operation_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_operation_name(
        ID4_QWEN3_VL_OPERATION_LINEAR, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_MLP_DOWN_PROJECTION, operation_name_buffer,
        IREE_ARRAYSIZE(operation_name_buffer), &operation_name));
    id4_qwen3_vl_program_matrix_parameter_t parameter;
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_matrix_parameter(
        options, layer_ordinal, ID4_QWEN3_VL_PARAMETER_LAYER_DOWN_PROJECTION,
        intermediate_size, hidden_size, &parameter));
    const id4_pipeline_program_matrix_options_t matrix_options = {
        .structure_size = sizeof(matrix_options),
        .name = operation_name,
        .problem = id4_qwen3_vl_program_make_matrix_problem(
            options, token_capacity, intermediate_size, hidden_size,
            ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_ADD),
        .operands =
            {
                .input = activation,
                .parameter = parameter.value,
                .addend = residual,
                .output = *out_output,
            },
    };
    return id4_pipeline_program_matrix(builder, &matrix_options);
  }

  id4_qwen3_vl_program_linear_kernel_selection_t selection;
  const id4_qwen3_vl_program_linear_wmma_tile_t* wmma_tile =
      id4_qwen3_vl_program_select_mlp_down_residual_wmma_tile(options,
                                                              token_capacity);
  if (!wmma_tile) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL fused MLP down residual requires a supported WMMA tile");
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_select_mlp_down_residual_kernel(
      options->parameter_format, options->weight_execution_strategy, wmma_tile,
      token_capacity, intermediate_size, hidden_size, &selection));
  id4_qwen3_vl_program_linear_weight_t weight;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_parameter_linear_weight(
      builder, options->parameter_scope, options->parameter_format,
      ID4_QWEN3_VL_PARAMETER_LAYER_DOWN_PROJECTION, layer_ordinal,
      intermediate_size, hidden_size, selection.weight_encoding, &weight));
  if (weight.requires_ready_barrier) {
    char operation_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t operation_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_operation_name(
        ID4_QWEN3_VL_OPERATION_LINEAR, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_MLP_DOWN_PROJECTION, operation_name_buffer,
        IREE_ARRAYSIZE(operation_name_buffer), &operation_name));
    char ready_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t ready_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_child_name(
        operation_name, IREE_SV("weight_ready"), ready_name_buffer,
        IREE_ARRAYSIZE(ready_name_buffer), &ready_name));
    IREE_RETURN_IF_ERROR(
        id4_qwen3_vl_program_barrier_named(builder, ready_name));
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_OUTPUT, layer_ordinal,
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_capacity, hidden_size),
      out_output));

  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_capacity},
      {ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT, token_capacity},
      {ID4_QWEN3_VL_CONFIG_INPUT_SIZE, intermediate_size},
      {ID4_QWEN3_VL_CONFIG_OUTPUT_SIZE, hidden_size},
  };
  char value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      selection.kernel_kind, IREE_ARRAYSIZE(config_values), config_values,
      value_buffers, config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[5];
  iree_host_size_t binding_count = 0;
  bindings[binding_count++] = id4_pipeline_program_read(activation);
  bindings[binding_count++] = id4_pipeline_program_read(weight.weight);
  if (weight.has_scale) {
    bindings[binding_count++] = id4_pipeline_program_read(weight.scale);
  }
  bindings[binding_count++] = id4_pipeline_program_read(residual);
  bindings[binding_count++] = id4_pipeline_program_write(*out_output);
  return id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_LINEAR, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_MLP_DOWN_PROJECTION, selection.kernel_kind,
      IREE_ARRAYSIZE(config_values), config_bindings, binding_count, bindings);
}

static iree_status_t id4_qwen3_vl_program_author_residual_add(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site, id4_pipeline_program_tensor_t lhs,
    id4_pipeline_program_tensor_t rhs, id4_qwen3_vl_tensor_kind_t output_kind,
    id4_pipeline_program_tensor_t* out_output) {
  const uint32_t hidden_size = options->model.hidden_size;
  uint32_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, &token_capacity));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, output_kind, layer_ordinal, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_capacity, hidden_size),
      out_output));
  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_capacity},
      {ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE, hidden_size},
  };
  char value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      ID4_QWEN3_VL_KERNEL_RESIDUAL_ADD, IREE_ARRAYSIZE(config_values),
      config_values, value_buffers, config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(lhs),
      id4_pipeline_program_read(rhs),
      id4_pipeline_program_write(*out_output),
  };
  return id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_RESIDUAL_ADD, layer_ordinal, site,
      ID4_QWEN3_VL_KERNEL_RESIDUAL_ADD, IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

static iree_status_t id4_qwen3_vl_program_author_residual_add_rmsnorm(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site,
    id4_qwen3_vl_parameter_kind_t parameter_kind,
    id4_qwen3_vl_tensor_kind_t residual_output_kind,
    id4_qwen3_vl_tensor_kind_t output_kind, id4_pipeline_program_tensor_t lhs,
    id4_pipeline_program_tensor_t rhs,
    id4_pipeline_program_tensor_t* out_residual_output,
    id4_pipeline_program_tensor_t* out_output) {
  const uint32_t hidden_size = options->model.hidden_size;
  uint32_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, &token_capacity));
  id4_pipeline_program_tensor_t weight = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_parameter(
      builder, options->parameter_scope, parameter_kind, layer_ordinal,
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(hidden_size), &weight));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, residual_output_kind, layer_ordinal,
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_capacity, hidden_size),
      out_residual_output));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, output_kind, layer_ordinal, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_capacity, hidden_size),
      out_output));
  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_capacity},
      {ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE, hidden_size},
  };
  char value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      ID4_QWEN3_VL_KERNEL_RESIDUAL_ADD_RMSNORM, IREE_ARRAYSIZE(config_values),
      config_values, value_buffers, config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(lhs),
      id4_pipeline_program_read(rhs),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_write(*out_residual_output),
      id4_pipeline_program_write(*out_output),
  };
  return id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_RMSNORM, layer_ordinal, site,
      ID4_QWEN3_VL_KERNEL_RESIDUAL_ADD_RMSNORM, IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

static iree_status_t id4_qwen3_vl_program_author_selected_hidden_pack(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    uint32_t selected_layer_index, id4_pipeline_program_tensor_t hidden_states,
    id4_pipeline_program_tensor_t selected_hidden_states) {
  const uint32_t token_count = options->request.token_count;
  const uint32_t hidden_size = options->model.hidden_size;
  const uint32_t selected_hidden_row_count =
      id4_qwen3_vl_program_selected_hidden_row_count(&options->model);
  uint32_t selected_layer_element_count = 0;
  if (!id4_qwen3_vl_program_checked_mul_u32(hidden_size, token_count,
                                            &selected_layer_element_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Qwen3-VL selected hidden pack element count overflow");
  }
  iree_device_size_t selected_layer_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_bf16_byte_length(
      selected_layer_element_count, &selected_layer_byte_length));
  iree_device_size_t selected_layer_byte_offset = 0;
  if (!iree_device_size_checked_mul(selected_layer_index,
                                    selected_layer_byte_length,
                                    &selected_layer_byte_offset)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Qwen3-VL selected hidden pack byte offset overflow");
  }
  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
      {ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE, hidden_size},
      {ID4_QWEN3_VL_CONFIG_HIDDEN_ROW_COUNT, selected_hidden_row_count},
      {ID4_QWEN3_VL_CONFIG_SELECTED_LAYER_INDEX, selected_layer_index},
  };
  char value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      ID4_QWEN3_VL_KERNEL_SELECTED_HIDDEN_PACK, IREE_ARRAYSIZE(config_values),
      config_values, value_buffers, config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(hidden_states),
      id4_pipeline_program_write_range(selected_hidden_states,
                                       selected_layer_byte_offset,
                                       selected_layer_byte_length),
  };
  return id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_SELECTED_HIDDEN_PACK, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_SELECTED_HIDDEN_PACK,
      ID4_QWEN3_VL_KERNEL_SELECTED_HIDDEN_PACK, IREE_ARRAYSIZE(config_values),
      config_bindings, IREE_ARRAYSIZE(bindings), bindings);
}

static iree_status_t id4_qwen3_vl_program_author_condition(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder,
    id4_pipeline_program_tensor_t selected_hidden_states,
    id4_pipeline_program_tensor_t token_weights,
    id4_pipeline_program_tensor_t* out_condition) {
  const uint32_t token_count = options->request.token_count;
  const uint32_t hidden_row_count =
      id4_qwen3_vl_program_selected_hidden_row_count(&options->model);
  uint32_t stats_row_count = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_condition_partial_count(
      hidden_row_count, token_count, &stats_row_count));
  id4_pipeline_program_tensor_t stats = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, ID4_QWEN3_VL_TENSOR_CONDITION_STATS, UINT32_MAX,
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(stats_row_count, 2), &stats));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_import_tensor(
      builder, ID4_QWEN3_VL_TENSOR_OUTPUT, /*flags=*/0,
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(hidden_row_count, token_count),
      out_condition));

  const id4_qwen3_vl_program_config_value_t condition_config_values[] = {
      {ID4_QWEN3_VL_CONFIG_HIDDEN_ROW_COUNT, hidden_row_count},
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
  };
  char apply_value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                          [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      apply_config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      ID4_QWEN3_VL_KERNEL_CONDITION_APPLY_TOKEN_WEIGHTS,
      IREE_ARRAYSIZE(condition_config_values), condition_config_values,
      apply_value_buffers, apply_config_bindings));
  id4_pipeline_program_dispatch_binding_t apply_bindings[] = {
      id4_pipeline_program_read(selected_hidden_states),
      id4_pipeline_program_read(token_weights),
      id4_pipeline_program_write(*out_condition),
      id4_pipeline_program_write(stats),
  };
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_CONDITION_APPLY_TOKEN_WEIGHTS, UINT32_MAX,
      ID4_QWEN3_VL_OPERATION_SITE_CONDITION_APPLY_TOKEN_WEIGHTS,
      ID4_QWEN3_VL_KERNEL_CONDITION_APPLY_TOKEN_WEIGHTS,
      IREE_ARRAYSIZE(condition_config_values), apply_config_bindings,
      IREE_ARRAYSIZE(apply_bindings), apply_bindings));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, UINT32_MAX,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_CONDITION_APPLY_TOKEN_WEIGHTS));

  char reduce_value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                           [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      reduce_config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      ID4_QWEN3_VL_KERNEL_CONDITION_REDUCE_TOKEN_WEIGHT_STATS,
      IREE_ARRAYSIZE(condition_config_values), condition_config_values,
      reduce_value_buffers, reduce_config_bindings));
  id4_pipeline_program_dispatch_binding_t reduce_bindings[] = {
      id4_pipeline_program_read_write(stats),
  };
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_CONDITION_REDUCE_TOKEN_WEIGHT_STATS,
      UINT32_MAX,
      ID4_QWEN3_VL_OPERATION_SITE_CONDITION_REDUCE_TOKEN_WEIGHT_STATS,
      ID4_QWEN3_VL_KERNEL_CONDITION_REDUCE_TOKEN_WEIGHT_STATS,
      IREE_ARRAYSIZE(condition_config_values), reduce_config_bindings,
      IREE_ARRAYSIZE(reduce_bindings), reduce_bindings));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, UINT32_MAX,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_CONDITION_REDUCE_TOKEN_WEIGHT_STATS));

  char normalize_value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                              [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      normalize_config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      ID4_QWEN3_VL_KERNEL_CONDITION_NORMALIZE_TOKEN_WEIGHTS,
      IREE_ARRAYSIZE(condition_config_values), condition_config_values,
      normalize_value_buffers, normalize_config_bindings));
  id4_pipeline_program_dispatch_binding_t normalize_bindings[] = {
      id4_pipeline_program_read_write(*out_condition),
      id4_pipeline_program_read(stats),
  };
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_CONDITION_NORMALIZE_TOKEN_WEIGHTS,
      UINT32_MAX, ID4_QWEN3_VL_OPERATION_SITE_CONDITION_NORMALIZE_TOKEN_WEIGHTS,
      ID4_QWEN3_VL_KERNEL_CONDITION_NORMALIZE_TOKEN_WEIGHTS,
      IREE_ARRAYSIZE(condition_config_values), normalize_config_bindings,
      IREE_ARRAYSIZE(normalize_bindings), normalize_bindings));
  return id4_qwen3_vl_program_barrier(
      builder, UINT32_MAX, ID4_QWEN3_VL_OPERATION_SITE_AFTER_CONDITION);
}

static iree_status_t id4_qwen3_vl_program_author_layer(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t attention_mask,
    id4_pipeline_program_tensor_t* out_output) {
  const uint32_t hidden_size = options->model.hidden_size;
  const uint32_t query_width =
      id4_qwen3_vl_program_query_width(&options->model);
  const uint32_t key_value_width =
      id4_qwen3_vl_program_key_value_width(&options->model);
  uint32_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, &token_capacity));
  uint32_t query_row_count = 0;
  if (!id4_qwen3_vl_program_checked_mul_u32(options->request.token_count,
                                            options->model.attention_head_count,
                                            &query_row_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL query RMSNorm row count overflow");
  }
  uint32_t key_row_count = 0;
  if (!id4_qwen3_vl_program_checked_mul_u32(options->request.token_count,
                                            options->model.key_value_head_count,
                                            &key_row_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL key RMSNorm row count overflow");
  }
  id4_pipeline_program_tensor_t input_norm =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_rmsnorm(
      options, builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_INPUT_LAYERNORM,
      ID4_QWEN3_VL_PARAMETER_LAYER_INPUT_NORM,
      ID4_QWEN3_VL_TENSOR_LAYER_INPUT_NORM, input, &input_norm));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_INPUT_LAYERNORM));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_INPUT_NORM, layer_ordinal,
      input_norm));

  bool uses_materialized_attention = false;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_attention_uses_materialized(
      options, layer_ordinal, &uses_materialized_attention));
  bool has_value_tap = false;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_has_diagnostic_tap(
      options, ID4_QWEN3_VL_TENSOR_LAYER_V, layer_ordinal, &has_value_tap));
  const bool uses_packed_value_projection =
      !uses_materialized_attention && !has_value_tap &&
      id4_qwen3_vl_program_can_author_value_projection_packed(options,
                                                              token_capacity);
  id4_pipeline_program_tensor_t query = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t key = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t value = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t packed_value =
      id4_pipeline_program_tensor_invalid();
  if (id4_qwen3_vl_program_uses_resident_fp8_matrix_family(options)) {
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_linear(
        options, builder, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_Q_PROJECTION,
        ID4_QWEN3_VL_PARAMETER_LAYER_Q_PROJECTION, ID4_QWEN3_VL_TENSOR_LAYER_Q,
        input_norm, hidden_size, query_width, &query));
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_linear(
        options, builder, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_K_PROJECTION,
        ID4_QWEN3_VL_PARAMETER_LAYER_K_PROJECTION, ID4_QWEN3_VL_TENSOR_LAYER_K,
        input_norm, hidden_size, key_value_width, &key));
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_linear(
        options, builder, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_V_PROJECTION,
        ID4_QWEN3_VL_PARAMETER_LAYER_V_PROJECTION, ID4_QWEN3_VL_TENSOR_LAYER_V,
        input_norm, hidden_size, key_value_width, &value));
  } else {
    uint32_t query_token_capacity = 0;
    uint32_t query_wmma_token_count = 0;
    id4_qwen3_vl_program_linear_kernel_selection_t query_selection;
    id4_qwen3_vl_program_linear_weight_t query_weight;
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_prepare_linear_weight(
        options, builder, layer_ordinal,
        ID4_QWEN3_VL_PARAMETER_LAYER_Q_PROJECTION, hidden_size, query_width,
        &query_token_capacity, &query_wmma_token_count, &query_selection,
        &query_weight));
    uint32_t key_token_capacity = 0;
    uint32_t key_wmma_token_count = 0;
    id4_qwen3_vl_program_linear_kernel_selection_t key_selection;
    id4_qwen3_vl_program_linear_weight_t key_weight;
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_prepare_linear_weight(
        options, builder, layer_ordinal,
        ID4_QWEN3_VL_PARAMETER_LAYER_K_PROJECTION, hidden_size, key_value_width,
        &key_token_capacity, &key_wmma_token_count, &key_selection,
        &key_weight));
    bool qkv_weights_require_ready_barrier =
        query_weight.requires_ready_barrier ||
        key_weight.requires_ready_barrier;
    uint32_t value_token_capacity = 0;
    uint32_t value_wmma_token_count = 0;
    id4_qwen3_vl_program_linear_kernel_selection_t value_selection;
    id4_qwen3_vl_program_linear_weight_t value_weight;
    if (uses_packed_value_projection) {
      IREE_RETURN_IF_ERROR(
          id4_qwen3_vl_program_prepare_value_projection_packed_weight(
              options, builder, layer_ordinal, &value_token_capacity,
              &value_selection, &value_weight));
    } else {
      IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_prepare_linear_weight(
          options, builder, layer_ordinal,
          ID4_QWEN3_VL_PARAMETER_LAYER_V_PROJECTION, hidden_size,
          key_value_width, &value_token_capacity, &value_wmma_token_count,
          &value_selection, &value_weight));
    }
    qkv_weights_require_ready_barrier |= value_weight.requires_ready_barrier;
    if (qkv_weights_require_ready_barrier) {
      char operation_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
      iree_string_view_t operation_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_operation_name(
          ID4_QWEN3_VL_OPERATION_LINEAR, layer_ordinal,
          ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_Q_PROJECTION,
          operation_name_buffer, IREE_ARRAYSIZE(operation_name_buffer),
          &operation_name));
      char ready_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
      iree_string_view_t ready_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_child_name(
          operation_name, IREE_SV("qkv_weights_ready"), ready_name_buffer,
          IREE_ARRAYSIZE(ready_name_buffer), &ready_name));
      IREE_RETURN_IF_ERROR(
          id4_qwen3_vl_program_barrier_named(builder, ready_name));
    }

    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_linear_prepared(
        options, builder, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_Q_PROJECTION,
        ID4_QWEN3_VL_TENSOR_LAYER_Q, input_norm, hidden_size, query_width,
        query_token_capacity, query_wmma_token_count, &query_selection,
        &query_weight, /*flags=*/0, &query));
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_linear_prepared(
        options, builder, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_K_PROJECTION,
        ID4_QWEN3_VL_TENSOR_LAYER_K, input_norm, hidden_size, key_value_width,
        key_token_capacity, key_wmma_token_count, &key_selection, &key_weight,
        /*flags=*/0, &key));
    if (uses_packed_value_projection) {
      IREE_RETURN_IF_ERROR(
          id4_qwen3_vl_program_author_value_projection_packed_prepared(
              options, builder, layer_ordinal, input_norm, value_token_capacity,
              &value_selection, &value_weight,
              /*flags=*/0, &packed_value));
    } else {
      IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_linear_prepared(
          options, builder, layer_ordinal,
          ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_V_PROJECTION,
          ID4_QWEN3_VL_TENSOR_LAYER_V, input_norm, hidden_size, key_value_width,
          value_token_capacity, value_wmma_token_count, &value_selection,
          &value_weight, /*flags=*/0, &value));
    }
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_QKV_PROJECTION));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_Q, layer_ordinal, query));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_K, layer_ordinal, key));
  if (!uses_packed_value_projection) {
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
        builder, ID4_QWEN3_VL_TENSOR_LAYER_V, layer_ordinal, value));
  }

  id4_pipeline_program_tensor_t query_rotary =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t key_rotary =
      id4_pipeline_program_tensor_invalid();
  bool has_query_norm_tap = false;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_has_diagnostic_tap(
      options, ID4_QWEN3_VL_TENSOR_LAYER_Q_NORM, layer_ordinal,
      &has_query_norm_tap));
  bool has_key_norm_tap = false;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_has_diagnostic_tap(
      options, ID4_QWEN3_VL_TENSOR_LAYER_K_NORM, layer_ordinal,
      &has_key_norm_tap));
  if (has_query_norm_tap || has_key_norm_tap) {
    id4_pipeline_program_tensor_t query_norm =
        id4_pipeline_program_tensor_invalid();
    if (has_query_norm_tap) {
      IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_rmsnorm_rows(
          options, builder, layer_ordinal,
          ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_Q_NORM,
          ID4_QWEN3_VL_PARAMETER_LAYER_Q_NORM, ID4_QWEN3_VL_TENSOR_LAYER_Q_NORM,
          query, query_row_count, options->model.head_size, query_width,
          &query_norm));
    }
    id4_pipeline_program_tensor_t key_norm =
        id4_pipeline_program_tensor_invalid();
    if (has_key_norm_tap) {
      IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_rmsnorm_rows(
          options, builder, layer_ordinal,
          ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_K_NORM,
          ID4_QWEN3_VL_PARAMETER_LAYER_K_NORM, ID4_QWEN3_VL_TENSOR_LAYER_K_NORM,
          key, key_row_count, options->model.head_size, key_value_width,
          &key_norm));
    }
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
        builder, layer_ordinal, ID4_QWEN3_VL_OPERATION_SITE_AFTER_QK_NORM));
    if (has_query_norm_tap) {
      IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
          builder, ID4_QWEN3_VL_TENSOR_LAYER_Q_NORM, layer_ordinal,
          query_norm));
      IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_rotary(
          options, builder, layer_ordinal,
          ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_Q, query_norm, query_width,
          ID4_QWEN3_VL_TENSOR_LAYER_Q_ROTARY, &query_rotary));
    } else {
      IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_rmsnorm_rotary(
          options, builder, layer_ordinal,
          ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_Q,
          ID4_QWEN3_VL_PARAMETER_LAYER_Q_NORM,
          ID4_QWEN3_VL_TENSOR_LAYER_Q_ROTARY, query,
          options->model.attention_head_count, &query_rotary));
    }
    if (has_key_norm_tap) {
      IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
          builder, ID4_QWEN3_VL_TENSOR_LAYER_K_NORM, layer_ordinal, key_norm));
      IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_rotary(
          options, builder, layer_ordinal,
          ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_K, key_norm,
          key_value_width, ID4_QWEN3_VL_TENSOR_LAYER_K_ROTARY, &key_rotary));
    } else {
      IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_rmsnorm_rotary(
          options, builder, layer_ordinal,
          ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_K,
          ID4_QWEN3_VL_PARAMETER_LAYER_K_NORM,
          ID4_QWEN3_VL_TENSOR_LAYER_K_ROTARY, key,
          options->model.key_value_head_count, &key_rotary));
    }
  } else {
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_rmsnorm_rotary(
        options, builder, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_Q,
        ID4_QWEN3_VL_PARAMETER_LAYER_Q_NORM, ID4_QWEN3_VL_TENSOR_LAYER_Q_ROTARY,
        query, options->model.attention_head_count, &query_rotary));
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_rmsnorm_rotary(
        options, builder, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_K,
        ID4_QWEN3_VL_PARAMETER_LAYER_K_NORM, ID4_QWEN3_VL_TENSOR_LAYER_K_ROTARY,
        key, options->model.key_value_head_count, &key_rotary));
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_ROTARY_EMBEDDING));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_Q_ROTARY, layer_ordinal,
      query_rotary));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_K_ROTARY, layer_ordinal, key_rotary));

  uint32_t attention_output_token_capacity = 0;
  uint32_t attention_output_wmma_token_count = 0;
  id4_qwen3_vl_program_linear_kernel_selection_t attention_output_selection;
  id4_qwen3_vl_program_linear_weight_t attention_output_weight;
  const bool uses_parameter_matrix =
      id4_qwen3_vl_program_uses_resident_fp8_matrix_family(options);
  if (!uses_parameter_matrix) {
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_prepare_linear_weight(
        options, builder, layer_ordinal,
        ID4_QWEN3_VL_PARAMETER_LAYER_O_PROJECTION, hidden_size, hidden_size,
        &attention_output_token_capacity, &attention_output_wmma_token_count,
        &attention_output_selection, &attention_output_weight));
  }

  id4_pipeline_program_tensor_t context = id4_pipeline_program_tensor_invalid();
  if (uses_packed_value_projection) {
    IREE_RETURN_IF_ERROR(
        id4_qwen3_vl_program_author_attention_online_packed_value(
            options, builder, layer_ordinal, query_rotary, key_rotary,
            packed_value, attention_mask, &context));
  } else {
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_attention(
        options, builder, layer_ordinal, query_rotary, key_rotary, value,
        attention_mask, &context));
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal, ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION));

  id4_pipeline_program_tensor_t attention_output =
      id4_pipeline_program_tensor_invalid();
  if (uses_parameter_matrix) {
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_linear(
        options, builder, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_O_PROJECTION,
        ID4_QWEN3_VL_PARAMETER_LAYER_O_PROJECTION,
        ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_OUTPUT, context, hidden_size,
        hidden_size, &attention_output));
  } else {
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_linear_prepared(
        options, builder, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_O_PROJECTION,
        ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_OUTPUT, context, hidden_size,
        hidden_size, attention_output_token_capacity,
        attention_output_wmma_token_count, &attention_output_selection,
        &attention_output_weight, /*flags=*/0, &attention_output));
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION_OUTPUT_PROJECTION));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_OUTPUT, layer_ordinal,
      attention_output));

  id4_pipeline_program_tensor_t after_attention =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t post_attention_norm =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_residual_add_rmsnorm(
      options, builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_POST_ATTENTION_LAYERNORM,
      ID4_QWEN3_VL_PARAMETER_LAYER_POST_ATTENTION_NORM,
      ID4_QWEN3_VL_TENSOR_LAYER_AFTER_ATTENTION,
      ID4_QWEN3_VL_TENSOR_LAYER_POST_ATTENTION_NORM, input, attention_output,
      &after_attention, &post_attention_norm));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_POST_ATTENTION_LAYERNORM));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_AFTER_ATTENTION, layer_ordinal,
      after_attention));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_POST_ATTENTION_NORM, layer_ordinal,
      post_attention_norm));

  id4_pipeline_program_tensor_t activation =
      id4_pipeline_program_tensor_invalid();
  bool requires_materialized_mlp_projections = false;
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_program_mlp_requires_materialized_projection_taps(
          options, layer_ordinal, &requires_materialized_mlp_projections));
  uint32_t mlp_token_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_options_token_capacity(
      options, &mlp_token_capacity));
  const bool uses_fused_mlp =
      !requires_materialized_mlp_projections &&
      (options->parameter_format == ID4_QWEN3_VL_PARAMETER_FORMAT_BF16 ||
       options->parameter_format ==
           ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED);
  if (!uses_fused_mlp) {
    id4_pipeline_program_tensor_t gate = id4_pipeline_program_tensor_invalid();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_linear(
        options, builder, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_MLP_GATE_PROJECTION,
        ID4_QWEN3_VL_PARAMETER_LAYER_GATE_PROJECTION,
        ID4_QWEN3_VL_TENSOR_LAYER_MLP_GATE, post_attention_norm, hidden_size,
        options->model.intermediate_size, &gate));
    id4_pipeline_program_tensor_t up = id4_pipeline_program_tensor_invalid();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_linear(
        options, builder, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_MLP_UP_PROJECTION,
        ID4_QWEN3_VL_PARAMETER_LAYER_UP_PROJECTION,
        ID4_QWEN3_VL_TENSOR_LAYER_MLP_UP, post_attention_norm, hidden_size,
        options->model.intermediate_size, &up));
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
        builder, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_AFTER_MLP_UP_GATE_PROJECTION));
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
        builder, ID4_QWEN3_VL_TENSOR_LAYER_MLP_GATE, layer_ordinal, gate));
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
        builder, ID4_QWEN3_VL_TENSOR_LAYER_MLP_UP, layer_ordinal, up));
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_silu_gate(
        options, builder, layer_ordinal, gate, up, &activation));
  } else {
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_mlp_gate_up_silu_product(
        options, builder, layer_ordinal, post_attention_norm, &activation));
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_MLP_ACTIVATION));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_MLP_ACTIVATION, layer_ordinal,
      activation));

  bool requires_materialized_mlp_down = false;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_has_diagnostic_tap(
      options, ID4_QWEN3_VL_TENSOR_LAYER_MLP_DOWN, layer_ordinal,
      &requires_materialized_mlp_down));
  const bool uses_fused_mlp_down =
      !requires_materialized_mlp_down &&
      id4_qwen3_vl_program_can_author_mlp_down_residual(options,
                                                        mlp_token_capacity);
  if (uses_fused_mlp_down) {
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_mlp_down_residual(
        options, builder, layer_ordinal, activation, after_attention,
        out_output));
  } else {
    id4_pipeline_program_tensor_t mlp_down =
        id4_pipeline_program_tensor_invalid();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_linear(
        options, builder, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_MLP_DOWN_PROJECTION,
        ID4_QWEN3_VL_PARAMETER_LAYER_DOWN_PROJECTION,
        ID4_QWEN3_VL_TENSOR_LAYER_MLP_DOWN, activation,
        options->model.intermediate_size, hidden_size, &mlp_down));
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
        builder, layer_ordinal,
        ID4_QWEN3_VL_OPERATION_SITE_AFTER_MLP_DOWN_PROJECTION));
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
        builder, ID4_QWEN3_VL_TENSOR_LAYER_MLP_DOWN, layer_ordinal, mlp_down));

    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_residual_add(
        options, builder, layer_ordinal, ID4_QWEN3_VL_OPERATION_SITE_MLP,
        after_attention, mlp_down, ID4_QWEN3_VL_TENSOR_LAYER_OUTPUT,
        out_output));
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal, ID4_QWEN3_VL_OPERATION_SITE_AFTER_LAYER_OUTPUT));
  return id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_OUTPUT, layer_ordinal, *out_output);
}

iree_status_t id4_qwen3_vl_program_author_forward(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder) {
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_validate_options(options, builder));

  id4_pipeline_program_tensor_t hidden_states =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t attention_mask =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t token_weights =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_token_embedding(
      options, builder, &hidden_states, &attention_mask, &token_weights));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_EMBEDDED_HIDDEN_STATES, UINT32_MAX,
      hidden_states));

  id4_pipeline_program_tensor_t selected_hidden_states =
      id4_pipeline_program_tensor_invalid();

  uint32_t selected_layer_index = 0;
  for (uint32_t i = 0; i < options->model.layer_count; ++i) {
    id4_pipeline_program_tensor_t layer_output =
        id4_pipeline_program_tensor_invalid();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_layer(
        options, builder, i, hidden_states, attention_mask, &layer_output));
    hidden_states = layer_output;
    if (selected_layer_index < options->model.selected_layer_count &&
        options->model.selected_layer_ordinals[selected_layer_index] == i) {
      if (!id4_pipeline_program_tensor_is_valid(selected_hidden_states)) {
        const uint32_t selected_hidden_row_count =
            id4_qwen3_vl_program_selected_hidden_row_count(&options->model);
        IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
            builder, ID4_QWEN3_VL_TENSOR_SELECTED_HIDDEN_STATES, UINT32_MAX,
            ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            id4_pipeline_program_make_shape_rank2(selected_hidden_row_count,
                                                  options->request.token_count),
            &selected_hidden_states));
      }
      IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_selected_hidden_pack(
          options, builder, i, selected_layer_index, hidden_states,
          selected_hidden_states));
      ++selected_layer_index;
    }
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, UINT32_MAX,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_SELECTED_HIDDEN_PACK));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_SELECTED_HIDDEN_STATES, UINT32_MAX,
      selected_hidden_states));

  id4_pipeline_program_tensor_t condition =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_condition(
      options, builder, selected_hidden_states, token_weights, &condition));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap(
      builder, UINT32_MAX, ID4_QWEN3_VL_OPERATION_SITE_OUTPUT, condition));
  char export_name_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t export_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_tensor_name(
      ID4_QWEN3_VL_TENSOR_OUTPUT, UINT32_MAX, export_name_buffer,
      IREE_ARRAYSIZE(export_name_buffer), &export_name));
  id4_pipeline_program_export_options_t export_options = {
      .structure_size = sizeof(export_options),
      .name = export_name,
      .tensor = condition,
  };
  return id4_pipeline_program_export(builder, &export_options);
}

const id4_qwen3_vl_model_config_t* id4_qwen3_vl_program_ideogram4_model_config(
    void) {
  return &id4_qwen3_vl_program_ideogram4_model_config_value;
}
