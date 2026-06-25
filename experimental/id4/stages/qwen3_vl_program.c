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
#include "iree/hal/command_buffer.h"

#define ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY 192

enum {
  ID4_QWEN3_VL_WORKGROUP_SIZE_X = 256,
  ID4_QWEN3_VL_WMMA_WORKGROUP_SIZE_X = 32,
  ID4_QWEN3_VL_LINEAR_SCALAR_INPUT_BLOCK = 4,
  ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK = 32,
  ID4_QWEN3_VL_LINEAR_WMMA_INPUT_BLOCK = 16,
  ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK = 64,
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
        // Number of post-layer hidden states used by Ideogram 4 conditioning.
        .selected_layer_count =
            IREE_ARRAYSIZE(id4_qwen3_vl_program_ideogram4_selected_layers),
        // Zero-based selected layer ordinals.
        .selected_layer_ordinals =
            id4_qwen3_vl_program_ideogram4_selected_layers,
};

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

static uint32_t id4_qwen3_vl_program_ceil_div_u32(uint32_t dividend,
                                                  uint32_t divisor) {
  return dividend / divisor + (dividend % divisor != 0 ? 1 : 0);
}

static iree_hal_dispatch_config_t id4_qwen3_vl_program_make_dispatch_config(
    uint32_t workgroup_count_x, uint32_t workgroup_count_y,
    uint32_t workgroup_count_z) {
  iree_hal_dispatch_config_t dispatch_config =
      iree_hal_make_static_dispatch_config(workgroup_count_x, workgroup_count_y,
                                           workgroup_count_z);
  dispatch_config.workgroup_size[0] = ID4_QWEN3_VL_WORKGROUP_SIZE_X;
  dispatch_config.workgroup_size[1] = 1;
  dispatch_config.workgroup_size[2] = 1;
  return dispatch_config;
}

static iree_hal_dispatch_config_t
id4_qwen3_vl_program_make_dispatch_config_with_workgroup_size(
    uint32_t workgroup_count_x, uint32_t workgroup_count_y,
    uint32_t workgroup_count_z, uint32_t workgroup_size_x) {
  iree_hal_dispatch_config_t dispatch_config =
      iree_hal_make_static_dispatch_config(workgroup_count_x, workgroup_count_y,
                                           workgroup_count_z);
  dispatch_config.workgroup_size[0] = workgroup_size_x;
  dispatch_config.workgroup_size[1] = 1;
  dispatch_config.workgroup_size[2] = 1;
  return dispatch_config;
}

static iree_hal_dispatch_config_t
id4_qwen3_vl_program_make_linear_dispatch_config(uint32_t token_count,
                                                 uint32_t output_size) {
  return id4_qwen3_vl_program_make_dispatch_config(token_count, output_size, 1);
}

static iree_hal_dispatch_config_t id4_qwen3_vl_program_make_row_dispatch_config(
    uint32_t row_count) {
  return id4_qwen3_vl_program_make_dispatch_config(row_count, 1, 1);
}

static iree_hal_dispatch_config_t
id4_qwen3_vl_program_make_reduction_dispatch_config(void) {
  return id4_qwen3_vl_program_make_dispatch_config(1, 1, 1);
}

static iree_status_t id4_qwen3_vl_program_make_element_dispatch_config(
    uint32_t element_count, iree_hal_dispatch_config_t* out_dispatch_config) {
  if (element_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL dispatch element count must be "
                            "nonzero");
  }
  const uint32_t workgroup_count_x = id4_qwen3_vl_program_ceil_div_u32(
      element_count, ID4_QWEN3_VL_WORKGROUP_SIZE_X);
  *out_dispatch_config =
      id4_qwen3_vl_program_make_dispatch_config(workgroup_count_x, 1, 1);
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_program_make_matrix_element_dispatch_config(
    uint32_t row_count, uint32_t column_count,
    iree_hal_dispatch_config_t* out_dispatch_config) {
  uint32_t element_count = 0;
  if (!id4_qwen3_vl_program_checked_mul_u32(row_count, column_count,
                                            &element_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL dispatch element count overflow");
  }
  return id4_qwen3_vl_program_make_element_dispatch_config(element_count,
                                                           out_dispatch_config);
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
  if (options->request.token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL token count must be nonzero");
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
  // Imported token id vector.
  ID4_QWEN3_VL_TENSOR_TOKEN_IDS = 0,
  // Imported additive attention mask matrix.
  ID4_QWEN3_VL_TENSOR_ATTENTION_MASK = 1,
  // Imported token weight vector.
  ID4_QWEN3_VL_TENSOR_TOKEN_WEIGHTS = 2,
  // Hidden states produced by token embedding.
  ID4_QWEN3_VL_TENSOR_EMBEDDED_HIDDEN_STATES = 3,
  // Per-layer input RMSNorm output.
  ID4_QWEN3_VL_TENSOR_LAYER_INPUT_NORM = 4,
  // Per-layer query projection output.
  ID4_QWEN3_VL_TENSOR_LAYER_Q = 5,
  // Per-layer key projection output.
  ID4_QWEN3_VL_TENSOR_LAYER_K = 6,
  // Per-layer value projection output.
  ID4_QWEN3_VL_TENSOR_LAYER_V = 7,
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
  // Two-scalar condition normalization statistics buffer.
  ID4_QWEN3_VL_TENSOR_CONDITION_STATS = 25,
  // Final decoder RMSNorm output.
  ID4_QWEN3_VL_TENSOR_FINAL_NORM = 26,
  // Exported condition tensor.
  ID4_QWEN3_VL_TENSOR_OUTPUT = 27,
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
  // Condition token-weight normalization operation.
  ID4_QWEN3_VL_OPERATION_CONDITION_NORMALIZE_TOKEN_WEIGHTS = 9,
  // Execution barrier operation.
  ID4_QWEN3_VL_OPERATION_BARRIER = 10,
  // Diagnostic tap operation.
  ID4_QWEN3_VL_OPERATION_TAP = 11,
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
  // Global condition token-weight normalization site.
  ID4_QWEN3_VL_OPERATION_SITE_CONDITION_NORMALIZE_TOKEN_WEIGHTS = 37,
  // Barrier after the exported condition tensor is complete.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_CONDITION = 38,
  // Final decoder RMSNorm site.
  ID4_QWEN3_VL_OPERATION_SITE_FINAL_NORM = 39,
  // Barrier after final decoder RMSNorm.
  ID4_QWEN3_VL_OPERATION_SITE_AFTER_FINAL_NORM = 40,
  // Forward output diagnostic site.
  ID4_QWEN3_VL_OPERATION_SITE_OUTPUT = 41,
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
  // Condition token-weight normalization kernel.
  ID4_QWEN3_VL_KERNEL_CONDITION_NORMALIZE_TOKEN_WEIGHTS = 11,
  // Dense BF16 activation/weight WMMA linear kernel.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA = 12,
  // Dense BF16 activation/weight scalar linear tail kernel.
  ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_TAIL = 13,
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
  // Per-head channel count config key.
  ID4_QWEN3_VL_CONFIG_HEAD_SIZE = 7,
  // Vocabulary row count config key.
  ID4_QWEN3_VL_CONFIG_VOCAB_SIZE = 8,
  // Selected hidden-state row count config key.
  ID4_QWEN3_VL_CONFIG_HIDDEN_ROW_COUNT = 9,
  // Selected-layer output slot config key.
  ID4_QWEN3_VL_CONFIG_SELECTED_LAYER_INDEX = 10,
  // Number of token rows handled by one dispatch.
  ID4_QWEN3_VL_CONFIG_DISPATCH_TOKEN_COUNT = 11,
  // First logical token row handled by a partial dispatch.
  ID4_QWEN3_VL_CONFIG_TOKEN_START = 12,
} id4_qwen3_vl_config_key_t;

enum {
  ID4_QWEN3_VL_PARAMETER_KIND_COUNT = ID4_QWEN3_VL_PARAMETER_FINAL_NORM + 1,
  ID4_QWEN3_VL_TENSOR_KIND_COUNT = ID4_QWEN3_VL_TENSOR_OUTPUT + 1,
  ID4_QWEN3_VL_OPERATION_KIND_COUNT = ID4_QWEN3_VL_OPERATION_TAP + 1,
  ID4_QWEN3_VL_OPERATION_SITE_COUNT = ID4_QWEN3_VL_OPERATION_SITE_OUTPUT + 1,
  ID4_QWEN3_VL_KERNEL_KIND_COUNT =
      ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_TAIL + 1,
  ID4_QWEN3_VL_CONFIG_KEY_COUNT = ID4_QWEN3_VL_CONFIG_TOKEN_START + 1,
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
        [ID4_QWEN3_VL_TENSOR_TOKEN_IDS] = {false, IREE_SVL("token_ids")},
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
        [ID4_QWEN3_VL_OPERATION_CONDITION_NORMALIZE_TOKEN_WEIGHTS] =
            {true, IREE_SVL("normalize_token_weights")},
        [ID4_QWEN3_VL_OPERATION_BARRIER] = {true, IREE_SVL("barrier")},
        [ID4_QWEN3_VL_OPERATION_TAP] = {true, IREE_SVL("tap")},
};

static const id4_pipeline_kernel_ref_t
    id4_qwen3_vl_program_kernel_refs[ID4_QWEN3_VL_KERNEL_KIND_COUNT] = {
        [ID4_QWEN3_VL_KERNEL_TOKEN_EMBEDDING] =
            {IREE_SVL("qwen3_vl/token_embedding"),
             IREE_SVL("id4_qwen3_vl_token_embedding_bf16_bf16")},
        [ID4_QWEN3_VL_KERNEL_RMSNORM] = {IREE_SVL("qwen3_vl/rmsnorm"),
                                         IREE_SVL("id4_qwen3_vl_rmsnorm_bf16")},
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16] =
            {IREE_SVL("qwen3_vl/linear_bf16_f32"),
             IREE_SVL("id4_qwen3_vl_linear_bf16_bf16")},
        [ID4_QWEN3_VL_KERNEL_ROTARY] =
            {IREE_SVL("qwen3_vl/rotary_embedding"),
             IREE_SVL("id4_qwen3_vl_rotary_embedding_bf16")},
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
                                           IREE_SVL(
                                               "id4_qwen3_vl_silu_gate_bf16")},
        [ID4_QWEN3_VL_KERNEL_RESIDUAL_ADD] =
            {IREE_SVL("qwen3_vl/residual_add_f32"),
             IREE_SVL("id4_qwen3_vl_residual_add_bf16")},
        [ID4_QWEN3_VL_KERNEL_SELECTED_HIDDEN_PACK] =
            {IREE_SVL("qwen3_vl/selected_hidden_pack_f32"),
             IREE_SVL("id4_qwen3_vl_selected_hidden_pack_bf16_f32")},
        [ID4_QWEN3_VL_KERNEL_CONDITION_APPLY_TOKEN_WEIGHTS] =
            {IREE_SVL("qwen3_vl/condition"),
             IREE_SVL("id4_qwen3_vl_condition_apply_token_weights_f32")},
        [ID4_QWEN3_VL_KERNEL_CONDITION_NORMALIZE_TOKEN_WEIGHTS] =
            {IREE_SVL("qwen3_vl/condition"),
             IREE_SVL("id4_qwen3_vl_condition_normalize_token_weights_f32")},
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA] =
            {IREE_SVL("qwen3_vl/linear_bf16_f32_wmma"),
             IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_wmma_m32n64")},
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_TAIL] =
            {IREE_SVL("qwen3_vl/linear_bf16_bf16_tail"),
             IREE_SVL("id4_qwen3_vl_linear_bf16_bf16_tail")},
};

static const iree_string_view_t id4_qwen3_vl_program_config_keys
    [ID4_QWEN3_VL_KERNEL_KIND_COUNT][ID4_QWEN3_VL_CONFIG_KEY_COUNT] = {
        [ID4_QWEN3_VL_KERNEL_TOKEN_EMBEDDING] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.token_embedding.token_count"),
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
        [ID4_QWEN3_VL_KERNEL_CONDITION_NORMALIZE_TOKEN_WEIGHTS] =
            {
                [ID4_QWEN3_VL_CONFIG_TOKEN_COUNT] =
                    IREE_SVL("id4.qwen3_vl.condition.token_count"),
                [ID4_QWEN3_VL_CONFIG_HIDDEN_ROW_COUNT] =
                    IREE_SVL("id4.qwen3_vl.condition.hidden_row_count"),
            },
        [ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA] =
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
    id4_pipeline_program_builder_t* builder, id4_qwen3_vl_parameter_kind_t kind,
    uint32_t layer_ordinal, id4_pipeline_program_dtype_t dtype,
    id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  char key_buffer[ID4_QWEN3_VL_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_format_parameter_key(
      kind, layer_ordinal, key_buffer, IREE_ARRAYSIZE(key_buffer), &key));
  id4_pipeline_program_parameter_options_t options = {
      .structure_size = sizeof(options),
      .key = key,
      .dtype = dtype,
      .shape = shape,
  };
  return id4_pipeline_program_parameter(builder, &options, out_tensor);
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
    iree_hal_dispatch_config_t dispatch_config,
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
      .dispatch_config = dispatch_config,
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
    iree_hal_dispatch_config_t dispatch_config,
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
      builder, name, kernel_kind, dispatch_config, config_binding_count,
      config_bindings, binding_count, bindings);
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

static iree_status_t id4_qwen3_vl_program_author_token_embedding(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder,
    id4_pipeline_program_tensor_t* out_hidden_states,
    id4_pipeline_program_tensor_t* out_attention_mask,
    id4_pipeline_program_tensor_t* out_token_weights) {
  const uint32_t token_count = options->request.token_count;
  const uint32_t hidden_size = options->model.hidden_size;
  id4_pipeline_program_tensor_t token_ids =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_import_tensor(
      builder, ID4_QWEN3_VL_TENSOR_TOKEN_IDS,
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      ID4_PIPELINE_PROGRAM_DTYPE_I32,
      id4_pipeline_program_make_shape_rank1(token_count), &token_ids));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_import_tensor(
      builder, ID4_QWEN3_VL_TENSOR_ATTENTION_MASK,
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(token_count, token_count),
      out_attention_mask));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_import_tensor(
      builder, ID4_QWEN3_VL_TENSOR_TOKEN_WEIGHTS,
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(token_count), out_token_weights));
  id4_pipeline_program_tensor_t embedding =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_parameter(
      builder, ID4_QWEN3_VL_PARAMETER_TOKEN_EMBEDDING, UINT32_MAX,
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(options->model.vocab_size,
                                            hidden_size),
      &embedding));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, ID4_QWEN3_VL_TENSOR_EMBEDDED_HIDDEN_STATES, UINT32_MAX,
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_count, hidden_size),
      out_hidden_states));

  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
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
      id4_pipeline_program_read(token_ids),
      id4_pipeline_program_read(embedding),
      id4_pipeline_program_write(*out_hidden_states),
  };
  iree_hal_dispatch_config_t dispatch_config;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_matrix_element_dispatch_config(
      token_count, hidden_size, &dispatch_config));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_TOKEN_EMBEDDING, UINT32_MAX,
      ID4_QWEN3_VL_OPERATION_SITE_TOKEN_EMBEDDING,
      ID4_QWEN3_VL_KERNEL_TOKEN_EMBEDDING, dispatch_config,
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings));

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
  id4_pipeline_program_tensor_t weight = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_parameter(
      builder, parameter_kind, layer_ordinal, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(channel_count), &weight));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, output_kind, layer_ordinal, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(options->request.token_count,
                                            output_width),
      out_output));

  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, row_count},
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
  const iree_hal_dispatch_config_t dispatch_config =
      id4_qwen3_vl_program_make_row_dispatch_config(row_count);
  return id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_RMSNORM, layer_ordinal, site,
      ID4_QWEN3_VL_KERNEL_RMSNORM, dispatch_config,
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
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

static iree_status_t id4_qwen3_vl_program_author_linear(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site,
    id4_qwen3_vl_parameter_kind_t parameter_kind,
    id4_qwen3_vl_tensor_kind_t output_kind, id4_pipeline_program_tensor_t input,
    uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t* out_output) {
  const uint32_t token_count = options->request.token_count;
  if ((input_size % ID4_QWEN3_VL_LINEAR_SCALAR_INPUT_BLOCK) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL scalar linear input size %" PRIu32
                            " must be a multiple of %u",
                            input_size, ID4_QWEN3_VL_LINEAR_SCALAR_INPUT_BLOCK);
  }
  id4_pipeline_program_tensor_t weight = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_parameter(
      builder, parameter_kind, layer_ordinal, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(output_size, input_size), &weight));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, output_kind, layer_ordinal, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_count, output_size),
      out_output));

  const bool can_use_wmma_schedule =
      (input_size % ID4_QWEN3_VL_LINEAR_WMMA_INPUT_BLOCK) == 0 &&
      (output_size % ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK) == 0;
  const uint32_t wmma_token_count =
      can_use_wmma_schedule
          ? token_count - (token_count % ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK)
          : 0;
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
        operation_name, IREE_SV("wmma"), wmma_name_buffer,
        IREE_ARRAYSIZE(wmma_name_buffer), &wmma_name));
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
        ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA,
        IREE_ARRAYSIZE(wmma_config_values), wmma_config_values,
        wmma_value_buffers, wmma_config_bindings));
    id4_pipeline_program_dispatch_binding_t wmma_bindings[] = {
        id4_pipeline_program_read(input),
        id4_pipeline_program_read(weight),
        id4_pipeline_program_write_range(*out_output, 0, wmma_byte_length),
    };
    const uint32_t wmma_token_tile_count =
        wmma_token_count / ID4_QWEN3_VL_LINEAR_WMMA_TOKEN_BLOCK;
    const uint32_t output_row_tile_count =
        output_size / ID4_QWEN3_VL_LINEAR_WMMA_OUTPUT_ROW_BLOCK;
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch_named(
        builder, wmma_name, ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_WMMA,
        id4_qwen3_vl_program_make_dispatch_config_with_workgroup_size(
            wmma_token_tile_count, output_row_tile_count, 1,
            ID4_QWEN3_VL_WMMA_WORKGROUP_SIZE_X),
        IREE_ARRAYSIZE(wmma_config_values), wmma_config_bindings,
        IREE_ARRAYSIZE(wmma_bindings), wmma_bindings));

    const uint32_t tail_token_count = token_count - wmma_token_count;
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
        id4_pipeline_program_read(weight),
        id4_pipeline_program_write_range(*out_output, wmma_byte_length,
                                         tail_byte_length),
    };
    return id4_qwen3_vl_program_dispatch_named(
        builder, tail_name, ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16_TAIL,
        id4_qwen3_vl_program_make_linear_dispatch_config(tail_token_count,
                                                         output_size),
        IREE_ARRAYSIZE(tail_config_values), tail_config_bindings,
        IREE_ARRAYSIZE(tail_bindings), tail_bindings);
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
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_write(*out_output),
  };
  const iree_hal_dispatch_config_t dispatch_config =
      id4_qwen3_vl_program_make_linear_dispatch_config(token_count,
                                                       output_size);
  return id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_LINEAR, layer_ordinal, site,
      ID4_QWEN3_VL_KERNEL_LINEAR_BF16_BF16, dispatch_config,
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_qwen3_vl_program_author_rotary(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site, id4_pipeline_program_tensor_t input,
    uint32_t channel_count, id4_qwen3_vl_tensor_kind_t output_kind,
    id4_pipeline_program_tensor_t* out_output) {
  const uint32_t token_count = options->request.token_count;
  if ((channel_count % 2) != 0 || (options->model.head_size % 2) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL rotary dimensions must be even");
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, output_kind, layer_ordinal, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_count, channel_count),
      out_output));
  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
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
  iree_hal_dispatch_config_t dispatch_config;
  const uint32_t pair_count = channel_count / 2;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_matrix_element_dispatch_config(
      token_count, pair_count, &dispatch_config));
  return id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_ROTARY, layer_ordinal, site,
      ID4_QWEN3_VL_KERNEL_ROTARY, dispatch_config,
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_qwen3_vl_program_dispatch_attention(
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site, id4_qwen3_vl_kernel_kind_t kernel_kind,
    iree_host_size_t config_value_count,
    const id4_qwen3_vl_program_config_value_t* config_values,
    iree_hal_dispatch_config_t dispatch_config, iree_host_size_t binding_count,
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
      kernel_kind, dispatch_config, config_value_count, config_bindings,
      binding_count, bindings);
}

static iree_status_t id4_qwen3_vl_program_author_attention(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t value,
    id4_pipeline_program_tensor_t attention_mask,
    id4_pipeline_program_tensor_t* out_context) {
  const uint32_t token_count = options->request.token_count;
  const uint32_t hidden_size = options->model.hidden_size;
  const uint32_t attention_head_count = options->model.attention_head_count;
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
      id4_pipeline_program_make_shape_rank2(token_count, hidden_size),
      out_context));

  const id4_pipeline_program_dispatch_binding_t score_bindings[] = {
      id4_pipeline_program_read(query),
      id4_pipeline_program_read(key),
      id4_pipeline_program_read(attention_mask),
      id4_pipeline_program_write(scores),
  };
  const id4_qwen3_vl_program_config_value_t score_config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
      {ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE, hidden_size},
      {ID4_QWEN3_VL_CONFIG_ATTENTION_HEAD_COUNT, attention_head_count},
      {ID4_QWEN3_VL_CONFIG_KEY_VALUE_HEAD_COUNT,
       options->model.key_value_head_count},
      {ID4_QWEN3_VL_CONFIG_HEAD_SIZE, options->model.head_size},
  };
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch_attention(
      builder, layer_ordinal, ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_SCORES,
      ID4_QWEN3_VL_KERNEL_ATTENTION_SCORES, IREE_ARRAYSIZE(score_config_values),
      score_config_values,
      id4_qwen3_vl_program_make_dispatch_config(
          token_count, attention_head_count, token_count),
      IREE_ARRAYSIZE(score_bindings), score_bindings));
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
      id4_qwen3_vl_program_make_dispatch_config(token_count,
                                                attention_head_count, 1),
      IREE_ARRAYSIZE(softmax_bindings), softmax_bindings));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION_SOFTMAX));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_PROBABILITIES, layer_ordinal,
      probabilities));

  const id4_pipeline_program_dispatch_binding_t pv_bindings[] = {
      id4_pipeline_program_read(probabilities),
      id4_pipeline_program_read(value),
      id4_pipeline_program_write(*out_context),
  };
  const id4_qwen3_vl_program_config_value_t pv_config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
      {ID4_QWEN3_VL_CONFIG_HIDDEN_SIZE, hidden_size},
      {ID4_QWEN3_VL_CONFIG_ATTENTION_HEAD_COUNT, attention_head_count},
      {ID4_QWEN3_VL_CONFIG_KEY_VALUE_HEAD_COUNT,
       options->model.key_value_head_count},
      {ID4_QWEN3_VL_CONFIG_HEAD_SIZE, options->model.head_size},
  };
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch_attention(
      builder, layer_ordinal, ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION,
      ID4_QWEN3_VL_KERNEL_ATTENTION_PV, IREE_ARRAYSIZE(pv_config_values),
      pv_config_values,
      id4_qwen3_vl_program_make_dispatch_config(token_count,
                                                attention_head_count, 1),
      IREE_ARRAYSIZE(pv_bindings), pv_bindings));
  return id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_CONTEXT, layer_ordinal,
      *out_context);
}

static iree_status_t id4_qwen3_vl_program_author_silu_gate(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_pipeline_program_tensor_t gate, id4_pipeline_program_tensor_t up,
    id4_pipeline_program_tensor_t* out_activation) {
  const uint32_t token_count = options->request.token_count;
  const uint32_t intermediate_size = options->model.intermediate_size;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_MLP_ACTIVATION, layer_ordinal,
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_count, intermediate_size),
      out_activation));
  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
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
  iree_hal_dispatch_config_t dispatch_config;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_matrix_element_dispatch_config(
      token_count, intermediate_size, &dispatch_config));
  return id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_SILU_GATE, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_MLP, ID4_QWEN3_VL_KERNEL_SILU_GATE,
      dispatch_config, IREE_ARRAYSIZE(config_values), config_bindings,
      IREE_ARRAYSIZE(bindings), bindings);
}

static iree_status_t id4_qwen3_vl_program_author_residual_add(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    id4_qwen3_vl_operation_site_t site, id4_pipeline_program_tensor_t lhs,
    id4_pipeline_program_tensor_t rhs, id4_qwen3_vl_tensor_kind_t output_kind,
    id4_pipeline_program_tensor_t* out_output) {
  const uint32_t token_count = options->request.token_count;
  const uint32_t hidden_size = options->model.hidden_size;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, output_kind, layer_ordinal, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_count, hidden_size),
      out_output));
  const id4_qwen3_vl_program_config_value_t config_values[] = {
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
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
  iree_hal_dispatch_config_t dispatch_config;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_matrix_element_dispatch_config(
      token_count, hidden_size, &dispatch_config));
  return id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_RESIDUAL_ADD, layer_ordinal, site,
      ID4_QWEN3_VL_KERNEL_RESIDUAL_ADD, dispatch_config,
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
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
      id4_pipeline_program_write(selected_hidden_states),
  };
  iree_hal_dispatch_config_t dispatch_config;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_matrix_element_dispatch_config(
      token_count, hidden_size, &dispatch_config));
  return id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_SELECTED_HIDDEN_PACK, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_SELECTED_HIDDEN_PACK,
      ID4_QWEN3_VL_KERNEL_SELECTED_HIDDEN_PACK, dispatch_config,
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
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
  id4_pipeline_program_tensor_t stats = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, ID4_QWEN3_VL_TENSOR_CONDITION_STATS, UINT32_MAX,
      ID4_PIPELINE_PROGRAM_DTYPE_F32, id4_pipeline_program_make_shape_rank1(2),
      &stats));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_import_tensor(
      builder, ID4_QWEN3_VL_TENSOR_OUTPUT, /*flags=*/0,
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(hidden_row_count, token_count),
      out_condition));

  const id4_qwen3_vl_program_config_value_t apply_config_values[] = {
      {ID4_QWEN3_VL_CONFIG_HIDDEN_ROW_COUNT, hidden_row_count},
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
  };
  char apply_value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                          [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      apply_config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      ID4_QWEN3_VL_KERNEL_CONDITION_APPLY_TOKEN_WEIGHTS,
      IREE_ARRAYSIZE(apply_config_values), apply_config_values,
      apply_value_buffers, apply_config_bindings));
  id4_pipeline_program_dispatch_binding_t apply_bindings[] = {
      id4_pipeline_program_read(selected_hidden_states),
      id4_pipeline_program_read(token_weights),
      id4_pipeline_program_write(*out_condition),
      id4_pipeline_program_write(stats),
  };
  const iree_hal_dispatch_config_t apply_dispatch_config =
      id4_qwen3_vl_program_make_reduction_dispatch_config();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_CONDITION_APPLY_TOKEN_WEIGHTS, UINT32_MAX,
      ID4_QWEN3_VL_OPERATION_SITE_CONDITION_APPLY_TOKEN_WEIGHTS,
      ID4_QWEN3_VL_KERNEL_CONDITION_APPLY_TOKEN_WEIGHTS, apply_dispatch_config,
      IREE_ARRAYSIZE(apply_config_values), apply_config_bindings,
      IREE_ARRAYSIZE(apply_bindings), apply_bindings));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, UINT32_MAX,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_CONDITION_APPLY_TOKEN_WEIGHTS));

  const id4_qwen3_vl_program_config_value_t normalize_config_values[] = {
      {ID4_QWEN3_VL_CONFIG_HIDDEN_ROW_COUNT, hidden_row_count},
      {ID4_QWEN3_VL_CONFIG_TOKEN_COUNT, token_count},
  };
  char normalize_value_buffers[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT]
                              [ID4_QWEN3_VL_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      normalize_config_bindings[ID4_QWEN3_VL_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_config_bindings(
      ID4_QWEN3_VL_KERNEL_CONDITION_NORMALIZE_TOKEN_WEIGHTS,
      IREE_ARRAYSIZE(normalize_config_values), normalize_config_values,
      normalize_value_buffers, normalize_config_bindings));
  id4_pipeline_program_dispatch_binding_t normalize_bindings[] = {
      id4_pipeline_program_read_write(*out_condition),
      id4_pipeline_program_read(stats),
  };
  iree_hal_dispatch_config_t normalize_dispatch_config;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_make_matrix_element_dispatch_config(
      hidden_row_count, token_count, &normalize_dispatch_config));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_dispatch(
      builder, ID4_QWEN3_VL_OPERATION_CONDITION_NORMALIZE_TOKEN_WEIGHTS,
      UINT32_MAX, ID4_QWEN3_VL_OPERATION_SITE_CONDITION_NORMALIZE_TOKEN_WEIGHTS,
      ID4_QWEN3_VL_KERNEL_CONDITION_NORMALIZE_TOKEN_WEIGHTS,
      normalize_dispatch_config, IREE_ARRAYSIZE(normalize_config_values),
      normalize_config_bindings, IREE_ARRAYSIZE(normalize_bindings),
      normalize_bindings));
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

  id4_pipeline_program_tensor_t query = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_linear(
      options, builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_Q_PROJECTION,
      ID4_QWEN3_VL_PARAMETER_LAYER_Q_PROJECTION, ID4_QWEN3_VL_TENSOR_LAYER_Q,
      input_norm, hidden_size, query_width, &query));
  id4_pipeline_program_tensor_t key = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_linear(
      options, builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_K_PROJECTION,
      ID4_QWEN3_VL_PARAMETER_LAYER_K_PROJECTION, ID4_QWEN3_VL_TENSOR_LAYER_K,
      input_norm, hidden_size, key_value_width, &key));
  id4_pipeline_program_tensor_t value = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_linear(
      options, builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_V_PROJECTION,
      ID4_QWEN3_VL_PARAMETER_LAYER_V_PROJECTION, ID4_QWEN3_VL_TENSOR_LAYER_V,
      input_norm, hidden_size, key_value_width, &value));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_QKV_PROJECTION));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_Q, layer_ordinal, query));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_K, layer_ordinal, key));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_V, layer_ordinal, value));

  id4_pipeline_program_tensor_t query_norm =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_rmsnorm_rows(
      options, builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_Q_NORM,
      ID4_QWEN3_VL_PARAMETER_LAYER_Q_NORM, ID4_QWEN3_VL_TENSOR_LAYER_Q_NORM,
      query, query_row_count, options->model.head_size, query_width,
      &query_norm));
  id4_pipeline_program_tensor_t key_norm =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_rmsnorm_rows(
      options, builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_K_NORM,
      ID4_QWEN3_VL_PARAMETER_LAYER_K_NORM, ID4_QWEN3_VL_TENSOR_LAYER_K_NORM,
      key, key_row_count, options->model.head_size, key_value_width,
      &key_norm));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal, ID4_QWEN3_VL_OPERATION_SITE_AFTER_QK_NORM));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_Q_NORM, layer_ordinal, query_norm));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_K_NORM, layer_ordinal, key_norm));

  id4_pipeline_program_tensor_t query_rotary =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_rotary(
      options, builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_Q, query_norm, query_width,
      ID4_QWEN3_VL_TENSOR_LAYER_Q_ROTARY, &query_rotary));
  id4_pipeline_program_tensor_t key_rotary =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_rotary(
      options, builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_K, key_norm, key_value_width,
      ID4_QWEN3_VL_TENSOR_LAYER_K_ROTARY, &key_rotary));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_ROTARY_EMBEDDING));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_Q_ROTARY, layer_ordinal,
      query_rotary));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_K_ROTARY, layer_ordinal, key_rotary));

  id4_pipeline_program_tensor_t context = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_attention(
      options, builder, layer_ordinal, query_rotary, key_rotary, value,
      attention_mask, &context));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal, ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION));

  id4_pipeline_program_tensor_t attention_output =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_linear(
      options, builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION_O_PROJECTION,
      ID4_QWEN3_VL_PARAMETER_LAYER_O_PROJECTION,
      ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_OUTPUT, context, hidden_size,
      hidden_size, &attention_output));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION_OUTPUT_PROJECTION));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_ATTENTION_OUTPUT, layer_ordinal,
      attention_output));

  id4_pipeline_program_tensor_t after_attention =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_residual_add(
      options, builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_SELF_ATTENTION, input, attention_output,
      ID4_QWEN3_VL_TENSOR_LAYER_AFTER_ATTENTION, &after_attention));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_ATTENTION_RESIDUAL));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_AFTER_ATTENTION, layer_ordinal,
      after_attention));

  id4_pipeline_program_tensor_t post_attention_norm =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_rmsnorm(
      options, builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_POST_ATTENTION_LAYERNORM,
      ID4_QWEN3_VL_PARAMETER_LAYER_POST_ATTENTION_NORM,
      ID4_QWEN3_VL_TENSOR_LAYER_POST_ATTENTION_NORM, after_attention,
      &post_attention_norm));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_POST_ATTENTION_LAYERNORM));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_POST_ATTENTION_NORM, layer_ordinal,
      post_attention_norm));

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

  id4_pipeline_program_tensor_t activation =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_silu_gate(
      options, builder, layer_ordinal, gate, up, &activation));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_barrier(
      builder, layer_ordinal,
      ID4_QWEN3_VL_OPERATION_SITE_AFTER_MLP_ACTIVATION));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_tap_tensor(
      builder, ID4_QWEN3_VL_TENSOR_LAYER_MLP_ACTIVATION, layer_ordinal,
      activation));

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
      after_attention, mlp_down, ID4_QWEN3_VL_TENSOR_LAYER_OUTPUT, out_output));
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

  const uint32_t selected_hidden_row_count =
      id4_qwen3_vl_program_selected_hidden_row_count(&options->model);
  id4_pipeline_program_tensor_t selected_hidden_states =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_acquire_tensor(
      builder, ID4_QWEN3_VL_TENSOR_SELECTED_HIDDEN_STATES, UINT32_MAX,
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(selected_hidden_row_count,
                                            options->request.token_count),
      &selected_hidden_states));

  uint32_t selected_layer_index = 0;
  for (uint32_t i = 0; i < options->model.layer_count; ++i) {
    id4_pipeline_program_tensor_t layer_output =
        id4_pipeline_program_tensor_invalid();
    IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_author_layer(
        options, builder, i, hidden_states, attention_mask, &layer_output));
    hidden_states = layer_output;
    if (selected_layer_index < options->model.selected_layer_count &&
        options->model.selected_layer_ordinals[selected_layer_index] == i) {
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
