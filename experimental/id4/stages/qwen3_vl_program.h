// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_QWEN3_VL_PROGRAM_H_
#define EXPERIMENTAL_ID4_STAGES_QWEN3_VL_PROGRAM_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Mutable builder used while authoring a semantic pipeline program.
typedef struct id4_pipeline_program_builder_t id4_pipeline_program_builder_t;

// Qwen3-VL model dimensions used when authoring the forward program.
typedef struct id4_qwen3_vl_model_config_t {
  // Number of decoder layers.
  uint32_t layer_count;
  // Vocabulary row count in the token embedding table.
  uint32_t vocab_size;
  // Hidden-state channel count.
  uint32_t hidden_size;
  // MLP intermediate channel count.
  uint32_t intermediate_size;
  // Number of query attention heads.
  uint32_t attention_head_count;
  // Number of key/value attention heads.
  uint32_t key_value_head_count;
  // Channel count per attention head.
  uint32_t head_size;
  // Maximum prompt token positions accepted by the model runner.
  uint32_t max_token_count;
  // Number of post-layer hidden states concatenated into the condition output.
  uint32_t selected_layer_count;
  // Zero-based layer ordinals selected after the layer output residual.
  const uint32_t* selected_layer_ordinals;
} id4_qwen3_vl_model_config_t;

// Dynamic request dimensions used when authoring the forward program.
typedef struct id4_qwen3_vl_request_config_t {
  // Number of token positions in the forward pass.
  uint32_t token_count;
} id4_qwen3_vl_request_config_t;

// Calculates the BF16 packed token capacity used by Qwen3-VL linear kernels.
iree_status_t id4_qwen3_vl_program_calculate_bf16_token_capacity(
    uint32_t token_count, uint32_t* out_token_capacity);

// Options for authoring a Qwen3-VL forward semantic program.
typedef struct id4_qwen3_vl_program_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Provider source scope used when loading Qwen3-VL parameters.
  iree_string_view_t parameter_scope;
  // Static model dimensions.
  id4_qwen3_vl_model_config_t model;
  // Dynamic request dimensions.
  id4_qwen3_vl_request_config_t request;
  // Diagnostic tap names requested by the caller during planning.
  iree_string_view_list_t diagnostic_tap_names;
} id4_qwen3_vl_program_options_t;

// Authors the Qwen3-VL forward program into |builder|.
iree_status_t id4_qwen3_vl_program_author_forward(
    const id4_qwen3_vl_program_options_t* options,
    id4_pipeline_program_builder_t* builder);

// Returns the Ideogram 4 Qwen3-VL text encoder model configuration.
const id4_qwen3_vl_model_config_t* id4_qwen3_vl_program_ideogram4_model_config(
    void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_QWEN3_VL_PROGRAM_H_
