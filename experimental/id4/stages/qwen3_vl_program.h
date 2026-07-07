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
  // Token ids used to assemble the compact prompt-local embedding rows.
  const int32_t* token_ids;
} id4_qwen3_vl_request_config_t;

// Qwen3-VL provider parameter format selected when creating the stage.
typedef enum id4_qwen3_vl_parameter_format_e {
  // Invalid provider parameter format.
  ID4_QWEN3_VL_PARAMETER_FORMAT_INVALID = 0,
  // Provider linear weights are dense BF16 tensors.
  ID4_QWEN3_VL_PARAMETER_FORMAT_BF16 = 1,
  // Provider linear weights are official FP8 e4m3 tensors with F32
  // weight_scale_inv tensors over 128x128 source blocks.
  ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED = 2,
} id4_qwen3_vl_parameter_format_t;

// Qwen3-VL linear weight execution strategy selected by the planner. Source
// precision is a model/provider contract, while this enum controls the
// resident execution layout.
typedef enum id4_qwen3_vl_weight_execution_strategy_e {
  // Invalid weight execution strategy.
  ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_INVALID = 0,
  // Consume provider row-major BF16 linear weights directly.
  ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR = 1,
  // Require every BF16 linear weight to use compact RHS WMMA execution tiles.
  ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_COMPACT_RHS = 2,
  // Use compact RHS for authored compact tile families and row-major for
  // row-major-only tile families.
  ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS = 3,
} id4_qwen3_vl_weight_execution_strategy_t;

// Qwen3-VL attention implementation selected by the planner.
typedef enum id4_qwen3_vl_attention_implementation_e {
  // Invalid attention implementation.
  ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_INVALID = 0,
  // Select the planner-owned stable attention implementation.
  ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_AUTO = 1,
  // Materialize score and probability tensors around scalar attention kernels.
  ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_MATERIALIZED = 2,
  // Use BF16 WMMA attention kernels without materialized score/probability
  // taps.
  ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_WMMA = 3,
} id4_qwen3_vl_attention_implementation_t;

// Parses a Qwen3-VL provider parameter format name.
iree_status_t id4_qwen3_vl_parameter_format_parse(
    iree_string_view_t value, id4_qwen3_vl_parameter_format_t* out_format);

// Returns the stable Qwen3-VL provider parameter format name.
iree_string_view_t id4_qwen3_vl_parameter_format_name(
    id4_qwen3_vl_parameter_format_t format);

// Parses a Qwen3-VL weight execution strategy name.
iree_status_t id4_qwen3_vl_weight_execution_strategy_parse(
    iree_string_view_t value,
    id4_qwen3_vl_weight_execution_strategy_t* out_strategy);

// Returns the stable Qwen3-VL weight execution strategy name.
iree_string_view_t id4_qwen3_vl_weight_execution_strategy_name(
    id4_qwen3_vl_weight_execution_strategy_t strategy);

// Parses a Qwen3-VL attention implementation name.
iree_status_t id4_qwen3_vl_attention_implementation_parse(
    iree_string_view_t value,
    id4_qwen3_vl_attention_implementation_t* out_implementation);

// Returns the stable Qwen3-VL attention implementation name.
iree_string_view_t id4_qwen3_vl_attention_implementation_name(
    id4_qwen3_vl_attention_implementation_t implementation);

// Calculates the packed token capacity used by Qwen3-VL kernels for a provider
// parameter format.
iree_status_t id4_qwen3_vl_program_calculate_token_capacity(
    id4_qwen3_vl_parameter_format_t parameter_format, uint32_t token_count,
    uint32_t* out_token_capacity);

// Calculates the BF16 packed token capacity used by Qwen3-VL kernels.
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
  // Host allocator used for transient authoring tables.
  iree_allocator_t host_allocator;
  // Provider parameter format selected for this program.
  id4_qwen3_vl_parameter_format_t parameter_format;
  // Linear weight execution strategy selected for this program.
  id4_qwen3_vl_weight_execution_strategy_t weight_execution_strategy;
  // Attention implementation selected for this program.
  id4_qwen3_vl_attention_implementation_t attention_implementation;
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
