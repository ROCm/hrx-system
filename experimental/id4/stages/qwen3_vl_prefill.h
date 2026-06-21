// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_QWEN3_VL_PREFILL_H_
#define EXPERIMENTAL_ID4_STAGES_QWEN3_VL_PREFILL_H_

#include <stdint.h>

#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/stage.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Byte length of one f32 tensor element.
#define ID4_QWEN3_VL_PREFILL_STAGE_F32_ELEMENT_BYTE_LENGTH 4

// Current Qwen3-VL parameter-provider scope.
#define ID4_QWEN3_VL_PREFILL_STAGE_PARAMETER_SCOPE ""

// Current Qwen3-VL hidden-state channel count.
#define ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE 4096

// Current Qwen3-VL first-layer input RMSNorm weight key.
#define ID4_QWEN3_VL_PREFILL_STAGE_INPUT_RMSNORM_WEIGHT_KEY \
  "text_encoders.llm.model.layers.0.input_layernorm.weight"

// Current Qwen3-VL first-layer Q projection weight key.
#define ID4_QWEN3_VL_PREFILL_STAGE_Q_PROJECTION_WEIGHT_KEY \
  "text_encoders.llm.model.layers.0.self_attn.q_proj.weight"

// Options for creating the Qwen3-VL prefill pipeline stage.
typedef struct id4_qwen3_vl_prefill_stage_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Services retained by the base pipeline stage.
  id4_pipeline_stage_services_t services;
  // Loom kernel cache used when preparing Qwen3-VL kernels.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Number of token rows in the prefill hidden-state tensor.
  uint32_t token_count;
} id4_qwen3_vl_prefill_stage_create_options_t;

// Creates the Qwen3-VL prefill stage.
iree_status_t id4_qwen3_vl_prefill_stage_create(
    const id4_qwen3_vl_prefill_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage);

// Returns the prefill input hidden-state buffer owned by a prepared bundle.
iree_hal_buffer_t* id4_qwen3_vl_prefill_stage_bundle_input_buffer(
    id4_pipeline_bundle_t* bundle);

// Returns the prefill output hidden-state buffer owned by a prepared bundle.
iree_hal_buffer_t* id4_qwen3_vl_prefill_stage_bundle_output_buffer(
    id4_pipeline_bundle_t* bundle);

// Returns the hidden-state tensor byte length owned by a prepared bundle.
iree_device_size_t id4_qwen3_vl_prefill_stage_bundle_hidden_states_byte_length(
    const id4_pipeline_bundle_t* bundle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_QWEN3_VL_PREFILL_H_
