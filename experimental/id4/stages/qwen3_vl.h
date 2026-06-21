// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_QWEN3_VL_H_
#define EXPERIMENTAL_ID4_STAGES_QWEN3_VL_H_

#include <stdint.h>

#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/stage.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Byte length of one f32 condition element.
#define ID4_QWEN3_VL_STAGE_CONDITION_ELEMENT_BYTE_LENGTH 4

// Options for creating the concrete Qwen3-VL forward pipeline stage.
typedef struct id4_qwen3_vl_stage_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Services retained by the base pipeline stage.
  id4_pipeline_stage_services_t services;
  // Loom kernel cache used to compile and prepare Qwen3-VL kernels.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Source identifier copied into stage-owned storage.
  iree_string_view_t source_identifier;
  // Textual Loom source contents copied into stage-owned storage.
  iree_const_byte_span_t source_contents;
  // Loom module name passed to the compiler.
  iree_string_view_t module_name;
  // HAL executable identifier assigned to the emitted artifact.
  iree_string_view_t executable_identifier;
  // Exported HAL function name resolved for the forward condition kernel.
  iree_string_view_t forward_function_name;
  // Number of selected text tokens copied into the condition tensor.
  uint32_t condition_token_count;
  // Hidden-state element count per selected text token.
  uint32_t hidden_size;
  // Configured X workgroup size for the condition forward kernel.
  uint32_t workgroup_size_x;
} id4_qwen3_vl_stage_create_options_t;

// Creates a concrete Qwen3-VL forward stage.
iree_status_t id4_qwen3_vl_stage_create(
    const id4_qwen3_vl_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage);

// Returns the selected-hidden-states input buffer owned by a prepared bundle.
iree_hal_buffer_t* id4_qwen3_vl_stage_bundle_selected_hidden_states_buffer(
    id4_pipeline_bundle_t* bundle);

// Returns the condition output buffer owned by a prepared bundle.
iree_hal_buffer_t* id4_qwen3_vl_stage_bundle_condition_buffer(
    id4_pipeline_bundle_t* bundle);

// Returns the condition tensor byte length owned by a prepared bundle.
iree_device_size_t id4_qwen3_vl_stage_bundle_condition_byte_length(
    const id4_pipeline_bundle_t* bundle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_QWEN3_VL_H_
