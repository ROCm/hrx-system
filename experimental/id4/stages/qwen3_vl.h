// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_QWEN3_VL_H_
#define EXPERIMENTAL_ID4_STAGES_QWEN3_VL_H_

#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/qwen3_vl_program.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Stable reference-comparable stage name for Qwen3-VL encoder outputs.
#define ID4_QWEN3_VL_STAGE_NAME "qwen.encoder"

// Options for creating a concrete Qwen3-VL forward stage.
typedef struct id4_qwen3_vl_stage_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Services retained by the base pipeline stage.
  id4_pipeline_stage_services_t services;
  // Loom kernel cache used when preparing Qwen3-VL kernels.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Static Qwen3-VL model dimensions.
  id4_qwen3_vl_model_config_t model;
} id4_qwen3_vl_stage_create_options_t;

// Stage-specific plan extension carrying dynamic Qwen3-VL request dimensions.
typedef struct id4_qwen3_vl_stage_plan_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Dynamic request dimensions for this plan.
  id4_qwen3_vl_request_config_t request;
} id4_qwen3_vl_stage_plan_options_t;

// Creates a Qwen3-VL forward stage.
iree_status_t id4_qwen3_vl_stage_create(
    const id4_qwen3_vl_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_QWEN3_VL_H_
