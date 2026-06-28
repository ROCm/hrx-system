// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_H_
#define EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_H_

#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/ideogram4_dit_program.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Stable reference-comparable stage name for Ideogram4 DiT forward execution.
#define ID4_IDEOGRAM4_DIT_STAGE_NAME "ideogram4.dit"

// Options for creating a concrete Ideogram4 DiT forward stage.
typedef struct id4_ideogram4_dit_stage_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Services retained by the base pipeline stage.
  id4_pipeline_stage_services_t services;
  // Loom kernel cache used when preparing Ideogram4 DiT kernels.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Parameter provider scope containing Ideogram4 DiT weights; empty selects
  // the anonymous scope. The stage copies this string during creation.
  iree_string_view_t parameter_scope;
  // Number of exact parameter source rules borrowed for creation.
  iree_host_size_t parameter_source_rule_count;
  // Exact parameter source rules copied by the stage during creation.
  const id4_ideogram4_dit_parameter_source_rule_t* parameter_source_rules;
  // Static Ideogram4 DiT model dimensions.
  id4_ideogram4_dit_model_config_t model;
} id4_ideogram4_dit_stage_create_options_t;

// Stage-specific plan extension carrying dynamic Ideogram4 DiT dimensions.
typedef struct id4_ideogram4_dit_stage_plan_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Dynamic request dimensions for this plan.
  id4_ideogram4_dit_request_config_t request;
  // Activation storage format for internal linear-input producers.
  id4_ideogram4_dit_activation_format_t activation_format;
  // Execution storage strategy selected for linear weights.
  id4_ideogram4_dit_weight_execution_format_t weight_execution_format;
  // Attention implementation selected for internal transformer blocks.
  id4_ideogram4_dit_attention_implementation_t attention_implementation;
  // Feed-forward implementation selected for internal transformer blocks.
  id4_ideogram4_dit_feed_forward_implementation_t feed_forward_implementation;
} id4_ideogram4_dit_stage_plan_options_t;

// Creates an Ideogram4 DiT forward stage.
iree_status_t id4_ideogram4_dit_stage_create(
    const id4_ideogram4_dit_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_H_
