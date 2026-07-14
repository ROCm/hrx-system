// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PROGRAM_STAGE_H_
#define EXPERIMENTAL_ID4_PIPELINE_PROGRAM_STAGE_H_

#include <stdint.h>

#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/pipeline/stage.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Asynchronously published parameter-domain replacement.
typedef struct id4_pipeline_parameter_materialization_t
    id4_pipeline_parameter_materialization_t;

// Options for deriving a pipeline plan from one stage-authored program.
typedef struct id4_pipeline_program_stage_plan_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Stable stage name copied into the created plan.
  iree_string_view_t stage_name;
  // Generic stage plan options driving device selection and diagnostics.
  const id4_pipeline_stage_plan_options_t* stage_options;
  // Semantic program authored by the concrete stage.
  const id4_pipeline_program_t* program;
  // Device group retained by the created plan.
  iree_hal_device_group_t* device_group;
  // Provider scope containing all parameter keys in the program.
  iree_string_view_t parameter_scope;
  // Device-local buffer alignment used for slabs and tensor suballocations.
  iree_device_size_t alignment;
} id4_pipeline_program_stage_plan_options_t;

// Options for preparing one reusable bundle from an authored program and plan.
typedef struct id4_pipeline_program_stage_prepare_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Stable stage name expected by the plan and used in diagnostics.
  iree_string_view_t stage_name;
  // Generic stage prepare options carrying providers, semaphores, and sinks.
  const id4_pipeline_stage_prepare_options_t* stage_options;
  // Program-backed plan retaining the immutable source program.
  const id4_pipeline_plan_t* plan;
  // Device group owned by the concrete stage.
  iree_hal_device_group_t* device_group;
  // Loom kernel cache used to JIT planned kernels.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // HAL executable cache used to prepare emitted kernel artifacts.
  iree_hal_executable_cache_t* executable_cache;
} id4_pipeline_program_stage_prepare_options_t;

// Derives a stage plan from a semantic program using the common single-region
// program-backed stage layout.
iree_status_t id4_pipeline_program_stage_create_plan(
    const id4_pipeline_program_stage_plan_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_plan_t** out_plan);

// Loads parameter slabs, prepares kernels, records the program region, and
// packages the prepared program in a reusable stage bundle.
iree_status_t id4_pipeline_program_stage_prepare(
    const id4_pipeline_program_stage_prepare_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_bundle_t** out_bundle);

// Derives a reusable bundle by rebinding one published parameter domain while
// retaining the source bundle's prepared executables and command buffers. The
// returned bundle uses the publication edge that causally dominates source
// readiness and retains the materialization until the bundle is released.
// |base_bundle| must be the prepared canonical bundle rather than another
// derived bundle.
iree_status_t id4_pipeline_program_stage_derive_bundle(
    id4_pipeline_bundle_t* base_bundle,
    id4_pipeline_parameter_materialization_t* materialization,
    iree_allocator_t host_allocator, id4_pipeline_bundle_t** out_bundle);

// Issues a bundle prepared by id4_pipeline_program_stage_prepare.
iree_status_t id4_pipeline_program_stage_issue(
    iree_string_view_t stage_name, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PROGRAM_STAGE_H_
