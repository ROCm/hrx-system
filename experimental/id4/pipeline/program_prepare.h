// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PROGRAM_PREPARE_H_
#define EXPERIMENTAL_ID4_PIPELINE_PROGRAM_PREPARE_H_

#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/pipeline/stage.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque prepared semantic program execution state.
typedef struct id4_pipeline_program_prepared_t id4_pipeline_program_prepared_t;

// Options for preparing executable state from a semantic program and plan.
typedef struct id4_pipeline_program_prepare_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Immutable semantic program to record.
  const id4_pipeline_program_t* program;
  // Plan derived from the same semantic program.
  const id4_pipeline_plan_t* plan;
  // Loom kernel cache used to JIT planned kernel specializations.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Kernel source library used to resolve planned module paths.
  id4_pipeline_kernel_library_t* kernel_library;
  // HAL executable cache that prepares emitted Loom artifacts.
  iree_hal_executable_cache_t* executable_cache;
  // HAL executable caching mode for all prepared kernels.
  iree_hal_executable_caching_mode_t executable_caching_mode;
  // HAL command-buffer mode used for the reusable recorded region.
  iree_hal_command_buffer_mode_t command_buffer_mode;
  // Diagnostic artifact classes requested from the Loom JIT path.
  id4_pipeline_kernel_diagnostic_artifact_flags_t diagnostic_artifact_flags;
  // HAL queue-alloca flags used for local transient slabs.
  iree_hal_alloca_flags_t local_slab_alloca_flags;
  // HAL queue-dealloca flags used for local transient slabs.
  iree_hal_dealloca_flags_t local_slab_dealloca_flags;
  // Diagnostics sink for JIT and recording events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_pipeline_program_prepare_options_t;

// Prepares kernels, records the semantic program into reusable command-buffer
// regions, and seals the result as an issueable prepared program.
iree_status_t id4_pipeline_program_prepare(
    const id4_pipeline_program_prepare_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_program_prepared_t** out_prepared);

// Retains |prepared| for the caller.
void id4_pipeline_program_prepared_retain(
    id4_pipeline_program_prepared_t* prepared);

// Releases |prepared| from the caller.
void id4_pipeline_program_prepared_release(
    id4_pipeline_program_prepared_t* prepared);

// Issues one execution of |prepared| using the retained parameter slabs in
// |bundle| and boundary bindings supplied in |options|.
iree_status_t id4_pipeline_program_prepared_issue(
    id4_pipeline_program_prepared_t* prepared, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PROGRAM_PREPARE_H_
