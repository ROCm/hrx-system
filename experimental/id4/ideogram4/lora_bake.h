// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_IDEOGRAM4_LORA_BAKE_H_
#define EXPERIMENTAL_ID4_IDEOGRAM4_LORA_BAKE_H_

#include "experimental/id4/ideogram4/lora_bake_plan.h"
#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/parameter_materialization.h"
#include "experimental/id4/pipeline/stage.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque immutable LoRA bake program shareable across concurrent submissions.
typedef struct id4_ideogram4_lora_bake_prepared_t
    id4_ideogram4_lora_bake_prepared_t;

// Options for preparing reusable LoRA bake executable state.
typedef struct id4_ideogram4_lora_bake_prepare_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Immutable bounded bake schedule retained by the prepared program.
  const id4_ideogram4_lora_bake_plan_t* plan;
  // In-memory Loom module library containing the bake kernels.
  id4_pipeline_kernel_library_t* kernel_library;
  // Reusable Loom JIT and prepared-executable cache.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // HAL executable cache used to prepare emitted kernel artifacts.
  iree_hal_executable_cache_t* executable_cache;
  // HAL executable caching mode used for bake-kernel preparation.
  iree_hal_executable_caching_mode_t executable_caching_mode;
  // Compiler artifacts retained for bake-kernel diagnostics.
  id4_pipeline_kernel_diagnostic_artifact_flags_t
      kernel_diagnostic_artifact_flags;
  // Diagnostics sink for JIT and preparation lifecycle events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_ideogram4_lora_bake_prepare_options_t;

// Options for asynchronously baking one resident LoRA parameter domain.
typedef struct id4_ideogram4_lora_bake_submit_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Prepared immutable bake program used for this materialization.
  const id4_ideogram4_lora_bake_prepared_t* prepared;
  // Validated execution-layout archive supplying pristine compact base bytes.
  id4_pipeline_parameter_source_t base_parameter_source;
  // Provider supporting every adapter scope retained by the bake topology.
  iree_io_parameter_provider_t* adapter_provider;
  // Optional HAL allocation pool used for variant and working storage.
  iree_hal_pool_t* allocation_pool;
  // Exact number of issue-time strengths in |strength_values|.
  iree_host_size_t strength_count;
  // Finite F32 strengths ordered by topology adapter ordinal.
  const float* strength_values;
  // Queue-allocation flags for the variant parameter domain.
  iree_hal_alloca_flags_t materialization_alloca_flags;
  // Queue-allocation flags for the bounded device working set.
  iree_hal_alloca_flags_t working_set_alloca_flags;
  // Queue-deallocation flags for the bounded device working set.
  iree_hal_dealloca_flags_t working_set_dealloca_flags;
  // Optional caller backpressure waits for variant and working-set allocation.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Caller publication edge signaled after compact weights are complete.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Diagnostics sink for loading and materialization lifecycle events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_ideogram4_lora_bake_submit_options_t;

// Prepares every Loom specialization required by one immutable bake plan.
// Device allocation and parameter I/O are deferred to submit calls.
iree_status_t id4_ideogram4_lora_bake_prepare(
    const id4_ideogram4_lora_bake_prepare_options_t* options,
    iree_allocator_t host_allocator,
    id4_ideogram4_lora_bake_prepared_t** out_prepared);

// Retains |prepared| for the caller.
void id4_ideogram4_lora_bake_prepared_retain(
    id4_ideogram4_lora_bake_prepared_t* prepared);

// Releases |prepared| from the caller.
void id4_ideogram4_lora_bake_prepared_release(
    id4_ideogram4_lora_bake_prepared_t* prepared);

// Asynchronously bakes the configured LoRA topology into resident FP8 weights.
//
// The submitted device work gathers the pristine patchable domain directly
// from its execution-layout archive, gathers and packs bounded adapter windows,
// applies each adapter in deterministic order, and requantizes each target
// directly into the dense or compact layout selected by the production plan.
// The patchable domain is allocated and populated independently of shared model
// domains. |prepared| is borrowed for this call; HAL submissions capture every
// executable and buffer they use.
//
// On success, |out_materialization| owns the published patchable domain. The
// caller combines it with independently published shared domains in one
// complete parameter binding, then releases those bindings and explicitly
// retires the materialization after its last-use edge.
iree_status_t id4_ideogram4_lora_bake_submit(
    const id4_ideogram4_lora_bake_submit_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_materialization_t** out_materialization);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_IDEOGRAM4_LORA_BAKE_H_
