// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_BINDING_CLI_LORA_VARIANT_H_
#define EXPERIMENTAL_ID4_BINDING_CLI_LORA_VARIANT_H_

#include "experimental/id4/binding/cli/lora_set.h"
#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/kernel_library.h"
#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque resident conditioned-DiT parameter variant with baked LoRA weights.
typedef struct id4_cli_lora_variant_t id4_cli_lora_variant_t;

// Opaque complete resident parameter slab set.
typedef struct id4_pipeline_parameter_slab_set_t
    id4_pipeline_parameter_slab_set_t;

// Options for synchronously materializing one compact LoRA variant.
typedef struct id4_cli_lora_variant_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Exact-base conditioned-DiT plan containing the patchable domain.
  const id4_pipeline_plan_t* conditioned_dit_plan;
  // Pristine conditioned-DiT execution-layout archive.
  id4_pipeline_parameter_source_t base_parameter_source;
  // Ordered adapters and strengths baked into the variant.
  const id4_cli_lora_set_t* lora_set;
  // Compiler cache used to prepare LoRA bake kernels.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // HAL executable cache used to prepare LoRA bake kernels.
  iree_hal_executable_cache_t* executable_cache;
  // Embedded Loom modules containing the LoRA bake kernels.
  id4_pipeline_kernel_library_t* kernel_library;
  // Maximum bounded device working bytes used while baking one target.
  iree_device_size_t working_set_byte_capacity;
  // Diagnostics sink for preparation and materialization events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_cli_lora_variant_create_options_t;

// Materializes every conditioned-DiT parameter domain, waits for publication,
// and returns a complete resident variant ready for generation preparation.
iree_status_t id4_cli_lora_variant_create(
    const id4_cli_lora_variant_create_options_t* options,
    iree_allocator_t host_allocator, id4_cli_lora_variant_t** out_variant);

// Returns the complete resident slabs owned by |variant|.
id4_pipeline_parameter_slab_set_t* id4_cli_lora_variant_parameter_slabs(
    const id4_cli_lora_variant_t* variant);

// Retires all variant domains after |last_use_wait_list| and waits for their
// allocations to become reusable. The caller must first release every session,
// bundle, and binding that can reference the returned resident slabs.
iree_status_t id4_cli_lora_variant_retire(
    id4_cli_lora_variant_t* variant,
    iree_hal_semaphore_list_t last_use_wait_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink);

// Releases a successfully retired variant.
void id4_cli_lora_variant_release(id4_cli_lora_variant_t* variant);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_BINDING_CLI_LORA_VARIANT_H_
