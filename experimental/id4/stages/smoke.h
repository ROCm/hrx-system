// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_SMOKE_H_
#define EXPERIMENTAL_ID4_STAGES_SMOKE_H_

#include <stdint.h>

#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/stage.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Byte length of the smoke stage output tensor.
#define ID4_SMOKE_STAGE_OUTPUT_BYTE_LENGTH 4

// Options for creating the concrete smoke pipeline stage.
typedef struct id4_smoke_stage_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Services retained by the base pipeline stage.
  id4_pipeline_stage_services_t services;
  // Loom kernel cache used to compile and prepare the smoke executable.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Source identifier copied into stage-owned storage.
  iree_string_view_t source_identifier;
  // Textual Loom source contents copied into stage-owned storage.
  iree_const_byte_span_t source_contents;
  // Loom module name passed to the compiler.
  iree_string_view_t module_name;
  // HAL executable identifier assigned to the emitted artifact.
  iree_string_view_t executable_identifier;
  // Exported HAL function name resolved after executable preparation.
  iree_string_view_t function_name;
  // Configured X workgroup count.
  uint32_t workgroups_x;
  // Configured X workgroup size.
  uint32_t workgroup_size_x;
} id4_smoke_stage_create_options_t;

// Creates a concrete smoke stage that exercises the shared pipeline lifecycle.
iree_status_t id4_smoke_stage_create(
    const id4_smoke_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage);

// Returns the output buffer owned by a prepared smoke bundle.
iree_hal_buffer_t* id4_smoke_stage_bundle_output_buffer(
    id4_pipeline_bundle_t* bundle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_SMOKE_H_
