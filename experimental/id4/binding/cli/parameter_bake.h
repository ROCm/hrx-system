// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_BINDING_CLI_PARAMETER_BAKE_H_
#define EXPERIMENTAL_ID4_BINDING_CLI_PARAMETER_BAKE_H_

#include "experimental/id4/ideogram4/session.h"
#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/kernel_library.h"
#include "experimental/id4/tooling/runtime.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Options for baking one generation plan's checkpoint parameters.
typedef struct id4_cli_parameter_bake_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Generation plan whose coarse stage parameter layouts are baked.
  const id4_ideogram4_generation_plan_t* generation_plan;
  // Checkpoint sources supplying each parameter-bearing coarse stage.
  const id4_ideogram4_generation_parameter_sources_t* parameter_sources;
  // Process-local HAL and Loom services used for parameter encoders.
  const id4_tooling_runtime_context_t* runtime_context;
  // Kernel library used by parameter encoding dispatches.
  id4_pipeline_kernel_library_t* kernel_library;
  // Coarse parameter-bearing stages selected for baking.
  id4_ideogram4_generation_resident_stage_mask_t stage_mask;
  // Directory receiving atomically published IRPA archives.
  iree_string_view_t output_directory;
  // Maximum device-local bytes staged by checkpoint parameter encoders.
  iree_device_size_t encoder_staging_chunk_byte_capacity;
  // Maximum device-local bytes relaying source-layout archive entries.
  iree_device_size_t archive_staging_chunk_byte_capacity;
  // Diagnostic artifacts requested while compiling parameter encoders.
  id4_pipeline_kernel_diagnostic_artifact_flags_t diagnostic_artifact_flags;
  // Diagnostics sink for parameter loading and archive population.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_cli_parameter_bake_options_t;

// Bakes selected coarse stage layouts serially and returns after archive data
// is flushed and published.
iree_status_t id4_cli_bake_parameter_layouts(
    const id4_cli_parameter_bake_options_t* options,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_BINDING_CLI_PARAMETER_BAKE_H_
