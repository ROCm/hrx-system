// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_VAE_PARAMETERS_H_
#define EXPERIMENTAL_ID4_STAGES_VAE_PARAMETERS_H_

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/kernel_library.h"
#include "experimental/id4/pipeline/plan.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Options for wrapping a model parameter provider with VAE-derived parameters.
typedef struct id4_vae_parameter_provider_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Source model parameter provider retained by the wrapper.
  iree_io_parameter_provider_t* source_provider;
  // Program-backed plan whose parameter tensors define derived layouts.
  const id4_pipeline_plan_t* plan;
  // Loom kernel library used for prepare-time parameter encoding kernels.
  id4_pipeline_kernel_library_t* kernel_library;
  // Loom kernel cache used for prepare-time parameter encoding kernels.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // HAL executable cache used for prepare-time parameter encoding kernels.
  iree_hal_executable_cache_t* executable_cache;
  // Diagnostics sink receiving encode kernel preparation diagnostics.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_vae_parameter_provider_create_options_t;

// Formats the virtual key for a 3x3 convolution weight packed as ICxKYxKXxOC.
iree_status_t id4_vae_parameter_format_packed_conv3x3_weight_key(
    iree_string_view_t source_key, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_key);

// Parses a packed 3x3 convolution virtual key and returns the source key.
bool id4_vae_parameter_parse_packed_conv3x3_weight_key(
    iree_string_view_t key, iree_string_view_t* out_source_key);

// Formats the virtual key for a dense F32 parameter materialized as BF16.
iree_status_t id4_vae_parameter_format_bf16_key(
    iree_string_view_t source_key, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_key);

// Parses a BF16 virtual parameter key and returns the F32 source key.
bool id4_vae_parameter_parse_bf16_key(iree_string_view_t key,
                                      iree_string_view_t* out_source_key);

// Creates a provider wrapper that materializes VAE-derived parameter layouts.
iree_status_t id4_vae_parameter_provider_create(
    const id4_vae_parameter_provider_create_options_t* options,
    iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_VAE_PARAMETERS_H_
