// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PROGRAM_REGION_H_
#define EXPERIMENTAL_ID4_PIPELINE_PROGRAM_REGION_H_

#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/pipeline/region.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Resolved HAL executable state for one semantic Loom dispatch.
typedef struct id4_pipeline_program_region_kernel_resolution_t {
  // HAL executable used when the region builder is in RECORD mode.
  iree_hal_executable_t* executable;
  // HAL executable function used when the region builder is in RECORD mode.
  iree_hal_executable_function_t function;
  // Static HAL dispatch configuration resolved from Loom launch config.
  iree_hal_dispatch_config_t dispatch_config;
} id4_pipeline_program_region_kernel_resolution_t;

// Diagnostic tap lowering mode.
typedef enum id4_pipeline_program_region_tap_mode_e {
  // Ignores semantic tap operations while lowering the region.
  ID4_PIPELINE_PROGRAM_REGION_TAP_MODE_IGNORE = 0,
  // Lowers semantic tap operations into explicit tensor copies.
  ID4_PIPELINE_PROGRAM_REGION_TAP_MODE_CAPTURE = 1,
} id4_pipeline_program_region_tap_mode_t;

// Resolves a semantic import operation into a concrete region tensor import.
typedef iree_status_t(
    IREE_API_PTR* id4_pipeline_program_region_resolve_import_fn_t)(
    void* user_data, const id4_pipeline_program_import_op_t* import_op,
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_host_size_t import_ordinal, id4_pipeline_tensor_import_t* out_import);

// Resolves a semantic parameter operation into a packed parameter slab import.
typedef iree_status_t(
    IREE_API_PTR* id4_pipeline_program_region_resolve_parameter_fn_t)(
    void* user_data, const id4_pipeline_program_parameter_op_t* parameter_op,
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_host_size_t parameter_ordinal,
    id4_pipeline_tensor_import_t* out_import);

// Resolves a semantic constant operation into a packed constant slab import.
typedef iree_status_t(
    IREE_API_PTR* id4_pipeline_program_region_resolve_constant_fn_t)(
    void* user_data, const id4_pipeline_program_constant_op_t* constant_op,
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_host_size_t constant_ordinal,
    id4_pipeline_tensor_import_t* out_import);

// Resolves a semantic Loom dispatch into prepared HAL executable state.
typedef iree_status_t(
    IREE_API_PTR* id4_pipeline_program_region_resolve_kernel_fn_t)(
    void* user_data, const id4_pipeline_program_dispatch_loom_op_t* dispatch_op,
    iree_string_view_t specialization_key, iree_host_size_t dispatch_ordinal,
    id4_pipeline_program_region_kernel_resolution_t* out_resolution);

// Resolves a semantic tap operation into a concrete capture tensor import.
typedef iree_status_t(
    IREE_API_PTR* id4_pipeline_program_region_resolve_tap_fn_t)(
    void* user_data, const id4_pipeline_program_tap_op_t* tap_op,
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_host_size_t tap_ordinal, id4_pipeline_tensor_import_t* out_import);

// Options for lowering a semantic program into a region builder.
typedef struct id4_pipeline_program_region_lower_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Immutable semantic program to lower.
  const id4_pipeline_program_t* program;
  // Region builder receiving lowered operations.
  id4_pipeline_region_builder_t* builder;
  // Diagnostic tap lowering mode.
  id4_pipeline_program_region_tap_mode_t tap_mode;
  // Caller-owned diagnostic tap names to capture.
  iree_string_view_list_t captured_tap_names;
  // Required alignment for local tensor suballocations.
  iree_device_size_t local_tensor_alignment;
  // Opaque pointer passed to resolver callbacks.
  void* user_data;
  // Import resolver required when the program contains import operations.
  id4_pipeline_program_region_resolve_import_fn_t resolve_import;
  // Parameter resolver required when the program contains parameter operations.
  id4_pipeline_program_region_resolve_parameter_fn_t resolve_parameter;
  // Constant resolver required when the program contains constant operations.
  id4_pipeline_program_region_resolve_constant_fn_t resolve_constant;
  // Kernel resolver required when the region builder is in RECORD mode.
  id4_pipeline_program_region_resolve_kernel_fn_t resolve_kernel;
  // Tap resolver required when tap_mode is CAPTURE.
  id4_pipeline_program_region_resolve_tap_fn_t resolve_tap;
} id4_pipeline_program_region_lower_options_t;

// Formats the stable specialization key for |dispatch_op|. The returned string
// is allocated with |host_allocator| and must be freed by the caller.
iree_status_t id4_pipeline_program_format_dispatch_specialization_key(
    const id4_pipeline_program_dispatch_loom_op_t* dispatch_op,
    iree_allocator_t host_allocator, iree_string_view_t* out_key);

// Converts a semantic program dtype into a region/plan tensor dtype.
id4_pipeline_tensor_dtype_t id4_pipeline_program_region_convert_dtype(
    id4_pipeline_program_dtype_t dtype);

// Lowers |options->program| into |options->builder| by walking semantic
// operations exactly once.
iree_status_t id4_pipeline_program_region_lower(
    const id4_pipeline_program_region_lower_options_t* options,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PROGRAM_REGION_H_
