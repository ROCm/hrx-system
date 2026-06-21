// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_KERNEL_CACHE_H_
#define EXPERIMENTAL_ID4_PIPELINE_KERNEL_CACHE_H_

#include <stdint.h>

#include "experimental/id4/pipeline/diagnostics.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Reusable Loom compiler state for ID4 pipeline kernels.
typedef struct id4_pipeline_kernel_cache_t id4_pipeline_kernel_cache_t;

// Prepared executable and copied compiler artifacts from a kernel cache miss.
typedef struct id4_pipeline_kernel_executable_t
    id4_pipeline_kernel_executable_t;

// Kernel artifact category copied from a Loom compiler result.
typedef enum id4_pipeline_kernel_artifact_kind_e {
  // Executable or loadable binary artifact.
  ID4_PIPELINE_KERNEL_ARTIFACT_KIND_EXECUTABLE = 0,
  // Human-readable textual artifact.
  ID4_PIPELINE_KERNEL_ARTIFACT_KIND_TEXT = 1,
  // Machine-readable report artifact.
  ID4_PIPELINE_KERNEL_ARTIFACT_KIND_REPORT = 2,
  // Loom module artifact such as text or bytecode.
  ID4_PIPELINE_KERNEL_ARTIFACT_KIND_MODULE = 3,
} id4_pipeline_kernel_artifact_kind_t;

// Copied kernel artifact view owned by a prepared kernel executable.
typedef struct id4_pipeline_kernel_artifact_t {
  // Artifact category.
  id4_pipeline_kernel_artifact_kind_t kind;
  // Stable artifact format string.
  iree_string_view_t format;
  // Human-readable artifact identifier.
  iree_string_view_t identifier;
  // Artifact bytes.
  iree_const_byte_span_t contents;
} id4_pipeline_kernel_artifact_t;

// Compile-time configuration binding passed to Loom config resolution.
typedef struct id4_pipeline_kernel_config_binding_t {
  // Config symbol key, with or without a leading at-sign.
  iree_string_view_t key;
  // Config value spelling.
  iree_string_view_t value;
} id4_pipeline_kernel_config_binding_t;

// Diagnostic artifacts requested from the Loom compiler path.
typedef uint32_t id4_pipeline_kernel_diagnostic_artifact_flags_t;

// Diagnostic artifact request bits.
typedef enum id4_pipeline_kernel_diagnostic_artifact_flag_bits_e {
  // Copy textual Loom module IR after successful compilation.
  ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_MODULE_TEXT = 1u << 0,
  // Copy binary Loom bytecode after successful compilation.
  ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_MODULE_BYTECODE = 1u << 1,
  // Copy the JSON compile report.
  ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_COMPILE_REPORT_JSON = 1u << 2,
  // Copy the JSON emit artifact manifest.
  ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_EMIT_MANIFEST_JSON = 1u << 3,
} id4_pipeline_kernel_diagnostic_artifact_flag_bits_t;

// Options for creating a reusable kernel cache.
typedef struct id4_pipeline_kernel_cache_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Explicit AMDGPU processor key used to build the Loom target profile.
  iree_string_view_t amdgpu_processor;
} id4_pipeline_kernel_cache_create_options_t;

// Options for preparing one kernel executable through Loom and the HAL.
typedef struct id4_pipeline_kernel_cache_prepare_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // HAL executable cache that prepares the emitted HSACO artifact.
  iree_hal_executable_cache_t* executable_cache;
  // Queue affinity the prepared executable may be dispatched on.
  iree_hal_queue_affinity_t queue_affinity;
  // HAL executable caching mode for preparation.
  iree_hal_executable_caching_mode_t caching_mode;
  // Source identifier used in Loom diagnostics and cache metadata.
  iree_string_view_t source_identifier;
  // Textual Loom source contents.
  iree_const_byte_span_t source_contents;
  // Runtime module name passed to the Loom compile invocation.
  iree_string_view_t module_name;
  // Identifier assigned to the primary emitted executable artifact.
  iree_string_view_t executable_identifier;
  // Number of Loom config bindings.
  iree_host_size_t config_binding_count;
  // Loom config bindings borrowed for the prepare call.
  const id4_pipeline_kernel_config_binding_t* config_bindings;
  // Diagnostic artifacts to copy into the prepared executable.
  id4_pipeline_kernel_diagnostic_artifact_flags_t diagnostic_artifact_flags;
  // Diagnostics sink for compiler and HAL preparation events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_pipeline_kernel_cache_prepare_options_t;

// Creates reusable Loom compiler state for AMDGPU pipeline kernels.
iree_status_t id4_pipeline_kernel_cache_create(
    const id4_pipeline_kernel_cache_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_kernel_cache_t** out_kernel_cache);

// Retains |kernel_cache| for the caller.
void id4_pipeline_kernel_cache_retain(
    id4_pipeline_kernel_cache_t* kernel_cache);

// Releases |kernel_cache| from the caller.
void id4_pipeline_kernel_cache_release(
    id4_pipeline_kernel_cache_t* kernel_cache);

// Returns the AMDGPU processor selected for |kernel_cache|.
iree_string_view_t id4_pipeline_kernel_cache_amdgpu_processor(
    const id4_pipeline_kernel_cache_t* kernel_cache);

// Compiles, emits, and prepares one kernel executable.
iree_status_t id4_pipeline_kernel_cache_prepare_executable(
    id4_pipeline_kernel_cache_t* kernel_cache,
    const id4_pipeline_kernel_cache_prepare_options_t* options,
    id4_pipeline_kernel_executable_t** out_executable);

// Retains |executable| for the caller.
void id4_pipeline_kernel_executable_retain(
    id4_pipeline_kernel_executable_t* executable);

// Releases |executable| from the caller.
void id4_pipeline_kernel_executable_release(
    id4_pipeline_kernel_executable_t* executable);

// Returns the retained HAL executable.
iree_hal_executable_t* id4_pipeline_kernel_executable_hal_executable(
    const id4_pipeline_kernel_executable_t* executable);

// Returns the HAL executable format inferred from the primary artifact bytes.
iree_string_view_t id4_pipeline_kernel_executable_hal_format(
    const id4_pipeline_kernel_executable_t* executable);

// Returns the primary executable artifact bytes.
iree_const_byte_span_t id4_pipeline_kernel_executable_primary_data(
    const id4_pipeline_kernel_executable_t* executable);

// Returns the number of copied compiler artifacts.
iree_host_size_t id4_pipeline_kernel_executable_artifact_count(
    const id4_pipeline_kernel_executable_t* executable);

// Returns copied artifact |index| or NULL when out of range.
const id4_pipeline_kernel_artifact_t*
id4_pipeline_kernel_executable_artifact_at(
    const id4_pipeline_kernel_executable_t* executable, iree_host_size_t index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_KERNEL_CACHE_H_
