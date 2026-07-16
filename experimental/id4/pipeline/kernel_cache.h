// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_KERNEL_CACHE_H_
#define EXPERIMENTAL_ID4_PIPELINE_KERNEL_CACHE_H_

#include <stdint.h>

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/kernel_library.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Reusable Loom compiler state for ID4 pipeline kernels.
typedef struct id4_pipeline_kernel_cache_t id4_pipeline_kernel_cache_t;

// Retained executable budget for interactive ID4 contexts.
//
// Zero keeps shared compiler state but disables prepared-executable retention.
#define ID4_PIPELINE_KERNEL_CACHE_INTERACTIVE_ENTRY_LIMIT 0

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

// Options for creating a reusable kernel cache.
typedef struct id4_pipeline_kernel_cache_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Target processor key used to build the Loom target profile.
  // Runtime callers derive this from the selected HAL device specification;
  // offline compiler callers may provide an explicit processor.
  iree_string_view_t target_processor;
  // Maximum retained entries before evicting the oldest prepared executable.
  // Zero disables prepared-executable retention.
  iree_host_size_t entry_limit;
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
  // Runtime module path passed to the Loom compile invocation.
  iree_string_view_t module_path;
  // Exported Loom function selected as the link root for this executable.
  iree_string_view_t function_name;
  // Number of Loom config bindings.
  iree_host_size_t config_binding_count;
  // Loom config bindings borrowed for the prepare call.
  const id4_pipeline_kernel_config_binding_t* config_bindings;
  // Diagnostic artifacts to copy into the prepared executable.
  id4_pipeline_kernel_diagnostic_artifact_flags_t diagnostic_artifact_flags;
  // Diagnostics sink for compiler and HAL preparation events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_pipeline_kernel_cache_prepare_options_t;

// Creates reusable Loom compiler state for pipeline kernels.
iree_status_t id4_pipeline_kernel_cache_create(
    const id4_pipeline_kernel_cache_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_kernel_cache_t** out_kernel_cache);

// Selects the exact AMDGPU processor advertised by |device_spec|.
//
// The returned view is borrowed from |device_spec| and remains valid for the
// lifetime of that specification.
iree_status_t id4_pipeline_kernel_cache_select_amdgpu_target_processor(
    const iree_hal_device_spec_t* device_spec,
    iree_string_view_t* out_target_processor);

// Retains |kernel_cache| for the caller.
void id4_pipeline_kernel_cache_retain(
    id4_pipeline_kernel_cache_t* kernel_cache);

// Releases |kernel_cache| from the caller.
void id4_pipeline_kernel_cache_release(
    id4_pipeline_kernel_cache_t* kernel_cache);

// Returns the target processor selected for |kernel_cache|.
iree_string_view_t id4_pipeline_kernel_cache_target_processor(
    const id4_pipeline_kernel_cache_t* kernel_cache);

// Returns a retained executable for one exact kernel specialization.
//
// Compiles, emits, and prepares the executable on miss. Reuses a retained
// executable when the target, HAL cache, source, function, config bindings,
// queue affinity, caching mode, and diagnostic artifact request all match a
// prior prepare call. Cache residency is bounded by the construction entry
// limit; a zero limit disables executable retention, and evicting an entry
// releases only the cache's reference without invalidating executables already
// retained by prepared bundles.
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

// Returns the resolved static HAL dispatch configuration.
iree_hal_dispatch_config_t id4_pipeline_kernel_executable_dispatch_config(
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
