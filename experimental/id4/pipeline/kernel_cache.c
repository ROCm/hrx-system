// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/kernel_cache.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "loomc/artifact_manifest.h"
#include "loomc/compile.h"
#include "loomc/context.h"
#include "loomc/emit.h"
#include "loomc/iree.h"
#include "loomc/launch_config.h"
#include "loomc/link.h"
#include "loomc/link_index.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/target.h"
#include "loomc/target/amdgpu.h"
#include "loomc/workspace.h"

#define ID4_PIPELINE_KERNEL_CACHE_ROOT_SYMBOL_CAPACITY 256

typedef struct id4_pipeline_kernel_cache_entry_t {
  // HAL executable cache used to prepare the retained executable.
  iree_hal_executable_cache_t* executable_cache;
  // Queue affinity the retained executable may be dispatched on.
  iree_hal_queue_affinity_t queue_affinity;
  // HAL executable caching mode used during preparation.
  iree_hal_executable_caching_mode_t caching_mode;
  // Diagnostic artifact request included in the compiled executable payload.
  id4_pipeline_kernel_diagnostic_artifact_flags_t diagnostic_artifact_flags;
  // Source identifier used in Loom diagnostics and cache metadata.
  iree_string_view_t source_identifier;
  // Textual Loom source contents used to produce the retained executable.
  iree_const_byte_span_t source_contents;
  // Runtime module path passed to the Loom compile invocation.
  iree_string_view_t module_path;
  // Exported Loom function selected as the link root.
  iree_string_view_t function_name;
  // Number of copied Loom config bindings.
  iree_host_size_t config_binding_count;
  // Copied Loom config bindings that define the specialization.
  id4_pipeline_kernel_config_binding_t* config_bindings;
  // Prepared executable retained by this cache entry.
  id4_pipeline_kernel_executable_t* executable;
} id4_pipeline_kernel_cache_entry_t;

struct id4_pipeline_kernel_cache_t {
  // Reference count for shared kernel cache ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for cache-owned metadata.
  iree_allocator_t host_allocator;
  // Explicit target processor used to create the Loom target profile.
  iree_string_view_t target_processor;
  // Target package linked into this embedding binary.
  loomc_target_environment_t* target_environment;
  // Shared Loom API context with target dialects registered.
  loomc_context_t* context;
  // Processor target profile.
  loomc_target_profile_t* target_profile;
  // Invocation-ready target selection derived from the profile.
  loomc_target_selection_t* target_selection;
  // Immutable linker used to select per-dispatch roots from kernel modules.
  loomc_linker_t* linker;
  // Immutable prepared compiler handle.
  loomc_compiler_t* compiler;
  // Prepared source-to-target-low pass program shared by invocations.
  loomc_pass_program_t* pass_program;
  // Number of retained prepared-executable cache entries.
  iree_host_size_t entry_count;
  // Allocated cache-entry capacity.
  iree_host_size_t entry_capacity;
  // Retained prepared-executable cache entries.
  id4_pipeline_kernel_cache_entry_t* entries;
};

struct id4_pipeline_kernel_executable_t {
  // Reference count for shared executable ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for executable-owned metadata.
  iree_allocator_t host_allocator;
  // Retained HAL executable prepared from the emitted artifact.
  iree_hal_executable_t* hal_executable;
  // Resolved static dispatch configuration for the prepared kernel.
  iree_hal_dispatch_config_t dispatch_config;
  // HAL executable format inferred from the primary artifact.
  iree_string_view_t hal_executable_format;
  // Valid executable byte length inferred by the HAL cache.
  iree_host_size_t inferred_executable_byte_length;
  // Index of the primary executable artifact in |artifacts|.
  iree_host_size_t primary_artifact_index;
  // Number of copied artifacts.
  iree_host_size_t artifact_count;
  // Copied artifacts produced by Loom compile and emit operations.
  id4_pipeline_kernel_artifact_t* artifacts;
};

static iree_status_t id4_pipeline_kernel_cache_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_pipeline_kernel_cache_copy_string(
    iree_string_view_t value, iree_allocator_t host_allocator,
    iree_string_view_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  memset(out_value, 0, sizeof(*out_value));
  if (iree_string_view_is_empty(value)) return iree_ok_status();

  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, value.size + 1, sizeof(storage[0]), (void**)&storage));
  memcpy(storage, value.data, value.size);
  storage[value.size] = 0;
  *out_value = iree_make_string_view(storage, value.size);
  return iree_ok_status();
}

static iree_status_t id4_pipeline_kernel_cache_copy_bytes(
    iree_const_byte_span_t value, iree_allocator_t host_allocator,
    iree_const_byte_span_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  memset(out_value, 0, sizeof(*out_value));
  if (value.data_length == 0) return iree_ok_status();

  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, value.data_length, sizeof(storage[0]), (void**)&storage));
  memcpy(storage, value.data, value.data_length);
  *out_value = iree_make_const_byte_span(storage, value.data_length);
  return iree_ok_status();
}

static void id4_pipeline_kernel_cache_free_string(
    iree_string_view_t* value, iree_allocator_t host_allocator) {
  if (!value) return;
  iree_allocator_free(host_allocator, (void*)value->data);
  memset(value, 0, sizeof(*value));
}

static void id4_pipeline_kernel_cache_free_bytes(
    iree_const_byte_span_t* value, iree_allocator_t host_allocator) {
  if (!value) return;
  iree_allocator_free(host_allocator, (void*)value->data);
  memset(value, 0, sizeof(*value));
}

static bool id4_pipeline_kernel_cache_byte_spans_equal(
    iree_const_byte_span_t lhs, iree_const_byte_span_t rhs) {
  if (lhs.data_length != rhs.data_length) return false;
  if (lhs.data_length == 0) return true;
  return memcmp(lhs.data, rhs.data, lhs.data_length) == 0;
}

static bool id4_pipeline_kernel_cache_config_bindings_equal(
    iree_host_size_t lhs_count, const id4_pipeline_kernel_config_binding_t* lhs,
    iree_host_size_t rhs_count,
    const id4_pipeline_kernel_config_binding_t* rhs) {
  if (lhs_count != rhs_count) return false;
  for (iree_host_size_t i = 0; i < lhs_count; ++i) {
    if (!iree_string_view_equal(lhs[i].key, rhs[i].key) ||
        !iree_string_view_equal(lhs[i].value, rhs[i].value)) {
      return false;
    }
  }
  return true;
}

static void id4_pipeline_kernel_cache_entry_deinitialize(
    id4_pipeline_kernel_cache_entry_t* entry, iree_allocator_t host_allocator) {
  if (!entry) return;
  id4_pipeline_kernel_executable_release(entry->executable);
  iree_hal_executable_cache_release(entry->executable_cache);
  if (entry->config_bindings) {
    for (iree_host_size_t i = 0; i < entry->config_binding_count; ++i) {
      id4_pipeline_kernel_cache_free_string(&entry->config_bindings[i].key,
                                            host_allocator);
      id4_pipeline_kernel_cache_free_string(&entry->config_bindings[i].value,
                                            host_allocator);
    }
  }
  iree_allocator_free(host_allocator, entry->config_bindings);
  id4_pipeline_kernel_cache_free_string(&entry->function_name, host_allocator);
  id4_pipeline_kernel_cache_free_string(&entry->module_path, host_allocator);
  id4_pipeline_kernel_cache_free_bytes(&entry->source_contents, host_allocator);
  id4_pipeline_kernel_cache_free_string(&entry->source_identifier,
                                        host_allocator);
  memset(entry, 0, sizeof(*entry));
}

static iree_status_t id4_pipeline_kernel_cache_entry_initialize(
    const id4_pipeline_kernel_cache_prepare_options_t* options,
    id4_pipeline_kernel_executable_t* executable,
    iree_allocator_t host_allocator,
    id4_pipeline_kernel_cache_entry_t* out_entry) {
  memset(out_entry, 0, sizeof(*out_entry));
  out_entry->executable_cache = options->executable_cache;
  iree_hal_executable_cache_retain(out_entry->executable_cache);
  out_entry->queue_affinity = options->queue_affinity;
  out_entry->caching_mode = options->caching_mode;
  out_entry->diagnostic_artifact_flags = options->diagnostic_artifact_flags;

  iree_status_t status = id4_pipeline_kernel_cache_copy_string(
      options->source_identifier, host_allocator,
      &out_entry->source_identifier);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_cache_copy_bytes(
        options->source_contents, host_allocator, &out_entry->source_contents);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_cache_copy_string(
        options->module_path, host_allocator, &out_entry->module_path);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_cache_copy_string(
        options->function_name, host_allocator, &out_entry->function_name);
  }
  out_entry->config_binding_count = options->config_binding_count;
  if (iree_status_is_ok(status) && options->config_binding_count != 0) {
    status = iree_allocator_malloc_array(host_allocator,
                                         options->config_binding_count,
                                         sizeof(out_entry->config_bindings[0]),
                                         (void**)&out_entry->config_bindings);
  }
  if (iree_status_is_ok(status) && options->config_binding_count != 0) {
    memset(
        out_entry->config_bindings, 0,
        options->config_binding_count * sizeof(out_entry->config_bindings[0]));
    for (iree_host_size_t i = 0;
         i < options->config_binding_count && iree_status_is_ok(status); ++i) {
      status = id4_pipeline_kernel_cache_copy_string(
          options->config_bindings[i].key, host_allocator,
          &out_entry->config_bindings[i].key);
      if (iree_status_is_ok(status)) {
        status = id4_pipeline_kernel_cache_copy_string(
            options->config_bindings[i].value, host_allocator,
            &out_entry->config_bindings[i].value);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    out_entry->executable = executable;
    id4_pipeline_kernel_executable_retain(out_entry->executable);
  } else {
    id4_pipeline_kernel_cache_entry_deinitialize(out_entry, host_allocator);
  }
  return status;
}

static bool id4_pipeline_kernel_cache_entry_matches(
    const id4_pipeline_kernel_cache_entry_t* entry,
    const id4_pipeline_kernel_cache_prepare_options_t* options) {
  return entry->executable_cache == options->executable_cache &&
         entry->queue_affinity == options->queue_affinity &&
         entry->caching_mode == options->caching_mode &&
         entry->diagnostic_artifact_flags ==
             options->diagnostic_artifact_flags &&
         iree_string_view_equal(entry->source_identifier,
                                options->source_identifier) &&
         id4_pipeline_kernel_cache_byte_spans_equal(entry->source_contents,
                                                    options->source_contents) &&
         iree_string_view_equal(entry->module_path, options->module_path) &&
         iree_string_view_equal(entry->function_name, options->function_name) &&
         id4_pipeline_kernel_cache_config_bindings_equal(
             entry->config_binding_count, entry->config_bindings,
             options->config_binding_count, options->config_bindings);
}

static id4_pipeline_kernel_cache_entry_t* id4_pipeline_kernel_cache_lookup(
    id4_pipeline_kernel_cache_t* kernel_cache,
    const id4_pipeline_kernel_cache_prepare_options_t* options) {
  for (iree_host_size_t i = 0; i < kernel_cache->entry_count; ++i) {
    id4_pipeline_kernel_cache_entry_t* entry = &kernel_cache->entries[i];
    if (id4_pipeline_kernel_cache_entry_matches(entry, options)) return entry;
  }
  return NULL;
}

static iree_status_t id4_pipeline_kernel_cache_store_entry(
    id4_pipeline_kernel_cache_t* kernel_cache,
    const id4_pipeline_kernel_cache_prepare_options_t* options,
    id4_pipeline_kernel_executable_t* executable) {
  if (kernel_cache->entry_count == kernel_cache->entry_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        kernel_cache->host_allocator, 8, sizeof(kernel_cache->entries[0]),
        &kernel_cache->entry_capacity, (void**)&kernel_cache->entries));
  }
  id4_pipeline_kernel_cache_entry_t* entry =
      &kernel_cache->entries[kernel_cache->entry_count];
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_cache_entry_initialize(
      options, executable, kernel_cache->host_allocator, entry));
  ++kernel_cache->entry_count;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_kernel_cache_validate_create_options(
    const id4_pipeline_kernel_cache_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel cache create options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_cache_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("kernel cache create")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "kernel cache create extension structures are not supported");
  }
  if (iree_string_view_is_empty(options->target_processor)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target processor is required");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_kernel_cache_validate_prepare_options(
    const id4_pipeline_kernel_cache_prepare_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel cache prepare options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_cache_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("kernel cache prepare")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "kernel cache prepare extension structures are not supported");
  }
  if (!options->executable_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL executable cache is required");
  }
  if (options->queue_affinity == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue affinity must not be empty");
  }
  if (iree_string_view_is_empty(options->source_identifier)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source identifier is required");
  }
  if (!options->source_contents.data ||
      options->source_contents.data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source contents are required");
  }
  if (iree_string_view_is_empty(options->module_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module path is required");
  }
  if (iree_string_view_is_empty(options->function_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function name is required");
  }
  if (options->config_binding_count != 0 && !options->config_bindings) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "config bindings are required when count is nonzero");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("kernel cache prepare")));
  for (iree_host_size_t i = 0; i < options->config_binding_count; ++i) {
    const id4_pipeline_kernel_config_binding_t* binding =
        &options->config_bindings[i];
    if (iree_string_view_is_empty(binding->key)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "config binding %" PRIhsz " key is required", i);
    }
    if (iree_string_view_is_empty(binding->value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "config binding %" PRIhsz " value is required",
                              i);
    }
  }
  return iree_ok_status();
}

static id4_pipeline_kernel_artifact_kind_t
id4_pipeline_kernel_artifact_kind_from_loomc(loomc_artifact_kind_t kind) {
  switch (kind) {
    case LOOMC_ARTIFACT_KIND_EXECUTABLE:
      return ID4_PIPELINE_KERNEL_ARTIFACT_KIND_EXECUTABLE;
    case LOOMC_ARTIFACT_KIND_TEXT:
      return ID4_PIPELINE_KERNEL_ARTIFACT_KIND_TEXT;
    case LOOMC_ARTIFACT_KIND_REPORT:
      return ID4_PIPELINE_KERNEL_ARTIFACT_KIND_REPORT;
    case LOOMC_ARTIFACT_KIND_MODULE:
      return ID4_PIPELINE_KERNEL_ARTIFACT_KIND_MODULE;
    default:
      return ID4_PIPELINE_KERNEL_ARTIFACT_KIND_REPORT;
  }
}

static void id4_pipeline_kernel_artifact_deinitialize(
    id4_pipeline_kernel_artifact_t* artifact, iree_allocator_t host_allocator) {
  if (!artifact) return;
  id4_pipeline_kernel_cache_free_string(&artifact->format, host_allocator);
  id4_pipeline_kernel_cache_free_string(&artifact->identifier, host_allocator);
  id4_pipeline_kernel_cache_free_bytes(&artifact->contents, host_allocator);
  memset(artifact, 0, sizeof(*artifact));
}

static iree_status_t id4_pipeline_kernel_artifact_initialize_from_loomc(
    const loomc_artifact_t* source, iree_allocator_t host_allocator,
    id4_pipeline_kernel_artifact_t* out_artifact) {
  IREE_ASSERT_ARGUMENT(source);
  IREE_ASSERT_ARGUMENT(out_artifact);
  memset(out_artifact, 0, sizeof(*out_artifact));
  out_artifact->kind =
      id4_pipeline_kernel_artifact_kind_from_loomc(source->kind);
  iree_status_t status = id4_pipeline_kernel_cache_copy_string(
      iree_string_view_from_loomc(source->format), host_allocator,
      &out_artifact->format);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_cache_copy_string(
        iree_string_view_from_loomc(source->identifier), host_allocator,
        &out_artifact->identifier);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_cache_copy_bytes(
        iree_const_byte_span_from_loomc(source->contents), host_allocator,
        &out_artifact->contents);
  }
  if (!iree_status_is_ok(status)) {
    id4_pipeline_kernel_artifact_deinitialize(out_artifact, host_allocator);
  }
  return status;
}

static iree_status_t id4_pipeline_kernel_cache_emit_event(
    id4_pipeline_diagnostics_sink_t* sink, iree_string_view_t key,
    iree_string_view_t message,
    const id4_pipeline_kernel_diagnostic_t* kernel) {
  return id4_pipeline_diagnostics_emit(
      sink, &(id4_pipeline_diagnostic_event_t){
                // This event describes kernel JIT or HAL preparation work.
                .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_KERNEL,
                // Kernel cache diagnostics are shared across stages.
                .stage_name = IREE_SV("id4.pipeline.kernel_cache"),
                // Stable event key within the kernel cache.
                .key = key,
                // Human-readable event message.
                .message = message,
                // No parameter slab payload is attached.
                .parameter_slab = NULL,
                // Kernel-specific payload for the event.
                .kernel = kernel,
            });
}

static iree_status_t id4_pipeline_kernel_cache_emit_loom_diagnostics(
    id4_pipeline_kernel_cache_t* kernel_cache,
    const id4_pipeline_kernel_cache_prepare_options_t* options,
    iree_string_view_t phase, const loomc_result_t* result) {
  if (!result) return iree_ok_status();
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    if (!diagnostic) continue;
    id4_pipeline_kernel_diagnostic_t kernel = {
        // Kernel-cache phase that produced this Loom diagnostic.
        .phase = phase,
        // Source identifier borrowed from the prepare options.
        .source_identifier = options->source_identifier,
        // Module path borrowed from the prepare options.
        .module_path = options->module_path,
        // Target processor owned by the cache.
        .target_processor = kernel_cache->target_processor,
        // Loom artifact format is not specific to a result diagnostic.
        .loom_artifact_format = iree_string_view_empty(),
        // HAL executable format is not known for compile diagnostics.
        .hal_executable_format = iree_string_view_empty(),
        // Number of config bindings supplied to the invocation.
        .config_binding_count = options->config_binding_count,
        // No artifact bytes are associated with a result diagnostic.
        .artifact_byte_length = 0,
        // No HAL-inferred executable length is associated with this diagnostic.
        .inferred_executable_byte_length = 0,
        // Result diagnostic ordinal.
        .diagnostic_index = i,
        // Loom diagnostic severity.
        .diagnostic_severity = (int32_t)diagnostic->severity,
        // Queue affinity selected for executable preparation.
        .queue_affinity = options->queue_affinity,
        // HAL caching mode selected for executable preparation.
        .caching_mode = options->caching_mode,
    };
    IREE_RETURN_IF_ERROR(id4_pipeline_kernel_cache_emit_event(
        options->diagnostics_sink,
        iree_string_view_from_loomc(diagnostic->code),
        iree_string_view_from_loomc(diagnostic->message), &kernel));
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_kernel_cache_require_loom_result(
    id4_pipeline_kernel_cache_t* kernel_cache,
    const id4_pipeline_kernel_cache_prepare_options_t* options,
    iree_string_view_t phase, const loomc_result_t* result) {
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_cache_emit_loom_diagnostics(
      kernel_cache, options, phase, result));
  if (result && loomc_result_succeeded(result)) return iree_ok_status();

  iree_string_view_t message = IREE_SV("Loom operation failed");
  if (result && loomc_result_diagnostic_count(result) != 0) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, 0);
    if (diagnostic) {
      message = iree_string_view_from_loomc(diagnostic->message);
    }
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION, "%.*s failed for %.*s:%.*s: %.*s",
      (int)phase.size, phase.data, (int)options->module_path.size,
      options->module_path.data, (int)options->function_name.size,
      options->function_name.data, (int)message.size, message.data);
}

static loomc_compile_artifact_flags_t
id4_pipeline_kernel_cache_compile_artifact_flags(
    id4_pipeline_kernel_diagnostic_artifact_flags_t flags) {
  loomc_compile_artifact_flags_t compile_flags = 0;
  if (iree_all_bits_set(
          flags, ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_MODULE_TEXT)) {
    compile_flags |= LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT;
  }
  if (iree_all_bits_set(
          flags,
          ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_MODULE_BYTECODE)) {
    compile_flags |= LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE;
  }
  if (iree_all_bits_set(
          flags,
          ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_COMPILE_REPORT_JSON)) {
    compile_flags |= LOOMC_COMPILE_ARTIFACT_FLAG_REPORT_JSON;
  }
  return compile_flags;
}

static const loomc_artifact_t* id4_pipeline_kernel_cache_find_artifact(
    const loomc_result_t* result, loomc_artifact_kind_t kind,
    loomc_string_view_t format) {
  if (!result) return NULL;
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (!artifact) continue;
    if (artifact->kind == kind &&
        loomc_string_view_equal(artifact->format, format)) {
      return artifact;
    }
  }
  return NULL;
}

static iree_status_t id4_pipeline_kernel_cache_make_dispatch_config(
    const loomc_launch_config_t* launch_config, iree_string_view_t module_path,
    iree_string_view_t function_name,
    iree_hal_dispatch_config_t* out_dispatch_config) {
  memset(out_dispatch_config, 0, sizeof(*out_dispatch_config));
  if (!iree_all_bits_set(launch_config->fields,
                         LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "launch config for %.*s:%.*s did not resolve workgroup count",
        (int)module_path.size, module_path.data, (int)function_name.size,
        function_name.data);
  }
  if (!iree_all_bits_set(launch_config->fields,
                         LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "launch config for %.*s:%.*s did not resolve workgroup size",
        (int)module_path.size, module_path.data, (int)function_name.size,
        function_name.data);
  }
  if (launch_config->workgroup_count.x == 0 ||
      launch_config->workgroup_count.y == 0 ||
      launch_config->workgroup_count.z == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "launch config for %.*s:%.*s resolved an empty workgroup count",
        (int)module_path.size, module_path.data, (int)function_name.size,
        function_name.data);
  }
  if (launch_config->workgroup_size.x == 0 ||
      launch_config->workgroup_size.y == 0 ||
      launch_config->workgroup_size.z == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "launch config for %.*s:%.*s resolved an empty workgroup size",
        (int)module_path.size, module_path.data, (int)function_name.size,
        function_name.data);
  }
  *out_dispatch_config = iree_hal_make_static_dispatch_config(
      launch_config->workgroup_count.x, launch_config->workgroup_count.y,
      launch_config->workgroup_count.z);
  out_dispatch_config->workgroup_size[0] = launch_config->workgroup_size.x;
  out_dispatch_config->workgroup_size[1] = launch_config->workgroup_size.y;
  out_dispatch_config->workgroup_size[2] = launch_config->workgroup_size.z;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_kernel_cache_emit_cache_hit(
    id4_pipeline_kernel_cache_t* kernel_cache,
    const id4_pipeline_kernel_cache_prepare_options_t* options,
    const id4_pipeline_kernel_cache_entry_t* entry) {
  const id4_pipeline_kernel_artifact_t* primary_artifact =
      id4_pipeline_kernel_executable_artifact_at(
          entry->executable, entry->executable->primary_artifact_index);
  id4_pipeline_kernel_diagnostic_t kernel = {
      // Kernel-cache phase associated with the event.
      .phase = IREE_SV("cache_hit"),
      // Source identifier borrowed from the prepare options.
      .source_identifier = options->source_identifier,
      // Module path borrowed from the prepare options.
      .module_path = options->module_path,
      // Target processor owned by the cache.
      .target_processor = kernel_cache->target_processor,
      // Loom primary artifact format retained by the cached executable.
      .loom_artifact_format = primary_artifact ? primary_artifact->format
                                               : iree_string_view_empty(),
      // HAL executable format retained by the cached executable.
      .hal_executable_format = entry->executable->hal_executable_format,
      // Number of config bindings supplied to the invocation.
      .config_binding_count = options->config_binding_count,
      // Primary artifact byte length retained by the cached executable.
      .artifact_byte_length =
          primary_artifact ? primary_artifact->contents.data_length : 0,
      // Inferred executable byte length retained by the cached executable.
      .inferred_executable_byte_length =
          entry->executable->inferred_executable_byte_length,
      // This event is not diagnostic-specific.
      .diagnostic_index = IREE_HOST_SIZE_MAX,
      // This event is not diagnostic-specific.
      .diagnostic_severity = -1,
      // Queue affinity selected for executable preparation.
      .queue_affinity = options->queue_affinity,
      // HAL caching mode selected for executable preparation.
      .caching_mode = options->caching_mode,
  };
  return id4_pipeline_kernel_cache_emit_event(
      options->diagnostics_sink, IREE_SV("kernel_cache.prepare.cache_hit"),
      IREE_SV("reused prepared HAL executable"), &kernel);
}

static iree_status_t id4_pipeline_kernel_executable_create(
    iree_string_view_t hal_executable_format,
    iree_host_size_t inferred_executable_byte_length,
    iree_hal_executable_t* hal_executable,
    iree_hal_dispatch_config_t dispatch_config,
    const loomc_result_t* compile_result, const loomc_result_t* emit_result,
    const loomc_artifact_t* primary_artifact, iree_allocator_t host_allocator,
    id4_pipeline_kernel_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(hal_executable);
  IREE_ASSERT_ARGUMENT(primary_artifact);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

  id4_pipeline_kernel_executable_t* executable = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable), (void**)&executable);
  if (iree_status_is_ok(status)) {
    memset(executable, 0, sizeof(*executable));
    iree_atomic_ref_count_init(&executable->ref_count);
    executable->host_allocator = host_allocator;
    executable->hal_executable = hal_executable;
    iree_hal_executable_retain(executable->hal_executable);
    executable->dispatch_config = dispatch_config;
    executable->inferred_executable_byte_length =
        inferred_executable_byte_length;
    executable->primary_artifact_index = IREE_HOST_SIZE_MAX;
    status = id4_pipeline_kernel_cache_copy_string(
        hal_executable_format, host_allocator,
        &executable->hal_executable_format);
  }

  iree_host_size_t artifact_count = 0;
  if (iree_status_is_ok(status)) {
    artifact_count +=
        compile_result ? loomc_result_artifact_count(compile_result) : 0;
    artifact_count +=
        emit_result ? loomc_result_artifact_count(emit_result) : 0;
    if (artifact_count != 0) {
      status = iree_allocator_malloc_array(host_allocator, artifact_count,
                                           sizeof(executable->artifacts[0]),
                                           (void**)&executable->artifacts);
    }
  }

  iree_host_size_t initialized_count = 0;
  if (iree_status_is_ok(status)) {
    memset(executable->artifacts, 0,
           artifact_count * sizeof(executable->artifacts[0]));
    for (iree_host_size_t result_index = 0;
         result_index < 2 && iree_status_is_ok(status); ++result_index) {
      const loomc_result_t* result =
          result_index == 0 ? compile_result : emit_result;
      if (!result) continue;
      for (loomc_host_size_t i = 0;
           i < loomc_result_artifact_count(result) && iree_status_is_ok(status);
           ++i) {
        const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
        if (!artifact) continue;
        status = id4_pipeline_kernel_artifact_initialize_from_loomc(
            artifact, host_allocator,
            &executable->artifacts[initialized_count]);
        if (iree_status_is_ok(status)) {
          if (artifact == primary_artifact) {
            executable->primary_artifact_index = initialized_count;
          }
          ++initialized_count;
        }
      }
    }
  }

  if (iree_status_is_ok(status) &&
      executable->primary_artifact_index == IREE_HOST_SIZE_MAX) {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "primary executable artifact was not copied");
  }

  if (iree_status_is_ok(status)) {
    executable->artifact_count = initialized_count;
    *out_executable = executable;
  } else if (executable) {
    for (iree_host_size_t i = 0; i < initialized_count; ++i) {
      id4_pipeline_kernel_artifact_deinitialize(&executable->artifacts[i],
                                                host_allocator);
    }
    iree_allocator_free(host_allocator, executable->artifacts);
    id4_pipeline_kernel_cache_free_string(&executable->hal_executable_format,
                                          host_allocator);
    iree_hal_executable_release(executable->hal_executable);
    iree_allocator_free(host_allocator, executable);
  }
  return status;
}

static iree_status_t id4_pipeline_kernel_cache_root_symbol(
    iree_string_view_t function_name, char* buffer,
    iree_host_size_t buffer_capacity, loomc_string_view_t* out_root_symbol) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_root_symbol);
  if (iree_string_view_starts_with(function_name, IREE_SV("@"))) {
    *out_root_symbol = loomc_string_view_from_iree(function_name);
    return iree_ok_status();
  }
  if (function_name.size + 1 >= buffer_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "kernel function name is too long for root symbol storage");
  }
  buffer[0] = '@';
  memcpy(buffer + 1, function_name.data, function_name.size);
  buffer[function_name.size + 1] = 0;
  *out_root_symbol = loomc_make_string_view(buffer, function_name.size + 1);
  return iree_ok_status();
}

static void id4_pipeline_kernel_cache_destroy(
    id4_pipeline_kernel_cache_t* kernel_cache) {
  iree_allocator_t host_allocator = kernel_cache->host_allocator;
  for (iree_host_size_t i = 0; i < kernel_cache->entry_count; ++i) {
    id4_pipeline_kernel_cache_entry_deinitialize(&kernel_cache->entries[i],
                                                 host_allocator);
  }
  iree_allocator_free(host_allocator, kernel_cache->entries);
  loomc_pass_program_release(kernel_cache->pass_program);
  loomc_compiler_release(kernel_cache->compiler);
  loomc_linker_release(kernel_cache->linker);
  loomc_target_selection_release(kernel_cache->target_selection);
  loomc_target_profile_release(kernel_cache->target_profile);
  loomc_context_release(kernel_cache->context);
  loomc_target_environment_release(kernel_cache->target_environment);
  id4_pipeline_kernel_cache_free_string(&kernel_cache->target_processor,
                                        host_allocator);
  iree_allocator_free(host_allocator, kernel_cache);
}

iree_status_t id4_pipeline_kernel_cache_create(
    const id4_pipeline_kernel_cache_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_kernel_cache_t** out_kernel_cache) {
  IREE_ASSERT_ARGUMENT(out_kernel_cache);
  *out_kernel_cache = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_kernel_cache_validate_create_options(options));

  id4_pipeline_kernel_cache_t* kernel_cache = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*kernel_cache), (void**)&kernel_cache);
  if (iree_status_is_ok(status)) {
    memset(kernel_cache, 0, sizeof(*kernel_cache));
    iree_atomic_ref_count_init(&kernel_cache->ref_count);
    kernel_cache->host_allocator = host_allocator;
    status = id4_pipeline_kernel_cache_copy_string(
        options->target_processor, host_allocator,
        &kernel_cache->target_processor);
  }

  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_target_environment_create_amdgpu(
        loomc_allocator_from_iree(host_allocator),
        &kernel_cache->target_environment));
  }
  if (iree_status_is_ok(status)) {
    loomc_context_target_options_t target_options = {
        // Registers the target environment with the context.
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        // Size of this extension descriptor.
        .structure_size = sizeof(target_options),
        // No additional context extensions are used.
        .next = NULL,
        // Target environment retained by the context on success.
        .target_environment = kernel_cache->target_environment,
    };
    loomc_context_options_t context_options = {
        // Context creation descriptor type.
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        // Size of this descriptor.
        .structure_size = sizeof(context_options),
        // Target environment extension.
        .next = &target_options,
    };
    status = iree_status_from_loomc(loomc_context_create(
        &context_options, loomc_allocator_from_iree(host_allocator),
        &kernel_cache->context));
  }
  if (iree_status_is_ok(status)) {
    loomc_amdgpu_profile_options_t profile_options = {
        // Target profile descriptor type.
        .type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
        // Size of this descriptor.
        .structure_size = sizeof(profile_options),
        // No target profile extensions are used.
        .next = NULL,
        // Use the explicit processor as the profile identifier.
        .identifier =
            loomc_string_view_from_iree(kernel_cache->target_processor),
        // Explicit target processor selected by the caller.
        .processor =
            loomc_string_view_from_iree(kernel_cache->target_processor),
    };
    status = iree_status_from_loomc(loomc_target_profile_create_amdgpu(
        kernel_cache->target_environment, &profile_options,
        loomc_allocator_from_iree(host_allocator),
        &kernel_cache->target_profile));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_target_selection_create_from_profile(
        kernel_cache->target_profile, loomc_allocator_from_iree(host_allocator),
        &kernel_cache->target_selection));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_linker_create(
        kernel_cache->context, NULL, loomc_allocator_from_iree(host_allocator),
        &kernel_cache->linker));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_compiler_create(
        kernel_cache->context, NULL, loomc_allocator_from_iree(host_allocator),
        &kernel_cache->compiler));
  }
  if (iree_status_is_ok(status)) {
    loomc_target_selection_options_t target_options = {
        // Target selection descriptor type.
        .type = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
        // Size of this descriptor.
        .structure_size = sizeof(target_options),
        // No additional target-selection extensions are used.
        .next = NULL,
        // Concrete target selection borrowed by pipeline creation.
        .target_selection = kernel_cache->target_selection,
    };
    loomc_target_pipeline_options_t pipeline_options = {
        // Target pipeline descriptor type.
        .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
        // Size of this descriptor.
        .structure_size = sizeof(pipeline_options),
        // Target-selection extension.
        .next = &target_options,
        // Stable identifier for diagnostics.
        .identifier = loomc_make_cstring_view("id4-target-prepared-low"),
        // Lower source/kernel IR to target-low IR ready for emission.
        .kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
        // Lower structured source control flow to CFG for now.
        .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
        // Keep enough source-to-low diagnostics for JIT failure reports.
        .source_to_low_max_errors = 20,
    };
    loomc_result_t* result = NULL;
    status =
        iree_status_from_loomc(loomc_pass_program_create_from_target_pipeline(
            kernel_cache->context, &pipeline_options,
            loomc_allocator_from_iree(host_allocator),
            &kernel_cache->pass_program, &result));
    if (iree_status_is_ok(status) && !loomc_result_succeeded(result)) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "target pipeline preparation failed");
    }
    loomc_result_release(result);
  }

  if (iree_status_is_ok(status)) {
    *out_kernel_cache = kernel_cache;
  } else if (kernel_cache) {
    id4_pipeline_kernel_cache_destroy(kernel_cache);
  }
  return status;
}

void id4_pipeline_kernel_cache_retain(
    id4_pipeline_kernel_cache_t* kernel_cache) {
  if (!kernel_cache) return;
  iree_atomic_ref_count_inc(&kernel_cache->ref_count);
}

void id4_pipeline_kernel_cache_release(
    id4_pipeline_kernel_cache_t* kernel_cache) {
  if (kernel_cache &&
      iree_atomic_ref_count_dec(&kernel_cache->ref_count) == 1) {
    id4_pipeline_kernel_cache_destroy(kernel_cache);
  }
}

iree_string_view_t id4_pipeline_kernel_cache_default_target_processor(void) {
  return IREE_SV("gfx1100");
}

iree_string_view_t id4_pipeline_kernel_cache_target_processor(
    const id4_pipeline_kernel_cache_t* kernel_cache) {
  return kernel_cache ? kernel_cache->target_processor
                      : iree_string_view_empty();
}

iree_status_t id4_pipeline_kernel_cache_prepare_executable(
    id4_pipeline_kernel_cache_t* kernel_cache,
    const id4_pipeline_kernel_cache_prepare_options_t* options,
    id4_pipeline_kernel_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(kernel_cache);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_kernel_cache_validate_prepare_options(options));

  id4_pipeline_kernel_cache_entry_t* cache_entry =
      id4_pipeline_kernel_cache_lookup(kernel_cache, options);
  if (cache_entry) {
    id4_pipeline_kernel_executable_retain(cache_entry->executable);
    iree_status_t status = id4_pipeline_kernel_cache_emit_cache_hit(
        kernel_cache, options, cache_entry);
    if (iree_status_is_ok(status)) {
      *out_executable = cache_entry->executable;
    } else {
      id4_pipeline_kernel_executable_release(cache_entry->executable);
    }
    return status;
  }

  loomc_config_binding_t* config_bindings = NULL;
  loomc_workspace_t* workspace = NULL;
  loomc_source_t* source = NULL;
  loomc_link_index_builder_t* link_index_builder = NULL;
  loomc_link_index_t* link_index = NULL;
  loomc_module_t* module = NULL;
  loomc_result_t* link_index_result = NULL;
  loomc_result_t* link_result = NULL;
  loomc_result_t* launch_config_result = NULL;
  loomc_result_t* compile_result = NULL;
  loomc_result_t* emit_result = NULL;
  iree_hal_executable_t* hal_executable = NULL;
  id4_pipeline_kernel_executable_t* prepared_executable = NULL;
  iree_hal_dispatch_config_t dispatch_config;
  memset(&dispatch_config, 0, sizeof(dispatch_config));
  iree_status_t status = iree_ok_status();

  if (options->config_binding_count != 0) {
    status = iree_allocator_malloc_array(
        kernel_cache->host_allocator, options->config_binding_count,
        sizeof(config_bindings[0]), (void**)&config_bindings);
  }
  for (iree_host_size_t i = 0;
       i < options->config_binding_count && iree_status_is_ok(status); ++i) {
    config_bindings[i] = (loomc_config_binding_t){
        // Config key borrowed for the compile invocation.
        .key = loomc_string_view_from_iree(options->config_bindings[i].key),
        // Config value borrowed for the compile invocation.
        .value = loomc_string_view_from_iree(options->config_bindings[i].value),
    };
  }

  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_workspace_create(
        NULL, loomc_allocator_from_iree(kernel_cache->host_allocator),
        &workspace));
  }
  if (iree_status_is_ok(status)) {
    loomc_source_options_t source_options = {
        // Source creation descriptor type.
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        // Size of this descriptor.
        .structure_size = sizeof(source_options),
        // No source extensions are used.
        .next = NULL,
        // ID4 passes textual .loom sources to the cache.
        .format = LOOMC_SOURCE_FORMAT_TEXT,
        // Stable identifier for Loom diagnostics.
        .identifier = loomc_string_view_from_iree(options->source_identifier),
        // Source bytes borrowed for creation and copied by Loom.
        .contents = loomc_byte_span_from_iree(options->source_contents),
        // Copy source bytes so caller storage need not outlive this call.
        .storage = LOOMC_SOURCE_STORAGE_COPY,
    };
    status = iree_status_from_loomc(loomc_source_create(
        &source_options,
        loomc_allocator_from_iree(kernel_cache->host_allocator), &source));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_link_index_builder_create(
        kernel_cache->context, NULL,
        loomc_allocator_from_iree(kernel_cache->host_allocator),
        &link_index_builder));
  }
  if (iree_status_is_ok(status)) {
    loomc_link_index_source_options_t source_options = {
        // Provider label used by linker diagnostics.
        .provider_name =
            loomc_string_view_from_iree(options->source_identifier),
        // ID4 prepares one primary source module per JIT invocation.
        .role = LOOMC_LINK_PROVIDER_ROLE_INPUT,
    };
    status = iree_status_from_loomc(loomc_link_index_builder_add_source(
        link_index_builder, source, &source_options, NULL));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_link_index_builder_finish(
        link_index_builder, &link_index, &link_index_result));
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_cache_require_loom_result(
        kernel_cache, options, IREE_SV("link_index"), link_index_result);
  }
  char root_symbol_storage[ID4_PIPELINE_KERNEL_CACHE_ROOT_SYMBOL_CAPACITY];
  loomc_string_view_t root_symbol = loomc_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_cache_root_symbol(
        options->function_name, root_symbol_storage,
        IREE_ARRAYSIZE(root_symbol_storage), &root_symbol);
  }
  if (iree_status_is_ok(status)) {
    loomc_target_selection_options_t target_options = {
        // Target selection descriptor type.
        .type = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
        // Size of this descriptor.
        .structure_size = sizeof(target_options),
        // No additional target-selection extensions are used.
        .next = NULL,
        // Concrete target selection borrowed by the link invocation.
        .target_selection = kernel_cache->target_selection,
    };
    loomc_link_options_t link_options = {
        // Link invocation descriptor type.
        .type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
        // Size of this descriptor.
        .structure_size = sizeof(link_options),
        // Target-selection extension.
        .next = &target_options,
        // Frozen index containing the source module for this invocation.
        .link_index = link_index,
        // Runtime module path for diagnostics and emitted objects.
        .module_name = loomc_string_view_from_iree(options->module_path),
        // Selected exported function root for this executable.
        .root_symbols = &root_symbol,
        // Exactly one dispatch-site function is materialized.
        .root_symbol_count = 1,
        // Strip check dialect testbench symbols from runtime modules.
        .flags = LOOMC_LINK_FLAG_STRIP_CHECK_SYMBOLS,
        // Strict config resolution for runtime JIT invocations.
        .config =
            {
                // Config binding table borrowed for this call.
                .bindings = config_bindings,
                // Number of config bindings.
                .binding_count = options->config_binding_count,
                // ID4 runtime scheduling does not use JSON config payloads.
                .json_object = loomc_string_view_empty(),
                // Unknown and unresolved config is a link failure.
                .flags = LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
                         LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
            },
    };
    status = iree_status_from_loomc(loomc_link_module(
        kernel_cache->linker, workspace, &link_options, &module, &link_result));
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_cache_require_loom_result(
        kernel_cache, options, IREE_SV("link"), link_result);
  }
  if (iree_status_is_ok(status)) {
    loomc_launch_config_eval_options_t launch_options = {
        // Launch-config evaluation descriptor type.
        .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_EVAL_OPTIONS,
        // Size of this descriptor.
        .structure_size = sizeof(launch_options),
        // No launch-config extensions are used.
        .next = NULL,
        // Kernel function selected as the link root.
        .function_symbol = root_symbol,
        // Configs were materialized by the selective link invocation.
        .config =
            {
                // No additional config bindings remain after linking.
                .bindings = NULL,
                // No additional config bindings remain after linking.
                .binding_count = 0,
                // ID4 runtime scheduling does not use JSON config payloads.
                .json_object = loomc_string_view_empty(),
                // No config work remains after linking selected roots.
                .flags = 0,
            },
        // This first integration pass uses config-resolved launch regions.
        .workload_arguments = NULL,
        // This first integration pass uses config-resolved launch regions.
        .workload_argument_count = 0,
        // HAL queue dispatch requires static group count and local size.
        .required_fields = LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
                           LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE,
    };
    loomc_launch_config_t launch_config = {
        // Launch-config result descriptor type.
        .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
        // Size of this result structure.
        .structure_size = sizeof(launch_config),
    };
    status = iree_status_from_loomc(loomc_module_evaluate_launch_config(
        module, workspace, &launch_options,
        loomc_allocator_from_iree(kernel_cache->host_allocator), &launch_config,
        &launch_config_result));
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_kernel_cache_require_loom_result(
          kernel_cache, options, IREE_SV("launch_config"),
          launch_config_result);
    }
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_kernel_cache_make_dispatch_config(
          &launch_config, options->module_path, options->function_name,
          &dispatch_config);
    }
  }
  if (iree_status_is_ok(status)) {
    loomc_target_selection_options_t target_options = {
        // Target selection descriptor type.
        .type = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
        // Size of this descriptor.
        .structure_size = sizeof(target_options),
        // No additional target-selection extensions are used.
        .next = NULL,
        // Concrete target selection borrowed by the compile invocation.
        .target_selection = kernel_cache->target_selection,
    };
    loomc_compile_options_t compile_options = {
        // Compile invocation descriptor type.
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        // Size of this descriptor.
        .structure_size = sizeof(compile_options),
        // Target-selection extension.
        .next = &target_options,
        // Runtime module path for diagnostics and emitted objects.
        .module_name = loomc_string_view_from_iree(options->module_path),
        // Optional diagnostic artifacts requested by the caller.
        .artifact_flags = id4_pipeline_kernel_cache_compile_artifact_flags(
            options->diagnostic_artifact_flags),
        // Strict config resolution for runtime JIT invocations.
        .config =
            {
                // Configs were materialized by the selective link invocation.
                .bindings = NULL,
                // Linked runtime modules should not retain config declarations.
                .binding_count = 0,
                // ID4 runtime scheduling does not use JSON config payloads.
                .json_object = loomc_string_view_empty(),
                // No config work remains after linking selected roots.
                .flags = 0,
            },
    };
    status = iree_status_from_loomc(loomc_compile_module(
        kernel_cache->compiler, workspace, kernel_cache->pass_program, module,
        &compile_options,
        loomc_allocator_from_iree(kernel_cache->host_allocator),
        &compile_result));
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_cache_require_loom_result(
        kernel_cache, options, IREE_SV("compile"), compile_result);
  }
  if (iree_status_is_ok(status)) {
    loomc_target_selection_options_t target_options = {
        // Target selection descriptor type.
        .type = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
        // Size of this descriptor.
        .structure_size = sizeof(target_options),
        // No additional target-selection extensions are used.
        .next = NULL,
        // Concrete target selection borrowed by the emit invocation.
        .target_selection = kernel_cache->target_selection,
    };
    loomc_amdgpu_emit_options_t amdgpu_options = {
        // Target emit descriptor type.
        .type = LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
        // Size of this descriptor.
        .structure_size = sizeof(amdgpu_options),
        // Target-selection extension.
        .next = &target_options,
        // Smoke kernels do not require runtime support globals.
        .runtime_globals = LOOMC_AMDGPU_RUNTIME_GLOBAL_NONE,
    };
    loomc_artifact_manifest_options_t manifest_options = {
        // Artifact manifest descriptor type.
        .type = LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS,
        // Size of this descriptor.
        .structure_size = sizeof(manifest_options),
        // Target emission extension follows the manifest descriptor.
        .next = &amdgpu_options,
        // Detailed manifests are useful for executable diagnostics.
        .mode = LOOMC_ARTIFACT_MANIFEST_MODE_DETAILS,
        // Empty derives a stable identifier from the emitted artifact.
        .identifier = loomc_string_view_empty(),
    };
    const bool request_manifest = iree_all_bits_set(
        options->diagnostic_artifact_flags,
        ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_EMIT_MANIFEST_JSON);
    loomc_emit_options_t emit_options = {
        // Emit invocation descriptor type.
        .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
        // Size of this descriptor.
        .structure_size = sizeof(emit_options),
        // Optional artifact manifest extension, then target emission options.
        .next = request_manifest ? (const void*)&manifest_options
                                 : (const void*)&amdgpu_options,
        // Request target executable bytes from Loom.
        .artifact_format =
            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
        // Empty selects the target's in-memory default artifact identifier.
        .identifier = loomc_string_view_empty(),
        // Request the primary executable artifact.
        .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
    };
    status = iree_status_from_loomc(loomc_emit_module(
        kernel_cache->target_environment, workspace, module, &emit_options,
        loomc_allocator_from_iree(kernel_cache->host_allocator), &emit_result));
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_cache_require_loom_result(
        kernel_cache, options, IREE_SV("emit"), emit_result);
  }

  const loomc_artifact_t* primary_artifact = NULL;
  if (iree_status_is_ok(status)) {
    primary_artifact = id4_pipeline_kernel_cache_find_artifact(
        emit_result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO));
    if (!primary_artifact) {
      status =
          iree_make_status(IREE_STATUS_NOT_FOUND,
                           "Loom did not emit a target executable artifact");
    }
  }

  char hal_executable_format_storage[128] = {0};
  iree_host_size_t inferred_executable_byte_length = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_cache_infer_format(
        options->executable_cache, options->caching_mode,
        iree_const_byte_span_from_loomc(primary_artifact->contents),
        sizeof(hal_executable_format_storage), hal_executable_format_storage,
        &inferred_executable_byte_length);
  }
  const iree_string_view_t hal_executable_format =
      iree_make_cstring_view(hal_executable_format_storage);
  if (iree_status_is_ok(status) && inferred_executable_byte_length == 0) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL executable cache inferred an empty executable payload");
  }
  if (iree_status_is_ok(status) &&
      !iree_hal_executable_cache_can_prepare_format(options->executable_cache,
                                                    options->caching_mode,
                                                    hal_executable_format)) {
    status = iree_make_status(
        IREE_STATUS_INCOMPATIBLE,
        "HAL executable cache cannot prepare format `%.*s`",
        (int)hal_executable_format.size, hal_executable_format.data);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_kernel_diagnostic_t kernel = {
        // Kernel-cache phase associated with the event.
        .phase = IREE_SV("hal.infer_format"),
        // Source identifier borrowed from the prepare options.
        .source_identifier = options->source_identifier,
        // Module path borrowed from the prepare options.
        .module_path = options->module_path,
        // Target processor owned by the cache.
        .target_processor = kernel_cache->target_processor,
        // Loom primary artifact format.
        .loom_artifact_format =
            iree_string_view_from_loomc(primary_artifact->format),
        // HAL executable format inferred from the HSACO bytes.
        .hal_executable_format = hal_executable_format,
        // Number of config bindings supplied to the invocation.
        .config_binding_count = options->config_binding_count,
        // Primary artifact byte length.
        .artifact_byte_length = primary_artifact->contents.data_length,
        // Valid executable byte length inferred by the HAL.
        .inferred_executable_byte_length = inferred_executable_byte_length,
        // This event is not diagnostic-specific.
        .diagnostic_index = IREE_HOST_SIZE_MAX,
        // This event is not diagnostic-specific.
        .diagnostic_severity = -1,
        // Queue affinity selected for executable preparation.
        .queue_affinity = options->queue_affinity,
        // HAL caching mode selected for executable preparation.
        .caching_mode = options->caching_mode,
    };
    status = id4_pipeline_kernel_cache_emit_event(
        options->diagnostics_sink, IREE_SV("kernel_cache.hal.infer_format"),
        IREE_SV("inferred HAL executable format"), &kernel);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_executable_params_t executable_params = {
        // Queues where dispatches using this executable may be submitted.
        .queue_affinity = options->queue_affinity,
        // Caller-selected HAL caching mode.
        .caching_mode = options->caching_mode,
        // HAL executable format inferred from the artifact bytes.
        .executable_format = hal_executable_format,
        // Valid executable byte span inferred by the HAL executable cache.
        .executable_data = iree_make_const_byte_span(
            primary_artifact->contents.data, inferred_executable_byte_length),
        // No executable-level specialization constants are used yet.
        .constant_count = 0,
        // No executable-level specialization constants are used yet.
        .constants = NULL,
    };
    status = iree_hal_executable_cache_prepare_executable(
        options->executable_cache, &executable_params, &hal_executable);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_executable_create(
        hal_executable_format, inferred_executable_byte_length, hal_executable,
        dispatch_config, compile_result, emit_result, primary_artifact,
        kernel_cache->host_allocator, &prepared_executable);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_kernel_diagnostic_t kernel = {
        // Kernel-cache phase associated with the event.
        .phase = IREE_SV("prepare"),
        // Source identifier borrowed from the prepare options.
        .source_identifier = options->source_identifier,
        // Module path borrowed from the prepare options.
        .module_path = options->module_path,
        // Target processor owned by the cache.
        .target_processor = kernel_cache->target_processor,
        // Loom primary artifact format.
        .loom_artifact_format =
            iree_string_view_from_loomc(primary_artifact->format),
        // HAL executable format inferred from the HSACO bytes.
        .hal_executable_format = hal_executable_format,
        // Number of config bindings supplied to the invocation.
        .config_binding_count = options->config_binding_count,
        // Primary artifact byte length.
        .artifact_byte_length = primary_artifact->contents.data_length,
        // Valid executable byte length inferred by the HAL.
        .inferred_executable_byte_length = inferred_executable_byte_length,
        // This event is not diagnostic-specific.
        .diagnostic_index = IREE_HOST_SIZE_MAX,
        // This event is not diagnostic-specific.
        .diagnostic_severity = -1,
        // Queue affinity selected for executable preparation.
        .queue_affinity = options->queue_affinity,
        // HAL caching mode selected for executable preparation.
        .caching_mode = options->caching_mode,
    };
    status = id4_pipeline_kernel_cache_emit_event(
        options->diagnostics_sink, IREE_SV("kernel_cache.prepare"),
        IREE_SV("prepared HAL executable"), &kernel);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_cache_store_entry(kernel_cache, options,
                                                   prepared_executable);
  }
  if (iree_status_is_ok(status)) {
    *out_executable = prepared_executable;
    prepared_executable = NULL;
  }

  id4_pipeline_kernel_executable_release(prepared_executable);
  iree_hal_executable_release(hal_executable);
  loomc_result_release(emit_result);
  loomc_result_release(compile_result);
  loomc_result_release(launch_config_result);
  loomc_result_release(link_result);
  loomc_result_release(link_index_result);
  loomc_module_release(module);
  loomc_link_index_release(link_index);
  loomc_link_index_builder_release(link_index_builder);
  loomc_source_release(source);
  loomc_workspace_release(workspace);
  iree_allocator_free(kernel_cache->host_allocator, config_bindings);
  return status;
}

static void id4_pipeline_kernel_executable_destroy(
    id4_pipeline_kernel_executable_t* executable) {
  iree_allocator_t host_allocator = executable->host_allocator;
  for (iree_host_size_t i = 0; i < executable->artifact_count; ++i) {
    id4_pipeline_kernel_artifact_deinitialize(&executable->artifacts[i],
                                              host_allocator);
  }
  iree_allocator_free(host_allocator, executable->artifacts);
  id4_pipeline_kernel_cache_free_string(&executable->hal_executable_format,
                                        host_allocator);
  iree_hal_executable_release(executable->hal_executable);
  iree_allocator_free(host_allocator, executable);
}

void id4_pipeline_kernel_executable_retain(
    id4_pipeline_kernel_executable_t* executable) {
  if (!executable) return;
  iree_atomic_ref_count_inc(&executable->ref_count);
}

void id4_pipeline_kernel_executable_release(
    id4_pipeline_kernel_executable_t* executable) {
  if (executable && iree_atomic_ref_count_dec(&executable->ref_count) == 1) {
    id4_pipeline_kernel_executable_destroy(executable);
  }
}

iree_hal_executable_t* id4_pipeline_kernel_executable_hal_executable(
    const id4_pipeline_kernel_executable_t* executable) {
  return executable ? executable->hal_executable : NULL;
}

iree_hal_dispatch_config_t id4_pipeline_kernel_executable_dispatch_config(
    const id4_pipeline_kernel_executable_t* executable) {
  iree_hal_dispatch_config_t dispatch_config;
  memset(&dispatch_config, 0, sizeof(dispatch_config));
  if (executable) {
    dispatch_config = executable->dispatch_config;
  }
  return dispatch_config;
}

iree_string_view_t id4_pipeline_kernel_executable_hal_format(
    const id4_pipeline_kernel_executable_t* executable) {
  return executable ? executable->hal_executable_format
                    : iree_string_view_empty();
}

iree_const_byte_span_t id4_pipeline_kernel_executable_primary_data(
    const id4_pipeline_kernel_executable_t* executable) {
  if (!executable ||
      executable->primary_artifact_index >= executable->artifact_count) {
    return iree_const_byte_span_empty();
  }
  return executable->artifacts[executable->primary_artifact_index].contents;
}

iree_host_size_t id4_pipeline_kernel_executable_artifact_count(
    const id4_pipeline_kernel_executable_t* executable) {
  return executable ? executable->artifact_count : 0;
}

const id4_pipeline_kernel_artifact_t*
id4_pipeline_kernel_executable_artifact_at(
    const id4_pipeline_kernel_executable_t* executable,
    iree_host_size_t index) {
  if (!executable || index >= executable->artifact_count) return NULL;
  return &executable->artifacts[index];
}
