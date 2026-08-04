// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/loom_jit.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "experimental/qwen/runtime/loom_compile_pool.h"
#include "iree/base/threading/mutex.h"
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
#include "loomc/target/amdgpu/iree_hal.h"
#include "loomc/workspace.h"

#define QWEN_LOOM_JIT_ROOT_SYMBOL_CAPACITY 256

typedef struct qwen_loom_jit_entry_t {
  // Stable runtime module path.
  iree_string_view_t module_path;
  // Source identifier used in compiler diagnostics.
  iree_string_view_t source_identifier;
  // Exact linked module source bytes.
  iree_const_byte_span_t source_contents;
  // Exported function selected as the link root.
  iree_string_view_t function_name;
  // Number of copied config bindings.
  iree_host_size_t config_binding_count;
  // Ordered copied config bindings.
  qwen_loom_config_binding_t* config_bindings;
  // Number of copied positional workload arguments.
  iree_host_size_t workload_argument_count;
  // Ordered copied positional workload arguments.
  int64_t* workload_arguments;
  // Prepared executable retained by this cache entry.
  qwen_loom_executable_t* executable;
} qwen_loom_jit_entry_t;

struct qwen_loom_jit_t {
  // Reference count for shared JIT ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for JIT-owned state.
  iree_allocator_t host_allocator;
  // Device used for target discovery and executable loading.
  iree_hal_device_t* device;
  // Queue affinity for all prepared executables.
  iree_hal_queue_affinity_t queue_affinity;
  // Maximum retained exact executable entries.
  iree_host_size_t entry_limit;
  // Number of initialized retained entries.
  iree_host_size_t entry_count;
  // Retained exact executable entries in oldest-first order.
  qwen_loom_jit_entry_t* entries;
  // Serializes complete prepare operations and compile-pool batches.
  iree_slim_mutex_t operation_mutex;
  // Serializes exact-entry lookup and cache mutation.
  iree_slim_mutex_t cache_mutex;
  // Bounded task workers used for independent code groups.
  qwen_loom_compile_pool_t compile_pool;
  // Number of initialized worker workspace slots.
  iree_host_size_t worker_count;
  // One mutable Loom workspace exclusively owned by each task worker.
  loomc_workspace_t** worker_workspaces;
  // AMDGPU compiler target package.
  loomc_target_environment_t* target_environment;
  // Loom API context registered with the target package.
  loomc_context_t* context;
  // Exact target profile derived from the selected HAL device.
  loomc_target_profile_t* target_profile;
  // Prepared selective linker.
  loomc_linker_t* linker;
  // Prepared source compiler.
  loomc_compiler_t* compiler;
  // Prepared source-to-target-low lowering pipeline.
  loomc_pass_program_t* pass_program;
  // AMDGPU support globals required by the configured sanitizer passes.
  loomc_amdgpu_runtime_global_flags_t runtime_globals;
};

struct qwen_loom_executable_t {
  // Reference count for shared executable ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for executable-owned state.
  iree_allocator_t host_allocator;
  // Loaded HAL executable.
  iree_hal_executable_t* hal_executable;
  // Resolved exported function.
  iree_hal_executable_function_t function;
  // Static workgroup count evaluated from exact workload arguments.
  iree_hal_dispatch_config_t dispatch_config;
};

static iree_status_t qwen_loom_jit_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "%.*s structure size %" PRIhsz " is smaller than required %" PRIhsz,
      (int)options_name.size, options_name.data, actual_size, expected_size);
}

static iree_status_t qwen_loom_jit_copy_string(iree_string_view_t source,
                                               iree_allocator_t host_allocator,
                                               iree_string_view_t* out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) return iree_ok_status();

  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, source.size + 1, sizeof(storage[0]), (void**)&storage));
  memcpy(storage, source.data, source.size);
  storage[source.size] = 0;
  *out_target = iree_make_string_view(storage, source.size);
  return iree_ok_status();
}

static iree_status_t qwen_loom_jit_copy_bytes(
    iree_const_byte_span_t source, iree_allocator_t host_allocator,
    iree_const_byte_span_t* out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = iree_const_byte_span_empty();
  if (source.data_length == 0) return iree_ok_status();

  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(host_allocator, source.data_length,
                                  sizeof(storage[0]), (void**)&storage));
  memcpy(storage, source.data, source.data_length);
  *out_target = iree_make_const_byte_span(storage, source.data_length);
  return iree_ok_status();
}

static void qwen_loom_jit_free_string(iree_string_view_t* value,
                                      iree_allocator_t host_allocator) {
  if (!value) return;
  iree_allocator_free(host_allocator, (void*)value->data);
  *value = iree_string_view_empty();
}

static void qwen_loom_jit_free_bytes(iree_const_byte_span_t* value,
                                     iree_allocator_t host_allocator) {
  if (!value) return;
  iree_allocator_free(host_allocator, (void*)value->data);
  *value = iree_const_byte_span_empty();
}

static bool qwen_loom_jit_byte_spans_equal(iree_const_byte_span_t lhs,
                                           iree_const_byte_span_t rhs) {
  if (lhs.data_length != rhs.data_length) return false;
  return lhs.data_length == 0 ||
         memcmp(lhs.data, rhs.data, lhs.data_length) == 0;
}

static bool qwen_loom_jit_config_bindings_equal(
    iree_host_size_t lhs_count, const qwen_loom_config_binding_t* lhs,
    iree_host_size_t rhs_count, const qwen_loom_config_binding_t* rhs) {
  if (lhs_count != rhs_count) return false;
  for (iree_host_size_t i = 0; i < lhs_count; ++i) {
    if (!iree_string_view_equal(lhs[i].key, rhs[i].key) ||
        !iree_string_view_equal(lhs[i].value, rhs[i].value)) {
      return false;
    }
  }
  return true;
}

static bool qwen_loom_jit_workload_arguments_equal(iree_host_size_t lhs_count,
                                                   const int64_t* lhs,
                                                   iree_host_size_t rhs_count,
                                                   const int64_t* rhs) {
  if (lhs_count != rhs_count) return false;
  return lhs_count == 0 || memcmp(lhs, rhs, lhs_count * sizeof(lhs[0])) == 0;
}

static bool qwen_loom_jit_prepare_options_match_code(
    const qwen_loom_jit_prepare_options_t* lhs,
    const qwen_loom_jit_prepare_options_t* rhs) {
  return iree_string_view_equal(lhs->source_module->module_path,
                                rhs->source_module->module_path) &&
         iree_string_view_equal(lhs->source_module->source_identifier,
                                rhs->source_module->source_identifier) &&
         qwen_loom_jit_byte_spans_equal(lhs->source_module->source_contents,
                                        rhs->source_module->source_contents) &&
         iree_string_view_equal(lhs->function_name, rhs->function_name) &&
         qwen_loom_jit_config_bindings_equal(
             lhs->config_binding_count, lhs->config_bindings,
             rhs->config_binding_count, rhs->config_bindings);
}

static bool qwen_loom_jit_prepare_options_match_exact(
    const qwen_loom_jit_prepare_options_t* lhs,
    const qwen_loom_jit_prepare_options_t* rhs) {
  return qwen_loom_jit_prepare_options_match_code(lhs, rhs) &&
         qwen_loom_jit_workload_arguments_equal(
             lhs->workload_argument_count, lhs->workload_arguments,
             rhs->workload_argument_count, rhs->workload_arguments);
}

static iree_status_t qwen_loom_jit_require_result(
    iree_string_view_t phase, iree_string_view_t module_path,
    iree_string_view_t function_name, const loomc_result_t* result) {
  if (result && loomc_result_succeeded(result)) return iree_ok_status();

  iree_string_view_t message = IREE_SV("Loom operation failed");
  if (result && loomc_result_diagnostic_count(result) != 0) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, 0);
    if (diagnostic) {
      message = iree_string_view_from_loomc(diagnostic->message);
    }
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "%.*s failed for %.*s:%.*s: %.*s", (int)phase.size,
                          phase.data, (int)module_path.size, module_path.data,
                          (int)function_name.size, function_name.data,
                          (int)message.size, message.data);
}

static iree_status_t qwen_loom_jit_validate_create_options(
    const qwen_loom_jit_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen Loom JIT options are required");
  }
  IREE_RETURN_IF_ERROR(qwen_loom_jit_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("JIT options")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Qwen Loom JIT option extensions are unsupported");
  }
  if (!options->device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL device is required");
  }
  if (options->entry_limit == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen Loom JIT requires a nonzero retained entry limit");
  }
  if (options->worker_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen Loom JIT requires a nonzero compiler worker count");
  }
  return iree_ok_status();
}

static iree_status_t qwen_loom_jit_validate_prepare_options(
    const qwen_loom_jit_prepare_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen Loom prepare options are required");
  }
  IREE_RETURN_IF_ERROR(qwen_loom_jit_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("JIT prepare options")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Qwen Loom prepare option extensions are unsupported");
  }
  if (!options->source_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "embedded Loom source module is required");
  }
  if (iree_string_view_is_empty(options->source_module->module_path) ||
      iree_string_view_is_empty(options->source_module->source_identifier) ||
      options->source_module->source_contents.data_length == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "embedded Loom source module must have path, identifier, and contents");
  }
  if (iree_string_view_is_empty(options->function_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "exported Loom function name is required");
  }
  if (iree_string_view_starts_with(options->function_name, IREE_SV("@"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "exported Loom function name must not include a leading at-sign");
  }
  if (options->config_binding_count != 0 && !options->config_bindings) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "config bindings are required when their count is nonzero");
  }
  for (iree_host_size_t i = 0; i < options->config_binding_count; ++i) {
    if (iree_string_view_is_empty(options->config_bindings[i].key) ||
        iree_string_view_is_empty(options->config_bindings[i].value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "config binding %" PRIhsz " requires a key and value", i);
    }
  }
  if (options->workload_argument_count != 0 && !options->workload_arguments) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "workload arguments are required when their count is nonzero");
  }
  return iree_ok_status();
}

static void qwen_loom_jit_entry_deinitialize(qwen_loom_jit_entry_t* entry,
                                             iree_allocator_t host_allocator) {
  if (!entry) return;
  qwen_loom_executable_release(entry->executable);
  iree_allocator_free(host_allocator, entry->workload_arguments);
  if (entry->config_bindings) {
    for (iree_host_size_t i = 0; i < entry->config_binding_count; ++i) {
      qwen_loom_jit_free_string(&entry->config_bindings[i].key, host_allocator);
      qwen_loom_jit_free_string(&entry->config_bindings[i].value,
                                host_allocator);
    }
  }
  iree_allocator_free(host_allocator, entry->config_bindings);
  qwen_loom_jit_free_string(&entry->function_name, host_allocator);
  qwen_loom_jit_free_bytes(&entry->source_contents, host_allocator);
  qwen_loom_jit_free_string(&entry->source_identifier, host_allocator);
  qwen_loom_jit_free_string(&entry->module_path, host_allocator);
  memset(entry, 0, sizeof(*entry));
}

static iree_status_t qwen_loom_jit_entry_initialize(
    const qwen_loom_jit_prepare_options_t* options,
    qwen_loom_executable_t* executable, iree_allocator_t host_allocator,
    qwen_loom_jit_entry_t* out_entry) {
  memset(out_entry, 0, sizeof(*out_entry));
  iree_status_t status =
      qwen_loom_jit_copy_string(options->source_module->module_path,
                                host_allocator, &out_entry->module_path);
  if (iree_status_is_ok(status)) {
    status = qwen_loom_jit_copy_string(
        options->source_module->source_identifier, host_allocator,
        &out_entry->source_identifier);
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_loom_jit_copy_bytes(options->source_module->source_contents,
                                 host_allocator, &out_entry->source_contents);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_loom_jit_copy_string(options->function_name, host_allocator,
                                       &out_entry->function_name);
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
      status = qwen_loom_jit_copy_string(options->config_bindings[i].key,
                                         host_allocator,
                                         &out_entry->config_bindings[i].key);
      if (iree_status_is_ok(status)) {
        status = qwen_loom_jit_copy_string(
            options->config_bindings[i].value, host_allocator,
            &out_entry->config_bindings[i].value);
      }
    }
  }

  out_entry->workload_argument_count = options->workload_argument_count;
  if (iree_status_is_ok(status) && options->workload_argument_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, options->workload_argument_count,
        sizeof(out_entry->workload_arguments[0]),
        (void**)&out_entry->workload_arguments);
  }
  if (iree_status_is_ok(status) && options->workload_argument_count != 0) {
    memcpy(out_entry->workload_arguments, options->workload_arguments,
           options->workload_argument_count *
               sizeof(out_entry->workload_arguments[0]));
  }

  if (iree_status_is_ok(status)) {
    out_entry->executable = executable;
    qwen_loom_executable_retain(executable);
  } else {
    qwen_loom_jit_entry_deinitialize(out_entry, host_allocator);
  }
  return status;
}

static bool qwen_loom_jit_entry_matches_code(
    const qwen_loom_jit_entry_t* entry,
    const qwen_loom_jit_prepare_options_t* options) {
  return iree_string_view_equal(entry->module_path,
                                options->source_module->module_path) &&
         iree_string_view_equal(entry->source_identifier,
                                options->source_module->source_identifier) &&
         qwen_loom_jit_byte_spans_equal(
             entry->source_contents, options->source_module->source_contents) &&
         iree_string_view_equal(entry->function_name, options->function_name) &&
         qwen_loom_jit_config_bindings_equal(
             entry->config_binding_count, entry->config_bindings,
             options->config_binding_count, options->config_bindings);
}

static bool qwen_loom_jit_entry_matches_exact(
    const qwen_loom_jit_entry_t* entry,
    const qwen_loom_jit_prepare_options_t* options) {
  return qwen_loom_jit_entry_matches_code(entry, options) &&
         qwen_loom_jit_workload_arguments_equal(
             entry->workload_argument_count, entry->workload_arguments,
             options->workload_argument_count, options->workload_arguments);
}

static qwen_loom_jit_entry_t* qwen_loom_jit_lookup_exact(
    qwen_loom_jit_t* jit, const qwen_loom_jit_prepare_options_t* options) {
  for (iree_host_size_t i = 0; i < jit->entry_count; ++i) {
    if (qwen_loom_jit_entry_matches_exact(&jit->entries[i], options)) {
      return &jit->entries[i];
    }
  }
  return NULL;
}

static qwen_loom_jit_entry_t* qwen_loom_jit_lookup_code(
    qwen_loom_jit_t* jit, const qwen_loom_jit_prepare_options_t* options) {
  for (iree_host_size_t i = 0; i < jit->entry_count; ++i) {
    if (qwen_loom_jit_entry_matches_code(&jit->entries[i], options)) {
      return &jit->entries[i];
    }
  }
  return NULL;
}

static void qwen_loom_jit_store_initialized_entry(
    qwen_loom_jit_t* jit, qwen_loom_jit_entry_t* entry) {
  if (jit->entry_count == jit->entry_limit) {
    qwen_loom_jit_entry_deinitialize(&jit->entries[0], jit->host_allocator);
    memmove(&jit->entries[0], &jit->entries[1],
            (jit->entry_count - 1) * sizeof(jit->entries[0]));
    --jit->entry_count;
    memset(&jit->entries[jit->entry_count], 0, sizeof(jit->entries[0]));
  }
  jit->entries[jit->entry_count] = *entry;
  ++jit->entry_count;
  memset(entry, 0, sizeof(*entry));
}

static iree_status_t qwen_loom_jit_make_root_symbol(
    iree_string_view_t function_name, char* storage,
    iree_host_size_t storage_capacity, loomc_string_view_t* out_root_symbol) {
  IREE_ASSERT_ARGUMENT(storage);
  IREE_ASSERT_ARGUMENT(out_root_symbol);
  if (function_name.size + 1 >= storage_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "exported Loom function name is too long");
  }
  storage[0] = '@';
  memcpy(storage + 1, function_name.data, function_name.size);
  storage[function_name.size + 1] = 0;
  *out_root_symbol = loomc_make_string_view(storage, function_name.size + 1);
  return iree_ok_status();
}

static iree_status_t qwen_loom_jit_make_dispatch_config(
    const loomc_launch_config_t* launch_config, iree_string_view_t module_path,
    iree_string_view_t function_name,
    iree_hal_dispatch_config_t* out_dispatch_config) {
  memset(out_dispatch_config, 0, sizeof(*out_dispatch_config));
  const loomc_launch_config_field_flags_t required_fields =
      LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
      LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE;
  if (!iree_all_bits_set(launch_config->fields, required_fields)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "launch config for %.*s:%.*s did not resolve static geometry",
        (int)module_path.size, module_path.data, (int)function_name.size,
        function_name.data);
  }
  if (launch_config->workgroup_count.x == 0 ||
      launch_config->workgroup_count.y == 0 ||
      launch_config->workgroup_count.z == 0 ||
      launch_config->workgroup_size.x == 0 ||
      launch_config->workgroup_size.y == 0 ||
      launch_config->workgroup_size.z == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "launch config for %.*s:%.*s resolved empty geometry",
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

static iree_status_t qwen_loom_jit_use_executable_workgroup_size(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    iree_string_view_t module_path, iree_string_view_t function_name,
    iree_hal_dispatch_config_t* dispatch_config) {
  iree_hal_executable_function_info_t function_info;
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_function_info(executable, function, &function_info));
  if (iree_any_bit_set(
          function_info.flags,
          IREE_HAL_EXECUTABLE_FUNCTION_FLAG_WORKGROUP_SIZE_DYNAMIC)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "compiled function %.*s:%.*s retains a dynamic workgroup size",
        (int)module_path.size, module_path.data, (int)function_name.size,
        function_name.data);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(function_info.workgroup_size);
       ++i) {
    if (function_info.workgroup_size[i] != dispatch_config->workgroup_size[i]) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "compiled function %.*s:%.*s workgroup size "
          "%" PRIu32 "x%" PRIu32 "x%" PRIu32
          " disagrees with evaluated launch size "
          "%" PRIu32 "x%" PRIu32 "x%" PRIu32,
          (int)module_path.size, module_path.data, (int)function_name.size,
          function_name.data, function_info.workgroup_size[0],
          function_info.workgroup_size[1], function_info.workgroup_size[2],
          dispatch_config->workgroup_size[0],
          dispatch_config->workgroup_size[1],
          dispatch_config->workgroup_size[2]);
    }
  }
  memset(dispatch_config->workgroup_size, 0,
         sizeof(dispatch_config->workgroup_size));
  return iree_ok_status();
}

static const loomc_artifact_t* qwen_loom_jit_find_executable_artifact(
    const loomc_result_t* result) {
  if (!result) return NULL;
  const loomc_string_view_t expected_format =
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact && artifact->kind == LOOMC_ARTIFACT_KIND_EXECUTABLE &&
        loomc_string_view_equal(artifact->format, expected_format)) {
      return artifact;
    }
  }
  return NULL;
}

static iree_status_t qwen_loom_executable_create(
    iree_hal_executable_t* hal_executable,
    iree_hal_executable_function_t function,
    iree_hal_dispatch_config_t dispatch_config, iree_allocator_t host_allocator,
    qwen_loom_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(hal_executable);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

  qwen_loom_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*executable), (void**)&executable));
  memset(executable, 0, sizeof(*executable));
  iree_atomic_ref_count_init(&executable->ref_count);
  executable->host_allocator = host_allocator;
  executable->hal_executable = hal_executable;
  iree_hal_executable_retain(hal_executable);
  executable->function = function;
  executable->dispatch_config = dispatch_config;
  *out_executable = executable;
  return iree_ok_status();
}

static void qwen_loom_jit_destroy(qwen_loom_jit_t* jit) {
  iree_allocator_t host_allocator = jit->host_allocator;
  qwen_loom_compile_pool_deinitialize(&jit->compile_pool);
  for (iree_host_size_t i = 0; i < jit->worker_count; ++i) {
    loomc_workspace_release(jit->worker_workspaces[i]);
  }
  iree_allocator_free(host_allocator, jit->worker_workspaces);
  for (iree_host_size_t i = 0; i < jit->entry_count; ++i) {
    qwen_loom_jit_entry_deinitialize(&jit->entries[i], host_allocator);
  }
  iree_allocator_free(host_allocator, jit->entries);
  loomc_pass_program_release(jit->pass_program);
  loomc_compiler_release(jit->compiler);
  loomc_linker_release(jit->linker);
  loomc_target_profile_release(jit->target_profile);
  loomc_context_release(jit->context);
  loomc_target_environment_release(jit->target_environment);
  iree_hal_device_release(jit->device);
  iree_slim_mutex_deinitialize(&jit->cache_mutex);
  iree_slim_mutex_deinitialize(&jit->operation_mutex);
  iree_allocator_free(host_allocator, jit);
}

iree_status_t qwen_loom_jit_create(const qwen_loom_jit_options_t* options,
                                   iree_allocator_t host_allocator,
                                   qwen_loom_jit_t** out_jit) {
  IREE_ASSERT_ARGUMENT(out_jit);
  *out_jit = NULL;
  IREE_RETURN_IF_ERROR(qwen_loom_jit_validate_create_options(options));

  qwen_loom_jit_t* jit = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*jit), (void**)&jit);
  if (iree_status_is_ok(status)) {
    memset(jit, 0, sizeof(*jit));
    iree_atomic_ref_count_init(&jit->ref_count);
    jit->host_allocator = host_allocator;
    jit->device = options->device;
    iree_hal_device_retain(jit->device);
    jit->queue_affinity = options->queue_affinity;
    jit->entry_limit = options->entry_limit;
    jit->runtime_globals = iree_any_bit_set(options->sanitizer_checks,
                                            LOOMC_SANITIZER_CHECK_ACCESS)
                               ? LOOMC_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG
                               : LOOMC_AMDGPU_RUNTIME_GLOBAL_NONE;
    iree_slim_mutex_initialize(&jit->operation_mutex);
    iree_slim_mutex_initialize(&jit->cache_mutex);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, jit->entry_limit,
                                         sizeof(jit->entries[0]),
                                         (void**)&jit->entries);
  }
  if (iree_status_is_ok(status)) {
    memset(jit->entries, 0, jit->entry_limit * sizeof(jit->entries[0]));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_target_environment_create_amdgpu(
        loomc_allocator_from_iree(host_allocator), &jit->target_environment));
  }
  if (iree_status_is_ok(status)) {
    const loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .next = NULL,
        .target_environment = jit->target_environment,
    };
    const loomc_context_options_t context_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(context_options),
        .next = &target_options,
    };
    status = iree_status_from_loomc(loomc_context_create(
        &context_options, loomc_allocator_from_iree(host_allocator),
        &jit->context));
  }

  loomc_result_t* target_profile_result = NULL;
  if (iree_status_is_ok(status)) {
    const loomc_amdgpu_iree_hal_profile_options_t profile_options = {
        .type = LOOMC_STRUCTURE_TYPE_AMDGPU_IREE_HAL_PROFILE_OPTIONS,
        .structure_size = sizeof(profile_options),
        .next = NULL,
        .identifier = loomc_make_cstring_view("qwen-live-amdgpu"),
        .device = jit->device,
        .physical_device_affinity = 0,
    };
    status = iree_status_from_loomc(loomc_target_profile_create_amdgpu_iree_hal(
        jit->target_environment, &profile_options,
        loomc_allocator_from_iree(host_allocator), &jit->target_profile,
        &target_profile_result));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_loom_jit_require_result(IREE_SV("target profile"), IREE_SV("qwen"),
                                     IREE_SV("device"), target_profile_result);
  }
  loomc_result_release(target_profile_result);

  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_linker_create(
        jit->context, NULL, loomc_allocator_from_iree(host_allocator),
        &jit->linker));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_compiler_create(
        jit->context, NULL, loomc_allocator_from_iree(host_allocator),
        &jit->compiler));
  }

  loomc_result_t* pass_program_result = NULL;
  if (iree_status_is_ok(status)) {
    const loomc_sanitizer_options_t sanitizer_options = {
        .type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
        .structure_size = sizeof(sanitizer_options),
        .next = NULL,
        .checks = options->sanitizer_checks,
        .flags = LOOMC_SANITIZER_FLAG_NONE,
        .reporting_mode = LOOMC_SANITIZER_REPORTING_MODE_TRAP,
    };
    const loomc_target_pipeline_options_t pipeline_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
        .structure_size = sizeof(pipeline_options),
        .next =
            options->sanitizer_checks ? (const void*)&sanitizer_options : NULL,
        .identifier = loomc_make_cstring_view("qwen-target-prepared-low"),
        .kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
        .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
        .source_to_low_max_errors = 20,
    };
    status =
        iree_status_from_loomc(loomc_pass_program_create_from_target_pipeline(
            jit->context, &pipeline_options,
            loomc_allocator_from_iree(host_allocator), &jit->pass_program,
            &pass_program_result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_loom_jit_require_result(
        IREE_SV("target pipeline"), IREE_SV("qwen"), IREE_SV("prepared-low"),
        pass_program_result);
  }
  loomc_result_release(pass_program_result);

  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, options->worker_count,
                                         sizeof(jit->worker_workspaces[0]),
                                         (void**)&jit->worker_workspaces);
  }
  if (iree_status_is_ok(status)) {
    memset(jit->worker_workspaces, 0,
           options->worker_count * sizeof(jit->worker_workspaces[0]));
    jit->worker_count = options->worker_count;
  }
  for (iree_host_size_t i = 0;
       i < jit->worker_count && iree_status_is_ok(status); ++i) {
    status = iree_status_from_loomc(
        loomc_workspace_create(NULL, loomc_allocator_from_iree(host_allocator),
                               &jit->worker_workspaces[i]));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_loom_compile_pool_initialize(
        jit->worker_count, host_allocator, &jit->compile_pool);
  }

  if (iree_status_is_ok(status)) {
    *out_jit = jit;
  } else if (jit) {
    qwen_loom_jit_destroy(jit);
  }
  return status;
}

void qwen_loom_jit_retain(qwen_loom_jit_t* jit) {
  if (!jit) return;
  iree_atomic_ref_count_inc(&jit->ref_count);
}

void qwen_loom_jit_release(qwen_loom_jit_t* jit) {
  if (jit && iree_atomic_ref_count_dec(&jit->ref_count) == 1) {
    qwen_loom_jit_destroy(jit);
  }
}

typedef struct qwen_loom_jit_linked_code_group_t {
  // Selectively linked source module retained through launch evaluation and
  // compilation.
  loomc_module_t* module;
  // Storage backing |root_symbol|.
  char root_symbol_storage[QWEN_LOOM_JIT_ROOT_SYMBOL_CAPACITY];
  // Exported root selected for the complete code group.
  loomc_string_view_t root_symbol;
} qwen_loom_jit_linked_code_group_t;

static void qwen_loom_jit_linked_code_group_deinitialize(
    qwen_loom_jit_linked_code_group_t* linked_group) {
  loomc_module_release(linked_group->module);
  memset(linked_group, 0, sizeof(*linked_group));
}

static iree_status_t qwen_loom_jit_link_code_group(
    qwen_loom_jit_t* jit, loomc_workspace_t* workspace,
    const qwen_loom_jit_prepare_options_t* options,
    qwen_loom_jit_linked_code_group_t* out_linked_group) {
  memset(out_linked_group, 0, sizeof(*out_linked_group));
  loomc_config_binding_t* config_bindings = NULL;
  loomc_source_t* source = NULL;
  loomc_link_index_builder_t* link_index_builder = NULL;
  loomc_link_index_t* link_index = NULL;
  loomc_result_t* link_index_result = NULL;
  loomc_result_t* link_result = NULL;
  iree_status_t status = iree_ok_status();

  if (options->config_binding_count != 0) {
    status = iree_allocator_malloc_array(
        jit->host_allocator, options->config_binding_count,
        sizeof(config_bindings[0]), (void**)&config_bindings);
  }
  for (iree_host_size_t i = 0;
       i < options->config_binding_count && iree_status_is_ok(status); ++i) {
    config_bindings[i] = (loomc_config_binding_t){
        .key = loomc_string_view_from_iree(options->config_bindings[i].key),
        .value = loomc_string_view_from_iree(options->config_bindings[i].value),
    };
  }

  if (iree_status_is_ok(status)) {
    const loomc_source_options_t source_options = {
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        .structure_size = sizeof(source_options),
        .next = NULL,
        .format = LOOMC_SOURCE_FORMAT_TEXT,
        .identifier = loomc_string_view_from_iree(
            options->source_module->source_identifier),
        .contents =
            loomc_byte_span_from_iree(options->source_module->source_contents),
        .storage = LOOMC_SOURCE_STORAGE_BORROWED,
        .release = NULL,
        .release_user_data = NULL,
    };
    status = iree_status_from_loomc(loomc_source_create(
        &source_options, loomc_allocator_from_iree(jit->host_allocator),
        &source));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_link_index_builder_create(
        jit->context, NULL, loomc_allocator_from_iree(jit->host_allocator),
        &link_index_builder));
  }
  if (iree_status_is_ok(status)) {
    const loomc_link_index_source_options_t source_options = {
        .provider_name = loomc_string_view_from_iree(
            options->source_module->source_identifier),
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
    status = qwen_loom_jit_require_result(
        IREE_SV("link index"), options->source_module->module_path,
        options->function_name, link_index_result);
  }

  if (iree_status_is_ok(status)) {
    status = qwen_loom_jit_make_root_symbol(
        options->function_name, out_linked_group->root_symbol_storage,
        IREE_ARRAYSIZE(out_linked_group->root_symbol_storage),
        &out_linked_group->root_symbol);
  }
  if (iree_status_is_ok(status)) {
    const loomc_link_options_t link_options = {
        .type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
        .structure_size = sizeof(link_options),
        .next = NULL,
        .link_index = link_index,
        .module_name =
            loomc_string_view_from_iree(options->source_module->module_path),
        .root_symbols = &out_linked_group->root_symbol,
        .root_symbol_count = 1,
        .flags = LOOMC_LINK_FLAG_STRIP_TEST_SYMBOLS,
        .config =
            {
                .bindings = config_bindings,
                .binding_count = options->config_binding_count,
                .json_object = loomc_string_view_empty(),
                .flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
            },
    };
    status = iree_status_from_loomc(
        loomc_link_module(jit->linker, workspace, &link_options,
                          &out_linked_group->module, &link_result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_loom_jit_require_result(IREE_SV("selective link"),
                                          options->source_module->module_path,
                                          options->function_name, link_result);
  }

  loomc_result_release(link_result);
  loomc_result_release(link_index_result);
  loomc_link_index_release(link_index);
  loomc_link_index_builder_release(link_index_builder);
  loomc_source_release(source);
  iree_allocator_free(jit->host_allocator, config_bindings);
  if (!iree_status_is_ok(status)) {
    qwen_loom_jit_linked_code_group_deinitialize(out_linked_group);
  }
  return status;
}

static iree_status_t qwen_loom_jit_evaluate_dispatch_config(
    qwen_loom_jit_t* jit, loomc_workspace_t* workspace,
    const qwen_loom_jit_linked_code_group_t* linked_group,
    const qwen_loom_jit_prepare_options_t* options,
    iree_hal_dispatch_config_t* out_dispatch_config) {
  const loomc_launch_config_eval_options_t launch_options = {
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_EVAL_OPTIONS,
      .structure_size = sizeof(launch_options),
      .next = NULL,
      .function_symbol = linked_group->root_symbol,
      .config = {0},
      .workload_arguments = options->workload_arguments,
      .workload_argument_count = options->workload_argument_count,
      .required_fields = LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
                         LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE,
  };
  loomc_launch_config_t launch_config = {
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      .structure_size = sizeof(launch_config),
  };
  loomc_result_t* launch_config_result = NULL;
  iree_status_t status =
      iree_status_from_loomc(loomc_module_evaluate_launch_config(
          linked_group->module, workspace, &launch_options,
          loomc_allocator_from_iree(jit->host_allocator), &launch_config,
          &launch_config_result));
  if (iree_status_is_ok(status)) {
    status = qwen_loom_jit_require_result(
        IREE_SV("launch config"), options->source_module->module_path,
        options->function_name, launch_config_result);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_loom_jit_make_dispatch_config(
        &launch_config, options->source_module->module_path,
        options->function_name, out_dispatch_config);
  }
  loomc_result_release(launch_config_result);
  return status;
}

static iree_status_t qwen_loom_jit_compile_code_group(
    qwen_loom_jit_t* jit, loomc_workspace_t* workspace,
    const qwen_loom_jit_prepare_options_t* options,
    qwen_loom_jit_linked_code_group_t* linked_group,
    iree_hal_executable_t** out_hal_executable,
    iree_hal_executable_function_t* out_function) {
  *out_hal_executable = NULL;
  memset(out_function, 0, sizeof(*out_function));
  loomc_result_t* compile_result = NULL;
  loomc_result_t* emit_result = NULL;
  iree_hal_executable_t* hal_executable = NULL;
  iree_hal_executable_function_t function = {0};

  const loomc_target_specialization_t specialization = {
      .function_symbol = linked_group->root_symbol,
      .target_profile = jit->target_profile,
  };
  const loomc_target_specialization_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .structure_size = sizeof(target_options),
      .next = NULL,
      .specializations = &specialization,
      .specialization_count = 1,
  };
  const loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = &target_options,
      .module_name =
          loomc_string_view_from_iree(options->source_module->module_path),
      .artifact_flags = 0,
      .config = {0},
  };
  iree_status_t status = iree_status_from_loomc(loomc_compile_module(
      jit->compiler, workspace, jit->pass_program, linked_group->module,
      &compile_options, loomc_allocator_from_iree(jit->host_allocator),
      &compile_result));
  if (iree_status_is_ok(status)) {
    status = qwen_loom_jit_require_result(
        IREE_SV("compile"), options->source_module->module_path,
        options->function_name, compile_result);
  }

  if (iree_status_is_ok(status)) {
    const loomc_amdgpu_emit_options_t amdgpu_options = {
        .type = LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
        .structure_size = sizeof(amdgpu_options),
        .next = NULL,
        .runtime_globals = jit->runtime_globals,
    };
    const loomc_emit_options_t emit_options = {
        .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
        .structure_size = sizeof(emit_options),
        .next = &amdgpu_options,
        .artifact_format =
            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
        .identifier = loomc_string_view_empty(),
        .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
    };
    status = iree_status_from_loomc(loomc_emit_module(
        jit->target_environment, workspace, linked_group->module, &emit_options,
        loomc_allocator_from_iree(jit->host_allocator), &emit_result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_loom_jit_require_result(IREE_SV("emit"),
                                          options->source_module->module_path,
                                          options->function_name, emit_result);
  }

  const loomc_artifact_t* executable_artifact = NULL;
  if (iree_status_is_ok(status)) {
    executable_artifact = qwen_loom_jit_find_executable_artifact(emit_result);
    if (!executable_artifact) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "Loom did not emit an AMDGPU HSACO executable artifact");
    }
  }

  const iree_hal_executable_target_t* executable_target = NULL;
  if (iree_status_is_ok(status)) {
    const iree_hal_executable_target_selection_t selection = {
        .family = IREE_SV("amdgpu"),
        .target_key = iree_string_view_empty(),
        .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
        .physical_device_affinity = 0,
    };
    const iree_hal_executable_target_selection_result_t selection_result =
        iree_hal_device_spec_select_executable_target(
            iree_hal_device_spec(jit->device), &selection);
    if (selection_result.outcome ==
        IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
      status =
          iree_make_status(IREE_STATUS_NOT_FOUND,
                           "HAL device has no exact AMDGPU executable target");
    } else if (selection_result.outcome ==
               IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "HAL device has multiple exact AMDGPU executable targets");
    } else {
      executable_target = selection_result.target;
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_executable_load_params_t load_params;
    iree_hal_executable_load_params_initialize(&load_params);
    load_params.executable_data =
        iree_const_byte_span_from_loomc(executable_artifact->contents);
    status = iree_hal_device_load_executable(jit->device, jit->queue_affinity,
                                             executable_target, &load_params,
                                             &hal_executable);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_lookup_function_by_name(
        hal_executable, options->function_name, &function);
  }
  if (iree_status_is_ok(status)) {
    *out_hal_executable = hal_executable;
    hal_executable = NULL;
    *out_function = function;
  }

  iree_hal_executable_release(hal_executable);
  loomc_result_release(emit_result);
  loomc_result_release(compile_result);
  return status;
}

typedef struct qwen_loom_jit_batch_item_t {
  // Exact prepare options borrowed from the public request array.
  const qwen_loom_jit_prepare_options_t* options;
  // First item with the same exact identity.
  iree_host_size_t canonical_item_ordinal;
  // Unique code group assigned to this cache miss.
  iree_host_size_t code_group_ordinal;
  // True when this canonical item requires a new exact cache entry.
  bool cache_miss;
  // Static launch geometry evaluated from this exact workload.
  iree_hal_dispatch_config_t dispatch_config;
  // Prepared executable owned by this canonical item.
  qwen_loom_executable_t* executable;
  // Fully initialized entry awaiting atomic cache publication.
  qwen_loom_jit_entry_t staged_entry;
} qwen_loom_jit_batch_item_t;

typedef struct qwen_loom_jit_code_group_t {
  // First canonical cache-miss item with this code identity.
  iree_host_size_t leader_item_ordinal;
  // Compatible code retained from the exact-entry cache, when available.
  qwen_loom_executable_t* cached_code_executable;
} qwen_loom_jit_code_group_t;

typedef struct qwen_loom_jit_batch_t {
  // JIT executing the batch.
  qwen_loom_jit_t* jit;
  // Number of public request items.
  iree_host_size_t item_count;
  // Per-request classification and temporary ownership.
  qwen_loom_jit_batch_item_t* items;
  // Number of independent code groups.
  iree_host_size_t code_group_count;
  // Per-code-group cache state.
  qwen_loom_jit_code_group_t* code_groups;
} qwen_loom_jit_batch_t;

static void qwen_loom_jit_batch_deinitialize(qwen_loom_jit_batch_t* batch) {
  if (!batch) return;
  if (batch->items) {
    for (iree_host_size_t i = 0; i < batch->item_count; ++i) {
      qwen_loom_jit_entry_deinitialize(&batch->items[i].staged_entry,
                                       batch->jit->host_allocator);
      qwen_loom_executable_release(batch->items[i].executable);
    }
  }
  if (batch->code_groups) {
    for (iree_host_size_t i = 0; i < batch->code_group_count; ++i) {
      qwen_loom_executable_release(
          batch->code_groups[i].cached_code_executable);
    }
  }
  iree_allocator_free(batch->jit->host_allocator, batch->code_groups);
  iree_allocator_free(batch->jit->host_allocator, batch->items);
  memset(batch, 0, sizeof(*batch));
}

static iree_status_t qwen_loom_jit_batch_initialize(
    qwen_loom_jit_t* jit, iree_host_size_t request_count,
    const qwen_loom_jit_prepare_options_t* requests,
    qwen_loom_jit_batch_t* out_batch) {
  memset(out_batch, 0, sizeof(*out_batch));
  out_batch->jit = jit;
  out_batch->item_count = request_count;

  iree_status_t status = iree_allocator_malloc_array(
      jit->host_allocator, request_count, sizeof(out_batch->items[0]),
      (void**)&out_batch->items);
  if (iree_status_is_ok(status)) {
    memset(out_batch->items, 0, request_count * sizeof(out_batch->items[0]));
    status = iree_allocator_malloc_array(jit->host_allocator, request_count,
                                         sizeof(out_batch->code_groups[0]),
                                         (void**)&out_batch->code_groups);
  }
  if (iree_status_is_ok(status)) {
    memset(out_batch->code_groups, 0,
           request_count * sizeof(out_batch->code_groups[0]));
  }
  if (!iree_status_is_ok(status)) {
    qwen_loom_jit_batch_deinitialize(out_batch);
    return status;
  }

  iree_slim_mutex_lock(&jit->cache_mutex);
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    qwen_loom_jit_batch_item_t* item = &out_batch->items[i];
    item->options = &requests[i];
    item->canonical_item_ordinal = i;
    item->code_group_ordinal = IREE_HOST_SIZE_MAX;

    for (iree_host_size_t j = 0; j < i; ++j) {
      if (qwen_loom_jit_prepare_options_match_exact(item->options,
                                                    &requests[j])) {
        item->canonical_item_ordinal =
            out_batch->items[j].canonical_item_ordinal;
        break;
      }
    }
    if (item->canonical_item_ordinal != i) continue;

    qwen_loom_jit_entry_t* exact_entry =
        qwen_loom_jit_lookup_exact(jit, item->options);
    if (exact_entry) {
      item->executable = exact_entry->executable;
      qwen_loom_executable_retain(item->executable);
      continue;
    }

    item->cache_miss = true;
    for (iree_host_size_t j = 0; j < out_batch->code_group_count; ++j) {
      const qwen_loom_jit_batch_item_t* leader =
          &out_batch->items[out_batch->code_groups[j].leader_item_ordinal];
      if (qwen_loom_jit_prepare_options_match_code(item->options,
                                                   leader->options)) {
        item->code_group_ordinal = j;
        break;
      }
    }
    if (item->code_group_ordinal != IREE_HOST_SIZE_MAX) continue;

    item->code_group_ordinal = out_batch->code_group_count;
    qwen_loom_jit_code_group_t* code_group =
        &out_batch->code_groups[out_batch->code_group_count++];
    code_group->leader_item_ordinal = i;
    qwen_loom_jit_entry_t* code_entry =
        qwen_loom_jit_lookup_code(jit, item->options);
    if (code_entry) {
      code_group->cached_code_executable = code_entry->executable;
      qwen_loom_executable_retain(code_group->cached_code_executable);
    }
  }
  iree_slim_mutex_unlock(&jit->cache_mutex);
  return iree_ok_status();
}

static iree_status_t qwen_loom_jit_batch_prepare_code_group(
    void* user_data, iree_host_size_t worker_ordinal,
    iree_host_size_t code_group_ordinal) {
  qwen_loom_jit_batch_t* batch = (qwen_loom_jit_batch_t*)user_data;
  qwen_loom_jit_t* jit = batch->jit;
  qwen_loom_jit_code_group_t* code_group =
      &batch->code_groups[code_group_ordinal];
  const qwen_loom_jit_prepare_options_t* leader_options =
      batch->items[code_group->leader_item_ordinal].options;
  loomc_workspace_t* workspace = jit->worker_workspaces[worker_ordinal];

  qwen_loom_jit_linked_code_group_t linked_group;
  iree_status_t status = qwen_loom_jit_link_code_group(
      jit, workspace, leader_options, &linked_group);
  for (iree_host_size_t i = 0;
       i < batch->item_count && iree_status_is_ok(status); ++i) {
    qwen_loom_jit_batch_item_t* item = &batch->items[i];
    if (item->canonical_item_ordinal != i || !item->cache_miss ||
        item->code_group_ordinal != code_group_ordinal) {
      continue;
    }
    status = qwen_loom_jit_evaluate_dispatch_config(
        jit, workspace, &linked_group, item->options, &item->dispatch_config);
  }

  iree_hal_executable_t* hal_executable = NULL;
  iree_hal_executable_function_t function = {0};
  if (iree_status_is_ok(status) && code_group->cached_code_executable) {
    hal_executable = code_group->cached_code_executable->hal_executable;
    iree_hal_executable_retain(hal_executable);
    function = code_group->cached_code_executable->function;
  } else if (iree_status_is_ok(status)) {
    status = qwen_loom_jit_compile_code_group(jit, workspace, leader_options,
                                              &linked_group, &hal_executable,
                                              &function);
  }

  for (iree_host_size_t i = 0;
       i < batch->item_count && iree_status_is_ok(status); ++i) {
    qwen_loom_jit_batch_item_t* item = &batch->items[i];
    if (item->canonical_item_ordinal != i || !item->cache_miss ||
        item->code_group_ordinal != code_group_ordinal) {
      continue;
    }
    status = qwen_loom_jit_use_executable_workgroup_size(
        hal_executable, function, item->options->source_module->module_path,
        item->options->function_name, &item->dispatch_config);
    if (iree_status_is_ok(status)) {
      status = qwen_loom_executable_create(
          hal_executable, function, item->dispatch_config, jit->host_allocator,
          &item->executable);
    }
  }

  iree_hal_executable_release(hal_executable);
  qwen_loom_jit_linked_code_group_deinitialize(&linked_group);
  return status;
}

static iree_status_t qwen_loom_jit_batch_stage_entries(
    qwen_loom_jit_batch_t* batch) {
  for (iree_host_size_t i = 0; i < batch->item_count; ++i) {
    qwen_loom_jit_batch_item_t* item = &batch->items[i];
    if (item->canonical_item_ordinal != i || !item->cache_miss) continue;
    IREE_RETURN_IF_ERROR(qwen_loom_jit_entry_initialize(
        item->options, item->executable, batch->jit->host_allocator,
        &item->staged_entry));
  }
  return iree_ok_status();
}

static void qwen_loom_jit_batch_publish_entries(qwen_loom_jit_batch_t* batch) {
  iree_slim_mutex_lock(&batch->jit->cache_mutex);
  for (iree_host_size_t i = 0; i < batch->item_count; ++i) {
    qwen_loom_jit_batch_item_t* item = &batch->items[i];
    if (item->canonical_item_ordinal == i && item->cache_miss) {
      qwen_loom_jit_store_initialized_entry(batch->jit, &item->staged_entry);
    }
  }
  iree_slim_mutex_unlock(&batch->jit->cache_mutex);
}

iree_status_t qwen_loom_jit_prepare_batch(
    qwen_loom_jit_t* jit, iree_host_size_t request_count,
    const qwen_loom_jit_prepare_options_t* requests,
    qwen_loom_executable_t** out_executables) {
  IREE_ASSERT_ARGUMENT(jit);
  IREE_ASSERT_ARGUMENT(out_executables);
  if (request_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen Loom prepare batch must not be empty");
  }
  if (!requests) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen Loom prepare requests are required");
  }
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    out_executables[i] = NULL;
  }
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    IREE_RETURN_IF_ERROR(qwen_loom_jit_validate_prepare_options(&requests[i]));
  }

  iree_slim_mutex_lock(&jit->operation_mutex);
  qwen_loom_jit_batch_t batch;
  iree_status_t status =
      qwen_loom_jit_batch_initialize(jit, request_count, requests, &batch);
  if (iree_status_is_ok(status) && batch.code_group_count != 0) {
    status = qwen_loom_compile_pool_run_batch(
        &jit->compile_pool, batch.code_group_count,
        qwen_loom_jit_batch_prepare_code_group, &batch);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_loom_jit_batch_stage_entries(&batch);
  }
  if (iree_status_is_ok(status)) {
    qwen_loom_jit_batch_publish_entries(&batch);
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      qwen_loom_executable_t* executable =
          batch.items[batch.items[i].canonical_item_ordinal].executable;
      qwen_loom_executable_retain(executable);
      out_executables[i] = executable;
    }
  }
  qwen_loom_jit_batch_deinitialize(&batch);
  iree_slim_mutex_unlock(&jit->operation_mutex);
  return status;
}

iree_status_t qwen_loom_jit_prepare(
    qwen_loom_jit_t* jit, const qwen_loom_jit_prepare_options_t* options,
    qwen_loom_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(jit);
  IREE_ASSERT_ARGUMENT(out_executable);
  return qwen_loom_jit_prepare_batch(jit, 1, options, out_executable);
}

iree_host_size_t qwen_loom_jit_entry_count(qwen_loom_jit_t* jit) {
  if (!jit) return 0;
  iree_slim_mutex_lock(&jit->cache_mutex);
  const iree_host_size_t entry_count = jit->entry_count;
  iree_slim_mutex_unlock(&jit->cache_mutex);
  return entry_count;
}

void qwen_loom_executable_retain(qwen_loom_executable_t* executable) {
  if (!executable) return;
  iree_atomic_ref_count_inc(&executable->ref_count);
}

static void qwen_loom_executable_destroy(qwen_loom_executable_t* executable) {
  iree_allocator_t host_allocator = executable->host_allocator;
  iree_hal_executable_release(executable->hal_executable);
  iree_allocator_free(host_allocator, executable);
}

void qwen_loom_executable_release(qwen_loom_executable_t* executable) {
  if (executable && iree_atomic_ref_count_dec(&executable->ref_count) == 1) {
    qwen_loom_executable_destroy(executable);
  }
}

iree_hal_executable_t* qwen_loom_executable_hal_executable(
    const qwen_loom_executable_t* executable) {
  return executable ? executable->hal_executable : NULL;
}

iree_hal_executable_function_t qwen_loom_executable_function(
    const qwen_loom_executable_t* executable) {
  iree_hal_executable_function_t function = {0};
  if (executable) function = executable->function;
  return function;
}

iree_hal_dispatch_config_t qwen_loom_executable_dispatch_config(
    const qwen_loom_executable_t* executable) {
  iree_hal_dispatch_config_t dispatch_config;
  memset(&dispatch_config, 0, sizeof(dispatch_config));
  if (executable) dispatch_config = executable->dispatch_config;
  return dispatch_config;
}
