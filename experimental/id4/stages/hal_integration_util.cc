// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/hal_integration_util.h"

#include "experimental/id4/kernels/embedded_loom_sources.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/scope_map.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/parameter_util.h"

namespace id4::test {

static iree_status_t CaptureDiagnostics(
    void* user_data, const id4_pipeline_diagnostic_event_t* event) {
  StageDiagnostics* diagnostics = static_cast<StageDiagnostics*>(user_data);
  ++diagnostics->event_count;
  if (event->kind == ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_KERNEL) {
    ++diagnostics->kernel_event_count;
  }
  return iree_ok_status();
}

id4_pipeline_diagnostics_sink_t DiagnosticsSink(StageDiagnostics* diagnostics) {
  return (id4_pipeline_diagnostics_sink_t){
      // Callback used to count diagnostics.
      /*.emit=*/CaptureDiagnostics,
      // Caller-owned diagnostics storage.
      /*.user_data=*/diagnostics,
  };
}

static iree_status_t RequireSingleDeviceFlag() {
  iree_string_view_list_t devices = iree_hal_device_flag_list();
  if (devices.count == 1) return iree_ok_status();
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "live stage integration tests require exactly one --device= flag; "
      "received %" PRIhsz,
      devices.count);
}

iree_status_t CreateLiveStageContextFromFlags(LiveStageContext* out_context) {
  IREE_ASSERT_ARGUMENT(out_context);
  IREE_RETURN_IF_ERROR(RequireSingleDeviceFlag());

  iree_async_proactor_pool_t* proactor_pool = nullptr;
  iree_status_t status = iree_async_proactor_pool_create(
      /*node_count=*/1, /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      &proactor_pool);
  if (iree_status_is_ok(status)) {
    out_context->proactor_pool.reset(proactor_pool);
  }

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = out_context->proactor_pool.get();
  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_device_from_flags(
        iree_hal_available_driver_registry(), iree_string_view_empty(),
        &create_params, iree_allocator_system(), &device);
  }
  if (iree_status_is_ok(status)) {
    out_context->device.reset(device);
  }

  iree_async_frontier_tracker_options_t tracker_options =
      iree_async_frontier_tracker_options_default();
  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_async_frontier_tracker_create(
        tracker_options, iree_allocator_system(), &frontier_tracker);
  }
  if (iree_status_is_ok(status)) {
    out_context->frontier_tracker.reset(frontier_tracker);
  }

  iree_hal_device_group_t* device_group = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_create_from_device(
        out_context->device.get(), out_context->frontier_tracker.get(),
        iree_allocator_system(), &device_group);
  }
  if (iree_status_is_ok(status)) {
    out_context->device_group.reset(device_group);
  }

  iree_hal_executable_cache_t* executable_cache = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_cache_create(
        out_context->device.get(), IREE_SV("id4.stage"), &executable_cache);
  }
  if (iree_status_is_ok(status)) {
    out_context->executable_cache.reset(executable_cache);
  }

  id4_pipeline_kernel_cache_create_options_t kernel_cache_options;
  memset(&kernel_cache_options, 0, sizeof(kernel_cache_options));
  kernel_cache_options.structure_size = sizeof(kernel_cache_options);
  kernel_cache_options.target_processor =
      id4_pipeline_kernel_cache_default_target_processor();
  id4_pipeline_kernel_cache_t* kernel_cache = nullptr;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_cache_create(
        &kernel_cache_options, iree_allocator_system(), &kernel_cache);
  }
  if (iree_status_is_ok(status)) {
    out_context->kernel_cache.reset(kernel_cache);
  }
  return status;
}

iree_status_t CreateEmbeddedKernelLibrary(
    id4_pipeline_kernel_library_t** out_library) {
  IREE_ASSERT_ARGUMENT(out_library);
  *out_library = nullptr;

  const iree_file_toc_t* toc = id4_kernel_embedded_loom_sources_create();
  const iree_host_size_t file_count = id4_kernel_embedded_loom_sources_size();
  id4_pipeline_kernel_source_file_t* source_files = nullptr;
  iree_status_t status = iree_ok_status();
  if (file_count != 0) {
    status = iree_allocator_malloc_array(
        iree_allocator_system(), file_count, sizeof(source_files[0]),
        reinterpret_cast<void**>(&source_files));
  }
  for (iree_host_size_t i = 0; i < file_count && iree_status_is_ok(status);
       ++i) {
    source_files[i] = id4_pipeline_kernel_source_file_t{
        // Source identifier formatted as <module_path>.loom.
        /*.source_identifier=*/iree_make_cstring_view(toc[i].name),
        // Embedded Loom source payload.
        /*.source_contents=*/
        iree_make_const_byte_span(toc[i].data, toc[i].size),
    };
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_library_create_from_source_files(
        file_count, source_files, iree_allocator_system(), out_library);
  }
  iree_allocator_free(iree_allocator_system(), source_files);
  return status;
}

iree_status_t CreateParameterProviderFromFlags(
    iree_string_view_t scope, iree_io_parameter_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = nullptr;

  iree_io_scope_map_t scope_map;
  iree_io_scope_map_initialize(iree_allocator_system(), &scope_map);
  iree_status_t status =
      iree_tooling_build_parameter_indices_from_flags(&scope_map);

  iree_io_parameter_index_t* index = nullptr;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < scope_map.count; ++i) {
      iree_io_scope_map_entry_t* entry = scope_map.entries[i];
      if (iree_string_view_equal(entry->scope, scope)) {
        index = entry->index;
        break;
      }
    }
    if (!index) {
      if (iree_string_view_is_empty(scope)) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "required anonymous parameter scope was not "
                                  "loaded; pass --parameters=<file>");
      } else {
        status = iree_make_status(
            IREE_STATUS_NOT_FOUND,
            "required parameter scope `%.*s` was not loaded; pass "
            "--parameters=%.*s=<file>",
            static_cast<int>(scope.size), scope.data,
            static_cast<int>(scope.size), scope.data);
      }
    }
  }
  if (iree_status_is_ok(status) && iree_io_parameter_index_count(index) == 0) {
    if (iree_string_view_is_empty(scope)) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "anonymous parameter scope was loaded with no parameters");
    } else {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "parameter scope `%.*s` was loaded with no parameters",
          static_cast<int>(scope.size), scope.data);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_provider_create(
        scope, index,
        IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
        iree_allocator_system(), out_provider);
  }
  iree_io_scope_map_deinitialize(&scope_map);
  return status;
}

}  // namespace id4::test
