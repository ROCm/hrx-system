// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/hal_integration_util.h"

#include <mutex>

#include "iree/hal/drivers/init.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/scope_map.h"
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

static iree_status_t RegisterHalDriversOnce() {
  static std::mutex mutex;
  static bool is_registered = false;
  std::lock_guard<std::mutex> lock(mutex);
  if (is_registered) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));
  is_registered = true;
  return iree_ok_status();
}

iree_status_t CreateLiveHalDevice(iree_string_view_t device_uri,
                                  LiveHalDevice* out_device) {
  IREE_RETURN_IF_ERROR(RegisterHalDriversOnce());

  iree_async_proactor_pool_t* proactor_pool = nullptr;
  iree_status_t status = iree_async_proactor_pool_create(
      /*node_count=*/1, /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      &proactor_pool);
  if (iree_status_is_ok(status)) {
    out_device->proactor_pool.reset(proactor_pool);
  }

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = out_device->proactor_pool.get();
  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_device(iree_hal_driver_registry_default(),
                                    device_uri, &create_params,
                                    iree_allocator_system(), &device);
  }
  if (iree_status_is_ok(status)) {
    out_device->device.reset(device);
  }

  iree_async_frontier_tracker_options_t tracker_options =
      iree_async_frontier_tracker_options_default();
  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_async_frontier_tracker_create(
        tracker_options, iree_allocator_system(), &frontier_tracker);
  }
  if (iree_status_is_ok(status)) {
    out_device->frontier_tracker.reset(frontier_tracker);
  }

  iree_hal_device_group_t* device_group = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_create_from_device(
        out_device->device.get(), out_device->frontier_tracker.get(),
        iree_allocator_system(), &device_group);
  }
  if (iree_status_is_ok(status)) {
    out_device->device_group.reset(device_group);
  }
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
