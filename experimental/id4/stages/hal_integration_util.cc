// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/hal_integration_util.h"

#include <cstring>
#include <mutex>

#include "iree/hal/drivers/init.h"

namespace id4::test {

iree_string_view_t StringView(const std::string& value) {
  return iree_make_string_view(value.data(), value.size());
}

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

bool ParseStringArgument(int argc, char** argv, const char* prefix,
                         std::string* out_value) {
  const iree_host_size_t prefix_length = std::strlen(prefix);
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], prefix, prefix_length) != 0) continue;
    *out_value = argv[i] + prefix_length;
    return !out_value->empty();
  }
  return false;
}

}  // namespace id4::test
