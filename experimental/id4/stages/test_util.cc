// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/test_util.h"

#include <algorithm>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/local_sync/sync_device.h"

namespace id4::test {

std::string ToString(iree_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

bool ContainsKey(const std::vector<std::string>& keys, const char* key) {
  return std::find(keys.begin(), keys.end(), key) != keys.end();
}

static iree_status_t CaptureDiagnostics(
    void* user_data, const id4_pipeline_diagnostic_event_t* event) {
  StageDiagnostics* diagnostics = static_cast<StageDiagnostics*>(user_data);
  ++diagnostics->count;
  diagnostics->keys.push_back(ToString(event->key));
  if (event->kind == ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_KERNEL) {
    ++diagnostics->kernel_event_count;
  }
  return iree_ok_status();
}

id4_pipeline_diagnostics_sink_t DiagnosticsSink(StageDiagnostics* diagnostics) {
  return (id4_pipeline_diagnostics_sink_t){
      // Callback used to capture events into StageDiagnostics.
      /*.emit=*/CaptureDiagnostics,
      // Caller-owned diagnostics structure.
      /*.user_data=*/diagnostics,
  };
}

iree_hal_device_group_t* CreateLocalSyncDeviceGroup() {
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      /*node_count=*/1, /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      &proactor_pool));

  iree_hal_allocator_t* device_allocator = nullptr;
  IREE_CHECK_OK(iree_hal_allocator_create_heap(
      IREE_SV("id4-stage-local-sync"), iree_allocator_system(),
      iree_allocator_system(), &device_allocator));

  iree_hal_sync_device_params_t sync_params;
  iree_hal_sync_device_params_initialize(&sync_params);
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;

  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_sync_device_create(
      IREE_SV("id4-stage-local-sync"), &sync_params, &create_params,
      /*loader_count=*/0, /*loaders=*/nullptr, device_allocator,
      iree_allocator_system(), &device);
  iree_hal_allocator_release(device_allocator);
  iree_async_proactor_pool_release(proactor_pool);
  IREE_CHECK_OK(status);

  iree_async_frontier_tracker_options_t tracker_options =
      iree_async_frontier_tracker_options_default();
  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      tracker_options, iree_allocator_system(), &frontier_tracker));

  iree_hal_device_group_t* device_group = nullptr;
  status = iree_hal_device_group_create_from_device(
      device, frontier_tracker, iree_allocator_system(), &device_group);
  iree_async_frontier_tracker_release(frontier_tracker);
  iree_hal_device_release(device);
  IREE_CHECK_OK(status);
  return device_group;
}

}  // namespace id4::test
