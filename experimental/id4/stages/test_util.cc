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
#include "iree/hal/testing/mock_device.h"

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

class SharedLocalSyncDeviceGroup {
 public:
  SharedLocalSyncDeviceGroup() {
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

    status = iree_hal_device_group_create_from_device(
        device, frontier_tracker, iree_allocator_system(), &device_group_);
    iree_async_frontier_tracker_release(frontier_tracker);
    iree_hal_device_release(device);
    IREE_CHECK_OK(status);
  }

  ~SharedLocalSyncDeviceGroup() {
    iree_hal_device_group_release(device_group_);
  }

  iree_hal_device_group_t* Acquire() const {
    iree_hal_device_group_retain(device_group_);
    return device_group_;
  }

 private:
  iree_hal_device_group_t* device_group_ = nullptr;
};

iree_hal_device_group_t* CreateLocalSyncDeviceGroup() {
  static const SharedLocalSyncDeviceGroup shared_device_group;
  return shared_device_group.Acquire();
}

class SharedGpuPlanningDeviceGroup {
 public:
  SharedGpuPlanningDeviceGroup() {
    iree_hal_physical_device_spec_t physical_device = {};
    physical_device.identity.display_name = IREE_SV("ID4 planning GPU");
    physical_device.identity.backend_path = IREE_SV("mock://id4-planning-gpu");
    physical_device.partition_count = 1;
    physical_device.physical_device_affinity = 1;

    iree_hal_device_identity_spec_t identity = {};
    identity.logical_device_id = IREE_SV("id4-planning-gpu");
    identity.display_name = IREE_SV("ID4 planning GPU");
    identity.driver_id = IREE_SV("mock");
    identity.driver_version = IREE_SV("1");
    identity.backend_id = IREE_SV("mock");
    identity.device_path = IREE_SV("mock://id4-planning-gpu");
    identity.vendor_name = IREE_SV("IREE");
    identity.physical_device_count = 1;
    identity.physical_devices = &physical_device;

    iree_hal_device_dispatch_spec_t dispatch = {};
    dispatch.launch.maximum_workgroup_invocations = 1024;
    dispatch.launch.maximum_workgroup_size[0] = 1024;
    dispatch.launch.maximum_workgroup_size[1] = 1024;
    dispatch.launch.maximum_workgroup_size[2] = 1024;
    dispatch.launch.maximum_workgroup_count[0] = UINT32_MAX;
    dispatch.launch.maximum_workgroup_count[1] = UINT32_MAX;
    dispatch.launch.maximum_workgroup_count[2] = UINT32_MAX;
    dispatch.subgroup.default_size = 32;
    dispatch.subgroup.minimum_size = 32;
    dispatch.subgroup.maximum_size = 32;
    dispatch.subgroup.supported_size_mask = 1ull << 32;
    dispatch.execution.unit_count = 48;
    dispatch.execution.group_count = 1;
    dispatch.execution.maximum_resident_workgroup_count = 64;
    dispatch.execution.maximum_resident_invocation_count = 2048;
    dispatch.execution.maximum_resident_subgroup_count = 64;
    dispatch.addressing.pointer_size_bits = 64;
    dispatch.addressing.address_space_bits = 64;

    iree_hal_device_spec_params_t spec_params = {};
    spec_params.identity = &identity;
    spec_params.dispatch = &dispatch;
    iree_hal_device_spec_t* device_spec = nullptr;
    IREE_CHECK_OK(iree_hal_device_spec_create(
        &spec_params, iree_allocator_system(), &device_spec));

    iree_hal_mock_device_options_t device_options;
    iree_hal_mock_device_options_initialize(&device_options);
    device_options.identifier = identity.logical_device_id;
    device_options.device_spec = device_spec;
    device_options.executable_cache_enabled = true;
    iree_hal_device_t* device = nullptr;
    iree_status_t status = iree_hal_mock_device_create(
        &device_options, iree_allocator_system(), &device);
    iree_hal_device_spec_release(device_spec);
    IREE_CHECK_OK(status);

    iree_async_frontier_tracker_options_t tracker_options =
        iree_async_frontier_tracker_options_default();
    iree_async_frontier_tracker_t* frontier_tracker = nullptr;
    IREE_CHECK_OK(iree_async_frontier_tracker_create(
        tracker_options, iree_allocator_system(), &frontier_tracker));
    status = iree_hal_device_group_create_from_device(
        device, frontier_tracker, iree_allocator_system(), &device_group_);
    iree_async_frontier_tracker_release(frontier_tracker);
    iree_hal_device_release(device);
    IREE_CHECK_OK(status);
  }

  ~SharedGpuPlanningDeviceGroup() {
    iree_hal_device_group_release(device_group_);
  }

  iree_hal_device_group_t* Acquire() const {
    iree_hal_device_group_retain(device_group_);
    return device_group_;
  }

 private:
  iree_hal_device_group_t* device_group_ = nullptr;
};

iree_hal_device_group_t* CreateGpuPlanningDeviceGroup() {
  static const SharedGpuPlanningDeviceGroup shared_device_group;
  return shared_device_group.Acquire();
}

}  // namespace id4::test
