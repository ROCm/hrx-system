// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/hal_integration_util.h"

#include <cstring>

#include "iree/async/frontier_tracker.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_status_t CreateLocalSyncDeviceGroup(
    iree_hal_device_group_t** out_device_group) {
  IREE_ASSERT_ARGUMENT(out_device_group);
  *out_device_group = nullptr;

  iree_async_proactor_pool_t* proactor_pool = nullptr;
  iree_status_t status = iree_async_proactor_pool_create(
      /*node_count=*/1, /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      &proactor_pool);

  iree_hal_allocator_t* device_allocator = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(
        IREE_SV("id4-hal-integration-util-local-sync"), iree_allocator_system(),
        iree_allocator_system(), &device_allocator);
  }

  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    iree_hal_sync_device_params_t sync_params;
    iree_hal_sync_device_params_initialize(&sync_params);
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    status = iree_hal_sync_device_create(
        IREE_SV("id4-hal-integration-util-local-sync"), &sync_params,
        &create_params, /*loader_count=*/0, /*loaders=*/nullptr,
        device_allocator, iree_allocator_system(), &device);
  }

  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  if (iree_status_is_ok(status)) {
    iree_async_frontier_tracker_options_t tracker_options =
        iree_async_frontier_tracker_options_default();
    status = iree_async_frontier_tracker_create(
        tracker_options, iree_allocator_system(), &frontier_tracker);
  }

  iree_hal_device_group_t* device_group = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_create_from_device(
        device, frontier_tracker, iree_allocator_system(), &device_group);
  }

  iree_async_frontier_tracker_release(frontier_tracker);
  iree_hal_device_release(device);
  iree_hal_allocator_release(device_allocator);
  iree_async_proactor_pool_release(proactor_pool);
  if (iree_status_is_ok(status)) {
    *out_device_group = device_group;
  } else {
    iree_hal_device_group_release(device_group);
  }
  return status;
}

TEST(HalIntegrationUtilTest, ComparesExpectedSliceAgainstFullActualBinding) {
  id4::test::HalDeviceGroupRef device_group;
  IREE_ASSERT_OK(CreateLocalSyncDeviceGroup(device_group.out()));
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group.get(), /*device_index=*/0);

  constexpr float kActualValues[12] = {
      1.0f, 2.0f,  3.0f,  4.0f,  //
      5.0f, 6.0f,  7.0f,  8.0f,  //
      9.0f, 10.0f, 11.0f, 12.0f,
  };
  id4::test::OwningRef<iree_hal_buffer_t, iree_hal_buffer_release> buffer;
  iree_hal_buffer_params_t params;
  std::memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                 IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE;
  params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), params, sizeof(kActualValues),
      buffer.out()));
  iree_hal_buffer_binding_t binding = {
      // Full actual tensor buffer.
      /*.buffer=*/buffer.get(),
      // Full actual tensor begins at the buffer base.
      /*.offset=*/0,
      // Full actual tensor byte length.
      /*.length=*/sizeof(kActualValues),
  };

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      update_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
                                           update_semaphore.out()));
  uint64_t update_value = 0;
  IREE_ASSERT_OK(id4::test::QueueUpdateBinding(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, &binding, kActualValues,
      sizeof(kActualValues), update_semaphore.get(), &update_value));
  id4::test::SemaphoreListStorage wait;
  wait.semaphore = update_semaphore.get();
  wait.payload_value = update_value;

  constexpr float kExpectedSlice[4] = {6.0f, 7.0f, 10.0f, 11.0f};
  id4::test::FixtureTensor expected_tensor;
  expected_tensor.name = "unit.slice";
  expected_tensor.role = "expected";
  expected_tensor.dtype = ID4_PIPELINE_TENSOR_DTYPE_F32;
  expected_tensor.shape.rank = 2;
  expected_tensor.shape.dims[0] = 2;
  expected_tensor.shape.dims[1] = 2;
  expected_tensor.source_shape.rank = 2;
  expected_tensor.source_shape.dims[0] = 3;
  expected_tensor.source_shape.dims[1] = 4;
  expected_tensor.slice_offsets.rank = 2;
  expected_tensor.slice_offsets.dims[0] = 1;
  expected_tensor.slice_offsets.dims[1] = 1;
  expected_tensor.absolute_tolerance = 0.0;
  expected_tensor.relative_tolerance = 0.0;
  expected_tensor.has_tolerance = true;
  const auto* expected_bytes = reinterpret_cast<const uint8_t*>(kExpectedSlice);
  expected_tensor.payload.assign(expected_bytes,
                                 expected_bytes + sizeof(kExpectedSlice));

  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, &binding, wait.list(),
      expected_tensor));
}

}  // namespace
