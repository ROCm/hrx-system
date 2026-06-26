// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/hal_integration_util.h"

#include <cstring>

#include "iree/async/frontier_tracker.h"
#include "iree/base/internal/math.h"
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
  expected_tensor.tolerance.mode =
      id4::test::FixtureToleranceMode::kElementwise;
  expected_tensor.tolerance.absolute_tolerance = 0.0;
  expected_tensor.tolerance.relative_tolerance = 0.0;
  const auto* expected_bytes = reinterpret_cast<const uint8_t*>(kExpectedSlice);
  expected_tensor.payload.assign(expected_bytes,
                                 expected_bytes + sizeof(kExpectedSlice));

  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, &binding, wait.list(),
      expected_tensor));
}

TEST(HalIntegrationUtilTest,
     ComparesAggregateToleranceAgainstFullActualBinding) {
  id4::test::HalDeviceGroupRef device_group;
  IREE_ASSERT_OK(CreateLocalSyncDeviceGroup(device_group.out()));
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group.get(), /*device_index=*/0);

  constexpr float kActualValues[4] = {1.0f, 2.2f, 3.1f, 4.0f};
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

  constexpr float kExpectedValues[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  id4::test::FixtureTensor expected_tensor;
  expected_tensor.name = "unit.aggregate";
  expected_tensor.role = "expected";
  expected_tensor.dtype = ID4_PIPELINE_TENSOR_DTYPE_F32;
  expected_tensor.shape.rank = 1;
  expected_tensor.shape.dims[0] = 4;
  expected_tensor.source_shape = expected_tensor.shape;
  expected_tensor.slice_offsets.rank = 1;
  expected_tensor.tolerance.mode = id4::test::FixtureToleranceMode::kAggregate;
  expected_tensor.tolerance.mean_absolute_error = 0.1;
  expected_tensor.tolerance.p99_absolute_error = 0.25;
  expected_tensor.tolerance.max_absolute_error = 0.25;
  const auto* expected_bytes =
      reinterpret_cast<const uint8_t*>(kExpectedValues);
  expected_tensor.payload.assign(expected_bytes,
                                 expected_bytes + sizeof(kExpectedValues));

  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, &binding, wait.list(),
      expected_tensor));

  expected_tensor.tolerance.max_absolute_error = 0.05;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        id4::test::CompareF32BindingWithFixtureTensor(
                            device, IREE_HAL_QUEUE_AFFINITY_ANY, &binding,
                            wait.list(), expected_tensor));
}

TEST(HalIntegrationUtilTest, ComparesF16ActualSliceAgainstF32Expected) {
  id4::test::HalDeviceGroupRef device_group;
  IREE_ASSERT_OK(CreateLocalSyncDeviceGroup(device_group.out()));
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group.get(), /*device_index=*/0);

  const uint16_t actual_values[12] = {
      iree_math_f32_to_f16(1.0f),  iree_math_f32_to_f16(2.0f),
      iree_math_f32_to_f16(3.0f),  iree_math_f32_to_f16(4.0f),
      iree_math_f32_to_f16(5.0f),  iree_math_f32_to_f16(6.0f),
      iree_math_f32_to_f16(7.0f),  iree_math_f32_to_f16(8.0f),
      iree_math_f32_to_f16(9.0f),  iree_math_f32_to_f16(10.0f),
      iree_math_f32_to_f16(11.0f), iree_math_f32_to_f16(12.0f),
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
      iree_hal_device_allocator(device), params, sizeof(actual_values),
      buffer.out()));
  iree_hal_buffer_binding_t binding = {
      // Full actual f16 tensor buffer.
      /*.buffer=*/buffer.get(),
      // Full actual tensor begins at the buffer base.
      /*.offset=*/0,
      // Full actual tensor byte length.
      /*.length=*/sizeof(actual_values),
  };

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      update_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
                                           update_semaphore.out()));
  uint64_t update_value = 0;
  IREE_ASSERT_OK(id4::test::QueueUpdateBinding(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, &binding, actual_values,
      sizeof(actual_values), update_semaphore.get(), &update_value));
  id4::test::SemaphoreListStorage wait;
  wait.semaphore = update_semaphore.get();
  wait.payload_value = update_value;

  constexpr float kExpectedSlice[4] = {6.0f, 7.0f, 10.0f, 11.0f};
  id4::test::FixtureTensor expected_tensor;
  expected_tensor.name = "unit.f16_slice";
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
  expected_tensor.tolerance.mode =
      id4::test::FixtureToleranceMode::kElementwise;
  expected_tensor.tolerance.absolute_tolerance = 0.0;
  expected_tensor.tolerance.relative_tolerance = 0.0;
  const auto* expected_bytes = reinterpret_cast<const uint8_t*>(kExpectedSlice);
  expected_tensor.payload.assign(expected_bytes,
                                 expected_bytes + sizeof(kExpectedSlice));

  id4_pipeline_tensor_layout_t actual_layout = {};
  actual_layout.name = IREE_SV("unit.f16_actual");
  actual_layout.dtype = ID4_PIPELINE_TENSOR_DTYPE_F16;
  actual_layout.shape = expected_tensor.source_shape;
  actual_layout.byte_length = sizeof(actual_values);

  IREE_ASSERT_OK(id4::test::CompareBindingWithFixtureTensor(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, &binding, wait.list(),
      &actual_layout, expected_tensor));
}

TEST(HalIntegrationUtilTest, ComparesBf16RoundedSliceAgainstF32Expected) {
  id4::test::HalDeviceGroupRef device_group;
  IREE_ASSERT_OK(CreateLocalSyncDeviceGroup(device_group.out()));
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group.get(), /*device_index=*/0);

  const uint16_t actual_values[12] = {
      iree_math_f32_to_bf16(1.125f), iree_math_f32_to_bf16(2.25f),
      iree_math_f32_to_bf16(3.375f), iree_math_f32_to_bf16(4.5f),
      iree_math_f32_to_bf16(5.625f), iree_math_f32_to_bf16(6.75f),
      iree_math_f32_to_bf16(7.875f), iree_math_f32_to_bf16(8.125f),
      iree_math_f32_to_bf16(9.25f),  iree_math_f32_to_bf16(10.375f),
      iree_math_f32_to_bf16(11.5f),  iree_math_f32_to_bf16(12.625f),
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
      iree_hal_device_allocator(device), params, sizeof(actual_values),
      buffer.out()));
  iree_hal_buffer_binding_t binding = {
      // Full actual bf16 tensor buffer.
      /*.buffer=*/buffer.get(),
      // Full actual tensor begins at the buffer base.
      /*.offset=*/0,
      // Full actual tensor byte length.
      /*.length=*/sizeof(actual_values),
  };

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      update_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
                                           update_semaphore.out()));
  uint64_t update_value = 0;
  IREE_ASSERT_OK(id4::test::QueueUpdateBinding(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, &binding, actual_values,
      sizeof(actual_values), update_semaphore.get(), &update_value));
  id4::test::SemaphoreListStorage wait;
  wait.semaphore = update_semaphore.get();
  wait.payload_value = update_value;

  constexpr float kExpectedSlice[4] = {6.75f, 7.875f, 10.375f, 11.5f};
  id4::test::FixtureTensor expected_tensor;
  expected_tensor.name = "unit.bf16_slice";
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
  const auto* expected_bytes = reinterpret_cast<const uint8_t*>(kExpectedSlice);
  expected_tensor.payload.assign(expected_bytes,
                                 expected_bytes + sizeof(kExpectedSlice));

  id4_pipeline_tensor_layout_t actual_layout = {};
  actual_layout.name = IREE_SV("unit.bf16_actual");
  actual_layout.dtype = ID4_PIPELINE_TENSOR_DTYPE_BF16;
  actual_layout.shape = expected_tensor.source_shape;
  actual_layout.byte_length = sizeof(actual_values);

  IREE_ASSERT_OK(id4::test::CompareBf16BindingWithRoundedF32FixtureTensor(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, &binding, wait.list(),
      &actual_layout, expected_tensor));
}

}  // namespace
