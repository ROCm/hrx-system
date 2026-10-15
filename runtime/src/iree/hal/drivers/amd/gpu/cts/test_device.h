// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_CTS_TEST_DEVICE_H_
#define IREE_HAL_DRIVERS_AMD_GPU_CTS_TEST_DEVICE_H_

#include <array>
#include <atomic>
#include <semaphore>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/amd/gpu/device.h"
#include "iree/testing/gtest.h"

namespace iree::hal::amd::gpu::cts {

// Observe actual resource reclamation, including both worker handles. Returning
// from device_release alone does not prove asynchronous cleanup has finished.
struct DeviceAllocator {
  // Live allocations made by the concrete device and its children.
  std::atomic<size_t> live_count = 0;
  // Set after successful construction, while the device still owns storage.
  std::atomic<bool> armed = false;
  // Final callback returns ownership of this test allocator to the test.
  std::binary_semaphore reclaimed{0};

  iree_allocator_t allocator() { return {this, Control}; }

  static iree_status_t Control(void* self_pointer,
                               iree_allocator_command_t command,
                               const void* params, void** inout_pointer);
};

class GpuNativeTest : public ::testing::Test {
 protected:
  void SetUp() override;

  void TearDown() override;

  void Create(iree_hal_device_runtime_feature_flags_t runtime_features = 0);

  void CreateSemaphore(iree_hal_semaphore_t** out_semaphore);

  // Independent native discovery ownership, released after device cleanup.
  iree_hal_amd_fabric_t* fabric_ = nullptr;
  // Constructing native owner, transferred to device on success.
  iree_hal_amd_gpu_context_t* context_ = nullptr;
  // Explicit provisioning used by each test.
  iree_hal_amd_gpu_device_options_t options_ = {};
  // Process-shared progress owners supplied through the public creation API.
  iree_async_proactor_pool_t* proactors_ = nullptr;
  // External tracker reference used to observe actual queue progress.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Allocator lifetime extends through the final worker cleanup.
  DeviceAllocator allocator_;
  // Owning public device reference, independently of its group.
  iree_hal_device_t* device_ = nullptr;
  // Immutable group assigning real memory scopes and queue axes.
  iree_hal_device_group_t* group_ = nullptr;
  // Borrowed provisioned queues, never separately retained by this fixture.
  std::array<iree_hal_queue_t*, 2> queues_ = {};
};

}  // namespace iree::hal::amd::gpu::cts

#endif  // IREE_HAL_DRIVERS_AMD_GPU_CTS_TEST_DEVICE_H_
