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
#include <thread>

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

// Holds the real allocation dependency while forwarding all native calls.
// Fixtures run serially and join fabric/device borrowers before destroying
// this table. The gate qualifies HAL ownership, not kernel lock isolation.
class NativeMemoryGate {
 public:
  ~NativeMemoryGate();

  iree_status_t CreateFabric(iree_hal_amd_fabric_t** out_fabric);

  // Read only after Hold::WaitEntered has published the native caller.
  std::thread::id native_thread() const { return native_thread_; }

  class Hold {
   public:
    explicit Hold(NativeMemoryGate* owner);
    ~Hold();

    void WaitEntered();
    void ReleaseAndHoldNext();

   private:
    // Borrowed gate outliving the native call and its explicit hold.
    NativeMemoryGate* owner_;
  };

 private:
  static amdf_status_t AMDF_CALL CreateMemory(
      amdf_memory_scope_t* scope, const amdf_memory_create_info_t* info,
      amdf_memory_t** out_memory);

  // Exact native API table, borrowed for all forwarded calls.
  const amdf_api_t* native_api_ = nullptr;
  // Same native API with an allocation-entry gate, never a substitute
  // allocator.
  amdf_api_t forwarding_api_ = {};
  // Serial fixture identity required by the native ABI's context-free table.
  inline static NativeMemoryGate* active_ = nullptr;
  // One-shot hold consumed by the next native backing acquisition.
  std::atomic<bool> hold_{false};
  // Publishes entry and thread identity of the held native call.
  std::binary_semaphore entered_{0};
  // Permission to continue the native call after independent work completes.
  std::binary_semaphore resume_{0};
  // Joins the forwarding function's final fixture access.
  std::binary_semaphore returned_{0};
  // Observed allocation owner, read only after entered_.
  std::thread::id native_thread_;
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
