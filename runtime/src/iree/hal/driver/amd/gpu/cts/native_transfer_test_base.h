// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_CTS_NATIVE_TRANSFER_TEST_BASE_H_
#define IREE_HAL_DRIVER_AMD_GPU_CTS_NATIVE_TRANSFER_TEST_BASE_H_

#include <array>
#include <utility>

#include "iree/hal/api.h"
#include "iree/hal/driver/amd/gpu/cts/test_device.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::cts {

class GpuNativeTransferTest : public GpuNativeTest,
                              public ::testing::WithParamInterface<
                                  std::pair<amdf_queue_command_type_t, bool>> {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return GetParam().first == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA
               ? AMDF_QUEUE_COMMAND_TYPE_UNKNOWN
               : GetParam().first;
  }

  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    // Start with minimal ABI storage and small retirement and primary-ring
    // capacities. Builtin dispatches must grow their argument storage; bulk
    // transfers grow staging and reuse it through native completion events.
    options_.queue.ring_byte_length = 4096;
    options_.queue.retirement_capacity = 4;
    options_.queue.argument_capacity = 8;
    Create();
    if (HasFatalFailure()) {
      return;
    }
    if (GetParam().first == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA) {
      AcquireTransferQueues();
      if (HasFatalFailure() || IsSkipped()) {
        return;
      }
    }
    const iree_hal_pool_family_access_t access = {
        .family = iree_hal_queue_family(queues_[0]),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
        .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &access,
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &options,
                                             iree_allocator_system(), &pool_));
    iree_hal_memory_scope_t queue_scope;
    IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
        group_, {IREE_HAL_MEMORY_SITE_QUEUE, access.family}, &queue_scope));
    const auto transition =
        iree_hal_pool_query_transition(pool_, queue_scope, queue_scope);
    ASSERT_TRUE(iree_hal_memory_effects_is_supported(transition.release));
    ASSERT_TRUE(iree_hal_memory_effects_is_supported(transition.acquire));
    ASSERT_FALSE(
        iree_hal_memory_effects_requires_resources(transition.release));
    ASSERT_FALSE(
        iree_hal_memory_effects_requires_resources(transition.acquire));
    acquire_.effects = transition.acquire;
    release_.effects = transition.release;
    for (auto& buffer : buffers_) {
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          pool_, iree_hal_make_pool_reservation_request(kLength, 4),
          iree_infinite_timeout(), &buffer));
    }
    CreateSemaphore(&timeline_);
    CreateSemaphore(&gate_);
  }

  void TearDown() override {
    iree_hal_semaphore_release(gate_);
    iree_hal_semaphore_release(timeline_);
    ReleaseBuffers();
    iree_hal_pool_release(pool_);
    GpuNativeTest::TearDown();
  }

  void ReleaseBuffers() {
    for (auto& buffer : buffers_) {
      iree_hal_buffer_release(buffer);
      buffer = nullptr;
    }
  }

  const iree_hal_queue_barriers_t* Barriers() {
    return GetParam().second ? &barriers_ : nullptr;
  }

  void Wait(uint64_t value) {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        timeline_, value, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  // Runs independent upload/download lanes through bounded native staging.
  // The first batch holds completion until all callers have submitted, forcing
  // capacity return to wake publication without relying on a later submit.
  void RunConcurrentTransfers(size_t lane_count, size_t iterations);

  // Odd extent exercises byte copies and staged fill tails.
  static constexpr size_t kLength = 32781;
  // Explicit source of buffers usable by the device's transfer family.
  iree_hal_pool_t* pool_ = nullptr;
  // Independent payload allocations retained by accepted transactions.
  std::array<iree_hal_buffer_t*, 2> buffers_ = {};
  // User-visible completion sequence, including host downloads.
  iree_hal_semaphore_t* timeline_ = nullptr;
  // Host-controlled readiness for borrowed uploads and captured updates.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Exact pool transitions between shader or copy-engine accesses. An empty
  // payload boundary is valid only when the queried transition is empty.
  iree_hal_barrier_t acquire_ = {};
  // Publishes every transfer in the batch before its completion semaphore.
  iree_hal_barrier_t release_ = {};
  // Stable descriptors shared by concurrent submitting threads.
  iree_hal_barrier_list_t before_ = {1, &acquire_};
  // Stable producer boundary paired with before_.
  iree_hal_barrier_list_t after_ = {1, &release_};
  // Explicit qualified alternatives to conservative default boundaries.
  iree_hal_queue_barriers_t barriers_ = {&before_, &after_};
};

}  // namespace iree::hal::amd::gpu::cts

#endif  // IREE_HAL_DRIVER_AMD_GPU_CTS_NATIVE_TRANSFER_TEST_BASE_H_
