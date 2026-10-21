// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <memory>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/queue/semaphore.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::cts {
namespace {

using iree::hal::cts::Ref;

class GpuNativeQueueLifetimeTest
    : public GpuNativeTest,
      public ::testing::WithParamInterface<amdf_queue_command_type_t> {
 protected:
  amdf_queue_command_type_t command_type() const override { return GetParam(); }

  void SetUp() override {
    IREE_ASSERT_OK(memory_gate_.CreateFabric(
        &fabric_, UINT64_MAX,
        command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
            ? AMDF_NATIVE_LIFETIME_PROCESS
            : AMDF_NATIVE_LIFETIME_INSTANCE));
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    domain_ = context_;
    Create();
    if (HasFatalFailure()) {
      return;
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
  }

  void TearDown() override {
    iree_hal_pool_release(pool_);
    GpuNativeTest::TearDown();
  }

  iree_status_t Acquire(iree_hal_queue_t** out_queue) {
    iree_hal_queue_params_t params;
    iree_hal_queue_params_initialize(&params);
    // Native lifecycle coverage enters the concrete device interface with
    // canonical family parameters; the shared CTS owns capability validation.
    const auto* resource =
        reinterpret_cast<const iree_hal_resource_t*>(device_);
    const auto* vtable =
        static_cast<const iree_hal_device_vtable_t*>(resource->vtable);
    return vtable->acquire_queue(device_, iree_hal_queue_family(queues_[0]),
                                 &params, out_queue);
  }

  // Native dependency gate, joined by base teardown before its table expires.
  NativeMemoryGate memory_gate_;
  // Explicit physical backing for recording and independent allocation work.
  iree_hal_pool_t* pool_ = nullptr;
  // Device's stable native signal domain, borrowed through device teardown.
  const void* domain_ = nullptr;
};

TEST_P(GpuNativeQueueLifetimeTest, FinalReleaseContinuesEpochsAcrossSlotReuse) {
  iree_async_single_frontier_t previous = {};
  for (uint32_t iteration = 0; iteration < 4; ++iteration) {
    iree_hal_queue_t* queue = nullptr;
    IREE_ASSERT_OK(Acquire(&queue));
    Ref<iree_hal_semaphore_t> done;
    CreateSemaphore(done.out());
    uint64_t one = 1;
    iree_hal_semaphore_t* handle = done;
    IREE_ASSERT_OK(
        iree_hal_queue_barrier(queue, {}, {1, &handle, &one}, nullptr, 0));
    // No preceding host wait: final release owns the complete accepted-work
    // and callback-tail join, while provisioned queues remain independently
    // live.
    iree_hal_queue_release(queue);
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(done, one, iree_infinite_timeout(), 0));
    iree_async_single_frontier_t completed = {};
    ASSERT_EQ(iree_async_semaphore_query_frontier(
                  reinterpret_cast<iree_async_semaphore_t*>(handle),
                  iree_async_single_frontier_as_frontier(&completed), 1),
              1);
    if (iteration) {
      EXPECT_EQ(completed.entries[0].axis, previous.entries[0].axis);
      EXPECT_GT(completed.entries[0].epoch, previous.entries[0].epoch);
      EXPECT_TRUE(iree_async_frontier_tracker_query_epoch(
          tracker_, previous.entries[0].axis, previous.entries[0].epoch));
    }
    previous = completed;
  }
}

TEST_P(GpuNativeQueueLifetimeTest, NativeSignalReaderSurvivesProducerRelease) {
  Ref<iree_hal_queue_t> producer;
  IREE_ASSERT_OK(Acquire(producer.out()));
  Ref<iree_hal_semaphore_t> paused, done;
  CreateSemaphore(paused.out());
  CreateSemaphore(done.out());
  iree_hal_semaphore_t* pause_handle = paused;
  iree_hal_semaphore_t* done_handle = done;
  uint64_t one = 1;
  CompletionPause pause{
      reinterpret_cast<iree_async_semaphore_t*>(pause_handle)};
  IREE_ASSERT_OK(pause.Arm(one));
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {},
                                        {1, &pause_handle, &one}, nullptr, 0));
  pause.entered.acquire();
  IREE_ASSERT_OK(iree_hal_queue_barrier(producer, {}, {1, &done_handle, &one},
                                        nullptr, 0));
  iree_hal_amd_gpu_semaphore_wait_t retained = {};
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_semaphore_acquire_wait(done, one, domain_, &retained));
  std::unique_ptr<iree_hal_amd_gpu_signal_t,
                  decltype(&iree_hal_amd_gpu_signal_release)>
      reader(retained.signal, iree_hal_amd_gpu_signal_release);
  ASSERT_EQ(retained.state, IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE);
  const uint64_t address = retained.signal->device_address;
  pause.Join();
  producer.reset();
  Ref<iree_hal_queue_t> replacement;
  IREE_ASSERT_OK(Acquire(replacement.out()));
  EXPECT_EQ(retained.signal->device_address, address);
  iree_status_code_t code = IREE_STATUS_UNKNOWN;
  EXPECT_TRUE(iree_hal_amd_gpu_signal_query_retirement(retained.signal, &code));
  EXPECT_EQ(code, IREE_STATUS_OK);
  EXPECT_EQ(*retained.signal->host_value,
            command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL ? 0u : 1u);
}

TEST_P(GpuNativeQueueLifetimeTest, RecordedOutputSurvivesMembershipGrowth) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(256, 4),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_command_buffer_t> recording;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0, recording.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  const uint32_t pattern = 0x1234CAFE;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      recording, iree_hal_make_buffer_ref(output, 0, 256), &pattern,
      sizeof(pattern), 0));
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  std::array<Ref<iree_hal_queue_t>, 3> acquired;
  for (auto& queue : acquired) {
    IREE_ASSERT_OK(Acquire(queue.out()));
  }
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  iree_hal_semaphore_t* handle = done;
  uint64_t one = 1, two = 2;
  IREE_ASSERT_OK(iree_hal_queue_execute(acquired.back(), {}, {1, &handle, &one},
                                        recording, {}, 0));
  std::array<uint32_t, 64> observed = {};
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = output,
              .target = observed.data(),
              .length = sizeof(observed),
          },
  };
  IREE_ASSERT_OK(iree_hal_queue_transfer(queues_[0], {1, &handle, &one},
                                         {1, &handle, &two}, 1, &download,
                                         nullptr));
  acquired.back().reset();
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, two, iree_infinite_timeout(), 0));
  for (uint32_t value : observed) {
    EXPECT_EQ(value, pattern);
  }
}

TEST_P(GpuNativeQueueLifetimeTest, QueueReleaseDoesNotDrainHeldAllocation) {
  Ref<iree_hal_queue_t> acquired;
  IREE_ASSERT_OK(Acquire(acquired.out()));
  Ref<iree_hal_buffer_t> pending;
  Ref<iree_hal_semaphore_t> allocated, completed;
  CreateSemaphore(allocated.out());
  CreateSemaphore(completed.out());
  iree_hal_semaphore_t* allocated_handle = allocated;
  iree_hal_semaphore_t* completed_handle = completed;
  uint64_t one = 1;
  {
    NativeMemoryGate::Hold hold(&memory_gate_);
    const auto request = iree_hal_make_pool_reservation_request(4096, 4);
    IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], {},
                                         {1, &allocated_handle, &one}, pool_, 1,
                                         &request, pending.out()));
    hold.WaitEntered();
    IREE_ASSERT_OK(iree_hal_queue_barrier(
        acquired, {}, {1, &completed_handle, &one}, nullptr, 0));
    acquired.reset();
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(completed, one, iree_infinite_timeout(), 0));
  }
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(allocated, one, iree_infinite_timeout(), 0));
}

INSTANTIATE_TEST_SUITE_P(NativeProtocols, GpuNativeQueueLifetimeTest,
                         ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                           AMDF_QUEUE_COMMAND_TYPE_GPU_AQL));

}  // namespace
}  // namespace iree::hal::amd::gpu::cts
