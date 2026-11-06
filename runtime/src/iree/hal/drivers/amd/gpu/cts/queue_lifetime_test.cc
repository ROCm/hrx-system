// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <memory>
#include <semaphore>
#include <thread>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/queue.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/queue.h"
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
    IREE_ASSERT_OK(memory_gate_.CreateFabric(&fabric_));
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
    IREE_ASSERT_OK(AcquireQueue(&queue));
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
                  reinterpret_cast<iree_async_semaphore_t*>(handle), 1,
                  iree_async_single_frontier_as_frontier(&completed)),
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
  IREE_ASSERT_OK(AcquireQueue(producer.out()));
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
  IREE_ASSERT_OK(AcquireQueue(replacement.out()));
  EXPECT_EQ(retained.signal->device_address, address);
  iree_status_code_t code = IREE_STATUS_UNKNOWN;
  EXPECT_TRUE(iree_hal_amd_gpu_signal_query_retirement(retained.signal, &code));
  EXPECT_EQ(code, IREE_STATUS_OK);
  EXPECT_EQ(*retained.signal->host_value,
            command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL ? 0u : 1u);
}

TEST_P(GpuNativeQueueLifetimeTest,
       RecordedOutputSurvivesConcurrentMembershipGrowth) {
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
  IREE_ASSERT_OK(AcquireQueue(acquired[0].out()));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  iree_hal_semaphore_t* handle = done;
  uint64_t one = 1, two = 2, three = 3;
  std::jthread constructor;
  {
    // Hold one native allocation before it reaches libamdf. Another acquired
    // coordinate must grow the frontier, run the existing resident recording,
    // and retire its output without waiting for that constructor.
    NativeMemoryGate::Hold hold(&memory_gate_);
    constructor =
        std::jthread([&] { IREE_EXPECT_OK(AcquireQueue(acquired[1].out())); });
    hold.WaitEntered();
    IREE_ASSERT_OK(AcquireQueue(acquired[2].out()));
    IREE_ASSERT_OK(iree_hal_device_trim(device_));
    IREE_ASSERT_OK(iree_hal_queue_execute(acquired[2], {}, {1, &handle, &one},
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
    acquired[2].reset();
    IREE_ASSERT_OK(iree_hal_device_trim(device_));
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(done, two, iree_infinite_timeout(), 0));
    for (uint32_t value : observed) {
      EXPECT_EQ(value, pattern);
    }
  }
  constructor.join();
  ASSERT_NE(acquired[1].get(), nullptr);
  // The delayed constructor catches up before accepting this frontier, which
  // transitively includes the wider coordinate of the already-released queue.
  IREE_ASSERT_OK(iree_hal_queue_barrier(acquired[1], {1, &handle, &two},
                                        {1, &handle, &three}, nullptr, 0));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, three, iree_infinite_timeout(), 0));
}

TEST_P(GpuNativeQueueLifetimeTest,
       FailedDependencyPreservesBorrowedDownloadJoin) {
  struct Download {
    // Host destination retained by the source view through its final access.
    std::array<uint32_t, 16> data = {};
    // Release of the exact source view captured by the download.
    std::atomic<bool> retired = false;
    // Cleanup joins the actual download even when the completion assertion
    // fails.
    std::binary_semaphore completed{0};
  };
  struct Gates {
    // Borrowed host gates, outliving this cleanup scope.
    std::array<iree_hal_semaphore_t*, 2> semaphores;
    void Release() {
      for (auto*& semaphore : semaphores) {
        if (semaphore) {
          IREE_EXPECT_OK(iree_hal_semaphore_signal(semaphore, 1, nullptr));
          semaphore = nullptr;
        }
      }
    }
    ~Gates() { Release(); }
  };
  Ref<iree_hal_buffer_t> source;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(64, 4),
      iree_infinite_timeout(), source.out()));
  Ref<iree_hal_semaphore_t> initialized;
  CreateSemaphore(initialized.out());
  uint64_t one = 1;
  iree_hal_semaphore_t* initialized_handle = initialized;
  constexpr uint32_t pattern = 0x1234CAFE;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queues_[0], {}, {1, &initialized_handle, &one}, source, 0, 64, &pattern,
      sizeof(pattern), nullptr, IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(initialized, 1, iree_infinite_timeout(), 0));
  for (bool already_failed : {false, true}) {
    for (bool failed_first : {false, true}) {
      for (bool shared_output : {false, true}) {
        for (bool transfer_join : {false, true}) {
          SCOPED_TRACE(already_failed);
          SCOPED_TRACE(failed_first);
          SCOPED_TRACE(shared_output);
          SCOPED_TRACE(transfer_join);
          Ref<iree_hal_queue_t> failing_queue;
          IREE_ASSERT_OK(AcquireQueue(failing_queue.out()));
          auto downloaded = std::make_shared<Download>();
          auto owner = std::make_unique<std::shared_ptr<Download>>(downloaded);
          Ref<iree_hal_buffer_t> view;
          IREE_ASSERT_OK(iree_hal_subspan_buffer_create_with_callback(
              source, iree_hal_buffer_byte_offset(source), 64,
              {[](void* user_data, iree_hal_buffer_t*) {
                 std::unique_ptr<std::shared_ptr<Download>> owner(
                     static_cast<std::shared_ptr<Download>*>(user_data));
                 (*owner)->retired.store(true);
                 (*owner)->completed.release();
               },
               owner.get()},
              iree_allocator_system(), view.out()));
          owner.release();
          Ref<iree_hal_semaphore_t> copy_gate, error_gate, copied, failed,
              joined;
          CreateSemaphore(copy_gate.out());
          CreateSemaphore(error_gate.out());
          CreateSemaphore(copied.out());
          CreateSemaphore(failed.out());
          CreateSemaphore(joined.out());
          iree_hal_semaphore_t* copy_gate_handle = copy_gate;
          iree_hal_semaphore_t* error_gate_handle = error_gate;
          iree_hal_semaphore_t* copied_handle = copied;
          iree_hal_semaphore_t* failed_handle = failed;
          Gates gates{{copy_gate, error_gate}};
          IREE_ASSERT_OK(
              iree_hal_queue_download(queues_[0], {1, &copy_gate_handle, &one},
                                      {1, &copied_handle, &one}, view, 0,
                                      downloaded->data.data(), 64, nullptr));
          view.reset();
          const auto call = iree_hal_make_host_call(
              [](void*, const uint64_t[4], iree_hal_host_call_context_t*) {
                return iree_make_status(IREE_STATUS_DATA_LOSS,
                                        "host operation failed");
              },
              nullptr);
          const uint64_t arguments[4] = {};
          IREE_ASSERT_OK(iree_hal_queue_host_call(
              failing_queue, {1, &error_gate_handle, &one},
              {1, &failed_handle, &one}, call, arguments,
              IREE_HAL_HOST_CALL_FLAG_NONE));
          if (already_failed) {
            IREE_ASSERT_OK(iree_hal_semaphore_signal(error_gate, 1, nullptr));
            gates.semaphores[1] = nullptr;
            failing_queue.reset();
          }
          iree_hal_semaphore_t* waits[] = {
              failed_first ? failed_handle : copied_handle,
              failed_first ? copied_handle : failed_handle};
          uint64_t values[] = {1, 1};
          iree_hal_semaphore_t* output = shared_output ? copied_handle : joined;
          uint64_t output_value = shared_output ? 2 : 1;
          if (transfer_join) {
            IREE_ASSERT_OK(iree_hal_queue_transfer(
                queues_[0], {2, waits, values}, {1, &output, &output_value}, 0,
                nullptr, nullptr));
          } else {
            IREE_ASSERT_OK(iree_hal_queue_barrier(
                queues_[0], {2, waits, values}, {1, &output, &output_value},
                nullptr, IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
          }
          if (!already_failed) {
            IREE_ASSERT_OK(iree_hal_semaphore_signal(error_gate, 1, nullptr));
            gates.semaphores[1] = nullptr;
            failing_queue.reset();
          }
          // Final release above joins the producer's signal callbacks. This
          // existing cold metadata transaction then joins publication of the
          // consumer's admission attempt, not its still-gated GPU work.
          iree_notification_t notification;
          iree_notification_initialize(&notification);
          iree_status_t admission_status =
              command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
                  ? iree_hal_amd_gpu_pm4_queue_reserve_frontiers(
                        queues_[0], options_.queue_count, &notification)
                  : iree_hal_amd_gpu_aql_queue_reserve_frontiers(
                        queues_[0], options_.queue_count, &notification);
          iree_notification_deinitialize(&notification);
          IREE_ASSERT_OK(admission_status);
          uint64_t value = UINT64_MAX;
          IREE_EXPECT_OK(iree_hal_semaphore_query(output, &value));
          EXPECT_EQ(value, 0u);
          EXPECT_FALSE(downloaded->retired.load());
          gates.Release();
          IREE_EXPECT_STATUS_IS(
              IREE_STATUS_DATA_LOSS,
              iree_hal_semaphore_wait(output, output_value,
                                      iree_infinite_timeout(), 0));
          EXPECT_TRUE(downloaded->retired.load());
          downloaded->completed.acquire();
          for (uint32_t actual : downloaded->data) {
            EXPECT_EQ(actual, pattern);
          }
        }
      }
    }
  }
}

TEST_P(GpuNativeQueueLifetimeTest, QueueReleaseDoesNotDrainHeldAllocation) {
  Ref<iree_hal_queue_t> acquired;
  IREE_ASSERT_OK(AcquireQueue(acquired.out()));
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
