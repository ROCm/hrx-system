// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/queue.h"

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/media/d3d12/command.h"
#include "libamdf/src/media/d3d12/object.h"
#include "libamdf/src/media/d3d12/queue_test_native.h"

namespace amdf::media::d3d12 {
namespace {

void QueryDevice(const amdf_device_t*, amdf_device_info_t* info) {
  info->reset_epoch = 1;
}
constexpr amdf_device_vtable_t kDevice = {.query_info = QueryDevice};

class MediaQueueTest : public ::testing::Test {
 protected:
  static void* AMDF_CALL AllocateHost(void* user_data, uint64_t byte_length,
                                      uint64_t minimum_alignment) {
    ++*static_cast<uint32_t*>(user_data);
    const auto allocator = amdf_allocator_system();
    return allocator.allocate(allocator.user_data, byte_length,
                              minimum_alignment);
  }

  void SetUp() override {
    native_ = amdf_media_queue_test_native_create();
    ASSERT_NE(native_, nullptr);
    device_.native.Attach(native_->device);
    device_.base.base.host_allocator = amdf_allocator_system();
    device_.base.base.host_allocator.user_data = &allocation_count_;
    device_.base.base.host_allocator.allocate = AllocateHost;
    device_.base.base.vtable = &kDevice;
    amdf_child_tracker_initialize(&device_.base.base.children);
    amdf_atomic_uint64_initialize(&device_.removed_status, AMDF_STATUS_OK);
    const amdf_media_kernel_queue_create_info_t create = {
        .maximum_pending_submission_count = 4,
    };
    amdf_kernel_queue_t* queue = nullptr;
    ASSERT_EQ(QueueCreate(&device_.base.base, &create, &queue), AMDF_STATUS_OK);
    queue_ = reinterpret_cast<Queue*>(queue);
    command_.base.queue = queue;
    command_.native.Attach(native_->command);
    command_.prepared = true;
  }

  void TearDown() override {
    for (auto& dependency : dependencies_) {
      if (dependency.native != nullptr) {
        EXPECT_EQ(dependency.native->destroy(dependency.native),
                  AMDF_STATUS_OK);
      }
    }
    command_.native.Reset();
    if (queue_ != nullptr) {
      // A failed signal or removed fake device deliberately leaves ownership
      // unretired. Ending this inert COM test domain is not a production
      // retirement operation; the tests assert that normal destruction refuses.
      queue_->native.Reset();
      queue_->fence.Reset();
      EXPECT_TRUE(CloseHandle(queue_->wait_event));
      amdf_kernel_queue_deinitialize(&queue_->base);
      Delete(amdf_allocator_system(), queue_);
    }
    device_.native.Reset();
    ASSERT_NE(native_, nullptr);
    EXPECT_EQ(native_->queue_references, 0u);
    EXPECT_EQ(native_->fence_references, 0u);
    EXPECT_EQ(native_->command_references, 0u);
    EXPECT_EQ(native_->device_references, 0u);
    amdf_media_queue_test_native_destroy(native_);
  }

  amdf_status_t Submit(uint64_t* accepted, uint32_t wait_count = 0) {
    const amdf_media_kernel_queue_submission_info_t submit = {
        .command = &command_.base,
        .wait_count = wait_count,
        .waits = waits_,
    };
    return QueueSubmit(&queue_->base, &submit, accepted);
  }

  void BindDependencies() {
    const amdf_windows_queue_progress_t progress = {
        .base = {AMDF_KERNEL_QUEUE_DEPENDENCY_PROTOCOL_WINDOWS_MONITORED_FENCE,
                 1},
    };
    for (uint32_t index = 0; index < 2; ++index) {
      ASSERT_EQ(
          queue_->base.vtable->create_native_dependency(
              &queue_->base, &progress.base, &dependencies_[index].native),
          AMDF_STATUS_OK);
      amdf_atomic_uint64_initialize(
          &dependencies_[index].last_consumer_submission, 0);
      waits_[index] = {&dependencies_[index], index + 1u};
    }
  }

  amdf_kernel_queue_status_t Refresh() {
    amdf_kernel_queue_status_t status = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
        .structure_size = sizeof(status),
    };
    EXPECT_EQ(amdf_kernel_queue_refresh_status(&queue_->base, &status),
              AMDF_STATUS_OK);
    return status;
  }

  // Mocked OS boundary, with independent acceptance and completion controls.
  amdf_media_queue_test_native_t* native_ = nullptr;
  // Provider device consuming the mocked COM factory.
  Device device_ = {};
  // Real queue owner built by the production constructor.
  Queue* queue_ = nullptr;
  // Prepared recording supplied by the queue's command dependency.
  Command command_ = {};
  // Cold consumer imports whose accepted-use records are checked below.
  amdf_kernel_queue_dependency_t dependencies_[2] = {};
  // Ordered native wait prefix under test.
  amdf_kernel_queue_dependency_wait_t waits_[2] = {};
  // Library allocations, independent of opaque driver recording allocations.
  uint32_t allocation_count_ = 0;
};

TEST_F(MediaQueueTest, ConstructionFailureRollsBackWithoutPublishing) {
  const auto before = amdf_child_tracker_count(&device_.base.base.children);
  native_->fence_create_status = E_OUTOFMEMORY;
  const amdf_media_kernel_queue_create_info_t create = {
      .maximum_pending_submission_count = 4,
  };
  amdf_kernel_queue_t* unchanged = &queue_->base;
  EXPECT_EQ(QueueCreate(&device_.base.base, &create, &unchanged),
            Hresult(E_OUTOFMEMORY));
  EXPECT_EQ(unchanged, &queue_->base);
  EXPECT_EQ(native_->queue_references, 1u);
  EXPECT_EQ(native_->fence_references, 1u);
  EXPECT_EQ(amdf_child_tracker_count(&device_.base.base.children), before);
}

TEST_F(MediaQueueTest, RejectedWaitAcceptsNothingAndDoesNotPoisonQueue) {
  ASSERT_NO_FATAL_FAILURE(BindDependencies());
  native_->reject_wait = 1;
  uint64_t accepted = 0;
  EXPECT_EQ(Submit(&accepted, 2), Hresult(E_FAIL));
  EXPECT_EQ(accepted, 0u);
  EXPECT_EQ(native_->execute_count, 0u);
  EXPECT_EQ(native_->signal_count, 0u);
  EXPECT_EQ(command_.submission, 0u);
  EXPECT_EQ(QueueTerminalStatus(queue_), AMDF_STATUS_OK);
  native_->reject_wait = 0;
  EXPECT_EQ(Submit(&accepted, 2), AMDF_STATUS_OK);
  EXPECT_EQ(accepted, 1u);
  native_->completed = accepted;
  EXPECT_EQ(Refresh().retired_submission, accepted);
}

TEST_F(MediaQueueTest,
       PartialWaitRetainsOnlyAcceptedBindingsAndHasNoCodecResult) {
  ASSERT_NO_FATAL_FAILURE(BindDependencies());
  native_->reject_wait = 2;
  uint64_t accepted = 0;
  EXPECT_EQ(Submit(&accepted, 2), Hresult(E_FAIL));
  EXPECT_EQ(accepted, 1u);
  EXPECT_EQ(native_->execute_count, 0u);
  EXPECT_EQ(native_->signaled, 1u);
  EXPECT_EQ(amdf_atomic_uint64_load_acquire(
                &dependencies_[0].last_consumer_submission),
            1u);
  EXPECT_EQ(amdf_atomic_uint64_load_acquire(
                &dependencies_[1].last_consumer_submission),
            0u);
  EXPECT_EQ(Refresh().retired_submission, 0u);
  amdf_media_command_result_info_t result = {};
  EXPECT_EQ(CommandQueryResult(&command_.base, &result),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  native_->completed = 1;
  const auto status = Refresh();
  EXPECT_EQ(status.retired_submission, 1u);
  EXPECT_EQ(status.terminal_status, Hresult(E_FAIL));
  EXPECT_EQ(CommandQueryResult(&command_.base, &result), AMDF_STATUS_OK);
  EXPECT_EQ(result.codec_result, AMDF_MEDIA_CODEC_RESULT_UNAVAILABLE);
}

TEST_F(MediaQueueTest,
       SignalFailureRetainsExecutionWithoutInventingRetirement) {
  native_->signal_status = E_FAIL;
  uint64_t accepted = 0;
  EXPECT_EQ(Submit(&accepted), Hresult(E_FAIL));
  EXPECT_EQ(accepted, 1u);
  EXPECT_EQ(native_->execute_count, 1u);
  EXPECT_TRUE(command_.executed);
  EXPECT_EQ(command_.submission, 1u);
  const auto status = Refresh();
  EXPECT_EQ(status.retired_submission, 0u);
  EXPECT_EQ(status.terminal_status, Hresult(E_FAIL));
  EXPECT_EQ(amdf_kernel_queue_destroy(&queue_->base),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  EXPECT_EQ(native_->queue_references, 1u);
  EXPECT_EQ(native_->fence_references, 1u);
  accepted = 0;
  EXPECT_EQ(Submit(&accepted), Hresult(E_FAIL));
  EXPECT_EQ(accepted, 0u);
  EXPECT_EQ(native_->signal_count, 1u);
}

TEST_F(MediaQueueTest, DuplicateRecordingDoesNotRepublishItsEarlierExecution) {
  const auto allocations = allocation_count_;
  uint64_t accepted = 0;
  ASSERT_EQ(Submit(&accepted), AMDF_STATUS_OK);
  accepted = 0;
  EXPECT_EQ(Submit(&accepted),
            amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION));
  EXPECT_EQ(accepted, 0u);
  EXPECT_EQ(native_->execute_count, 1u);
  EXPECT_EQ(native_->signal_count, 1u);
  EXPECT_EQ(native_->completion_query_count, 0u);
  EXPECT_EQ(native_->removal_query_count, 0u);
  EXPECT_EQ(allocation_count_, allocations);
  native_->completed = 1;
  EXPECT_EQ(Refresh().retired_submission, 1u);
}

TEST_F(MediaQueueTest, DeviceRemovalCannotRetireAnApparentlyCompletedPrefix) {
  uint64_t accepted = 0;
  ASSERT_EQ(Submit(&accepted), AMDF_STATUS_OK);
  native_->completed = 1;
  native_->removed_status = DXGI_ERROR_DEVICE_REMOVED;
  const auto status = Refresh();
  EXPECT_EQ(status.retired_submission, 0u);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_DEVICE_LOST);
  EXPECT_EQ(status.terminal_status, Hresult(DXGI_ERROR_DEVICE_REMOVED));
  EXPECT_EQ(amdf_kernel_queue_destroy(&queue_->base),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
}

TEST_F(MediaQueueTest, RemovalSentinelCannotAdvanceCheckedProgress) {
  uint64_t accepted = 0;
  ASSERT_EQ(Submit(&accepted), AMDF_STATUS_OK);
  native_->completed = UINT64_MAX;
  const auto status = Refresh();
  EXPECT_EQ(status.retired_submission, 0u);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_DEVICE_LOST);
  EXPECT_EQ(status.terminal_status, Hresult(DXGI_ERROR_DEVICE_REMOVED));
}

}  // namespace
}  // namespace amdf::media::d3d12
