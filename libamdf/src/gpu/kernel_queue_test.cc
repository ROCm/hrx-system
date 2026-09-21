// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/kernel_queue.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/device.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/gpu/device.h"
#include "libamdf/src/gpu/umd/kernel_queue.h"
#include "libamdf/src/kernel_queue.h"
#include "libamdf/src/memory_resource.h"

// Only native dependencies are controlled. Public submission and checked
// retirement use the production shared queue implementation.
struct amdf_gpu_umd_kernel_queue_t {
  // Native completion frontier independent of software retirement.
  std::atomic<uint64_t> progress{0};
  // Greatest completion point accepted by the native dependency.
  uint64_t submitted = 0;
  // Outcome of the next native publication attempt.
  amdf_status_t submit_status = AMDF_STATUS_OK;
  // Cached terminal device failure sampled by the shared queue.
  amdf_status_t terminal_status = AMDF_STATUS_OK;
  // Outcome of the next native wait after it publishes completion.
  amdf_status_t wait_status = AMDF_STATUS_OK;
  // Native cleanup result after the shared layer establishes retirement.
  amdf_status_t destroy_status = AMDF_STATUS_OK;
  // Number of final native release attempts.
  uint32_t destroy_count = 0;
  // Native publication schedule controlled by the concurrency scenario.
  enum class PublicationPhase { kOpen, kPauseOnAccept, kWaiting, kResume };
  // Protects the test dependency's publication schedule, not the queue under
  // test.
  std::mutex publication_mutex;
  // Signals explicit readiness and release of the paused native call.
  std::condition_variable publication_condition;
  // Current publication schedule while holding publication_mutex.
  PublicationPhase publication_phase = PublicationPhase::kOpen;
};

struct amdf_gpu_umd_device_t {
  // Device-owned native queue borrowed by the shared implementation.
  amdf_gpu_umd_kernel_queue_t queue;
};

namespace {

struct Device {
  // Generic device base consumed by production queue ownership.
  amdf_device_t base = {};
  // Immutable identity and reset epoch returned by the device dependency.
  amdf_gpu_device_info_t info = {};
  // Native dependency controlled independently of shared queue state.
  amdf_gpu_umd_device_t native;
  // Command representation advertised by the endpoint dependency.
  amdf_queue_command_type_t command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
};

class GpuKernelQueueTest
    : public ::testing::TestWithParam<amdf_queue_command_type_t> {
 protected:
  void SetUp() override {
    device.base.engine_kind = AMDF_ENGINE_KIND_GPU;
    device.base.host_allocator = amdf_allocator_system();
    device.base.endpoint = reinterpret_cast<amdf_endpoint_t*>(&device);
    amdf_child_tracker_initialize(&device.base.children);
    device.info.reset_epoch = 1;
    device.command_type = GetParam();
    ASSERT_EQ(
        amdf_memory_resource_allocate(amdf_allocator_system(), 1, &memory),
        AMDF_STATUS_OK);
    memory->info.byte_length = 4096;
    memory->accesses[0].device = &device.base;
    memory->accesses[0].info.access = AMDF_MEMORY_ACCESS_EXECUTE;
    memory->accesses[0].info.reset_epoch = device.info.reset_epoch;
    memory->accesses[0].info.address_kinds = uint64_t{1}
                                             << AMDF_MEMORY_ADDRESS_GPU;
    memory->accesses[0].addresses[AMDF_MEMORY_ADDRESS_GPU] = 0x100000000;
    command.memory = memory;
    command.byte_offset = 64;
    command.byte_length = 20;
    ASSERT_EQ(CreateQueue(0), AMDF_STATUS_OK);
  }

  void TearDown() override {
    device.native.queue.progress = device.native.queue.submitted;
    if (queue) {
      EXPECT_EQ(amdf_kernel_queue_destroy(queue), AMDF_STATUS_OK);
    }
    if (memory) {
      amdf_free(amdf_allocator_system(), memory);
    }
    EXPECT_EQ(amdf_child_tracker_count(&device.base.children), 0u);
  }

  amdf_status_t Submit(uint64_t* out_submission) {
    amdf_gpu_kernel_queue_submission_info_t submit = {};
    submit.type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO;
    submit.structure_size = sizeof(submit);
    submit.command_count = 1;
    submit.commands = &command;
    return amdf_gpu_kernel_queue_submit(queue, &submit, out_submission);
  }

  amdf_status_t CreateQueue(uint32_t capacity) {
    if (queue) {
      const auto status = amdf_kernel_queue_destroy(queue);
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      queue = nullptr;
    }
    amdf_gpu_kernel_queue_create_info_t create = {};
    create.type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO;
    create.structure_size = sizeof(create);
    create.maximum_pending_submission_count = capacity;
    return amdf_gpu_kernel_queue_create(&device.base, &create, &queue);
  }

  amdf_kernel_queue_status_t Query() {
    amdf_kernel_queue_status_t status = {};
    status.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS;
    status.structure_size = sizeof(status);
    EXPECT_EQ(amdf_kernel_queue_query_status(queue, &status), AMDF_STATUS_OK);
    return status;
  }

  amdf_kernel_queue_status_t Refresh() {
    amdf_kernel_queue_status_t status = {};
    status.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS;
    status.structure_size = sizeof(status);
    EXPECT_EQ(amdf_kernel_queue_refresh_status(queue, &status), AMDF_STATUS_OK);
    return status;
  }

  // Device dependency retained until queue destruction.
  Device device;
  // Real memory owner kept live by the caller until native last use.
  amdf_memory_t* memory = nullptr;
  // Command range passed directly to the native dependency.
  amdf_gpu_kernel_command_t command = {};
  // Production queue under test.
  amdf_kernel_queue_t* queue = nullptr;
};

TEST_P(GpuKernelQueueTest, QueryDoesNotRetireNativeCompletion) {
  uint64_t submission = 0;
  ASSERT_EQ(Submit(&submission), AMDF_STATUS_OK);
  device.native.queue.progress = device.native.queue.submitted;
  EXPECT_EQ(Query().retired_submission, 0u);
  uint64_t later_submission = 0;
  ASSERT_EQ(Submit(&later_submission), AMDF_STATUS_OK);
  EXPECT_GT(later_submission, submission);
  EXPECT_EQ(Query().retired_submission, 0u);
  ASSERT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, submission);
  ASSERT_EQ(Submit(&submission), AMDF_STATUS_OK);
}

TEST_P(GpuKernelQueueTest, UnsupportedNotificationDoesNotChangeAcceptedWork) {
  amdf_kernel_queue_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO;
  info.structure_size = sizeof(info);
  ASSERT_EQ(amdf_kernel_queue_query_info(queue, &info), AMDF_STATUS_OK);
  ASSERT_EQ(info.notification_types, 0u);
  uint64_t submission = 0;
  ASSERT_EQ(Submit(&submission), AMDF_STATUS_OK);
  amdf_native_event_t event = {};
  event.type = AMDF_NATIVE_EVENT_TYPE_EVENTFD;
  event.payload.file_descriptor = 5;
  EXPECT_EQ(amdf_status_code(amdf_kernel_queue_request_notification(
                queue, submission, &event)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(Query().retired_submission, 0u);
  EXPECT_EQ(Query().terminal_status, AMDF_STATUS_OK);
  EXPECT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, submission);
}

TEST_P(GpuKernelQueueTest, DefaultCapacityAcceptsAnEntirePendingWindow) {
  amdf_kernel_queue_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO;
  info.structure_size = sizeof(info);
  ASSERT_EQ(amdf_kernel_queue_query_info(queue, &info), AMDF_STATUS_OK);
  ASSERT_GT(info.maximum_pending_submission_count, 1u);
  uint64_t submission = 0;
  for (uint32_t i = 0; i < info.maximum_pending_submission_count; ++i) {
    ASSERT_EQ(Submit(&submission), AMDF_STATUS_OK);
  }
  EXPECT_EQ(device.native.queue.progress.load(), 0u);
  EXPECT_EQ(Query().retired_submission, 0u);
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(Submit(&rejected), amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  EXPECT_EQ(rejected, UINT64_MAX);
  ASSERT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, submission);
}

TEST_P(GpuKernelQueueTest, RefreshChecksCompletedPrefixBeforePendingTail) {
  EXPECT_EQ(Refresh().retired_submission, 0u);
  ASSERT_EQ(CreateQueue(3), AMDF_STATUS_OK);
  uint64_t points[3] = {};
  for (auto& point : points) {
    ASSERT_EQ(Submit(&point), AMDF_STATUS_OK);
  }
  device.native.queue.progress = points[1];
  EXPECT_EQ(Query().retired_submission, 0u);
  auto checked = Refresh();
  EXPECT_EQ(checked.retired_submission, points[1]);
  EXPECT_EQ(checked.terminal_status, AMDF_STATUS_OK);
  EXPECT_EQ(checked.state, AMDF_QUEUE_STATE_ACTIVE);
  EXPECT_EQ(device.native.queue.progress.load(), points[1]);
  EXPECT_EQ(Query().retired_submission, points[1]);
  EXPECT_EQ(Refresh().retired_submission, points[1]);
  device.native.queue.progress = points[2];
  EXPECT_EQ(Refresh().retired_submission, points[2]);
}

TEST_P(GpuKernelQueueTest, RefreshSeparatesTerminalFailureFromProgress) {
  uint64_t submission = 0;
  ASSERT_EQ(Submit(&submission), AMDF_STATUS_OK);
  const auto failure = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  device.native.queue.terminal_status = failure;
  auto checked = Refresh();
  EXPECT_EQ(checked.retired_submission, 0u);
  EXPECT_EQ(checked.terminal_status, failure);
  EXPECT_EQ(checked.state, AMDF_QUEUE_STATE_DEVICE_LOST);
  device.native.queue.progress = submission;
  checked = Refresh();
  EXPECT_EQ(checked.retired_submission, submission);
  EXPECT_EQ(checked.terminal_status, failure);
}

TEST_P(GpuKernelQueueTest, ConfiguredCapacityReclaimsCompletedCreditsOnSubmit) {
  ASSERT_EQ(CreateQueue(3), AMDF_STATUS_OK);
  uint64_t submission = 0;
  for (uint32_t round = 0; round < 12; ++round) {
    for (uint32_t i = 0; i < 3; ++i) {
      ASSERT_EQ(Submit(&submission), AMDF_STATUS_OK);
    }
    uint64_t rejected = UINT64_MAX;
    EXPECT_EQ(Submit(&rejected), amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
    EXPECT_EQ(rejected, UINT64_MAX);
    device.native.queue.progress = submission;
    // No explicit wait or retirement operation intervenes before the next
    // round.
  }
  ASSERT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, submission);
}

TEST_P(GpuKernelQueueTest, RejectionPreservesTheAcceptedPrefixAndCapacity) {
  ASSERT_EQ(CreateQueue(2), AMDF_STATUS_OK);
  uint64_t first = 0;
  ASSERT_EQ(Submit(&first), AMDF_STATUS_OK);
  const auto failure =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  device.native.queue.submit_status = failure;
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(Submit(&rejected), failure);
  EXPECT_EQ(rejected, UINT64_MAX);
  EXPECT_EQ(device.native.queue.submitted, first);
  device.native.queue.submit_status = AMDF_STATUS_OK;
  uint64_t second = 0;
  ASSERT_EQ(Submit(&second), AMDF_STATUS_OK);
  EXPECT_GT(second, first);
  EXPECT_EQ(Submit(&rejected), amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  ASSERT_EQ(amdf_kernel_queue_wait(queue, second, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, second);
  EXPECT_EQ(amdf_kernel_queue_wait(queue, first, 0, 0), AMDF_STATUS_OK);
}

TEST_P(GpuKernelQueueTest, SingleEntryWindowNeedsNoIntermediateHostWait) {
  ASSERT_EQ(CreateQueue(1), AMDF_STATUS_OK);
  uint64_t first = 0;
  ASSERT_EQ(Submit(&first), AMDF_STATUS_OK);
  uint64_t second = UINT64_MAX;
  EXPECT_EQ(Submit(&second), amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  EXPECT_EQ(second, UINT64_MAX);
  device.native.queue.progress = first;
  ASSERT_EQ(Submit(&second), AMDF_STATUS_OK);
  EXPECT_GT(second, first);
  EXPECT_EQ(Query().retired_submission, first);
  ASSERT_EQ(amdf_kernel_queue_wait(queue, second, 0, 0), AMDF_STATUS_OK);
}

TEST_P(GpuKernelQueueTest, TerminalFailureTakesPrecedenceOverFullCapacity) {
  ASSERT_EQ(CreateQueue(1), AMDF_STATUS_OK);
  uint64_t submission = 0;
  ASSERT_EQ(Submit(&submission), AMDF_STATUS_OK);
  const auto failure = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  device.native.queue.terminal_status = failure;
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(Submit(&rejected), failure);
  EXPECT_EQ(rejected, UINT64_MAX);
  EXPECT_EQ(Query().retired_submission, 0u);
}

TEST_P(GpuKernelQueueTest, WaitingForPrefixDoesNotRetireLaterWork) {
  ASSERT_EQ(CreateQueue(2), AMDF_STATUS_OK);
  uint64_t first = 0;
  uint64_t second = 0;
  ASSERT_EQ(Submit(&first), AMDF_STATUS_OK);
  ASSERT_EQ(Submit(&second), AMDF_STATUS_OK);
  ASSERT_EQ(amdf_kernel_queue_wait(queue, first, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, first);
  EXPECT_EQ(amdf_kernel_queue_destroy(queue),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  uint64_t third = 0;
  ASSERT_EQ(Submit(&third), AMDF_STATUS_OK);
  EXPECT_GT(third, second);
  ASSERT_EQ(amdf_kernel_queue_wait(queue, third, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, third);
}

TEST_P(GpuKernelQueueTest, NativeCompletionCannotExposeUnpublishedAcceptance) {
  ASSERT_EQ(CreateQueue(3), AMDF_STATUS_OK);
  uint64_t first = 0;
  ASSERT_EQ(Submit(&first), AMDF_STATUS_OK);
  auto& native = device.native.queue;
  using Phase = amdf_gpu_umd_kernel_queue_t::PublicationPhase;
  native.publication_phase = Phase::kPauseOnAccept;
  uint64_t second = 0;
  amdf_status_t second_status = AMDF_STATUS_OK;
  std::thread publisher([&] { second_status = Submit(&second); });
  {
    std::unique_lock<std::mutex> lock(native.publication_mutex);
    native.publication_condition.wait(
        lock, [&] { return native.publication_phase == Phase::kWaiting; });
  }
  EXPECT_EQ(Refresh().retired_submission, first);
  EXPECT_EQ(amdf_kernel_queue_wait(queue, first, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, first);
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(Submit(&rejected), amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  EXPECT_EQ(rejected, UINT64_MAX);
  {
    std::lock_guard<std::mutex> lock(native.publication_mutex);
    native.publication_phase = Phase::kResume;
  }
  native.publication_condition.notify_one();
  publisher.join();
  ASSERT_EQ(second_status, AMDF_STATUS_OK);
  EXPECT_GT(second, first);
  ASSERT_EQ(amdf_kernel_queue_wait(queue, second, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, second);
}

TEST_P(GpuKernelQueueTest, ZeroTimeoutRefreshesNativeProgress) {
  uint64_t submission = 0;
  ASSERT_EQ(Submit(&submission), AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, 0u);
  ASSERT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(device.native.queue.progress, device.native.queue.submitted);
  EXPECT_EQ(Query().retired_submission, submission);
}

TEST_P(GpuKernelQueueTest, CachedTerminalFailureDoesNotProveRetirement) {
  uint64_t submission = 0;
  ASSERT_EQ(Submit(&submission), AMDF_STATUS_OK);
  const auto failure = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  device.native.queue.terminal_status = failure;
  const auto status = Query();
  EXPECT_EQ(status.retired_submission, 0u);
  EXPECT_EQ(status.terminal_status, failure);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_DEVICE_LOST);
  EXPECT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0), failure);
  EXPECT_EQ(Query().retired_submission, 0u);
  device.native.queue.progress = device.native.queue.submitted;
  EXPECT_EQ(Query().retired_submission, 0u);
  EXPECT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0), failure);
  EXPECT_EQ(Query().retired_submission, submission);
}

TEST_P(GpuKernelQueueTest, NativeWaitErrorDoesNotEraseConfirmedRetirement) {
  uint64_t submission = 0;
  ASSERT_EQ(Submit(&submission), AMDF_STATUS_OK);
  device.native.queue.wait_status =
      amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  EXPECT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0),
            device.native.queue.wait_status);
  const auto status = Query();
  EXPECT_EQ(status.retired_submission, submission);
  EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
}

TEST_P(GpuKernelQueueTest, RejectedDestroyRetainsButNativeFailureConsumes) {
  uint64_t submission = 0;
  ASSERT_EQ(Submit(&submission), AMDF_STATUS_OK);
  EXPECT_EQ(amdf_kernel_queue_destroy(queue),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  EXPECT_EQ(device.native.queue.destroy_count, 0u);
  EXPECT_EQ(amdf_child_tracker_count(&device.base.children), 1u);

  // ERROR_INVALID_HANDLE numerically overlaps API-domain BUSY. Ownership is
  // determined by the complete status, never by its domain-specific code alone.
  const auto failure = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, 6);
  device.native.queue.destroy_status = failure;
  device.native.queue.progress = device.native.queue.submitted;
  EXPECT_EQ(amdf_kernel_queue_destroy(queue), failure);
  queue = nullptr;
  EXPECT_EQ(device.native.queue.destroy_count, 1u);
  EXPECT_EQ(amdf_child_tracker_count(&device.base.children), 0u);
}

INSTANTIATE_TEST_SUITE_P(CommandFormats, GpuKernelQueueTest,
                         ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                           AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA));

}  // namespace

extern "C" {

amdf_status_t amdf_endpoint_register_device(amdf_endpoint_t*) {
  return AMDF_STATUS_OK;
}
void amdf_endpoint_unregister_device(amdf_endpoint_t*) {}
amdf_instance_t* amdf_endpoint_get_instance(const amdf_endpoint_t*) {
  return reinterpret_cast<amdf_instance_t*>(uintptr_t{1});
}
amdf_allocator_t amdf_endpoint_host_allocator(const amdf_endpoint_t*) {
  return amdf_allocator_system();
}
amdf_status_t AMDF_CALL amdf_endpoint_query_queue_family_info(
    amdf_endpoint_t* endpoint, uint32_t ordinal,
    amdf_queue_family_info_t* out_info) {
  out_info->ordinal = ordinal;
  out_info->command_type = reinterpret_cast<Device*>(endpoint)->command_type;
  out_info->publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL;
  return AMDF_STATUS_OK;
}
const amdf_gpu_device_info_t* amdf_gpu_device_get_info(
    const amdf_device_t* device) {
  return &reinterpret_cast<const Device*>(device)->info;
}
uint64_t amdf_gpu_device_query_reset_epoch(const amdf_device_t* device) {
  return amdf_gpu_device_get_info(device)->reset_epoch;
}
amdf_gpu_umd_device_t* amdf_gpu_device_get_umd(amdf_device_t* device) {
  return &reinterpret_cast<Device*>(device)->native;
}
amdf_status_t amdf_gpu_umd_kernel_queue_create(
    amdf_gpu_umd_device_t* device, amdf_queue_command_type_t,
    amdf_gpu_umd_kernel_queue_t** out_queue) {
  *out_queue = &device->queue;
  return AMDF_STATUS_OK;
}
amdf_status_t amdf_gpu_umd_kernel_queue_submit(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t, uint64_t,
    uint64_t submission) {
  if (!amdf_status_is_ok(queue->terminal_status)) {
    return queue->terminal_status;
  }
  if (!amdf_status_is_ok(queue->submit_status)) {
    return queue->submit_status;
  }
  EXPECT_EQ(submission, queue->submitted + 1);
  queue->submitted = submission;
  using Phase = amdf_gpu_umd_kernel_queue_t::PublicationPhase;
  std::unique_lock<std::mutex> lock(queue->publication_mutex);
  if (queue->publication_phase == Phase::kPauseOnAccept) {
    queue->progress.store(submission, std::memory_order_release);
    queue->publication_phase = Phase::kWaiting;
    queue->publication_condition.notify_one();
    queue->publication_condition.wait(
        lock, [&] { return queue->publication_phase == Phase::kResume; });
    queue->publication_phase = Phase::kOpen;
  }
  return AMDF_STATUS_OK;
}
uint64_t amdf_gpu_umd_kernel_queue_query_progress(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  return queue->progress.load(std::memory_order_acquire);
}
amdf_status_t amdf_gpu_umd_kernel_queue_query_terminal_status(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  return queue->terminal_status;
}
amdf_status_t amdf_gpu_umd_kernel_queue_wait(amdf_gpu_umd_kernel_queue_t* queue,
                                             uint64_t submission,
                                             const amdf_wait_deadline_t*) {
  queue->progress = submission;
  return queue->wait_status;
}
amdf_status_t amdf_gpu_umd_kernel_queue_destroy(
    amdf_gpu_umd_kernel_queue_t* queue) {
  ++queue->destroy_count;
  return queue->destroy_status;
}

}  // extern "C"
