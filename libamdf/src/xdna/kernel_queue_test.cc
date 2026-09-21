// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/kernel_queue.h"

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/device.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/kernel_queue.h"
#include "libamdf/src/memory_resource.h"
#include "libamdf/src/memory_scope.h"
#include "libamdf/src/xdna/device.h"
#include "libamdf/src/xdna/device_profile.h"
#include "libamdf/src/xdna/umd/kernel_queue.h"

// Native dependencies are controlled here; submission ownership, retirement,
// public status, and wait-budget handling use the production queue code.
struct amdf_xdna_umd_kernel_queue_t {
  // Native completion frontier published independently of host retirement.
  std::atomic<uint64_t> progress{0};
  // Driver progress made visible by a nonblocking native refresh.
  uint64_t available_progress = 0;
  // Native query failure independent of execution or accepted ownership.
  amdf_status_t refresh_status = AMDF_STATUS_OK;
  // Number of explicit native progress refreshes.
  std::atomic<size_t> refresh_count{0};
  // Native wake representation qualified before publication of the queue.
  amdf_native_event_types_t notification_types =
      AMDF_NATIVE_EVENT_TYPE_BIT_EVENTFD;
  // Exact native point requested, or zero for an already-checked public point.
  uint64_t notification_submission = UINT64_MAX;
  // Number of independent requests received, including repeated points.
  uint32_t notification_count = 0;
  // Native registration error independent of accepted command ownership.
  amdf_status_t notification_status = AMDF_STATUS_OK;
  // Packet slots provided by the controlled native dependency.
  struct Slot {
    // Accepted native identity, or zero after result consumption.
    uint64_t native_submission = 0;
    // Number of checked results consumed from this storage location.
    uint32_t retirement_count = 0;
  };
  // Stable storage for every native packet, allocated only at creation.
  std::vector<Slot> slots;
  // Sticky execution failure observed only during protected retirement.
  std::atomic<amdf_status_t> terminal_status{AMDF_STATUS_OK};
  // Native wait behavior selected before a test starts observing the queue.
  enum class WaitAction {
    kTimeout,
    kComplete,
    kCompleteWithError
  } action = WaitAction::kComplete;
  // Native execution result consumed after completion proves retirement.
  amdf_status_t completion_status = AMDF_STATUS_OK;
  // Number of explicit native wait/poll calls.
  std::atomic<size_t> wait_count{0};
  // Native rejection selected before publication.
  amdf_status_t submission_status = AMDF_STATUS_OK;
  // Native cleanup result after the shared layer establishes retirement.
  amdf_status_t destroy_status = AMDF_STATUS_OK;
  // Number of final native release attempts.
  uint32_t destroy_count = 0;
  // Monotonic native submission count.
  uint64_t submitted = 0;
  // Exact instruction range passed directly to the native transport.
  uint64_t pending_address = 0;
  // Byte length passed without instruction interpretation.
  uint32_t pending_byte_length = 0;
  // Coordinates a preempted observer inside its protected retirement section.
  mutable std::mutex mutex;
  // Explicit readiness/release condition for the retirement observer.
  mutable std::condition_variable condition;
  // Retirement pause phase; all accesses occur under mutex.
  mutable enum class Phase {
    kRunning,
    kPauseRequested,
    kPaused,
    kReleased
  } phase = Phase::kRunning;
  // Separate publication schedule, protected by the same test-only mutex.
  Phase publication_phase = Phase::kRunning;
};

struct amdf_xdna_umd_context_t {
  // Context-owned native queue dependency leased by the production queue.
  amdf_xdna_umd_kernel_queue_t queue;
};

namespace {

struct Device {
  // Production generic device base consumed by queue creation.
  amdf_device_t base = {};
  // Immutable identity and instruction contract returned by the device.
  amdf_xdna_device_info_t info = {};
  // Cached profile consumed by production input validation.
  amdf_xdna_device_profile_t profile = {};
};

struct Context {
  // Device borrowed by the scheduling context.
  amdf_device_t* device = nullptr;
  // Immutable context identity used to initialize queue information.
  amdf_xdna_context_info_t info = {};
  // Controlled native queue dependency.
  amdf_xdna_umd_context_t native;
  // Production child ownership observed during queue lifetime.
  amdf_child_tracker_t children;
  // Exact context qualification for private instruction addresses.
  amdf_memory_scope_t memory_scope = {};
};

class XdnaKernelQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device.base.engine_kind = AMDF_ENGINE_KIND_XDNA;
    device.base.host_allocator = amdf_allocator_system();
    amdf_child_tracker_initialize(&device.base.children);
    device.info.reset_epoch = 1;
    device.info.instruction.maximum_byte_length = UINT32_MAX & ~uint64_t{3};
    device.info.instruction.address_alignment = 4;
    device.info.instruction.byte_length_granularity = 4;
    device.profile.info = &device.info;
    context.device = &device.base;
    context.info.device_id = device.info.id;
    context.info.reset_epoch = 1;
    amdf_child_tracker_initialize(&context.children);
    ASSERT_EQ(
        amdf_memory_resource_allocate(amdf_allocator_system(), 1, &memory),
        AMDF_STATUS_OK);
    memory->scope = &context.memory_scope;
    memory->info.byte_length = 4096;
    memory->accesses[0].info.access = AMDF_MEMORY_ACCESS_EXECUTE;
    memory->accesses[0].info.reset_epoch = 1;
    memory->accesses[0].info.address_kinds =
        uint64_t{1} << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE;
    memory->accesses[0].addresses[AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE] =
        0x100000000;
    command.memory = memory;
    command.byte_offset = 64;
    command.byte_length = 20;
    ASSERT_EQ(CreateQueue(0), AMDF_STATUS_OK);
  }

  amdf_status_t CreateQueue(uint32_t capacity) {
    if (queue) {
      const auto status = amdf_kernel_queue_destroy(queue);
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      queue = nullptr;
    }
    amdf_xdna_kernel_queue_create_info_t create = {};
    create.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO;
    create.structure_size = sizeof(create);
    create.maximum_pending_submission_count = capacity;
    return amdf_xdna_kernel_queue_create(
        reinterpret_cast<amdf_xdna_context_t*>(&context), &create, &queue);
  }

  void TearDown() override {
    context.native.queue.progress = context.native.queue.submitted;
    if (queue) {
      EXPECT_EQ(amdf_kernel_queue_destroy(queue), AMDF_STATUS_OK);
    }
    amdf_free(amdf_allocator_system(), memory);
    EXPECT_EQ(amdf_child_tracker_count(&device.base.children), 0u);
    EXPECT_EQ(amdf_child_tracker_count(&context.children), 0u);
  }

  amdf_status_t SubmitCommand(uint64_t* out_submission) {
    amdf_xdna_kernel_queue_submission_info_t submit = {};
    submit.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
    submit.structure_size = sizeof(submit);
    submit.command_count = 1;
    submit.commands = &command;
    return amdf_xdna_kernel_queue_submit(queue, &submit, out_submission);
  }

  void Submit() {
    ASSERT_EQ(SubmitCommand(&submission), AMDF_STATUS_OK);
    ASSERT_EQ(submission, context.native.queue.submitted);
    EXPECT_EQ(context.native.queue.pending_address,
              memory->accesses[0].addresses[AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE] +
                  command.byte_offset);
    EXPECT_EQ(context.native.queue.pending_byte_length, command.byte_length);
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

  // Device dependency retained until production queue teardown.
  Device device;
  // Context dependency retained until production queue teardown.
  Context context;
  // Caller-owned memory kept live until command retirement.
  amdf_memory_t* memory = nullptr;
  // Public instruction range; the native dependency receives only its address.
  amdf_xdna_kernel_command_t command = {};
  // Production queue under test.
  amdf_kernel_queue_t* queue = nullptr;
  // Accepted public submission identity.
  uint64_t submission = 0;
};

TEST_F(XdnaKernelQueueTest, DefaultCapacityHoldsAnEntirePendingWindow) {
  amdf_kernel_queue_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO;
  info.structure_size = sizeof(info);
  ASSERT_EQ(amdf_kernel_queue_query_info(queue, &info), AMDF_STATUS_OK);
  ASSERT_GT(info.maximum_pending_submission_count, 1u);
  uint64_t last = 0;
  for (uint32_t i = 0; i < info.maximum_pending_submission_count; ++i) {
    ASSERT_EQ(SubmitCommand(&last), AMDF_STATUS_OK);
  }
  EXPECT_EQ(context.native.queue.progress.load(), 0u);
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(SubmitCommand(&rejected),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  EXPECT_EQ(rejected, UINT64_MAX);
  ASSERT_EQ(amdf_kernel_queue_wait(queue, last, AMDF_TIMEOUT_INFINITE, 0),
            AMDF_STATUS_OK);
  for (const auto& slot : context.native.queue.slots) {
    EXPECT_EQ(slot.native_submission, 0u);
    EXPECT_EQ(slot.retirement_count, 1u);
  }
}

TEST_F(XdnaKernelQueueTest, WrapsAndReclaimsResultsWithoutHostWaits) {
  ASSERT_EQ(CreateQueue(3), AMDF_STATUS_OK);
  uint64_t last = 0;
  for (uint32_t round = 0; round < 12; ++round) {
    for (uint32_t i = 0; i < 3; ++i) {
      ASSERT_EQ(SubmitCommand(&last), AMDF_STATUS_OK);
    }
    uint64_t rejected = UINT64_MAX;
    EXPECT_EQ(SubmitCommand(&rejected),
              amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
    context.native.queue.available_progress = context.native.queue.submitted;
  }
  ASSERT_EQ(amdf_kernel_queue_wait(queue, last, AMDF_TIMEOUT_INFINITE, 0),
            AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, last);
  for (const auto& slot : context.native.queue.slots) {
    EXPECT_EQ(slot.retirement_count, 12u);
  }
}

TEST_F(XdnaKernelQueueTest, ReclaimsOnlyCompletedPrefixAndPreservesRejections) {
  ASSERT_EQ(CreateQueue(2), AMDF_STATUS_OK);
  uint64_t first = 0, second = 0, third = 0;
  ASSERT_EQ(SubmitCommand(&first), AMDF_STATUS_OK);
  const auto failure = amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM);
  context.native.queue.submission_status = failure;
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(SubmitCommand(&rejected), failure);
  EXPECT_EQ(rejected, UINT64_MAX);
  context.native.queue.submission_status = AMDF_STATUS_OK;
  ASSERT_EQ(SubmitCommand(&second), AMDF_STATUS_OK);
  context.native.queue.available_progress = first;
  ASSERT_EQ(SubmitCommand(&third), AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, first);
  EXPECT_EQ(amdf_kernel_queue_wait(queue, first, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(context.native.queue.wait_count.load(), 0u);
  EXPECT_EQ(SubmitCommand(&rejected),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  ASSERT_EQ(amdf_kernel_queue_wait(queue, third, AMDF_TIMEOUT_INFINITE, 0),
            AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, third);
}

TEST_F(XdnaKernelQueueTest, RefreshErrorPreservesEveryAcceptedPacket) {
  ASSERT_EQ(CreateQueue(1), AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(Submit());
  const auto failure = amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO);
  context.native.queue.refresh_status = failure;
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(SubmitCommand(&rejected), failure);
  EXPECT_EQ(rejected, UINT64_MAX);
  EXPECT_EQ(Query().retired_submission, 0u);
  EXPECT_EQ(context.native.queue.slots[0].native_submission, submission);
  context.native.queue.refresh_status = AMDF_STATUS_OK;
  context.native.queue.available_progress = submission;
  ASSERT_EQ(SubmitCommand(&rejected), AMDF_STATUS_OK);
  EXPECT_GT(rejected, submission);
}

TEST_F(XdnaKernelQueueTest, CompletedFailureReclaimsWholeNativePrefix) {
  ASSERT_EQ(CreateQueue(3), AMDF_STATUS_OK);
  uint64_t last = 0;
  for (uint32_t i = 0; i < 3; ++i) {
    ASSERT_EQ(SubmitCommand(&last), AMDF_STATUS_OK);
  }
  context.native.queue.available_progress = last;
  const auto failure = amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, 5);
  context.native.queue.completion_status = failure;
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(SubmitCommand(&rejected), failure);
  EXPECT_EQ(rejected, UINT64_MAX);
  EXPECT_EQ(Query().retired_submission, last);
  for (const auto& slot : context.native.queue.slots) {
    EXPECT_EQ(slot.retirement_count, 1u);
  }
}

TEST_F(XdnaKernelQueueTest, NativePointsAreIndependentOfPublicNumbering) {
  context.native.queue.submitted = 37;
  context.native.queue.progress = 37;
  uint64_t point = 0;
  ASSERT_EQ(SubmitCommand(&point), AMDF_STATUS_OK);
  EXPECT_NE(point, context.native.queue.submitted);
  ASSERT_EQ(amdf_kernel_queue_wait(queue, point, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(context.native.queue.progress.load(), 38u);
  EXPECT_EQ(Query().retired_submission, point);
}

TEST_F(XdnaKernelQueueTest, CompletionCannotExposeUnpublishedAcceptance) {
  ASSERT_EQ(CreateQueue(2), AMDF_STATUS_OK);
  uint64_t first = 0;
  ASSERT_EQ(SubmitCommand(&first), AMDF_STATUS_OK);
  auto& native = context.native.queue;
  using Phase = amdf_xdna_umd_kernel_queue_t::Phase;
  native.publication_phase = Phase::kPauseRequested;
  uint64_t second = 0;
  amdf_status_t status = AMDF_STATUS_OK;
  std::thread publisher([&] { status = SubmitCommand(&second); });
  {
    std::unique_lock<std::mutex> lock(native.mutex);
    native.condition.wait(
        lock, [&] { return native.publication_phase == Phase::kPaused; });
  }
  EXPECT_EQ(amdf_kernel_queue_wait(queue, first, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, first);
  EXPECT_EQ(native.slots[1].retirement_count, 0u);
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(SubmitCommand(&rejected),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  {
    std::lock_guard<std::mutex> lock(native.mutex);
    native.publication_phase = Phase::kReleased;
  }
  native.condition.notify_all();
  publisher.join();
  ASSERT_EQ(status, AMDF_STATUS_OK);
  ASSERT_EQ(amdf_kernel_queue_wait(queue, second, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(native.slots[1].retirement_count, 1u);
}

TEST_F(XdnaKernelQueueTest, RetirementDoesNotPreventPublicationIntoFreeSlots) {
  ASSERT_EQ(CreateQueue(2), AMDF_STATUS_OK);
  uint64_t first = 0;
  ASSERT_EQ(SubmitCommand(&first), AMDF_STATUS_OK);
  auto& native = context.native.queue;
  using Phase = amdf_xdna_umd_kernel_queue_t::Phase;
  native.phase = Phase::kPauseRequested;
  native.progress = native.submitted;
  amdf_status_t status = AMDF_STATUS_OK;
  std::thread waiter([&] {
    status = amdf_kernel_queue_wait(queue, first, AMDF_TIMEOUT_INFINITE, 0);
  });
  {
    std::unique_lock<std::mutex> lock(native.mutex);
    native.condition.wait(lock, [&] { return native.phase == Phase::kPaused; });
  }
  uint64_t second = 0;
  EXPECT_EQ(SubmitCommand(&second), AMDF_STATUS_OK);
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(SubmitCommand(&rejected),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  EXPECT_EQ(Query().retired_submission, 0u);
  amdf_native_event_t event = {};
  event.type = AMDF_NATIVE_EVENT_TYPE_EVENTFD;
  event.payload.file_descriptor = 5;
  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_EQ(amdf_kernel_queue_request_notification(queue, first, &event),
              AMDF_STATUS_OK);
    EXPECT_EQ(native.notification_submission, first);
    // A busy result consumer does not own notification or force a handoff.
    EXPECT_EQ(Refresh().retired_submission, 0u);
  }
  {
    std::lock_guard<std::mutex> lock(native.mutex);
    native.phase = Phase::kReleased;
  }
  native.condition.notify_all();
  waiter.join();
  ASSERT_EQ(status, AMDF_STATUS_OK);
  EXPECT_EQ(amdf_kernel_queue_request_notification(queue, first, &event),
            AMDF_STATUS_OK);
  EXPECT_EQ(native.notification_submission, 0u);
  EXPECT_EQ(native.notification_count, 3u);
  ASSERT_EQ(amdf_kernel_queue_wait(queue, second, AMDF_TIMEOUT_INFINITE, 0),
            AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, second);
}

TEST_F(XdnaKernelQueueTest, TerminalFailureStillRefreshesLaterNativeProgress) {
  ASSERT_EQ(CreateQueue(2), AMDF_STATUS_OK);
  uint64_t first = 0;
  uint64_t second = 0;
  ASSERT_EQ(SubmitCommand(&first), AMDF_STATUS_OK);
  ASSERT_EQ(SubmitCommand(&second), AMDF_STATUS_OK);
  auto& native = context.native.queue;
  const auto failure = amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, 5);
  native.completion_status = failure;
  EXPECT_EQ(amdf_kernel_queue_wait(queue, first, AMDF_TIMEOUT_INFINITE, 0),
            failure);
  EXPECT_EQ(Query().retired_submission, first);
  native.available_progress = native.submitted;
  EXPECT_EQ(amdf_kernel_queue_wait(queue, second, 0, 0), failure);
  EXPECT_EQ(Query().retired_submission, second);
}

TEST_F(XdnaKernelQueueTest, ZeroTimeoutRefreshesNativeProgress) {
  ASSERT_NO_FATAL_FAILURE(Submit());
  EXPECT_EQ(Query().retired_submission, 0u);
  EXPECT_EQ(context.native.queue.wait_count.load(), 0u);
  EXPECT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(context.native.queue.wait_count.load(), 1u);
  EXPECT_EQ(Query().retired_submission, submission);
}

TEST_F(XdnaKernelQueueTest, RefreshChecksAvailablePrefixWithoutWaiting) {
  ASSERT_EQ(CreateQueue(3), AMDF_STATUS_OK);
  uint64_t points[3] = {};
  for (auto& point : points) {
    ASSERT_EQ(SubmitCommand(&point), AMDF_STATUS_OK);
  }
  context.native.queue.available_progress = points[1];
  EXPECT_EQ(Query().retired_submission, 0u);
  const auto checked = Refresh();
  EXPECT_EQ(checked.retired_submission, points[1]);
  EXPECT_EQ(checked.terminal_status, AMDF_STATUS_OK);
  EXPECT_EQ(checked.state, AMDF_QUEUE_STATE_ACTIVE);
  EXPECT_EQ(Query().retired_submission, points[1]);
  EXPECT_EQ(context.native.queue.wait_count.load(), 0u);
  EXPECT_EQ(context.native.queue.slots[0].retirement_count, 1u);
  EXPECT_EQ(context.native.queue.slots[1].retirement_count, 1u);
  EXPECT_EQ(context.native.queue.slots[2].retirement_count, 0u);
  EXPECT_EQ(Refresh().retired_submission, points[1]);
  context.native.queue.available_progress = points[2];
  EXPECT_EQ(Refresh().retired_submission, points[2]);
  for (const auto& slot : context.native.queue.slots) {
    EXPECT_EQ(slot.retirement_count, 1u);
  }
}

TEST_F(XdnaKernelQueueTest, NotificationResolvesNativePointsWithoutRetirement) {
  auto& native = context.native.queue;
  native.submitted = 37;
  native.progress = 37;
  ASSERT_EQ(SubmitCommand(&submission), AMDF_STATUS_OK);
  amdf_kernel_queue_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO;
  info.structure_size = sizeof(info);
  ASSERT_EQ(amdf_kernel_queue_query_info(queue, &info), AMDF_STATUS_OK);
  ASSERT_NE(info.notification_types & AMDF_NATIVE_EVENT_TYPE_BIT_EVENTFD, 0u);
  amdf_native_event_t event = {};
  event.type = AMDF_NATIVE_EVENT_TYPE_EVENTFD;
  event.payload.file_descriptor = 5;
  EXPECT_EQ(amdf_kernel_queue_request_notification(queue, submission, &event),
            AMDF_STATUS_OK);
  EXPECT_EQ(native.notification_submission, 38u);
  EXPECT_EQ(native.notification_count, 1u);
  EXPECT_EQ(Query().retired_submission, 0u);
  native.progress = native.submitted;
  EXPECT_EQ(amdf_kernel_queue_request_notification(queue, submission, &event),
            AMDF_STATUS_OK);
  EXPECT_EQ(native.notification_submission, 38u);
  EXPECT_EQ(native.notification_count, 2u);
  EXPECT_EQ(Query().retired_submission, 0u);
  EXPECT_EQ(native.wait_count.load(), 0u);
  EXPECT_EQ(native.refresh_count.load(), 0u);
}

TEST_F(XdnaKernelQueueTest, NotificationOfRecycledPointRequestsFreshHint) {
  ASSERT_EQ(CreateQueue(1), AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(Submit());
  auto& native = context.native.queue;
  native.available_progress = native.submitted;
  uint64_t next = 0;
  ASSERT_EQ(SubmitCommand(&next), AMDF_STATUS_OK);
  ASSERT_EQ(Query().retired_submission, submission);
  ASSERT_EQ(native.slots[0].native_submission, next);
  amdf_native_event_t event = {};
  event.type = AMDF_NATIVE_EVENT_TYPE_EVENTFD;
  event.payload.file_descriptor = 5;
  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_EQ(amdf_kernel_queue_request_notification(queue, submission, &event),
              AMDF_STATUS_OK);
    EXPECT_EQ(native.notification_submission, 0u);
    EXPECT_EQ(native.notification_count, i + 1);
  }
  EXPECT_EQ(native.slots[0].native_submission, next);
  EXPECT_EQ(Query().retired_submission, submission);
}

TEST_F(XdnaKernelQueueTest, NotificationErrorDoesNotRejectAcceptedWork) {
  ASSERT_NO_FATAL_FAILURE(Submit());
  auto& native = context.native.queue;
  native.notification_status =
      amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM);
  amdf_native_event_t event = {};
  event.type = AMDF_NATIVE_EVENT_TYPE_EVENTFD;
  event.payload.file_descriptor = 5;
  EXPECT_EQ(amdf_kernel_queue_request_notification(queue, submission, &event),
            native.notification_status);
  EXPECT_EQ(native.submitted, submission);
  EXPECT_EQ(native.slots[0].native_submission, submission);
  EXPECT_EQ(Query().retired_submission, 0u);
  EXPECT_EQ(Query().terminal_status, AMDF_STATUS_OK);
  ASSERT_EQ(amdf_kernel_queue_wait(queue, submission, AMDF_TIMEOUT_INFINITE, 0),
            AMDF_STATUS_OK);
  EXPECT_EQ(native.submitted, submission);
}

TEST_F(XdnaKernelQueueTest, NotificationRejectsInvalidOrUnsupportedInputs) {
  ASSERT_NO_FATAL_FAILURE(Submit());
  amdf_native_event_t event = {};
  event.type = AMDF_NATIVE_EVENT_TYPE_EVENTFD;
  event.payload.file_descriptor = 5;
  EXPECT_EQ(amdf_status_code(amdf_kernel_queue_request_notification(
                nullptr, submission, &event)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(amdf_status_code(amdf_kernel_queue_request_notification(
                queue, submission, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(amdf_status_code(
                amdf_kernel_queue_request_notification(queue, 0, &event)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(amdf_status_code(amdf_kernel_queue_request_notification(
                queue, submission + 1, &event)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  event.reserved = 1;
  EXPECT_EQ(amdf_status_code(amdf_kernel_queue_request_notification(
                queue, submission, &event)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  event.reserved = 0;
  event.payload.file_descriptor = -1;
  EXPECT_EQ(amdf_status_code(amdf_kernel_queue_request_notification(
                queue, submission, &event)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  event.payload.file_descriptor = INT64_MAX;
  EXPECT_EQ(amdf_status_code(amdf_kernel_queue_request_notification(
                queue, submission, &event)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  event.type = UINT32_MAX;
  EXPECT_EQ(amdf_status_code(amdf_kernel_queue_request_notification(
                queue, submission, &event)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  event.type = AMDF_NATIVE_EVENT_TYPE_WIN32_EVENT;
  event.payload.native_handle = nullptr;
  EXPECT_EQ(amdf_status_code(amdf_kernel_queue_request_notification(
                queue, submission, &event)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  event.payload.native_handle = reinterpret_cast<void*>(UINTPTR_MAX);
  EXPECT_EQ(amdf_status_code(amdf_kernel_queue_request_notification(
                queue, submission, &event)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  event.payload.native_handle = reinterpret_cast<void*>(uintptr_t{5});
  EXPECT_EQ(amdf_status_code(amdf_kernel_queue_request_notification(
                queue, submission, &event)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(context.native.queue.notification_count, 0u);
  EXPECT_EQ(Query().retired_submission, 0u);
}

TEST_F(XdnaKernelQueueTest, IdleRefreshDoesNotObserveAnUnsubmittedFence) {
  auto& native = context.native.queue;
  const auto failure = amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO);
  native.refresh_status = failure;
  EXPECT_EQ(Refresh().retired_submission, 0u);
  EXPECT_EQ(native.refresh_count.load(), 0u);
  native.refresh_status = AMDF_STATUS_OK;
  ASSERT_NO_FATAL_FAILURE(Submit());
  native.available_progress = native.submitted;
  EXPECT_EQ(Refresh().retired_submission, submission);
  EXPECT_EQ(native.refresh_count.load(), 1u);
  native.refresh_status = failure;
  EXPECT_EQ(Refresh().retired_submission, submission);
  EXPECT_EQ(native.refresh_count.load(), 1u);
}

TEST_F(XdnaKernelQueueTest, RefreshErrorPreservesOutputAndKnownRetirement) {
  ASSERT_NO_FATAL_FAILURE(Submit());
  auto& native = context.native.queue;
  native.refresh_status = amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO);
  amdf_kernel_queue_status_t output = {};
  output.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS;
  output.structure_size = sizeof(output);
  output.retired_submission = UINT64_MAX;
  const auto original = output;
  EXPECT_EQ(amdf_kernel_queue_refresh_status(queue, &output),
            native.refresh_status);
  EXPECT_EQ(std::memcmp(&output, &original, sizeof(output)), 0);
  EXPECT_EQ(Query().retired_submission, 0u);
  EXPECT_EQ(native.slots[0].native_submission, native.submitted);
  // Another native observer can establish progress independently of this
  // call's failing query. That progress still permits checked retirement.
  native.progress = native.submitted;
  EXPECT_EQ(amdf_kernel_queue_refresh_status(queue, &output),
            native.refresh_status);
  EXPECT_EQ(std::memcmp(&output, &original, sizeof(output)), 0);
  EXPECT_EQ(Query().retired_submission, submission);
  EXPECT_EQ(Query().terminal_status, AMDF_STATUS_OK);
  EXPECT_EQ(native.slots[0].retirement_count, 1u);
}

TEST_F(XdnaKernelQueueTest, RefreshReportsExecutionFailureInSnapshot) {
  ASSERT_EQ(CreateQueue(2), AMDF_STATUS_OK);
  uint64_t first = 0, second = 0;
  ASSERT_EQ(SubmitCommand(&first), AMDF_STATUS_OK);
  ASSERT_EQ(SubmitCommand(&second), AMDF_STATUS_OK);
  auto& native = context.native.queue;
  const auto failure = amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, 5);
  native.completion_status = failure;
  native.available_progress = first;
  auto checked = Refresh();
  EXPECT_EQ(checked.retired_submission, first);
  EXPECT_EQ(checked.terminal_status, failure);
  EXPECT_EQ(checked.state, AMDF_QUEUE_STATE_DEVICE_LOST);
  native.available_progress = second;
  checked = Refresh();
  EXPECT_EQ(checked.retired_submission, second);
  EXPECT_EQ(checked.terminal_status, failure);
  EXPECT_EQ(amdf_kernel_queue_wait(queue, first, 0, 0), failure);
}

TEST_F(XdnaKernelQueueTest, RefreshValidatesOutputBeforeNativeObservation) {
  ASSERT_NO_FATAL_FAILURE(Submit());
  amdf_kernel_queue_status_t output = {};
  output.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO;
  output.structure_size = sizeof(output);
  EXPECT_EQ(amdf_status_code(amdf_kernel_queue_refresh_status(queue, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  output.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS;
  output.structure_size = sizeof(output) - 1;
  EXPECT_EQ(amdf_status_code(amdf_kernel_queue_refresh_status(queue, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(amdf_status_code(amdf_kernel_queue_refresh_status(queue, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(
      amdf_status_code(amdf_kernel_queue_refresh_status(nullptr, &output)),
      AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(context.native.queue.refresh_count.load(), 0u);
  EXPECT_EQ(context.native.queue.slots[0].retirement_count, 0u);
}

TEST_F(XdnaKernelQueueTest, NativeReleaseFailureConsumesBothLifetimeBorrows) {
  ASSERT_NO_FATAL_FAILURE(Submit());
  EXPECT_EQ(amdf_kernel_queue_destroy(queue),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  EXPECT_EQ(context.native.queue.destroy_count, 0u);
  EXPECT_EQ(amdf_child_tracker_count(&context.children), 1u);
  EXPECT_EQ(amdf_child_tracker_count(&device.base.children), 1u);

  const auto failure = amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBUSY);
  context.native.queue.destroy_status = failure;
  context.native.queue.progress = context.native.queue.submitted;
  EXPECT_EQ(amdf_kernel_queue_destroy(queue), failure);
  queue = nullptr;
  EXPECT_EQ(context.native.queue.destroy_count, 1u);
  EXPECT_EQ(amdf_child_tracker_count(&context.children), 0u);
  EXPECT_EQ(amdf_child_tracker_count(&device.base.children), 0u);
}

TEST_F(XdnaKernelQueueTest, QueryDoesNotRetireNativeCompletion) {
  ASSERT_NO_FATAL_FAILURE(Submit());
  context.native.queue.progress = context.native.queue.submitted;
  EXPECT_EQ(Query().retired_submission, 0u);
  uint64_t later = 0;
  EXPECT_EQ(SubmitCommand(&later), AMDF_STATUS_OK);
  EXPECT_GT(later, submission);
  EXPECT_EQ(Query().retired_submission, 0u);
  ASSERT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, submission);
  ASSERT_NO_FATAL_FAILURE(Submit());
}

TEST_F(XdnaKernelQueueTest, QueryDoesNotConsumeCompletedCommandFailure) {
  const auto failure = amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, 5);
  context.native.queue.completion_status = failure;
  ASSERT_NO_FATAL_FAILURE(Submit());
  context.native.queue.progress = context.native.queue.submitted;
  const auto pending = Query();
  EXPECT_EQ(pending.retired_submission, 0u);
  EXPECT_EQ(pending.terminal_status, AMDF_STATUS_OK);
  EXPECT_EQ(pending.state, AMDF_QUEUE_STATE_ACTIVE);
  EXPECT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0), failure);
  const auto retired = Query();
  EXPECT_EQ(retired.retired_submission, submission);
  EXPECT_EQ(retired.terminal_status, failure);
  EXPECT_EQ(retired.state, AMDF_QUEUE_STATE_DEVICE_LOST);
}

TEST_F(XdnaKernelQueueTest, TimeoutPreservesAcceptedProgress) {
  context.native.queue.action =
      amdf_xdna_umd_kernel_queue_t::WaitAction::kTimeout;
  ASSERT_NO_FATAL_FAILURE(Submit());
  EXPECT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0),
            amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED));
  EXPECT_EQ(context.native.queue.wait_count.load(), 1u);
  EXPECT_EQ(Query().retired_submission, 0u);
  EXPECT_EQ(amdf_kernel_queue_destroy(queue),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  context.native.queue.progress = 1;
  EXPECT_EQ(Query().retired_submission, 0u);
  EXPECT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, submission);
}

TEST_F(XdnaKernelQueueTest, QueryAndFiniteWaitDoNotBlockBehindRetiringWaiter) {
  ASSERT_NO_FATAL_FAILURE(Submit());
  context.native.queue.phase =
      amdf_xdna_umd_kernel_queue_t::Phase::kPauseRequested;
  context.native.queue.progress = 1;
  amdf_status_t waiter_status = AMDF_STATUS_OK;
  std::thread waiter([&] {
    waiter_status =
        amdf_kernel_queue_wait(queue, submission, AMDF_TIMEOUT_INFINITE, 0);
  });
  {
    std::unique_lock<std::mutex> lock(context.native.queue.mutex);
    context.native.queue.condition.wait(lock, [&] {
      return context.native.queue.phase ==
             amdf_xdna_umd_kernel_queue_t::Phase::kPaused;
    });
  }
  EXPECT_EQ(Query().retired_submission, 0u);
  EXPECT_EQ(Refresh().retired_submission, 0u);
  EXPECT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0),
            amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED));
  EXPECT_EQ(context.native.queue.wait_count.load(), 0u);
  {
    std::lock_guard<std::mutex> lock(context.native.queue.mutex);
    context.native.queue.phase = amdf_xdna_umd_kernel_queue_t::Phase::kReleased;
    context.native.queue.condition.notify_all();
  }
  waiter.join();
  EXPECT_EQ(waiter_status, AMDF_STATUS_OK);
  EXPECT_EQ(Query().retired_submission, submission);
}

TEST_F(XdnaKernelQueueTest, CompletedFailureRetiresBeforeReportingError) {
  const auto failure = amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, 5);
  context.native.queue.completion_status = failure;
  ASSERT_NO_FATAL_FAILURE(Submit());
  EXPECT_EQ(amdf_kernel_queue_wait(queue, submission, AMDF_TIMEOUT_INFINITE, 0),
            failure);
  const auto status = Query();
  EXPECT_EQ(status.retired_submission, submission);
  EXPECT_EQ(status.terminal_status, failure);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_DEVICE_LOST);
}

TEST_F(XdnaKernelQueueTest, NativeWaitErrorDoesNotEraseConfirmedRetirement) {
  context.native.queue.action =
      amdf_xdna_umd_kernel_queue_t::WaitAction::kCompleteWithError;
  ASSERT_NO_FATAL_FAILURE(Submit());
  EXPECT_EQ(amdf_kernel_queue_wait(queue, submission, AMDF_TIMEOUT_INFINITE, 0),
            amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL));
  EXPECT_EQ(Query().retired_submission, submission);
  EXPECT_EQ(Query().terminal_status, AMDF_STATUS_OK);
}

TEST_F(XdnaKernelQueueTest, ReusesBackingWithDifferentInstructionRanges) {
  for (uint32_t i = 0; i < 3; ++i) {
    command.byte_offset = i * 64;
    ASSERT_NO_FATAL_FAILURE(Submit());
    EXPECT_EQ(
        amdf_kernel_queue_wait(queue, submission, AMDF_TIMEOUT_INFINITE, 0),
        AMDF_STATUS_OK);
    EXPECT_EQ(Query().retired_submission, submission);
  }
}

TEST_F(XdnaKernelQueueTest, RejectsForeignScopeBeforeNativeSubmission) {
  amdf_memory_scope_t foreign_scope = {};
  memory->scope = &foreign_scope;
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(SubmitCommand(&rejected),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(rejected, UINT64_MAX);
  EXPECT_EQ(context.native.queue.submitted, 0u);
}

TEST_F(XdnaKernelQueueTest, ValidatesInstructionRangeAtPublicBoundary) {
  const amdf_xdna_kernel_command_t valid = command;
  const auto expect_rejected = [&](amdf_status_code_t expected) {
    uint64_t rejected = UINT64_MAX;
    EXPECT_EQ(amdf_status_code(SubmitCommand(&rejected)), expected);
    EXPECT_EQ(rejected, UINT64_MAX);
    EXPECT_EQ(context.native.queue.submitted, 0u);
  };
  command.access_ordinal = 1;
  expect_rejected(AMDF_STATUS_CODE_OUT_OF_RANGE);
  command = valid;
  command.byte_offset = 1;
  expect_rejected(AMDF_STATUS_CODE_OUT_OF_RANGE);
  command = valid;
  command.byte_length = 0;
  expect_rejected(AMDF_STATUS_CODE_OUT_OF_RANGE);
  command.byte_length = 19;
  expect_rejected(AMDF_STATUS_CODE_OUT_OF_RANGE);
  command = valid;
  command.byte_offset = UINT64_MAX - 3;
  expect_rejected(AMDF_STATUS_CODE_OUT_OF_RANGE);
  command.byte_offset = memory->info.byte_length - 16;
  expect_rejected(AMDF_STATUS_CODE_OUT_OF_RANGE);
  command = valid;
  memory->accesses[0].info.access = AMDF_MEMORY_ACCESS_READ;
  expect_rejected(AMDF_STATUS_CODE_UNSUPPORTED);
  memory->accesses[0].info.access = AMDF_MEMORY_ACCESS_EXECUTE;
  memory->accesses[0].info.address_kinds = 0;
  expect_rejected(AMDF_STATUS_CODE_UNSUPPORTED);
}

TEST_F(XdnaKernelQueueTest, UsesProfileLimitsAndAbsoluteInstructionAddress) {
  device.info.instruction.address_alignment = 256;
  device.info.instruction.byte_length_granularity = 16;
  device.info.instruction.maximum_byte_length = 80;
  memory->accesses[0].addresses[AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE] += 64;
  command.byte_offset = 0;
  command.byte_length = 64;
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(amdf_status_code(SubmitCommand(&rejected)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  command.byte_offset = 192;
  command.byte_length = 68;
  EXPECT_EQ(amdf_status_code(SubmitCommand(&rejected)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  command.byte_length = 96;
  EXPECT_EQ(amdf_status_code(SubmitCommand(&rejected)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(rejected, UINT64_MAX);
  EXPECT_EQ(context.native.queue.submitted, 0u);
  command.byte_length = 64;
  ASSERT_NO_FATAL_FAILURE(Submit());
}

TEST_F(XdnaKernelQueueTest, NativeRejectionPreservesSlotAndOutput) {
  context.native.queue.submission_status =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(SubmitCommand(&rejected), context.native.queue.submission_status);
  EXPECT_EQ(rejected, UINT64_MAX);
  EXPECT_EQ(Query().retired_submission, 0u);
  context.native.queue.submission_status = AMDF_STATUS_OK;
  ASSERT_NO_FATAL_FAILURE(Submit());
}

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
    amdf_endpoint_t*, uint32_t ordinal, amdf_queue_family_info_t* out_info) {
  out_info->ordinal = ordinal;
  out_info->command_type = AMDF_QUEUE_COMMAND_TYPE_XDNA;
  out_info->publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL;
  out_info->format_version = AMDF_XDNA_QUEUE_FORMAT_VERSION_1;
  out_info->roles = AMDF_QUEUE_ROLE_COMPUTE;
  return AMDF_STATUS_OK;
}
const amdf_xdna_device_info_t* amdf_xdna_device_get_info(
    const amdf_device_t* device) {
  return &reinterpret_cast<const Device*>(device)->info;
}
uint64_t amdf_xdna_device_query_reset_epoch(const amdf_device_t* device) {
  return amdf_xdna_device_get_info(device)->reset_epoch;
}
amdf_device_t* amdf_xdna_context_get_device(amdf_xdna_context_t* context) {
  return reinterpret_cast<Context*>(context)->device;
}
const amdf_xdna_context_info_t* amdf_xdna_context_get_info(
    const amdf_xdna_context_t* context) {
  return &reinterpret_cast<const Context*>(context)->info;
}
amdf_xdna_umd_context_t* amdf_xdna_context_get_umd(
    amdf_xdna_context_t* context) {
  return &reinterpret_cast<Context*>(context)->native;
}
amdf_status_t amdf_xdna_context_register_child(amdf_xdna_context_t* context) {
  return amdf_child_tracker_register(
      &reinterpret_cast<Context*>(context)->children);
}
void amdf_xdna_context_unregister_child(amdf_xdna_context_t* context) {
  amdf_child_tracker_unregister(&reinterpret_cast<Context*>(context)->children);
}
const amdf_xdna_device_profile_t* amdf_xdna_device_get_profile(
    const amdf_device_t* device) {
  return &reinterpret_cast<const Device*>(device)->profile;
}
amdf_memory_scope_t* amdf_xdna_context_get_memory_scope(
    amdf_xdna_context_t* context) {
  return &reinterpret_cast<Context*>(context)->memory_scope;
}
amdf_status_t amdf_xdna_umd_kernel_queue_create(
    amdf_xdna_umd_context_t* context, uint32_t capacity,
    amdf_xdna_umd_kernel_queue_t** out_queue) {
  context->queue.slots.resize(capacity);
  *out_queue = &context->queue;
  return AMDF_STATUS_OK;
}
amdf_native_event_types_t amdf_xdna_umd_kernel_queue_query_notification_types(
    const amdf_xdna_umd_kernel_queue_t* queue) {
  return queue->notification_types;
}
amdf_status_t amdf_xdna_umd_kernel_queue_request_notification(
    amdf_xdna_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_native_event_t*) {
  queue->notification_submission = native_submission;
  ++queue->notification_count;
  return queue->notification_status;
}
amdf_status_t amdf_xdna_umd_kernel_queue_submit(
    amdf_xdna_umd_kernel_queue_t* queue, uint32_t slot,
    uint64_t instruction_address, uint32_t instruction_byte_length,
    uint64_t* out_submission) {
  if (!amdf_status_is_ok(queue->submission_status)) {
    return queue->submission_status;
  }
  if (!amdf_status_is_ok(queue->terminal_status.load())) {
    return queue->terminal_status.load();
  }
  EXPECT_EQ(queue->slots[slot].native_submission, 0u);
  queue->pending_address = instruction_address;
  queue->pending_byte_length = instruction_byte_length;
  *out_submission = ++queue->submitted;
  queue->slots[slot].native_submission = queue->submitted;
  std::unique_lock<std::mutex> lock(queue->mutex);
  using Phase = amdf_xdna_umd_kernel_queue_t::Phase;
  if (queue->publication_phase == Phase::kPauseRequested) {
    queue->progress.store(queue->submitted, std::memory_order_release);
    queue->publication_phase = Phase::kPaused;
    queue->condition.notify_all();
    queue->condition.wait(
        lock, [&] { return queue->publication_phase == Phase::kReleased; });
    queue->publication_phase = Phase::kRunning;
  }
  return AMDF_STATUS_OK;
}
uint64_t amdf_xdna_umd_kernel_queue_query_progress(
    const amdf_xdna_umd_kernel_queue_t* queue) {
  return queue->progress.load();
}
amdf_status_t amdf_xdna_umd_kernel_queue_refresh_progress(
    amdf_xdna_umd_kernel_queue_t* queue) {
  ++queue->refresh_count;
  if (!amdf_status_is_ok(queue->refresh_status)) {
    return queue->refresh_status;
  }
  uint64_t observed = queue->progress.load();
  while (observed < queue->available_progress &&
         !queue->progress.compare_exchange_weak(observed,
                                                queue->available_progress)) {
  }
  return AMDF_STATUS_OK;
}
void amdf_xdna_umd_kernel_queue_retire_command(
    amdf_xdna_umd_kernel_queue_t* queue, uint32_t slot) {
  EXPECT_NE(queue->slots[slot].native_submission, 0u);
  EXPECT_LE(queue->slots[slot].native_submission, queue->progress.load());
  std::unique_lock<std::mutex> lock(queue->mutex);
  if (queue->phase == amdf_xdna_umd_kernel_queue_t::Phase::kPauseRequested) {
    queue->phase = amdf_xdna_umd_kernel_queue_t::Phase::kPaused;
    queue->condition.notify_all();
    queue->condition.wait(lock, [&] {
      return queue->phase == amdf_xdna_umd_kernel_queue_t::Phase::kReleased;
    });
  }
  if (!amdf_status_is_ok(queue->completion_status)) {
    queue->terminal_status = queue->completion_status;
  }
  queue->slots[slot].native_submission = 0;
  ++queue->slots[slot].retirement_count;
}
amdf_status_t amdf_xdna_umd_kernel_queue_query_terminal_status(
    const amdf_xdna_umd_kernel_queue_t* queue) {
  return queue->terminal_status.load();
}
amdf_status_t amdf_xdna_umd_kernel_queue_wait(
    amdf_xdna_umd_kernel_queue_t* queue, uint64_t submission,
    const amdf_wait_deadline_t*) {
  ++queue->wait_count;
  if (queue->action == amdf_xdna_umd_kernel_queue_t::WaitAction::kTimeout) {
    return amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
  }
  uint64_t observed = queue->progress.load();
  while (observed < submission &&
         !queue->progress.compare_exchange_weak(observed, submission)) {
  }

  return queue->action ==
                 amdf_xdna_umd_kernel_queue_t::WaitAction::kCompleteWithError
             ? amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL)
             : AMDF_STATUS_OK;
}
amdf_status_t amdf_xdna_umd_kernel_queue_destroy(
    amdf_xdna_umd_kernel_queue_t* queue) {
  ++queue->destroy_count;
  return queue->destroy_status;
}

}  // extern "C"
