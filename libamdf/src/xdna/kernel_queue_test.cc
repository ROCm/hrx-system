// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/kernel_queue.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/device.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/kernel_queue.h"
#include "libamdf/src/memory_resource.h"
#include "libamdf/src/memory_scope.h"
#include "libamdf/src/xdna/device.h"
#include "libamdf/src/xdna/endpoint_profile.h"
#include "libamdf/src/xdna/umd/kernel_queue.h"

// Native dependencies are controlled here; submission ownership, retirement,
// public status, and wait-budget handling use the production queue code.
struct amdf_xdna_umd_kernel_queue_t {
  // Native completion frontier published independently of host retirement.
  std::atomic<uint64_t> progress{0};
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
};

struct amdf_xdna_umd_context_t {
  // Context-owned native queue dependency leased by the production queue.
  amdf_xdna_umd_kernel_queue_t queue;
};

namespace {

struct Device {
  // Production generic device base consumed by queue creation.
  amdf_device_t base = {};
  // Immutable device identity returned by the XDNA device dependency.
  amdf_xdna_device_info_t info = {};
  // Immutable native instruction contract.
  amdf_xdna_endpoint_info_t endpoint_info = {};
  // Cached profile consumed by production input validation.
  amdf_xdna_endpoint_profile_t profile = {};
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
    device.endpoint_info.instruction.maximum_byte_length =
        UINT32_MAX & ~uint64_t{3};
    device.endpoint_info.instruction.address_alignment = 4;
    device.endpoint_info.instruction.byte_length_granularity = 4;
    device.profile.info = &device.endpoint_info;
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
    amdf_xdna_kernel_queue_create_info_t create = {};
    create.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO;
    create.structure_size = sizeof(create);
    ASSERT_EQ(
        amdf_xdna_kernel_queue_create(
            reinterpret_cast<amdf_xdna_context_t*>(&context), &create, &queue),
        AMDF_STATUS_OK);
  }

  void TearDown() override {
    context.native.queue.progress = context.native.queue.submitted;
    if (queue) {
      EXPECT_EQ(amdf_kernel_queue_destroy(queue), AMDF_STATUS_OK);
    }
    EXPECT_EQ(amdf_child_tracker_count(&memory->children), 0u);
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
    EXPECT_EQ(amdf_child_tracker_count(&memory->children), 1u);
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

  // Device dependency retained until production queue teardown.
  Device device;
  // Context dependency retained until production queue teardown.
  Context context;
  // Actual memory owner borrowed by production submission and retirement.
  amdf_memory_t* memory = nullptr;
  // Public instruction range; the native dependency receives only its address.
  amdf_xdna_kernel_command_t command = {};
  // Production queue under test.
  amdf_kernel_queue_t* queue = nullptr;
  // Accepted public submission identity.
  uint64_t submission = 0;
};

TEST_F(XdnaKernelQueueTest, ZeroTimeoutRefreshesNativeProgress) {
  ASSERT_NO_FATAL_FAILURE(Submit());
  EXPECT_EQ(Query().retired_submission, 0u);
  EXPECT_EQ(context.native.queue.wait_count.load(), 0u);
  EXPECT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0), AMDF_STATUS_OK);
  EXPECT_EQ(context.native.queue.wait_count.load(), 1u);
  EXPECT_EQ(amdf_child_tracker_count(&memory->children), 0u);
  EXPECT_EQ(Query().retired_submission, submission);
}

TEST_F(XdnaKernelQueueTest, TimeoutRetainsAcceptedCommand) {
  context.native.queue.action =
      amdf_xdna_umd_kernel_queue_t::WaitAction::kTimeout;
  ASSERT_NO_FATAL_FAILURE(Submit());
  EXPECT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0),
            amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED));
  EXPECT_EQ(context.native.queue.wait_count.load(), 1u);
  EXPECT_EQ(amdf_child_tracker_count(&memory->children), 1u);
  EXPECT_EQ(Query().retired_submission, 0u);
  EXPECT_EQ(amdf_kernel_queue_destroy(queue),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  context.native.queue.progress = 1;
  EXPECT_EQ(Query().retired_submission, submission);
  EXPECT_EQ(amdf_child_tracker_count(&memory->children), 0u);
}

TEST_F(XdnaKernelQueueTest, FiniteWaitDoesNotBlockBehindRetirementObserver) {
  ASSERT_NO_FATAL_FAILURE(Submit());
  context.native.queue.phase =
      amdf_xdna_umd_kernel_queue_t::Phase::kPauseRequested;
  context.native.queue.progress = 1;
  std::thread observer([&] { Query(); });
  {
    std::unique_lock<std::mutex> lock(context.native.queue.mutex);
    context.native.queue.condition.wait(lock, [&] {
      return context.native.queue.phase ==
             amdf_xdna_umd_kernel_queue_t::Phase::kPaused;
    });
  }
  EXPECT_EQ(amdf_kernel_queue_wait(queue, submission, 0, 0),
            amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED));
  EXPECT_EQ(amdf_child_tracker_count(&memory->children), 1u);
  EXPECT_EQ(context.native.queue.wait_count.load(), 0u);
  {
    std::lock_guard<std::mutex> lock(context.native.queue.mutex);
    context.native.queue.phase = amdf_xdna_umd_kernel_queue_t::Phase::kReleased;
    context.native.queue.condition.notify_all();
  }
  observer.join();
  EXPECT_EQ(Query().retired_submission, submission);
  EXPECT_EQ(amdf_child_tracker_count(&memory->children), 0u);
}

TEST_F(XdnaKernelQueueTest, CompletedFailureRetiresBeforeReportingError) {
  const auto failure = amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, 5);
  context.native.queue.completion_status = failure;
  ASSERT_NO_FATAL_FAILURE(Submit());
  EXPECT_EQ(amdf_kernel_queue_wait(queue, submission, AMDF_TIMEOUT_INFINITE, 0),
            failure);
  EXPECT_EQ(amdf_child_tracker_count(&memory->children), 0u);
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
  EXPECT_EQ(amdf_child_tracker_count(&memory->children), 0u);
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
    EXPECT_EQ(amdf_child_tracker_count(&memory->children), 0u);
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
  EXPECT_EQ(amdf_child_tracker_count(&memory->children), 0u);
}

TEST_F(XdnaKernelQueueTest, ValidatesInstructionRangeAtPublicBoundary) {
  const amdf_xdna_kernel_command_t valid = command;
  const auto expect_rejected = [&](amdf_status_code_t expected) {
    uint64_t rejected = UINT64_MAX;
    EXPECT_EQ(amdf_status_code(SubmitCommand(&rejected)), expected);
    EXPECT_EQ(rejected, UINT64_MAX);
    EXPECT_EQ(context.native.queue.submitted, 0u);
    EXPECT_EQ(amdf_child_tracker_count(&memory->children), 0u);
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
  device.endpoint_info.instruction.address_alignment = 256;
  device.endpoint_info.instruction.byte_length_granularity = 16;
  device.endpoint_info.instruction.maximum_byte_length = 80;
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
  EXPECT_EQ(amdf_child_tracker_count(&memory->children), 0u);
  command.byte_length = 64;
  ASSERT_NO_FATAL_FAILURE(Submit());
}

TEST_F(XdnaKernelQueueTest, NativeRejectionReleasesBorrowAndPreservesOutput) {
  context.native.queue.submission_status =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(SubmitCommand(&rejected), context.native.queue.submission_status);
  EXPECT_EQ(rejected, UINT64_MAX);
  EXPECT_EQ(Query().retired_submission, 0u);
  EXPECT_EQ(amdf_child_tracker_count(&memory->children), 0u);
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
const amdf_xdna_endpoint_profile_t* amdf_xdna_device_get_profile(
    const amdf_device_t* device) {
  return &reinterpret_cast<const Device*>(device)->profile;
}
amdf_memory_scope_t* amdf_xdna_context_get_memory_scope(
    amdf_xdna_context_t* context) {
  return &reinterpret_cast<Context*>(context)->memory_scope;
}
amdf_status_t amdf_xdna_umd_kernel_queue_create(
    amdf_xdna_umd_context_t* context,
    amdf_xdna_umd_kernel_queue_t** out_queue) {
  *out_queue = &context->queue;
  return AMDF_STATUS_OK;
}
amdf_status_t amdf_xdna_umd_kernel_queue_submit(
    amdf_xdna_umd_kernel_queue_t* queue, uint64_t instruction_address,
    uint32_t instruction_byte_length, uint64_t* out_submission) {
  if (!amdf_status_is_ok(queue->submission_status))
    return queue->submission_status;
  EXPECT_EQ(queue->pending_address, 0u);
  queue->pending_address = instruction_address;
  queue->pending_byte_length = instruction_byte_length;
  *out_submission = ++queue->submitted;
  return AMDF_STATUS_OK;
}
uint64_t amdf_xdna_umd_kernel_queue_query_progress(
    const amdf_xdna_umd_kernel_queue_t* queue) {
  return queue->progress.load();
}
void amdf_xdna_umd_kernel_queue_retire_command(
    amdf_xdna_umd_kernel_queue_t* queue) {
  EXPECT_NE(queue->pending_address, 0u);
  std::unique_lock<std::mutex> lock(queue->mutex);
  if (queue->phase == amdf_xdna_umd_kernel_queue_t::Phase::kPauseRequested) {
    queue->phase = amdf_xdna_umd_kernel_queue_t::Phase::kPaused;
    queue->condition.notify_all();
    queue->condition.wait(lock, [&] {
      return queue->phase == amdf_xdna_umd_kernel_queue_t::Phase::kReleased;
    });
  }
  queue->terminal_status = queue->completion_status;
  queue->pending_address = 0;
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
  queue->progress = submission;
  return queue->action ==
                 amdf_xdna_umd_kernel_queue_t::WaitAction::kCompleteWithError
             ? amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL)
             : AMDF_STATUS_OK;
}
amdf_status_t amdf_xdna_umd_kernel_queue_destroy(
    amdf_xdna_umd_kernel_queue_t*) {
  return AMDF_STATUS_OK;
}

}  // extern "C"
