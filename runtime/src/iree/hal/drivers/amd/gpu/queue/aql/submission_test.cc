// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/submission.h"

#include <array>
#include <cstring>

#include "iree/async/proactor_platform.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/signal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct Executable {
  // Real HAL resource whose payload dependency has no native work.
  iree_hal_executable_t base;
  // Destructor observation retained through the test.
  bool destroyed = false;
  static void Destroy(iree_hal_executable_t* base) {
    reinterpret_cast<Executable*>(base)->destroyed = true;
  }
};

class AqlSubmissionTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    iree_hal_amd_gpu_capture_pool_initialize(512, iree_allocator_system(),
                                             &pool_);
  }
  void TearDown() override {
    iree_hal_amd_gpu_capture_pool_deinitialize(&pool_);
  }
  // Actual bounded capture storage, shared by admission and retirement.
  iree_hal_amd_gpu_capture_pool_t pool_;
};

TEST_P(AqlSubmissionTest, CapturesInputsAndReleasesPayloadBeforeCompletion) {
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_create(
      proactor, 0, 2, iree_allocator_system(), &semaphore));
  const iree_hal_queue_family_t family = {};
  const iree_hal_executable_vtable_t vtable = {.destroy = Executable::Destroy};
  Executable executable;
  iree_hal_executable_initialize(&family, &vtable, &executable.base);
  const iree_hal_amd_gpu_aql_function_t function = {};
  std::array<uint8_t, 16> bytes = {};
  bool buffer_destroyed = false;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_STORAGE, bytes.size(),
      iree_make_byte_span(bytes.data(), bytes.size()),
      {[](void* data, iree_hal_buffer_t*) { *static_cast<bool*>(data) = true; },
       &buffer_destroyed},
      iree_allocator_system(), &buffer));
  auto reference = iree_hal_make_buffer_ref(buffer, 4, 8);
  uint32_t constant = 0x12345678;
  const iree_hal_amd_gpu_aql_dispatch_request_t request = {
      .executable = &executable.base,
      .function = &function,
      .constants = iree_make_const_byte_span(&constant, sizeof(constant)),
      .bindings = {1, &reference},
  };
  uint64_t value = 4;
  iree_alignas(8) uint8_t inline_storage[496];
  iree_hal_amd_gpu_aql_submission_t* capture = nullptr;
  if (GetParam()) {
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_retirement(
        &pool_, iree_make_byte_span(inline_storage, sizeof(inline_storage)),
        &request, 0, nullptr, 0, {1, &semaphore, &value}, &capture));
    EXPECT_EQ(capture->pool, nullptr);
    EXPECT_EQ(capture->dispatch.constants.data_length, 0u);
  } else {
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_capture(
        &pool_, &request, {}, {1, &semaphore, &value}, 17, 23, &capture));
    constant = 0;
    uint32_t copied = 0;
    std::memcpy(&copied, capture->dispatch.constants.data, sizeof(copied));
    EXPECT_EQ(copied, 0x12345678u);
    EXPECT_EQ(capture->before, 17u);
    EXPECT_EQ(capture->after, 23u);
  }
  reference = {};
  value = 99;
  iree_hal_executable_release(&executable.base);
  iree_hal_buffer_release(buffer);
  iree_hal_semaphore_release(semaphore);
  EXPECT_FALSE(executable.destroyed);
  EXPECT_FALSE(buffer_destroyed);
  EXPECT_EQ(capture->dispatch.bindings.values[0].offset, 4u);
  EXPECT_EQ(capture->signals.payload_values[0], 4u);
  iree_hal_amd_gpu_aql_submission_release_payload(capture);
  EXPECT_TRUE(executable.destroyed);
  EXPECT_TRUE(buffer_destroyed);
  uint64_t current = UINT64_MAX;
  IREE_ASSERT_OK(
      iree_hal_semaphore_query(capture->signals.semaphores[0], &current));
  EXPECT_EQ(current, 0u);
  IREE_ASSERT_OK(
      iree_hal_semaphore_signal(capture->signals.semaphores[0], 4, nullptr));
  iree_hal_amd_gpu_aql_submission_release(capture);
  iree_async_proactor_release(proactor);
}

TEST_P(AqlSubmissionTest, DependencyReadersJoinProducerHostRetirement) {
  iree_hal_amd_gpu_signal_pool_t signals;
  iree_hal_amd_gpu_signal_pool_initialize({}, &signals);
  iree_hal_amd_gpu_aql_signal_t native;
  iree_hal_amd_gpu_signal_t owner;
  uint64_t frontier;
  iree_hal_amd_gpu_signal_block_t block;
  const amdf_gpu_event_info_t event = {};
  iree_hal_amd_gpu_aql_signal_block_initialize(
      &signals, this, &event, &native, 0x1000, &owner, 1, 1, &frontier, &block);
  auto* producer = iree_hal_amd_gpu_signal_pool_try_acquire(&signals, 1);
  iree_hal_amd_gpu_signal_retain(producer);
  const iree_hal_amd_gpu_aql_wait_t dependency = {.signal = producer};
  const iree_hal_amd_gpu_aql_dispatch_request_t barrier = {};
  iree_alignas(8) uint8_t inline_storage[496];
  iree_hal_amd_gpu_aql_submission_t* consumer = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_retirement(
      &pool_,
      GetParam() ? iree_make_byte_span(inline_storage, sizeof(inline_storage))
                 : iree_byte_span_empty(),
      &barrier, 1, &dependency, 1, {}, &consumer));
  native.value = 0;  // Firmware has finished; the host still owns the payload.
  iree_status_code_t code = IREE_STATUS_OK;
  EXPECT_FALSE(
      iree_hal_amd_gpu_aql_submission_retirement_ready(consumer, &code));
  iree_hal_amd_gpu_signal_retire(producer, IREE_STATUS_ABORTED);
  iree_hal_amd_gpu_signal_release(producer);
  EXPECT_TRUE(
      iree_hal_amd_gpu_aql_submission_retirement_ready(consumer, &code));
  EXPECT_EQ(code, IREE_STATUS_ABORTED);
  EXPECT_EQ(iree_hal_amd_gpu_signal_pool_try_acquire(&signals, 1), nullptr);
  iree_hal_amd_gpu_aql_submission_release(consumer);
  auto* reused = iree_hal_amd_gpu_signal_pool_try_acquire(&signals, 1);
  EXPECT_EQ(reused, producer);
  EXPECT_EQ(native.value, 1u);
  iree_hal_amd_gpu_signal_release(reused);
  iree_hal_amd_gpu_signal_block_deinitialize(&block);
  iree_hal_amd_gpu_signal_pool_deinitialize(&signals);
}

INSTANTIATE_TEST_SUITE_P(Storage, AqlSubmissionTest,
                         ::testing::Values(false, true));

TEST(AqlTransferSubmissionTest, RejectedSuffixWaitsForAcceptedHostWrites) {
  struct Buffer {
    // Real HAL resource retained by a captured transfer.
    iree_hal_buffer_t base;
    // Destruction observation independent of capture storage.
    bool destroyed = false;
    static void Destroy(iree_hal_buffer_t* base) {
      reinterpret_cast<Buffer*>(base)->destroyed = true;
    }
  };
  iree_hal_amd_gpu_capture_pool_t pool;
  iree_hal_amd_gpu_capture_pool_initialize(512, iree_allocator_system(), &pool);
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_create(
      proactor, 0, 2, iree_allocator_system(), &semaphore));
  const iree_hal_buffer_vtable_t vtable = {.destroy = Buffer::Destroy};
  Buffer buffer = {};
  iree_hal_buffer_initialize(
      {}, &buffer.base, 8, 0, 8, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_TRANSFER, &vtable,
      &buffer.base);
  // The native binding belongs to the dependency, not the capture under test.
  // Address commitment is deliberately deferred until native publication.
  const iree_hal_queue_family_t family = {.memory = {.queue_scope_id = 1}};
  std::array<iree_hal_memory_scope_access_t, 3> scopes = {};
  scopes[2].bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] = 0;
  iree_hal_memory_contract_t contract = {.scopes = scopes.data()};
  buffer.base.memory.contract = &contract;
  std::array<uint8_t, 8> target = {};
  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = &buffer.base,
              .target = target.data(),
              .length = target.size(),
          },
  };
  uint64_t value = 1;
  iree_hal_amd_gpu_aql_submission_t* submission = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_transfer(
      &pool, &family, {}, {1, &semaphore, &value}, 1, &operation, 0, 0,
      &submission));
  struct Completion {
    // File callbacks distinguish failure notification from ending borrows.
    bool retired = false;
    // Borrowed dependency must be destroyed before handing ownership back.
    Buffer* buffer;
  } completion = {.buffer = &buffer};
  submission->transfer.callback = {
      [](void* user_data, iree_status_t status,
         const iree_async_single_frontier_t* frontier, bool retired) {
        auto* completion = static_cast<Completion*>(user_data);
        EXPECT_TRUE(retired);
        EXPECT_TRUE(completion->buffer->destroyed);
        EXPECT_EQ(frontier, nullptr);
        EXPECT_EQ(iree_status_consume_code(status), IREE_STATUS_UNAVAILABLE);
        completion->retired = true;
      },
      &completion,
  };
  iree_hal_buffer_release(&buffer.base);
  // One accepted native piece retains the capture while a later, unpublished
  // suffix still owns its initial reference.
  iree_atomic_ref_count_inc(&submission->transfer.ref_count);
  iree_atomic_store(&submission->transfer.failure, IREE_STATUS_UNAVAILABLE,
                    iree_memory_order_release);
  iree_hal_amd_gpu_aql_submission_release(submission);
  EXPECT_FALSE(buffer.destroyed);
  EXPECT_FALSE(completion.retired);
  uint64_t observed = UINT64_MAX;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &observed));
  EXPECT_EQ(observed, 0u);
  EXPECT_EQ(submission->transfer.capture.operations[0].target.host,
            target.data());
  // The capture may report failure only after its last accepted piece has
  // stopped using the borrowed host destination.
  iree_hal_amd_gpu_aql_submission_release(submission);
  EXPECT_TRUE(buffer.destroyed);
  EXPECT_TRUE(completion.retired);
  EXPECT_THAT(iree_hal_semaphore_query(semaphore, &observed),
              iree::testing::status::StatusIs(iree::StatusCode::kUnavailable));
  iree_hal_semaphore_release(semaphore);
  iree_async_proactor_release(proactor);
  iree_hal_amd_gpu_capture_pool_deinitialize(&pool);
}

TEST(AqlTransferSubmissionTest, FileBoundaryJoinsFailureAndRetirement) {
  iree_hal_amd_gpu_capture_pool_t pool;
  iree_hal_amd_gpu_capture_pool_initialize(512, iree_allocator_system(), &pool);
  const iree_hal_queue_family_t family = {};
  iree_hal_amd_gpu_aql_submission_t* submission = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_transfer(
      &pool, &family, {}, {}, 0, nullptr, 17, 23, &submission));
  EXPECT_EQ(submission->entry.kind, IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER);
  EXPECT_EQ(submission->transfer.capture.count, 0u);
  EXPECT_EQ(submission->before, 17u);
  EXPECT_EQ(submission->after, 23u);
  struct Completion {
    // Early notification count, independent of ownership handoff.
    unsigned failures = 0;
    // Final ownership handoff count.
    unsigned retirements = 0;
    // Exact last accepted queue epoch returned with retirement.
    iree_async_single_frontier_t frontier = {};
  } completion;
  submission->transfer.callback = {
      [](void* user_data, iree_status_t status,
         const iree_async_single_frontier_t* frontier, bool retired) {
        auto* completion = static_cast<Completion*>(user_data);
        EXPECT_EQ(iree_status_consume_code(status), IREE_STATUS_UNAVAILABLE);
        if (retired) {
          ++completion->retirements;
          ASSERT_NE(frontier, nullptr);
          completion->frontier = *frontier;
        } else {
          ++completion->failures;
          EXPECT_EQ(frontier, nullptr);
        }
      },
      &completion,
  };
  iree_atomic_ref_count_inc(&submission->transfer.ref_count);
  iree_hal_amd_gpu_aql_submission_fail(
      submission, iree_status_from_code(IREE_STATUS_UNAVAILABLE));
  iree_hal_amd_gpu_aql_submission_fail(
      submission, iree_status_from_code(IREE_STATUS_ABORTED));
  EXPECT_EQ(completion.failures, 1u);
  EXPECT_EQ(completion.retirements, 0u);
  iree_hal_amd_gpu_aql_submission_release(submission);
  EXPECT_EQ(completion.retirements, 0u);
  iree_async_single_frontier_t frontier;
  iree_async_single_frontier_initialize(&frontier, 3, 41);
  iree_hal_amd_gpu_aql_submission_set_frontier(submission, &frontier);
  iree_hal_amd_gpu_aql_submission_release(submission);
  EXPECT_EQ(completion.failures, 1u);
  EXPECT_EQ(completion.retirements, 1u);
  EXPECT_EQ(completion.frontier.entry_count, frontier.entry_count);
  EXPECT_EQ(completion.frontier.entries[0].axis, frontier.entries[0].axis);
  EXPECT_EQ(completion.frontier.entries[0].epoch, frontier.entries[0].epoch);
  iree_hal_amd_gpu_capture_pool_deinitialize(&pool);
}

}  // namespace
