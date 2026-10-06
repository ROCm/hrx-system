// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/retirement.h"

#include <array>

#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/storage/memory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class RetirementTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    loom_serve_retirement_initialize(&retirement);
    const loom_serve_device_options_t options = {
        .uri = IREE_SV("amdgpu"),
        .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC,
        .slab_size = kSlabBytes,
        .memory_limit = kSlabBytes};
    IREE_ASSERT_OK(
        loom_serve_device_create(&options, &device, iree_allocator_system()));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_create(
        loom_serve_device_memory_pool(device), kSlabBytes, 256, &statistics,
        &reservation));
    IREE_ASSERT_OK(
        loom_serve_virtual_buffer_commit(reservation, 0, kSlabBytes));
    IREE_ASSERT_OK(
        iree_hal_buffer_subspan(loom_serve_virtual_buffer_handle(reservation),
                                256, 1024, iree_allocator_system(), &view));
    for (auto** semaphore : {&gate, &completion}) {
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
          0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, semaphore));
    }
  }

  void TearDown() override {
    // A failed assertion must still release the accepted operation's gate.
    // Two is beyond the test body's successful signal, if it reached one.
    if (gate) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 2, nullptr));
    }
    iree_hal_buffer_release(view);
    loom_serve_retirement_deinitialize(&retirement);
    iree_hal_semaphore_release(completion);
    iree_hal_semaphore_release(gate);
    IREE_EXPECT_OK(loom_serve_virtual_buffer_destroy(reservation));
    IREE_EXPECT_OK(loom_serve_device_destroy(device));
  }

  // One mapped slab remains owned until all transfer views retire.
  static constexpr uint64_t kSlabBytes = 2 * 1024 * 1024;
  // Shared physical owner enclosing the test's queue operations.
  loom_serve_device_t* device = nullptr;
  // Parameter/state-like virtual reservation, independent of view references.
  loom_serve_virtual_buffer_t* reservation = nullptr;
  // Reservation accounting retained through destruction.
  loom_serve_memory_statistics_t statistics = {};
  // Final ownership counter shared with queue completion callbacks.
  loom_serve_retirement_t retirement = {};
  // Host reference released after the queue has accepted its nested child.
  iree_hal_buffer_t* view = nullptr;
  // Unsignaled dependency holding accepted work without sleeps or GPU loops.
  iree_hal_semaphore_t* gate = nullptr;
  // Readiness edge that may fail before the accepted operation can retire.
  iree_hal_semaphore_t* completion = nullptr;
  // Borrowed upload and download bytes alive until TearDown joins retirement.
  struct {
    // Nonuniform upload payload.
    std::array<uint32_t, 128> input = {};
    // Feedback destination written by the transfer before final view release.
    std::array<uint32_t, 128> output = {};
  } payload;
};

TEST_P(RetirementTest, NestedQueueViewsOutliveFailedReadiness) {
  auto* original = view;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_retirement_track(&retirement, &view, iree_allocator_null()));
  EXPECT_EQ(view, original);
  IREE_ASSERT_OK(
      loom_serve_retirement_track(&retirement, &view, iree_allocator_system()));
  iree_hal_buffer_t* child = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(view, 64, sizeof(payload.input),
                                         iree_allocator_system(), &child));
  EXPECT_EQ(iree_hal_buffer_byte_offset(child), 320u);
  for (size_t i = 0; i < payload.input.size(); ++i) {
    payload.input[i] = static_cast<uint32_t>(i * 17 + 13);
  }
  const iree_hal_transfer_operation_t operations[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = payload.input.data(),
                  .target_buffer = child,
                  .length = sizeof(payload.input)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
       .download = {.source_buffer = child,
                    .target = payload.output.data(),
                    .length = sizeof(payload.output)}},
  };
  uint64_t value = 1;
  auto status = iree_hal_queue_transfer(
      loom_serve_device_transfer_queue(device), {1, &gate, &value},
      {1, &completion, &value}, IREE_ARRAYSIZE(operations), operations);
  iree_hal_buffer_release(child);
  IREE_ASSERT_OK(status);
  iree_hal_buffer_release(view);
  view = nullptr;
  if (GetParam()) {
    iree_hal_semaphore_fail(completion,
                            iree_status_from_code(IREE_STATUS_CANCELLED));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_CANCELLED,
        iree_hal_semaphore_wait(completion, 1, iree_infinite_timeout(),
                                IREE_ASYNC_WAIT_FLAG_NONE));
  }
  // Readiness may have failed, but the accepted transfer still owns the view.
  iree_slim_mutex_lock(&retirement.mutex);
  EXPECT_EQ(retirement.count, 1u);
  iree_slim_mutex_unlock(&retirement.mutex);
  EXPECT_EQ(statistics.committed_bytes, kSlabBytes);
  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  loom_serve_retirement_await(&retirement);
  if (!GetParam()) {
    IREE_EXPECT_OK(iree_hal_semaphore_wait(
        completion, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_EQ(payload.output, payload.input);
  }
  // Neither successful nor failed readiness is used to authorize this unmap.
  IREE_ASSERT_OK(loom_serve_virtual_buffer_destroy(reservation));
  reservation = nullptr;
  EXPECT_EQ(statistics.committed_bytes, 0u);
}

INSTANTIATE_TEST_SUITE_P(Readiness, RetirementTest,
                         ::testing::Values(false, true));

}  // namespace
