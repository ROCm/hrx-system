// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/retirement.h"

#include <array>
#include <condition_variable>
#include <mutex>

#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/storage/memory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class RetirementTest : public ::testing::TestWithParam<bool> {
 protected:
  static void Wake(void* user_data) {
    auto* self = static_cast<RetirementTest*>(user_data);
    std::lock_guard<std::mutex> lock(self->wake_mutex);
    ++self->wake_count;
    self->wake_condition.notify_one();
  }

  void SetUp() override {
    loom_serve_retirement_initialize(&retirement);
    const loom_serve_device_options_t options = {
        .uri = IREE_SV("amdgpu"),
        .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC,
        .slab_size = kSlabBytes,
        .memory_limit = kSlabBytes};
    loom_serve_host_t* host = nullptr;
    IREE_ASSERT_OK(loom_serve_host_create(1, &host, iree_allocator_system()));
    auto status = loom_serve_device_group_create(host, 1, &options, &group,
                                                 iree_allocator_system());
    loom_serve_host_release(host);
    IREE_ASSERT_OK(status);
    device = loom_serve_device_group_at(group, 0);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_create(
        loom_serve_device_memory_pool(device), kSlabBytes, 256, &statistics,
        &reservation));
    IREE_ASSERT_OK(
        loom_serve_virtual_buffer_commit(reservation, 0, kSlabBytes));
    IREE_ASSERT_OK(
        iree_hal_buffer_subspan(loom_serve_virtual_buffer_handle(reservation),
                                256, 1024, iree_allocator_system(), &view));
    for (auto** semaphore : {&gate, &uploaded, &completion}) {
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
    loom_serve_retirement_await(&retirement);
    loom_serve_retirement_deinitialize(&retirement);
    iree_hal_semaphore_release(completion);
    iree_hal_semaphore_release(uploaded);
    iree_hal_semaphore_release(gate);
    IREE_EXPECT_OK(loom_serve_virtual_buffer_destroy(reservation));
    IREE_EXPECT_OK(loom_serve_device_group_destroy(group));
  }

  // One mapped slab remains owned until all transfer views retire.
  static constexpr uint64_t kSlabBytes = 2 * 1024 * 1024;
  // Application group enclosing all device resources.
  loom_serve_device_group_t* group = nullptr;
  // Borrowed device record within the application group.
  loom_serve_device_t* device = nullptr;
  // Parameter/state-like virtual reservation, independent of view references.
  loom_serve_virtual_buffer_t* reservation = nullptr;
  // Reservation accounting retained through destruction.
  loom_serve_memory_statistics_t statistics = {};
  // Final ownership counter shared with queue completion callbacks.
  loom_serve_retirement_t retirement = {};
  // Protects the nonblocking callback publication to the outer test owner.
  std::mutex wake_mutex;
  // Outer wait for the actual final buffer release, not semaphore readiness.
  std::condition_variable wake_condition;
  // Final retirement notifications; detachment prevents subsequent callbacks.
  uint32_t wake_count = 0;
  // Host reference released after the queue has accepted its nested child.
  iree_hal_buffer_t* view = nullptr;
  // Unsignaled dependency holding accepted work without sleeps or GPU loops.
  iree_hal_semaphore_t* gate = nullptr;
  // Explicit edge from upload to download; transaction siblings are unordered.
  iree_hal_semaphore_t* uploaded = nullptr;
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
  loom_serve_retirement_set_observer(&retirement, Wake, this);
  EXPECT_TRUE(loom_serve_retirement_is_idle(&retirement));
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
      {1, &uploaded, &value}, 1, operations, nullptr);
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_transfer(
        loom_serve_device_transfer_queue(device), {1, &uploaded, &value},
        {1, &completion, &value}, 1, operations + 1, nullptr);
  }
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
  EXPECT_FALSE(loom_serve_retirement_is_idle(&retirement));
  {
    std::lock_guard<std::mutex> lock(wake_mutex);
    EXPECT_EQ(wake_count, 0u);
  }
  EXPECT_EQ(statistics.committed_bytes, kSlabBytes);
  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  {
    std::unique_lock<std::mutex> lock(wake_mutex);
    wake_condition.wait(lock, [&] { return wake_count != 0; });
    EXPECT_EQ(wake_count, 1u);
  }
  EXPECT_TRUE(loom_serve_retirement_is_idle(&retirement));
  loom_serve_retirement_set_observer(&retirement, nullptr, nullptr);
  if (!GetParam()) {
    IREE_EXPECT_OK(iree_hal_semaphore_wait(
        completion, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_EQ(payload.output, payload.input);
  }
  // A later, caller-owned view remains tracked, but a detached observer never
  // borrows the old owner payload again.
  view = loom_serve_virtual_buffer_handle(reservation);
  iree_hal_buffer_retain(view);
  IREE_ASSERT_OK(
      loom_serve_retirement_track(&retirement, &view, iree_allocator_system()));
  EXPECT_FALSE(loom_serve_retirement_is_idle(&retirement));
  iree_hal_buffer_release(view);
  view = nullptr;
  EXPECT_TRUE(loom_serve_retirement_is_idle(&retirement));
  EXPECT_EQ(wake_count, 1u);
  // Neither successful nor failed readiness is used to authorize this unmap.
  IREE_ASSERT_OK(loom_serve_virtual_buffer_destroy(reservation));
  reservation = nullptr;
  EXPECT_EQ(statistics.committed_bytes, 0u);
}

INSTANTIATE_TEST_SUITE_P(Readiness, RetirementTest,
                         ::testing::Values(false, true));

}  // namespace
