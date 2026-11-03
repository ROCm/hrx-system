// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "iree/async/proactor_platform.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/state.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Only the native queue is a dependency. Captures, buffers, semaphores,
// frontier tracking, publication ownership, and failed retirement are real.
class Pm4LifecycleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_.gpu = &gpu_;
    gpu_.user_queue_inactivate = [](amdf_user_queue_t* native) {
      auto* self = reinterpret_cast<Pm4LifecycleTest*>(native);
      EXPECT_EQ(self->destroyed_, 0u);
      EXPECT_NE(self->queue_.storage.mapping, nullptr);
      ++self->native_stops_;
      self->native_stopped_ = true;
      return AMDF_STATUS_OK;
    };
    queue_.context = &context_;
    queue_.options.retirement_capacity = issued_.size();
    queue_.issued = issued_.data();
    queue_.storage = {
        .context = &context_,
        .queue = reinterpret_cast<amdf_user_queue_t*>(this),
        .mapping = reinterpret_cast<amdf_user_queue_mapping_t*>(this),
    };
    iree_hal_amd_gpu_publisher_initialize(&queue_.publisher);
    iree_hal_amd_gpu_capture_pool_initialize(2048, iree_allocator_system(),
                                             &queue_.captures);
    iree_slim_mutex_initialize(&queue_.pending.completion_mutex);
    iree_hal_amd_gpu_signal_pool_initialize({}, &signal_pool_);
    iree_hal_amd_gpu_signal_block_initialize(
        &signal_pool_, this, native_values_.data(), 0x1000, sizeof(uint64_t), 0,
        native_signals_.data(), native_signals_.size(), 1,
        native_frontiers_.data(), &signal_block_);
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_pm4_barrier_profile_initialize({11, 0, 0}, &profile_));
    queue_.retirement.capacity_mask = 3;
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_create(
        proactor_, nullptr, 0, 2, iree_allocator_system(), &semaphore_));
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &queue_.frontier.tracker));
    queue_.frontier.axis = iree_async_axis_make_queue(1, 0, 0, 0, 0);
    IREE_ASSERT_OK(iree_async_frontier_tracker_register_axis(
        queue_.frontier.tracker, queue_.frontier.axis, nullptr));
  }
  void TearDown() override {
    EXPECT_EQ(destroyed_, 6u);
    iree_hal_amd_gpu_capture_pool_deinitialize(&queue_.captures);
    for (auto* reader : readers_) {
      iree_hal_amd_gpu_signal_release(reader);
    }
    iree_hal_amd_gpu_signal_block_deinitialize(&signal_block_);
    iree_hal_amd_gpu_signal_pool_deinitialize(&signal_pool_);
    iree_hal_amd_gpu_publisher_deinitialize(&queue_.publisher);
    iree_slim_mutex_deinitialize(&queue_.pending.completion_mutex);
    iree_async_frontier_tracker_release(queue_.frontier.tracker);
    iree_hal_semaphore_release(semaphore_);
    iree_async_proactor_release(proactor_);
  }
  void Capture(bool partial = false) {
    std::array<iree_hal_buffer_ref_t, 6> references;
    for (size_t i = 0; i < references.size(); ++i) {
      iree_hal_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
          {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
          IREE_HAL_BUFFER_USAGE_STORAGE, bytes_[i].size(),
          iree_make_byte_span(bytes_[i].data(), bytes_[i].size()),
          {[](void* data, iree_hal_buffer_t*) {
             auto* self = static_cast<Pm4LifecycleTest*>(data);
             EXPECT_TRUE(self->native_stopped_);
             self->ExpectPending();
             ++self->destroyed_;
           },
           this},
          iree_allocator_system(), &buffer));
      references[i] = iree_hal_make_buffer_ref(buffer, 0, bytes_[i].size());
    }
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
        &queue_.captures, false, {}, {1, &semaphore_, &signal_value_}, nullptr,
        &function_, {}, {}, {}, {references.size(), references.data()},
        &profile_, nullptr, 0, &capture_));
    for (auto reference : references) {
      iree_hal_buffer_release(reference.buffer);
    }
    if (!partial) {
      capture_->signal =
          iree_hal_amd_gpu_signal_pool_try_acquire(&signal_pool_, 0);
      readers_[0] = capture_->signal;
      iree_hal_amd_gpu_signal_retain(readers_[0]);
    }
    const uint64_t count = partial ? 2 : 1;
    for (uint64_t i = 0; i < count; ++i) {
      if (partial) {
        iree_hal_amd_gpu_pm4_submission_retain(capture_);
      }
      issued_[i].action = {
          .submission = capture_,
          .consumed_words = (i + 1) * 64,
          .consumed_arguments = (i + 1) * 128,
          .terminal = !partial,
      };
    }
    iree_atomic_store(&queue_.producer.count, count, iree_memory_order_release);
    iree_atomic_store(&queue_.native_failure,
                      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST),
                      iree_memory_order_relaxed);
    iree_hal_amd_gpu_pm4_queue_close(&queue_.base);
  }
  void ExpectPending() {
    uint64_t value = UINT64_MAX;
    IREE_EXPECT_OK(iree_hal_semaphore_query(semaphore_, &value));
    EXPECT_EQ(value, 0u);
  }
  void ExpectFailed() {
    uint64_t value = 0;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                          iree_hal_semaphore_query(semaphore_, &value));
  }
  void FailIssued() {
    iree_hal_amd_gpu_pm4_queue_fail_issued(&queue_, IREE_STATUS_ABORTED);
    ExpectPending();
    iree_async_single_frontier_t frontier;
    iree_async_single_frontier_initialize(&frontier, queue_.frontier.axis, 1);
    bool satisfied = true;
    IREE_EXPECT_OK(iree_async_frontier_tracker_query(
        queue_.frontier.tracker,
        iree_async_single_frontier_as_const_frontier(&frontier), &satisfied));
    EXPECT_FALSE(satisfied);
  }
  void StopAndRetire() {
    amdf_status_t status = AMDF_STATUS_OK;
    ASSERT_TRUE(
        iree_hal_amd_gpu_pm4_queue_try_inactivate(&queue_.base, &status));
    EXPECT_EQ(status, AMDF_STATUS_OK);
    EXPECT_EQ(destroyed_, 0u);
    for (auto* reader : readers_) {
      if (reader) {
        iree_status_code_t code = IREE_STATUS_OK;
        EXPECT_FALSE(iree_hal_amd_gpu_signal_query_retirement(reader, &code));
      }
    }
    ASSERT_TRUE(iree_hal_amd_gpu_pm4_queue_retire_failed(&queue_.base,
                                                         IREE_STATUS_ABORTED));
    EXPECT_EQ(native_stops_, 1u);
    for (auto* reader : readers_) {
      if (reader) {
        iree_status_code_t code = IREE_STATUS_OK;
        EXPECT_TRUE(iree_hal_amd_gpu_signal_query_retirement(reader, &code));
        EXPECT_EQ(code, IREE_STATUS_ABORTED);
      }
    }
    EXPECT_EQ(
        iree_atomic_load(&queue_.completed.count, iree_memory_order_acquire),
        iree_atomic_load(&queue_.producer.count, iree_memory_order_acquire));
    iree_async_single_frontier_t frontier;
    iree_async_single_frontier_initialize(&frontier, queue_.frontier.axis, 1);
    bool satisfied = false;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_ABORTED,
        iree_async_frontier_tracker_query(
            queue_.frontier.tracker,
            iree_async_single_frontier_as_const_frontier(&frontier),
            &satisfied));
  }

  // Actual concrete protocol ownership state, without creating native hardware.
  iree_hal_amd_gpu_pm4_queue_t queue_ = {};
  // Host issued actions; GPU completion storage is deliberately absent.
  std::array<iree_hal_amd_gpu_pm4_retirement_slot_t, 4> issued_ = {};
  // Native dependency exposing only the synchronous stop boundary.
  amdf_gpu_api_t gpu_ = {};
  // Borrowed API owner, live through every capture and stop call.
  iree_hal_amd_gpu_context_t context_ = {};
  // Immutable dispatch metadata, unused by this ownership-only path.
  iree_hal_amd_gpu_pm4_function_t function_ = {};
  // Native dependency cells retain their original uncompleted values.
  std::array<uint64_t, 4> native_values_ = {};
  // Real independent dependency metadata, not borrowed capture pointers.
  std::array<iree_hal_amd_gpu_signal_t, 4> native_signals_ = {};
  // Prepared native causal vectors for each dependency cell.
  std::array<uint64_t, 6> native_frontiers_ = {};
  // Real finite signal owner, retained through consumer observations.
  iree_hal_amd_gpu_signal_pool_t signal_pool_ = {};
  // Stable storage lease for the native values and host metadata.
  iree_hal_amd_gpu_signal_block_t signal_block_ = {};
  // Independent readers verify failed lifetime handoff after capture release.
  std::array<iree_hal_amd_gpu_signal_t*, 4> readers_ = {};
  // Six application buffers with final-release observations.
  std::array<std::array<uint8_t, 16>, 6> bytes_ = {};
  // Non-owning published capture, or owning unpublished suffix in partial
  // tests.
  iree_hal_amd_gpu_pm4_submission_t* capture_ = nullptr;
  // Existing public completion object, independent of safe buffer retirement.
  iree_hal_semaphore_t* semaphore_ = nullptr;
  // Host semaphore service used by the real public completion object.
  iree_async_proactor_t* proactor_ = nullptr;
  // Requested completion value that must fail rather than advance successfully.
  uint64_t signal_value_ = 1;
  // Number of native removal calls; repeated terminal cleanup is idempotent.
  uint32_t native_stops_ = 0;
  // Native access proof observed by every buffer final-release callback.
  bool native_stopped_ = false;
  // Number of buffer references whose final release has run.
  uint32_t destroyed_ = 0;
  // Public barrier policy consumed only when the capture is constructed.
  iree_hal_amd_gpu_pm4_barrier_profile_t profile_ = {};
};

TEST_F(Pm4LifecycleTest, NativeStopPrecedesSixBufferReleases) {
  Capture();
  FailIssued();
  EXPECT_TRUE(iree_hal_amd_gpu_pm4_queue_is_host_idle(&queue_.base));
  EXPECT_FALSE(iree_hal_amd_gpu_pm4_queue_is_idle(&queue_.base));
  ASSERT_TRUE(iree_hal_amd_gpu_publisher_try_acquire(&queue_.publisher));
  EXPECT_FALSE(iree_hal_amd_gpu_pm4_queue_is_host_idle(&queue_.base));
  amdf_status_t status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  EXPECT_FALSE(
      iree_hal_amd_gpu_pm4_queue_try_inactivate(&queue_.base, &status));
  EXPECT_EQ(status, amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL));
  EXPECT_EQ(native_stops_, 0u);
  EXPECT_EQ(destroyed_, 0u);
  EXPECT_FALSE(iree_hal_amd_gpu_publisher_handoff(&queue_.publisher, nullptr));
  StopAndRetire();
  EXPECT_EQ(destroyed_, 6u);
  ExpectFailed();
  EXPECT_TRUE(iree_hal_amd_gpu_pm4_queue_is_idle(&queue_.base));
  ASSERT_TRUE(iree_hal_amd_gpu_pm4_queue_try_inactivate(&queue_.base, &status));
  EXPECT_EQ(native_stops_, 1u);
  EXPECT_TRUE(iree_hal_amd_gpu_pm4_queue_retire_failed(&queue_.base,
                                                       IREE_STATUS_ABORTED));
  EXPECT_EQ(destroyed_, 6u);
}

TEST_F(Pm4LifecycleTest, UnpublishedSuffixRetainsItsIndependentHostOwnership) {
  Capture(true);
  FailIssued();
  StopAndRetire();
  EXPECT_EQ(destroyed_, 0u);
  ExpectPending();
  iree_hal_amd_gpu_pm4_submission_release(capture_);
  EXPECT_EQ(destroyed_, 6u);
  ExpectFailed();
}

}  // namespace
