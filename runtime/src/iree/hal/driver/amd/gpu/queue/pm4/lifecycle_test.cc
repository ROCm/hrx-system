// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "iree/async/proactor_platform.h"
#include "iree/async/semaphore.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/state.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Only the native queue is a dependency. Captures, buffers, semaphores,
// frontier tracking, publication ownership, and failed retirement are real.
class Pm4LifecycleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const uint16_t binding_types[] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS};
    const iree_hal_buffer_binding_layout_t layout = {
        sizeof(iree_hal_buffer_native_binding_t), 1, 0, binding_types};
    IREE_ASSERT_OK(iree_hal_memory_contract_create(
        this, 2, &layout, iree_allocator_system(), &contract_));
    for (uint32_t i = 0; i < 2; ++i) {
      contract_->scopes[i].interfaces =
          1u << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
      contract_->scopes[i].bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] =
          0;
      contract_->scopes[i].usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    }
    family_.spec = &family_spec_;
    family_.memory.domain = this;
    queue_.base.queue_family = &family_;
    context_.event_sink = {
        [](void* user_data, const iree_hal_device_event_t* event) {
          EXPECT_EQ(event->severity, IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR);
          ++static_cast<Pm4LifecycleTest*>(user_data)->diagnostics_;
        },
        this};
    iree_slim_mutex_initialize(&completion_.drain_mutex);
    context_.gpu = &gpu_;
    gpu_.user_queue_inactivate = [](amdf_user_queue_t* native) {
      auto* self = reinterpret_cast<Pm4LifecycleTest*>(native);
      EXPECT_EQ(self->destroyed_, 0u);
      EXPECT_NE(self->queue_.storage.mapping, nullptr);
      ++self->native_stops_;
      self->native_access_ended_ = true;
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
        native_signals_.data(), native_signals_.size(), 1,
        native_frontiers_.data(), &signal_block_);
    iree_hal_amd_gpu_signal_block_attach(&signal_pool_, this,
                                         native_values_.data(), 0x1000,
                                         sizeof(uint64_t), 0, &signal_block_);
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
    iree_hal_memory_contract_release(contract_);
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
    iree_slim_mutex_deinitialize(&completion_.drain_mutex);
  }
  void Capture(bool partial = false,
               iree_hal_semaphore_t* second_signal = nullptr) {
    iree_hal_semaphore_t* signal_semaphores[] = {semaphore_, second_signal};
    uint64_t signal_values[] = {signal_value_, signal_value_};
    const iree_hal_semaphore_list_t signals = {
        second_signal ? 2u : 1u, signal_semaphores, signal_values};
    std::array<iree_hal_buffer_ref_t, 6> references;
    for (size_t i = 0; i < references.size(); ++i) {
      iree_hal_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
          {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
          IREE_HAL_BUFFER_USAGE_STORAGE, bytes_[i].size(),
          iree_make_byte_span(bytes_[i].data(), bytes_[i].size()),
          {[](void* data, iree_hal_buffer_t*) {
             auto* self = static_cast<Pm4LifecycleTest*>(data);
             EXPECT_TRUE(self->native_access_ended_);
             self->ExpectPending();
             ++self->destroyed_;
           },
           this},
          iree_allocator_system(), &buffer));
      references[i] = iree_hal_make_buffer_ref(buffer, 0, bytes_[i].size());
    }
    if (partial) {
      std::array<iree_hal_transfer_operation_t, 6> operations = {};
      const uint32_t pattern = 0;
      for (size_t i = 0; i < references.size(); ++i) {
        references[i].buffer->memory = {
            .contract = contract_,
            .bindings = &bindings_[i],
        };
        operations[i] = {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
            .fill =
                {
                    .target_buffer = references[i].buffer,
                    .length = bytes_[i].size(),
                    .pattern = &pattern,
                    .pattern_length = sizeof(pattern),
                },
        };
      }
      IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create_transfer(
          &queue_.captures, false, &family_, {}, signals, operations.size(),
          operations.data(), &profile_, nullptr, {}, &capture_));
    } else {
      IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
          &queue_.captures, false, {}, signals, nullptr, &function_, {}, {}, {},
          {references.size(), references.data()}, &profile_, nullptr, 0,
          &capture_));
    }
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
          .tagged_submission = reinterpret_cast<uintptr_t>(capture_) |
                               static_cast<uintptr_t>(!partial),
          .consumed_words = (i + 1) * 64,
          .consumed_arguments = (i + 1) * 128,
      };
    }
    iree_atomic_store(&queue_.producer.count, count, iree_memory_order_release);
    iree_atomic_store(&queue_.native_failure,
                      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST),
                      iree_memory_order_relaxed);
    iree_hal_amd_gpu_pm4_queue_close(&queue_.base);
  }
  void FailFeedback() {
    feedback_.context = &context_;
    feedback_.control = &feedback_control_;
    iree_atomic_store(&feedback_.failure_code, IREE_STATUS_ABORTED,
                      iree_memory_order_release);
    queue_.options.feedback = &feedback_;
    EnableNativeCompletion();
  }
  void EnableNativeCompletion() {
    iree_atomic_store(&queue_.native_failure, AMDF_STATUS_OK,
                      iree_memory_order_release);
    iree_hal_amd_gpu_pm4_retirement_ring_initialize(
        records_.data(), 0x2000, records_.size(), &queue_.retirement);
  }
  bool Poll(iree_hal_amd_gpu_completion_wait_t* out_wait = nullptr) {
    auto source = iree_hal_amd_gpu_pm4_queue_completion_source(&queue_.base);
    iree_slim_mutex_lock(&completion_.drain_mutex);
    iree_hal_amd_gpu_completion_scope_t scope = {
        .completion = &completion_,
        .source = &source,
        .fn = source.fn,
    };
    iree_hal_amd_gpu_completion_wait_t wait = {};
    const bool progressed =
        source.fn(source.user_data, AMDF_STATUS_OK, &scope, &wait, nullptr);
    EXPECT_FALSE(iree_hal_amd_gpu_completion_scope_finish(&scope));
    iree_slim_mutex_unlock(&completion_.drain_mutex);
    if (out_wait) {
      *out_wait = wait;
    }
    return progressed;
  }
  void ExpectFrontier(uint64_t value, bool expected) {
    iree_async_single_frontier_t frontier;
    iree_async_single_frontier_initialize(&frontier, queue_.frontier.axis,
                                          value);
    bool satisfied = !expected;
    IREE_EXPECT_OK(iree_async_frontier_tracker_query(
        queue_.frontier.tracker,
        iree_async_single_frontier_as_const_frontier(&frontier), &satisfied));
    EXPECT_EQ(satisfied, expected);
  }
  void CompletePiece(uint32_t index) {
    records_[index].action =
        reinterpret_cast<uintptr_t>(&issued_[index].action);
    iree_atomic_store(&records_[index].generation, 1,
                      iree_memory_order_release);
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

  // Family metadata borrowed by the production queue and failure diagnostics.
  iree_hal_queue_family_t family_ = {};
  // Real transfer capabilities used by split-operation capture construction.
  const iree_hal_queue_family_spec_t family_spec_ = {
      .role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER,
  };
  // The real completion scope serializes report parsing and queue ownership.
  iree_hal_amd_gpu_completion_t completion_ = {};
  // Empty report ring after the dependency has published its first failure.
  iree_hal_amd_gpu_feedback_control_t feedback_control_ = {};
  // Logical failure is independent of each native completion word.
  iree_hal_amd_gpu_feedback_t feedback_ = {};
  // Immediate diagnostic delivery is distinct from final output publication.
  uint32_t diagnostics_ = 0;
  // Actual concrete protocol ownership state, without creating native hardware.
  iree_hal_amd_gpu_pm4_queue_t queue_ = {};
  // Coherent native completion records, published only by the dependency.
  std::array<iree_hal_amd_gpu_pm4_retirement_record_t, 4> records_ = {};
  // Host issued actions parallel to native completion storage.
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
  // Stable native-address contract retained through all payload release.
  iree_hal_memory_contract_t* contract_ = nullptr;
  // One qualified address slot per buffer; cleanup never consumes the address.
  std::array<iree_hal_buffer_native_binding_t, 6> bindings_ = {};
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
  bool native_access_ended_ = false;
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

TEST_F(Pm4LifecycleTest, PublicationFailureRetiresWithoutSignalingSuccess) {
  iree_hal_semaphore_t* rejected_signal = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_create(
      proactor_, nullptr, 0, 2, iree_allocator_system(), &rejected_signal));
  Capture(false, rejected_signal);
  EnableNativeCompletion();
  IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_publish_signal(
      semaphore_, signal_value_, readers_[0]));
  iree_hal_semaphore_fail(rejected_signal,
                          iree_status_from_code(IREE_STATUS_ABORTED));
  iree_status_t status = iree_hal_amd_gpu_semaphore_publish_signal(
      rejected_signal, signal_value_, readers_[0]);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_ABORTED);
  iree_hal_amd_gpu_pm4_submission_fail(capture_, status);
  iree_status_free(status);

  struct Observation {
    // Payload retirement must precede the first public completion notification.
    uint32_t* destroyed;
    // Each registered output wait receives exactly one terminal callback.
    uint32_t calls = 0;
    // Captured callback result distinguishes transient success from final
    // failure.
    iree_status_code_t code = IREE_STATUS_UNKNOWN;
  } observation{&destroyed_};
  iree_async_semaphore_timepoint_t timepoint = {
      .callback =
          [](void* user_data, iree_async_semaphore_timepoint_t*,
             iree_status_t status) {
            auto* observation = static_cast<Observation*>(user_data);
            EXPECT_EQ(*observation->destroyed, 6u);
            ++observation->calls;
            observation->code = iree_status_code(status);
            iree_status_free(status);
          },
      .user_data = &observation,
  };
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      reinterpret_cast<iree_async_semaphore_t*>(semaphore_), signal_value_,
      &timepoint));
  EXPECT_FALSE(Poll());
  ExpectPending();
  EXPECT_EQ(observation.calls, 0u);
  EXPECT_EQ(destroyed_, 0u);
  native_access_ended_ = true;
  CompletePiece(0);
  EXPECT_TRUE(Poll());
  EXPECT_EQ(observation.calls, 1u);
  EXPECT_EQ(observation.code, IREE_STATUS_ABORTED);
  ExpectFailed();
  iree_status_code_t completion_code = IREE_STATUS_OK;
  EXPECT_TRUE(
      iree_hal_amd_gpu_signal_query_retirement(readers_[0], &completion_code));
  EXPECT_EQ(completion_code, IREE_STATUS_ABORTED);
  iree_hal_semaphore_release(rejected_signal);
}

TEST_F(Pm4LifecycleTest, ShaderFailureWaitsForIssuedCompletion) {
  EnableNativeCompletion();
  iree_hal_amd_gpu_completion_wait_t idle = {};
  EXPECT_FALSE(Poll(&idle));
  ASSERT_NE(idle.address, nullptr);
  EXPECT_TRUE(idle.is_64_bit);
  EXPECT_FALSE(idle.pending);
  Capture();
  // Publication becomes visible through the idle source's retained hint.
  EXPECT_NE(
      iree_atomic_load(static_cast<const iree_atomic_int64_t*>(idle.address),
                       iree_memory_order_acquire),
      idle.value);
  FailFeedback();
  iree_hal_amd_gpu_completion_wait_t pending = {};
  EXPECT_FALSE(Poll(&pending));
  ASSERT_NE(pending.address, nullptr);
  EXPECT_TRUE(pending.pending);
  EXPECT_EQ(pending.is_64_bit, false);
  EXPECT_GT(diagnostics_, 0u);
  ExpectPending();
  ExpectFrontier(1, false);
  EXPECT_EQ(destroyed_, 0u);
  native_access_ended_ = true;
  CompletePiece(0);
  EXPECT_TRUE(Poll());
  EXPECT_EQ(destroyed_, 6u);
  ExpectFailed();
  ExpectFrontier(1, true);
  EXPECT_EQ(native_stops_, 0u);
}

TEST_F(Pm4LifecycleTest,
       ShaderFailureJoinsEveryIssuedPieceAndUnpublishedSuffix) {
  Capture(true);
  FailFeedback();
  EXPECT_FALSE(Poll());
  ExpectPending();
  CompletePiece(0);
  EXPECT_TRUE(Poll());
  ExpectPending();
  ExpectFrontier(1, true);
  ExpectFrontier(2, false);
  EXPECT_EQ(destroyed_, 0u);
  native_access_ended_ = true;
  CompletePiece(1);
  EXPECT_TRUE(Poll());
  ExpectPending();
  EXPECT_EQ(destroyed_, 0u);
  iree_hal_amd_gpu_pm4_submission_release(capture_);
  EXPECT_EQ(destroyed_, 6u);
  ExpectFailed();
  ExpectFrontier(2, true);
  EXPECT_EQ(native_stops_, 0u);
}

TEST_F(Pm4LifecycleTest,
       ShaderFailureRetiresRejectedSuffixBeforeLastNativePiece) {
  Capture(true);
  FailFeedback();
  EXPECT_FALSE(Poll());
  iree_hal_amd_gpu_pm4_submission_release(capture_);
  ExpectPending();
  CompletePiece(0);
  EXPECT_TRUE(Poll());
  ExpectPending();
  EXPECT_EQ(destroyed_, 0u);
  native_access_ended_ = true;
  CompletePiece(1);
  EXPECT_TRUE(Poll());
  EXPECT_EQ(destroyed_, 6u);
  ExpectFailed();
  ExpectFrontier(2, true);
  EXPECT_EQ(native_stops_, 0u);
}

}  // namespace
