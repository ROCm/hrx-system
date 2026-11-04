// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/state.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Only the native queue is a dependency. Captures, buffers, semaphores,
// frontier tracking, publication ownership, and failed retirement are real.
class AqlLifecycleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                         sizeof(*contract_),
                                         reinterpret_cast<void**>(&contract_)));
    contract_->scopes = scopes_.data();
    context_.gpu = &gpu_;
    gpu_.user_queue_inactivate = [](amdf_user_queue_t* native) {
      auto* self = reinterpret_cast<AqlLifecycleTest*>(native);
      EXPECT_EQ(self->destroyed_, 0u);
      EXPECT_NE(self->queue_.storage.mapping, nullptr);
      ++self->native_stops_;
      self->native_access_ended_ = true;
      return AMDF_STATUS_OK;
    };
    queue_.context = &context_;
    context_.event = reinterpret_cast<amdf_gpu_event_t*>(this);
    gpu_.event_signal = [](amdf_gpu_event_t*) { return AMDF_STATUS_OK; };
    queue_.options.signals = &signal_cache_;
    queue_.options.publication = &publication_;
    iree_notification_initialize(&publication_.notification);
    iree_async_semaphore_wait_context_initialize(&queue_.wait_context);
    iree_hal_amd_gpu_host_wait_set_initialize(&queue_.wait_context,
                                              &queue_.pending.waits);
    queue_.growth.context = &context_;
    queue_.growth.publisher = &queue_.publisher;
    queue_.growth.publication = &publication_;
    queue_.capacity.entry.kind = IREE_HAL_AMD_GPU_AQL_ENTRY_CAPACITY;
    queue_.scratch.growth = &queue_.growth;
    gpu_.user_queue_query_scratch = [](amdf_user_queue_t*,
                                       amdf_gpu_aql_scratch_request_t*) {
      return AMDF_STATUS_OK;
    };

    queue_.options.retirement_capacity = issued_.size();
    queue_.issued = issued_.data();
    queue_.storage = {
        .context = &context_,
        .queue = reinterpret_cast<amdf_user_queue_t*>(this),
        .mapping = reinterpret_cast<amdf_user_queue_mapping_t*>(this),
    };
    iree_hal_amd_gpu_publisher_initialize(&queue_.publisher);
    const iree_allocator_t capture_allocator = {
        this, [](void* user_data, iree_allocator_command_t command,
                 const void* params, void** inout_pointer) {
          auto* self = static_cast<AqlLifecycleTest*>(user_data);
          if (self->deny_allocations_ &&
              command != IREE_ALLOCATOR_COMMAND_FREE) {
            ++self->denied_allocations_;
            return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
          }
          const auto allocator = iree_allocator_system();
          return allocator.ctl(allocator.self, command, params, inout_pointer);
        }};
    iree_hal_amd_gpu_capture_pool_initialize(2048, capture_allocator,
                                             &queue_.captures);
    // A full capture block supplies the deferred wait's private sidecar.
    queue_.pending_size = queue_.captures.block_size;
    iree_slim_mutex_initialize(&queue_.pending.completion_mutex);
    iree_hal_amd_gpu_signal_pool_initialize({}, &signal_pool_);
    iree_hal_amd_gpu_signal_block_initialize(
        &signal_pool_, this, native_values_.data(), 0x1000, sizeof(uint64_t), 0,
        native_signals_.data(), native_signals_.size(), 1,
        native_frontiers_.data(), &signal_block_);

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
    iree_allocator_free(iree_allocator_system(), contract_);
    iree_hal_amd_gpu_capture_pool_deinitialize(&queue_.captures);
    for (auto* reader : readers_) {
      iree_hal_amd_gpu_signal_release(reader);
    }
    iree_hal_amd_gpu_signal_block_deinitialize(&signal_block_);
    iree_hal_amd_gpu_signal_pool_deinitialize(&signal_pool_);
    iree_hal_amd_gpu_publisher_deinitialize(&queue_.publisher);
    iree_hal_amd_gpu_host_wait_set_deinitialize(&queue_.pending.waits);
    iree_async_semaphore_wait_context_deinitialize(&queue_.wait_context);
    iree_notification_deinitialize(&publication_.notification);
    iree_slim_mutex_deinitialize(&queue_.pending.completion_mutex);

    iree_async_frontier_tracker_release(queue_.frontier.tracker);
    iree_hal_semaphore_release(semaphore_);
    iree_async_proactor_release(proactor_);
  }
  void Capture(bool partial = false, iree_hal_semaphore_list_t waits =
                                         iree_hal_semaphore_list_empty()) {
    std::array<iree_hal_buffer_ref_t, 6> references;
    for (size_t i = 0; i < references.size(); ++i) {
      iree_hal_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
          {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
          IREE_HAL_BUFFER_USAGE_STORAGE, bytes_[i].size(),
          iree_make_byte_span(bytes_[i].data(), bytes_[i].size()),
          {[](void* data, iree_hal_buffer_t*) {
             auto* self = static_cast<AqlLifecycleTest*>(data);
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
      const iree_hal_queue_family_t family = {};
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
      IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_transfer(
          &queue_.captures, queue_.pending_size, nullptr, &family, waits,
          {1, &semaphore_, &signal_value_}, operations.size(),
          operations.data(), 0, 0, &capture_));
    } else {
      const iree_hal_amd_gpu_aql_dispatch_request_t request = {
          .function = &function_,
          .bindings = {references.size(), references.data()},
      };
      IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_capture(
          &queue_.captures, queue_.pending_size, nullptr, &request, waits,
          {1, &semaphore_, &signal_value_}, 0, 0, &capture_));
    }
    if (waits.count) {
      EXPECT_GE(capture_->allocation_size,
                iree_host_align(capture_->capture_size, iree_max_align_t) +
                    queue_.pending_size);
    }
    for (auto reference : references) {
      iree_hal_buffer_release(reference.buffer);
    }
    const uint64_t count = partial ? 2 : 1;
    for (uint64_t i = 0; i < count; ++i) {
      if (partial) {
        iree_atomic_ref_count_inc(&capture_->transfer.ref_count);
      }
      readers_[i] = iree_hal_amd_gpu_signal_pool_try_acquire(&signal_pool_, 1);
      iree_hal_amd_gpu_signal_retain(readers_[i]);
      issued_[i] = {
          .submission = capture_,
          .signal = readers_[i],
          .arguments = (i + 1) * 128,
          .terminal = !partial,
      };
    }
    iree_atomic_store(&queue_.producer.count, count, iree_memory_order_release);
    iree_atomic_store(&queue_.failure, IREE_STATUS_ABORTED,
                      iree_memory_order_relaxed);
    iree_hal_amd_gpu_aql_queue_close(&queue_.base);
  }
  void CheckShaderFailure(bool partial, bool abandon_suffix,
                          bool pending_producer = false) {
    iree_hal_semaphore_t* producer = nullptr;
    iree_hal_amd_gpu_signal_t* producer_signal = nullptr;
    uint64_t producer_value = 1;
    if (pending_producer) {
      IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_create(
          proactor_, nullptr, 0, 2, iree_allocator_system(), &producer));
      producer_signal =
          iree_hal_amd_gpu_signal_pool_try_acquire(&signal_pool_, 0);
    }
    Capture(partial,
            producer ? iree_hal_semaphore_list_t{1, &producer, &producer_value}
                     : iree_hal_semaphore_list_empty());
    if (pending_producer) {
      capture_->native_wait_count = 1;
      capture_->waits[0].signal = producer_signal;
    }
    iree_atomic_store(&queue_.failure, IREE_STATUS_OK,
                      iree_memory_order_relaxed);
    iree_hal_amd_gpu_aql_growth_context_t growth = {
        .context = &context_,
    };
    queue_.scratch.growth = &growth;
    gpu_.user_queue_query_scratch = [](amdf_user_queue_t*,
                                       amdf_gpu_aql_scratch_request_t*) {
      return AMDF_STATUS_OK;
    };
    iree_hal_queue_family_t family = {};
    queue_.base.queue_family = &family;
    iree_hal_amd_gpu_feedback_control_t control = {};
    iree_hal_amd_gpu_feedback_t feedback = {.control = &control};
    queue_.options.feedback = &feedback;
    uint32_t diagnostics = 0;
    context_.event_sink = {
        [](void* data, const iree_hal_device_event_t* event) {
          EXPECT_EQ(event->type, IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE);
          ++*static_cast<uint32_t*>(data);
        },
        &diagnostics};
    iree_atomic_store(&feedback.failure_code, IREE_STATUS_ABORTED,
                      iree_memory_order_release);
    const auto source =
        iree_hal_amd_gpu_aql_queue_completion_source(&queue_.base);
    auto poll = [&]() {
      iree_hal_amd_gpu_completion_wait_t wait = {};
      return source.fn(source.user_data, AMDF_STATUS_OK, &wait);
    };
    const uint64_t count = partial ? 2 : 1;
    iree_async_single_frontier_t frontier;
    iree_async_single_frontier_initialize(&frontier, queue_.frontier.axis,
                                          count);
    auto expect_frontier = [&](bool expected) {
      bool satisfied = !expected;
      IREE_EXPECT_OK(iree_async_frontier_tracker_query(
          queue_.frontier.tracker,
          iree_async_single_frontier_as_const_frontier(&frontier), &satisfied));
      EXPECT_EQ(satisfied, expected);
    };

    EXPECT_FALSE(poll());
    EXPECT_GT(diagnostics, 0u);
    ExpectPending();
    expect_frontier(false);
    EXPECT_EQ(destroyed_, 0u);
    if (partial && abandon_suffix) {
      // The publisher drops its suffix while accepted pieces still own it.
      iree_hal_amd_gpu_aql_submission_release(capture_);
      ExpectPending();
    }
    for (uint64_t i = 0; i < count; ++i) {
      native_access_ended_ = i + 1 == count;
      iree_atomic_store(
          reinterpret_cast<iree_atomic_int64_t*>(readers_[i]->host_value), 0,
          iree_memory_order_release);
      if (pending_producer && i == 0) {
        EXPECT_FALSE(poll());
        ExpectPending();
        expect_frontier(false);
        EXPECT_EQ(destroyed_, 0u);
        iree_hal_amd_gpu_signal_retire(producer_signal, IREE_STATUS_OK);
        IREE_ASSERT_OK(iree_hal_semaphore_signal(producer, 1, nullptr));
      }
      EXPECT_TRUE(poll());
      expect_frontier(native_access_ended_);
      if (!native_access_ended_ || (partial && !abandon_suffix)) {
        ExpectPending();
        EXPECT_EQ(destroyed_, 0u);
      }
    }
    if (partial && !abandon_suffix) {
      // GPU completion alone cannot end the publisher's remaining borrow.
      iree_hal_amd_gpu_aql_submission_release(capture_);
    }
    ExpectFailed();
    EXPECT_EQ(destroyed_, 6u);
    EXPECT_EQ(native_stops_, 0u);
    EXPECT_FALSE(poll());
    iree_hal_semaphore_release(producer);
  }
  void CheckRejectedFollower(bool admitted, bool already_waiting,
                             bool deny_allocations = false) {
    Capture();
    iree_atomic_store(&queue_.failure, 0, iree_memory_order_relaxed);
    iree_hal_queue_family_t family = {};
    queue_.base.queue_family = &family;
    iree_hal_amd_gpu_feedback_control_t control = {};
    iree_hal_amd_gpu_feedback_t feedback = {.control = &control};
    queue_.options.feedback = &feedback;
    uint32_t diagnostics = 0;
    context_.event_sink = {
        [](void* data, const iree_hal_device_event_t* event) {
          EXPECT_EQ(event->type, IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE);
          ++*static_cast<uint32_t*>(data);
        },
        &diagnostics};
    iree_hal_semaphore_t* peer = nullptr;
    iree_hal_semaphore_t* completion = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_create(
        proactor_, nullptr, 0, 2, iree_allocator_system(), &peer));
    IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_create(
        proactor_, nullptr, 0, 2, iree_allocator_system(), &completion));
    std::vector<iree_hal_semaphore_t*> waits(deny_allocations ? 128 : 2, peer);
    waits[0] = semaphore_;
    std::vector<uint64_t> values(waits.size(), 1);
    std::array<iree_hal_semaphore_t*, 2> signals = {semaphore_, completion};
    std::array<uint64_t, 2> signal_values = {2, 1};
    iree_hal_amd_gpu_aql_submission_t* follower = nullptr;
    auto* peer_signal =
        admitted ? iree_hal_amd_gpu_signal_pool_try_acquire(&signal_pool_, 0)
                 : nullptr;
    const iree_hal_amd_gpu_aql_dispatch_request_t request = {};
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_capture(
        &queue_.captures, queue_.pending_size, nullptr, &request,
        {waits.size(), waits.data(), values.data()},
        {signals.size(), signals.data(), signal_values.data()}, 0, 0,
        &follower));
    auto* dependencies = follower->waits;
    if (admitted) {
      // Every unresolved admitted dependency owns its native producer point.
      follower->native_wait_count = waits.size();
      dependencies[0].signal = readers_[0];
      iree_hal_amd_gpu_signal_retain(readers_[0]);
      dependencies[1].signal = peer_signal;
    }
    std::vector<iree_byte_span_t> held_blocks;
    if (deny_allocations) {
      ASSERT_GT(follower->capture_size, queue_.captures.block_size);
      // Exhaust reusable capacity before denying backing allocations. The
      // accepted follower must use its own tail for each successive wait.
      while (queue_.captures.available.head || queue_.captures.returned.head) {
        iree_byte_span_t storage;
        IREE_ASSERT_OK(iree_hal_amd_gpu_capture_pool_acquire(&queue_.captures,
                                                             1, &storage));
        held_blocks.push_back(storage);
      }
      deny_allocations_ = true;
    }
    if (!already_waiting) {
      iree_atomic_store(&feedback.failure_code, IREE_STATUS_ABORTED,
                        iree_memory_order_release);
    }
    const auto publish =
        iree_hal_amd_gpu_aql_queue_publication_source(&queue_.base);
    iree_hal_amd_gpu_publication_enqueue(&publication_, 0, &queue_.publisher,
                                         &follower->entry.publication);
    EXPECT_FALSE(publish.publish(publish.user_data));
    if (already_waiting) {
      iree_atomic_store(&feedback.failure_code, IREE_STATUS_ABORTED,
                        iree_memory_order_release);
      iree_hal_amd_gpu_aql_queue_notify_failure(&queue_.base);
      EXPECT_FALSE(publish.publish(publish.user_data));
    }
    uint64_t value = UINT64_MAX;
    IREE_EXPECT_OK(iree_hal_semaphore_query(completion, &value));
    EXPECT_EQ(value, 0u);
    ExpectPending();
    EXPECT_EQ(destroyed_, 0u);
    EXPECT_GT(iree_atomic_load(&queue_.pending.host_owners,
                               iree_memory_order_acquire),
              0);

    // Retiring the copy may fail the stream, but the follower still joins its
    // other input before publishing its own private completion.
    iree_atomic_store(
        reinterpret_cast<iree_atomic_int64_t*>(readers_[0]->host_value), 0,
        iree_memory_order_release);
    const auto retire =
        iree_hal_amd_gpu_aql_queue_completion_source(&queue_.base);
    iree_hal_amd_gpu_completion_wait_t wait = {};
    native_access_ended_ = true;
    EXPECT_TRUE(retire.fn(retire.user_data, AMDF_STATUS_OK, &wait));
    EXPECT_EQ(destroyed_, 6u);
    ExpectFailed();
    IREE_EXPECT_OK(iree_hal_semaphore_query(completion, &value));
    EXPECT_EQ(value, 0u);
    if (admitted) {
      EXPECT_EQ(follower->waits, dependencies);
      EXPECT_EQ(follower->wait_count, waits.size());
      EXPECT_EQ(dependencies[0].signal, readers_[0]);
    }
    if (peer_signal) {
      iree_hal_amd_gpu_signal_retire(peer_signal, IREE_STATUS_DATA_LOSS);
    }
    iree_hal_semaphore_fail(peer, iree_status_from_code(IREE_STATUS_DATA_LOSS));
    EXPECT_FALSE(publish.publish(publish.user_data));
    IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                          iree_hal_semaphore_query(completion, &value));
    EXPECT_EQ(iree_atomic_load(&queue_.pending.host_owners,
                               iree_memory_order_acquire),
              0);
    EXPECT_TRUE(iree_hal_amd_gpu_publisher_is_idle(&queue_.publisher));
    EXPECT_GT(diagnostics, 0u);
    EXPECT_EQ(native_stops_, 0u);
    iree_hal_semaphore_release(completion);
    iree_hal_semaphore_release(peer);
    EXPECT_EQ(denied_allocations_, 0u);
    for (auto storage : held_blocks) {
      iree_hal_amd_gpu_capture_pool_release(&queue_.captures, storage);
    }
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
    iree_hal_amd_gpu_aql_queue_fail_issued(&queue_, IREE_STATUS_ABORTED);
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
        iree_hal_amd_gpu_aql_queue_try_inactivate(&queue_.base, &status));
    EXPECT_EQ(status, AMDF_STATUS_OK);
    EXPECT_EQ(destroyed_, 0u);
    for (auto* reader : readers_) {
      if (reader) {
        iree_status_code_t code = IREE_STATUS_OK;
        EXPECT_FALSE(iree_hal_amd_gpu_signal_query_retirement(reader, &code));
      }
    }
    ASSERT_TRUE(iree_hal_amd_gpu_aql_queue_retire_failed(&queue_.base,
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

  // Allocator dependency denies new backing only after capture acceptance.
  bool deny_allocations_ = false;
  // Any denied request exposes an allocation on the dependency cleanup path.
  uint32_t denied_allocations_ = 0;
  // Device progress notification storage, driven synchronously by the test.
  iree_hal_amd_gpu_publication_t publication_ = {};
  // Native signal-capacity dependency; these cases require no acquisition.
  iree_hal_amd_gpu_signal_cache_t signal_cache_ = {};
  // Actual concrete protocol ownership state, without creating native hardware.
  iree_hal_amd_gpu_aql_queue_t queue_ = {};
  // Host issued actions; GPU completion storage is deliberately absent.
  std::array<iree_hal_amd_gpu_aql_retirement_t, 4> issued_ = {};
  // Native dependency exposing only the synchronous stop boundary.
  amdf_gpu_api_t gpu_ = {};
  // Borrowed API owner, live through every capture and stop call.
  iree_hal_amd_gpu_context_t context_ = {};
  // Immutable dispatch metadata, unused by this ownership-only path.
  iree_hal_amd_gpu_aql_function_t function_ = {};
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
  // Queue/program address contract for the split transfer capture.
  std::array<iree_hal_memory_scope_access_t, 2> scopes_ = {};
  // Stable contract retained by the six actual HAL buffers.
  iree_hal_memory_contract_t* contract_ = nullptr;
  // One qualified address slot per buffer; no address is consumed by cleanup.
  std::array<iree_hal_buffer_native_binding_t, 6> bindings_ = {};
  // Non-owning published capture, or owning unpublished suffix in partial
  // tests.
  iree_hal_amd_gpu_aql_submission_t* capture_ = nullptr;
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
};

TEST_F(AqlLifecycleTest, NativeStopPrecedesSixBufferReleases) {
  Capture();
  FailIssued();
  EXPECT_TRUE(iree_hal_amd_gpu_aql_queue_is_host_idle(&queue_.base));
  EXPECT_FALSE(iree_hal_amd_gpu_aql_queue_is_idle(&queue_.base));
  ASSERT_TRUE(iree_hal_amd_gpu_publisher_try_acquire(&queue_.publisher));
  amdf_status_t status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  EXPECT_FALSE(
      iree_hal_amd_gpu_aql_queue_try_inactivate(&queue_.base, &status));
  EXPECT_EQ(status, amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL));
  EXPECT_EQ(native_stops_, 0u);
  EXPECT_EQ(destroyed_, 0u);
  EXPECT_FALSE(iree_hal_amd_gpu_publisher_handoff(&queue_.publisher, nullptr));
  StopAndRetire();
  EXPECT_EQ(destroyed_, 6u);
  ExpectFailed();
  EXPECT_TRUE(iree_hal_amd_gpu_aql_queue_is_idle(&queue_.base));
  ASSERT_TRUE(iree_hal_amd_gpu_aql_queue_try_inactivate(&queue_.base, &status));
  EXPECT_EQ(native_stops_, 1u);
  EXPECT_TRUE(iree_hal_amd_gpu_aql_queue_retire_failed(&queue_.base,
                                                       IREE_STATUS_ABORTED));
  EXPECT_EQ(destroyed_, 6u);
}

TEST_F(AqlLifecycleTest, UnpublishedSuffixRetainsItsIndependentHostOwnership) {
  Capture(true);
  FailIssued();
  StopAndRetire();
  EXPECT_EQ(destroyed_, 0u);
  ExpectPending();
  iree_hal_amd_gpu_aql_submission_release(capture_);
  EXPECT_EQ(destroyed_, 6u);
  ExpectFailed();
}

TEST_F(AqlLifecycleTest, ShaderFailureWaitsForNativeCompletion) {
  CheckShaderFailure(/*partial=*/false, /*abandon_suffix=*/false);
}

TEST_F(AqlLifecycleTest, ShaderFailureJoinsEveryAcceptedPiece) {
  CheckShaderFailure(/*partial=*/true, /*abandon_suffix=*/true);
}

TEST_F(AqlLifecycleTest, ShaderFailureJoinsUnpublishedSuffix) {
  CheckShaderFailure(/*partial=*/true, /*abandon_suffix=*/false);
}

TEST_F(AqlLifecycleTest, ShaderFailureJoinsUnadmittedFollowers) {
  CheckRejectedFollower(/*admitted=*/false, /*already_waiting=*/false);
}

TEST_F(AqlLifecycleTest, ShaderFailureJoinsCapacityDeferredFollowers) {
  CheckRejectedFollower(/*admitted=*/true, /*already_waiting=*/false);
}

TEST_F(AqlLifecycleTest, ShaderFailurePreservesEnrolledDependencyWaits) {
  CheckRejectedFollower(/*admitted=*/false, /*already_waiting=*/true);
}

TEST_F(AqlLifecycleTest, ShaderFailureJoinsProducerHostRetirement) {
  CheckShaderFailure(/*partial=*/false, /*abandon_suffix=*/false,
                     /*pending_producer=*/true);
}

TEST_F(AqlLifecycleTest, AbandonedSuffixKeepsNonterminalProducerJoin) {
  CheckShaderFailure(/*partial=*/true, /*abandon_suffix=*/true,
                     /*pending_producer=*/true);
}

TEST_F(AqlLifecycleTest, OversizedFailureJoinsWithoutAllocating) {
  CheckRejectedFollower(/*admitted=*/false, /*already_waiting=*/false,
                        /*deny_allocations=*/true);
}

}  // namespace
