// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/aql/retirement.h"

#include <array>
#include <vector>

#include "iree/async/event.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/driver/amd/gpu/queue/aql/publication.h"
#include "iree/hal/driver/amd/gpu/queue/aql/signal.h"
#include "iree/hal/driver/amd/gpu/queue/aql/state.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Real admission, packet publication, captures, aliases, timelines, and
// retirement. The fixture supplies native storage and completion observations.
class AqlRetirementTest : public ::testing::Test {
 protected:
  struct Signals {
    // Stable native-domain identity and allocation pool.
    iree_hal_amd_gpu_signal_cache_t cache = {};
    // Backing owner for independently retained native readers.
    iree_hal_amd_gpu_signal_block_t block = {};
    // Host logical ownership records.
    std::array<iree_hal_amd_gpu_signal_t, 128> signals = {};
    // Native descriptors written by the controlled hardware dependency.
    std::array<iree_hal_amd_gpu_aql_signal_t, 128> values = {};
    // Two-queue causal vectors for every signal.
    std::array<uint64_t, 2 + 128 * 2> frontiers = {};
  };

  struct Payload {
    // Real HAL resource lifetime; no executable lookup is needed by retirement.
    iree_hal_executable_t executable = {};
    // Fixture outliving the final resource callback.
    AqlRetirementTest* owner = nullptr;
    // Independently retained reader proving that native use has ended.
    iree_hal_amd_gpu_signal_t* signal = nullptr;
  };

  struct Observation {
    // Fixture outliving the registered timepoint.
    AqlRetirementTest* owner;
    // Payload count required before this public success notification.
    size_t expected_destroyed;
    // Number of delivered callbacks.
    size_t calls = 0;
  };

  void SetUp() override {
    IREE_ASSERT_OK(iree_async_event_native_initialize(&context_.kernel_event));
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    gpu_.user_queue_query_scratch =
        [](amdf_user_queue_t*, amdf_gpu_aql_scratch_request_t* request) {
          request->byte_length = 0;
          return AMDF_STATUS_OK;
        };
    context_.gpu = &gpu_;
    context_.event_sink = {
        [](void* user_data, const iree_hal_device_event_t* event) {
          EXPECT_EQ(event->severity, IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR);
          ++static_cast<AqlRetirementTest*>(user_data)->diagnostics_;
        },
        this};
    for (size_t i = 0; i < signals_.size(); ++i) {
      auto& signals = signals_[i];
      iree_hal_amd_gpu_signal_pool_initialize({}, &signals.cache.pool);
      iree_hal_amd_gpu_signal_block_initialize(
          signals.signals.data(), signals.signals.size(), 2,
          signals.frontiers.data(), &signals.block);
      const amdf_gpu_event_info_t event = {};
      iree_hal_amd_gpu_aql_signal_block_attach(
          &signals.cache.pool, &context_, &event, signals.values.data(),
          0x10000 + i * 0x10000, &signals.block);
    }
    queue_.base.queue_family = &family_;
    queue_.context = &context_;
    queue_.storage.context = &context_;
    queue_.growth.context = &context_;
    queue_.options.signals = &signals_[0].cache;
    queue_.options.retirement_capacity = issued_.size();
    queue_.issued = issued_.data();
    queue_.producer.arguments = &arguments_;
    queue_.producer.frontier = frontier_.data();
    queue_.producer.queue_count = frontier_.size();
    arguments_.capacity = argument_bytes_.size();
    arguments_.address = 0x50000;
    arguments_.memory.contents =
        iree_make_byte_span(argument_bytes_.data(), argument_bytes_.size());
    iree_hal_amd_gpu_aql_scratch_initialize(&queue_.growth, nullptr, nullptr,
                                            &queue_.scratch);
    iree_hal_amd_gpu_publisher_initialize(&queue_.publisher);
    iree_hal_amd_gpu_capture_pool_initialize(512, iree_allocator_system(),
                                             &queue_.captures);
    iree_hal_amd_gpu_aql_ring_initialize(
        packets_.data(), 0x40000, packets_.size(), &read_index_, &write_index_,
        &doorbell_, AMDF_CACHE_TRANSITION_EXECUTOR_NONE, &queue_.producer.ring);
    iree_slim_mutex_initialize(&completion_.drain_mutex);
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &queue_.frontier.tracker));
    queue_.frontier.axis = iree_async_axis_make_queue(1, 0, 0, 0, 0);
    IREE_ASSERT_OK(iree_async_frontier_tracker_register_axis(
        queue_.frontier.tracker, queue_.frontier.axis, nullptr));
    function_.base.metadata = &metadata_;
    function_.base.kernarg = &layout_;
    function_.kernel_object = 0x60000;
  }

  void TearDown() override {
    CompleteAll();
    Poll();
    EXPECT_EQ(
        iree_atomic_load(&queue_.completed.count, iree_memory_order_acquire),
        iree_atomic_load(&queue_.producer.count, iree_memory_order_acquire));
    EXPECT_EQ(destroyed_, payload_count_);
    for (auto* semaphore : semaphores_) {
      iree_hal_semaphore_release(semaphore);
    }
    for (auto* signal : readers_) {
      iree_hal_amd_gpu_signal_release(signal);
    }
    iree_hal_amd_gpu_capture_pool_deinitialize(&queue_.captures);
    iree_hal_amd_gpu_publisher_deinitialize(&queue_.publisher);
    iree_async_frontier_tracker_release(queue_.frontier.tracker);
    for (auto& signals : signals_) {
      iree_hal_amd_gpu_signal_block_deinitialize(&signals.block);
      iree_hal_amd_gpu_signal_pool_deinitialize(&signals.cache.pool);
    }
    iree_slim_mutex_deinitialize(&completion_.drain_mutex);
    iree_async_proactor_release(proactor_);
    iree_async_event_native_deinitialize(&context_.kernel_event);
  }

  iree_hal_semaphore_t* Semaphore() {
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_CHECK_OK(iree_hal_amd_gpu_semaphore_create(
        proactor_, nullptr, 0, 2, iree_allocator_system(), &semaphore));
    semaphores_.push_back(semaphore);
    return semaphore;
  }

  void Publish(iree_hal_semaphore_t* output, uint64_t value,
               iree_hal_semaphore_t* predecessor = nullptr,
               uint64_t predecessor_value = 0) {
    std::vector<iree_hal_amd_gpu_aql_wait_t> waits;
    if (predecessor) {
      waits.push_back({.semaphore = predecessor, .value = predecessor_value});
    }
    iree_hal_amd_gpu_aql_wait_counts_t counts;
    iree_host_size_t wait_index = 0;
    iree_hal_amd_gpu_aql_wait_kind_t kind;
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_admit_waits(
        &context_, &signals_[0].cache.pool, waits.size(), waits.data(), true,
        &counts, &wait_index, &kind));
    ASSERT_EQ(kind, IREE_HAL_AMD_GPU_AQL_WAIT_READY);
    ASSERT_LT(payload_count_, payloads_.size());
    auto& payload = payloads_[payload_count_++];
    payload.owner = this;
    static const iree_hal_executable_vtable_t vtable = {
        .destroy =
            [](iree_hal_executable_t* executable) {
              auto* payload = reinterpret_cast<Payload*>(executable);
              EXPECT_EQ(iree_atomic_load(reinterpret_cast<iree_atomic_int64_t*>(
                                             payload->signal->host_value),
                                         iree_memory_order_acquire),
                        0);
              ++payload->owner->destroyed_;
            },
    };
    iree_hal_executable_initialize(&family_, &vtable, &payload.executable);
    const uint32_t argument = static_cast<uint32_t>(value);
    const iree_hal_amd_gpu_aql_dispatch_request_t request = {
        .executable = &payload.executable,
        .function = &function_,
        .geometry =
            {
                .workgroup_size = {64, 1, 1},
                .workgroup_count = {1, 1, 1},
            },
        .constants = iree_make_const_byte_span(&argument, sizeof(argument)),
        .flags = IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS,
    };
    const uint64_t position =
        iree_atomic_load(&queue_.producer.count, iree_memory_order_relaxed);
    iree_hal_amd_gpu_aql_submission_t* submission = nullptr;
    iree_status_t status = iree_ok_status();
    ASSERT_EQ(iree_hal_amd_gpu_aql_publish(&queue_, &request, waits.size(),
                                           waits.data(), counts,
                                           {1, &output, &value}, 0, 0, nullptr,
                                           nullptr, &submission, &status),
              IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE);
    IREE_ASSERT_OK(status);
    auto* signal = queue_.issued[position & (issued_.size() - 1)].signal;
    iree_hal_amd_gpu_signal_retain(signal);
    readers_.push_back(signal);
    payload.signal = signal;
    iree_hal_executable_release(&payload.executable);
  }

  void CompleteAll() {
    ASSERT_EQ(iree_hal_amd_gpu_aql_publish_run(&queue_), AMDF_STATUS_OK);
    for (auto* signal : readers_) {
      iree_atomic_store(
          reinterpret_cast<iree_atomic_int64_t*>(signal->host_value), 0,
          iree_memory_order_release);
    }
    // Native consumption invalidates packet headers before returning credit.
    for (auto& packet : packets_) {
      packet.words[0] = IREE_HAL_AMD_GPU_AQL_HEADER_INVALID;
    }
    iree_atomic_store(&read_index_, queue_.producer.ring.reserved,
                      iree_memory_order_release);
  }

  bool Poll() {
    auto source = iree_hal_amd_gpu_aql_queue_completion_source(&queue_.base);
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
    return progressed;
  }

  void Observe(iree_hal_semaphore_t* semaphore, uint64_t value,
               Observation* observation,
               iree_async_semaphore_timepoint_t* timepoint) {
    *timepoint = {
        .callback =
            [](void* user_data, iree_async_semaphore_timepoint_t*,
               iree_status_t status) {
              auto* observation = static_cast<Observation*>(user_data);
              IREE_EXPECT_OK(status);
              EXPECT_EQ(observation->owner->destroyed_,
                        observation->expected_destroyed);
              ++observation->calls;
            },
        .user_data = observation,
    };
    IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
        reinterpret_cast<iree_async_semaphore_t*>(semaphore), value,
        timepoint));
  }

  void ExpectRetired(size_t first, iree_status_code_t expected) {
    for (size_t i = first; i < readers_.size(); ++i) {
      iree_status_code_t code = IREE_STATUS_UNKNOWN;
      ASSERT_TRUE(iree_hal_amd_gpu_signal_query_retirement(readers_[i], &code));
      EXPECT_EQ(code, expected);
    }
  }

  // Native scratch query dependency, with no allocation request.
  amdf_gpu_api_t gpu_ = {};
  // Shared native domain and completion wake destination.
  iree_hal_amd_gpu_context_t context_ = {};
  // Real host timeline scheduler.
  iree_async_proactor_t* proactor_ = nullptr;
  // Family borrowed by the queue and payload resources.
  iree_hal_queue_family_t family_ = {};
  // Actual queue publication and retirement state.
  iree_hal_amd_gpu_aql_queue_t queue_ = {};
  // Same completion serialization used by the device's completion source.
  iree_hal_amd_gpu_completion_t completion_ = {};
  // Finite retirement capacity deliberately smaller than signal storage.
  std::array<iree_hal_amd_gpu_aql_retirement_t, 16> issued_ = {};
  // Native storage for this queue and an independent producer.
  std::array<Signals, 2> signals_;
  // Native packet ring with independently wrapping retirement coordinates.
  std::array<iree_hal_amd_gpu_aql_packet_t, 32> packets_ = {};
  // Native consumption credit.
  iree_atomic_int64_t read_index_ = 0;
  // Host reservation extent.
  iree_atomic_int64_t write_index_ = 0;
  // Native notification operand.
  uint64_t doorbell_ = 0;
  // Dedicated ABI backing, with enough bytes for all live retirement slots.
  std::array<uint8_t, 2048> argument_bytes_ = {};
  // Argument generation borrowing the fixture's native storage.
  iree_hal_amd_gpu_byte_storage_t arguments_ = {};
  // Publisher-owned native causal frontier.
  std::array<uint64_t, 2> frontier_ = {};
  // Immutable metadata for a single-word native argument.
  iree_hal_amd_gpu_metadata_kernel_t metadata_ = {};
  // Prepared layout consumed directly by production argument emplacement.
  iree_hal_amd_gpu_kernarg_layout_t layout_ = {
      .byte_length = 4,
      .alignment = 4,
      .native_byte_length = 4,
  };
  // Placed native descriptor dependency used by packet publication.
  iree_hal_amd_gpu_aql_function_t function_ = {};
  // Application payload resources with observable final release.
  std::array<Payload, 128> payloads_;
  // Number of captures published through the production path.
  size_t payload_count_ = 0;
  // Number of payloads whose last native borrow has ended.
  size_t destroyed_ = 0;
  // Terminal errors reported through the required device event sink.
  size_t diagnostics_ = 0;
  // External signal readers retained beyond producer retirement.
  std::vector<iree_hal_amd_gpu_signal_t*> readers_;
  // Application-owned public timelines.
  std::vector<iree_hal_semaphore_t*> semaphores_;
};

TEST_F(AqlRetirementTest, SharedRunReleasesPayloadsBeforePublicCallbacks) {
  auto* semaphore = Semaphore();
  for (uint64_t value = 1; value <= 8; ++value) {
    ASSERT_NO_FATAL_FAILURE(
        Publish(semaphore, value, value > 1 ? semaphore : nullptr, value - 1));
  }
  ASSERT_NE(readers_[0]->host_value, readers_[1]->host_value);
  ASSERT_EQ(readers_[1]->host_value, readers_.back()->host_value);
  Observation observation{this, 8};
  iree_async_semaphore_timepoint_t timepoint;
  Observe(semaphore, 2, &observation, &timepoint);
  CompleteAll();
  ASSERT_TRUE(Poll());
  EXPECT_EQ(observation.calls, 1u);
  ExpectRetired(0, IREE_STATUS_OK);
  EXPECT_EQ(
      iree_atomic_load(&queue_.completed.arguments, iree_memory_order_acquire),
      arguments_.end);
  // The retained alias still owns its physical descriptor and causal snapshot.
  std::array<uint64_t, 2> frontier = {};
  iree_hal_amd_gpu_signal_join_frontier(readers_[2], frontier.size(),
                                        frontier.data());
  EXPECT_EQ(frontier[0], 3u);
  EXPECT_EQ(*readers_[2]->host_value, 0u);
}

TEST_F(AqlRetirementTest, DifferentTimelinesSplitOnePhysicalRun) {
  auto* first = Semaphore();
  auto* second = Semaphore();
  for (uint64_t value = 1; value <= 4; ++value) {
    ASSERT_NO_FATAL_FAILURE(
        Publish(first, value, value > 1 ? first : nullptr, value - 1));
  }
  Publish(second, 1, first, 4);
  Publish(second, 2, second, 1);
  Publish(first, 5, second, 2);
  Publish(first, 6, first, 5);
  ASSERT_EQ(readers_[1]->host_value, readers_.back()->host_value);
  std::array<Observation, 3> observations = {
      Observation{this, 4}, Observation{this, 6}, Observation{this, 8}};
  std::array<iree_async_semaphore_timepoint_t, 3> timepoints;
  Observe(first, 2, &observations[0], &timepoints[0]);
  Observe(second, 1, &observations[1], &timepoints[1]);
  Observe(first, 5, &observations[2], &timepoints[2]);
  CompleteAll();
  ASSERT_TRUE(Poll());
  for (const auto& observation : observations) {
    EXPECT_EQ(observation.calls, 1u);
  }
  ExpectRetired(0, IREE_STATUS_OK);
}

TEST_F(AqlRetirementTest, RepeatedValuesRetainIndependentOutcomes) {
  auto* semaphore = Semaphore();
  Publish(semaphore, 1);
  Publish(semaphore, 2, semaphore, 1);
  Publish(semaphore, 2, semaphore, 2);
  Publish(semaphore, 3, semaphore, 2);
  Observation observation{this, 4};
  iree_async_semaphore_timepoint_t timepoint;
  Observe(semaphore, 2, &observation, &timepoint);
  CompleteAll();
  ASSERT_TRUE(Poll());
  EXPECT_EQ(observation.calls, 1u);
  ExpectRetired(0, IREE_STATUS_OK);
}

TEST_F(AqlRetirementTest, PendingPhysicalRunKeepsPayloadAndPublicTimeline) {
  auto* semaphore = Semaphore();
  for (uint64_t value = 1; value <= 8; ++value) {
    ASSERT_NO_FATAL_FAILURE(
        Publish(semaphore, value, value > 1 ? semaphore : nullptr, value - 1));
  }
  ASSERT_EQ(iree_hal_amd_gpu_aql_publish_run(&queue_), AMDF_STATUS_OK);
  EXPECT_FALSE(Poll());
  EXPECT_EQ(destroyed_, 0u);
  iree_atomic_store(
      reinterpret_cast<iree_atomic_int64_t*>(readers_[0]->host_value), 0,
      iree_memory_order_release);
  ASSERT_TRUE(Poll());
  EXPECT_EQ(destroyed_, 1u);
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(value, 1u);
  for (size_t i = 1; i < readers_.size(); ++i) {
    iree_status_code_t code = IREE_STATUS_UNKNOWN;
    EXPECT_FALSE(iree_hal_amd_gpu_signal_query_retirement(readers_[i], &code));
  }
  CompleteAll();
  ASSERT_TRUE(Poll());
  ExpectRetired(0, IREE_STATUS_OK);
}

TEST_F(AqlRetirementTest, CaptureFailureSplitsTheCompletedPrefix) {
  auto* semaphore = Semaphore();
  for (uint64_t value = 1; value <= 8; ++value) {
    ASSERT_NO_FATAL_FAILURE(Publish(semaphore, value));
  }
  // A capture may report publication failure while native ownership is live.
  const iree_status_t failure = iree_status_from_code(IREE_STATUS_ABORTED);
  iree_hal_amd_gpu_aql_submission_fail(issued_[4].submission, failure);
  iree_status_free(failure);
  Observation observation{this, 4};
  iree_async_semaphore_timepoint_t timepoint;
  Observe(semaphore, 2, &observation, &timepoint);
  CompleteAll();
  ASSERT_TRUE(Poll());
  EXPECT_EQ(observation.calls, 1u);
  for (size_t i = 0; i < 4; ++i) {
    iree_status_code_t code = IREE_STATUS_UNKNOWN;
    ASSERT_TRUE(iree_hal_amd_gpu_signal_query_retirement(readers_[i], &code));
    EXPECT_EQ(code, IREE_STATUS_OK);
  }
  ExpectRetired(4, IREE_STATUS_ABORTED);
}

TEST_F(AqlRetirementTest, FailedPublicTimelineReachesEveryLogicalAlias) {
  auto* semaphore = Semaphore();
  for (uint64_t value = 1; value <= 8; ++value) {
    ASSERT_NO_FATAL_FAILURE(Publish(semaphore, value));
  }
  iree_hal_semaphore_fail(semaphore,
                          iree_status_from_code(IREE_STATUS_ABORTED));
  CompleteAll();
  ASSERT_TRUE(Poll());
  ExpectRetired(0, IREE_STATUS_ABORTED);
  EXPECT_EQ(destroyed_, 8u);
  EXPECT_GT(diagnostics_, 0u);
}

TEST_F(AqlRetirementTest, ForeignOutcomePrecedesSharedRunRetirement) {
  auto* foreign = Semaphore();
  auto* output = Semaphore();
  auto* producer =
      iree_hal_amd_gpu_signal_pool_try_acquire(&signals_[1].cache.pool, 1);
  ASSERT_NE(producer, nullptr);
  readers_.push_back(producer);
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_semaphore_publish_signal(foreign, 1, producer));
  Publish(output, 1, foreign, 1);
  for (uint64_t value = 2; value <= 8; ++value) {
    ASSERT_NO_FATAL_FAILURE(Publish(output, value, output, value - 1));
  }
  CompleteAll();
  EXPECT_FALSE(Poll());
  EXPECT_EQ(destroyed_, 0u);
  uint64_t value = UINT64_MAX;
  IREE_ASSERT_OK(iree_hal_semaphore_query(output, &value));
  EXPECT_EQ(value, 0u);
  iree_hal_amd_gpu_signal_retire(producer, IREE_STATUS_ABORTED);
  ASSERT_TRUE(Poll());
  ExpectRetired(1, IREE_STATUS_ABORTED);
  EXPECT_GT(diagnostics_, 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_semaphore_query(output, &value));
}

TEST_F(AqlRetirementTest, RetainedAliasesSurviveArgumentAndRetirementWrap) {
  auto* semaphore = Semaphore();
  for (uint64_t batch = 0; batch < 8; ++batch) {
    for (uint64_t i = 1; i <= 8; ++i) {
      const uint64_t value = batch * 8 + i;
      ASSERT_NO_FATAL_FAILURE(Publish(
          semaphore, value, value > 1 ? semaphore : nullptr, value - 1));
    }
    CompleteAll();
    ASSERT_TRUE(Poll());
    ExpectRetired(0, IREE_STATUS_OK);
    EXPECT_EQ(destroyed_, (batch + 1) * 8);
    EXPECT_EQ(iree_atomic_load(&queue_.completed.arguments,
                               iree_memory_order_acquire),
              arguments_.end);
  }
  EXPECT_GT(arguments_.end, argument_bytes_.size());
  EXPECT_GT(queue_.producer.ring.reserved, packets_.size());
  EXPECT_EQ(
      iree_atomic_load(&queue_.completed.count, iree_memory_order_acquire), 64);
}

}  // namespace
