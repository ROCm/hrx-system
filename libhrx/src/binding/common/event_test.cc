// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/event.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <semaphore>
#include <thread>
#include <utility>

#include "common/context.h"
#include "common/device.h"
#include "common/graph.h"
#include "common/hrx_bridge.h"
#include "common/stream.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/device.h"

namespace {

template <typename Cleanup>
class ScopeExit {
 public:
  explicit ScopeExit(Cleanup cleanup) : cleanup_(std::move(cleanup)) {}
  ~ScopeExit() { cleanup_(); }
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;

 private:
  // Called once when this object leaves scope.
  Cleanup cleanup_;
};
template <typename Cleanup>
ScopeExit(Cleanup) -> ScopeExit<Cleanup>;

//===----------------------------------------------------------------------===//
// Tick pair to duration
//===----------------------------------------------------------------------===//

constexpr uint64_t kGigahertz = 1000000000ull;

// A kilohertz domain makes one tick one millisecond, so a duration reads back
// as the tick count the conversion reduced the pair to.
constexpr uint64_t kKilohertz = 1000ull;

iree_hal_streaming_timestamp_domain_t Domain(uint64_t frequency_hz,
                                             uint32_t valid_bits) {
  iree_hal_streaming_timestamp_domain_t domain = {
      .frequency_hz = frequency_hz,
      .valid_bits = valid_bits,
  };
  return domain;
}

float ElapsedMs(iree_hal_streaming_timestamp_domain_t domain,
                uint64_t start_tick, uint64_t stop_tick) {
  return iree_hal_streaming_timestamp_domain_elapsed_ms(domain, start_tick,
                                                        stop_tick);
}

TEST(TimestampDomainElapsedMsTest, ConvertsOnTheAdvertisedFrequency) {
  EXPECT_FLOAT_EQ(1000.0f, ElapsedMs(Domain(kGigahertz, 64), 0, kGigahertz));
  EXPECT_FLOAT_EQ(2000.0f,
                  ElapsedMs(Domain(kGigahertz / 2, 64), 0, kGigahertz));
  EXPECT_FLOAT_EQ(
      1.0f, ElapsedMs(Domain(kGigahertz, 64), 1000, 1000 + kGigahertz / 1000));
}

TEST(TimestampDomainElapsedMsTest, ReportsZeroForIdenticalTicks) {
  EXPECT_FLOAT_EQ(0.0f, ElapsedMs(Domain(kGigahertz, 64), 12345, 12345));
}

TEST(TimestampDomainElapsedMsTest, ReportsNegativeForAReversedPair) {
  EXPECT_FLOAT_EQ(-1000.0f, ElapsedMs(Domain(kGigahertz, 64), kGigahertz, 0));
}

// A counter narrower than 64 bits wraps at its own width, so a pair straddling
// the wrap is the interval between the two captures and not the counter range
// minus it.
TEST(TimestampDomainElapsedMsTest, ReducesAPairStraddlingAThirtyTwoBitWrap) {
  EXPECT_FLOAT_EQ(
      31.0f, ElapsedMs(Domain(kKilohertz, 32), 0xFFFFFFF0ull, 0x0000000Full));
  EXPECT_FLOAT_EQ(
      -31.0f, ElapsedMs(Domain(kKilohertz, 32), 0x0000000Full, 0xFFFFFFF0ull));
}

TEST(TimestampDomainElapsedMsTest, ReducesAPairStraddlingAFortyEightBitWrap) {
  constexpr uint64_t kWidth = 1ull << 48;
  EXPECT_FLOAT_EQ(15.0f,
                  ElapsedMs(Domain(kKilohertz, 48), kWidth - 10ull, 5ull));
  EXPECT_FLOAT_EQ(-15.0f,
                  ElapsedMs(Domain(kKilohertz, 48), 5ull, kWidth - 10ull));
}

// At full width the reduction is the identity, so ticks the signed range cannot
// hold convert as exactly as any others.
TEST(TimestampDomainElapsedMsTest, HandlesTicksAboveTheSignedRange) {
  constexpr uint64_t kHigh = std::numeric_limits<uint64_t>::max() - 4096ull;
  EXPECT_FLOAT_EQ(1000.0f,
                  ElapsedMs(Domain(kGigahertz, 64), kHigh - kGigahertz, kHigh));
  EXPECT_FLOAT_EQ(-1000.0f,
                  ElapsedMs(Domain(kGigahertz, 64), kHigh, kHigh - kGigahertz));
}

TEST(TimestampDomainElapsedMsTest, HandlesAnIntervalCrossingTheSignedBoundary) {
  constexpr uint64_t kSignedMax =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  EXPECT_FLOAT_EQ(1000.0f,
                  ElapsedMs(Domain(kGigahertz, 64), kSignedMax - kGigahertz / 2,
                            kSignedMax + kGigahertz / 2));
}

// Exactly half the counter range is the one difference with no int64_t
// representation at full width; it reads as the most negative offset the
// counter can express.
TEST(TimestampDomainElapsedMsTest, HandlesHalfTheCounterRange) {
  EXPECT_FLOAT_EQ(-9223372036854775808.0f,
                  ElapsedMs(Domain(kKilohertz, 64), 0, 1ull << 63));
  EXPECT_FLOAT_EQ(9223372036854775808.0f,
                  ElapsedMs(Domain(kKilohertz, 64), 0, (1ull << 63) - 1ull));
}

// A zeroed domain names no clock, and this conversion is declared beside the
// type it converts with, so a call carrying one has to stay inert: every pair
// reduces to no ticks, and no frequency divides them into a duration. That
// outcome is all a test can see here, since with no width every candidate for
// the counter's top bit leaves the reduced pair on the same side of it.
TEST(TimestampDomainElapsedMsTest, ReportsNoDurationForAZeroedDomain) {
  EXPECT_TRUE(std::isnan(ElapsedMs(Domain(0, 0), 4096, 8192)));
  EXPECT_TRUE(std::isnan(ElapsedMs(Domain(0, 0), 8192, 4096)));
}

//===----------------------------------------------------------------------===//
// Which devices can be timed
//===----------------------------------------------------------------------===//

// Physical device bits are their own namespace and a spec built without an
// identity facet advertises none, so any nonzero value names a valid set.
constexpr iree_hal_physical_device_affinity_t kFirstPhysicalDevice = 1ull << 0;
constexpr iree_hal_physical_device_affinity_t kSecondPhysicalDevice = 1ull << 1;
constexpr iree_hal_queue_priority_t kNormalQueuePriority =
    IREE_HAL_QUEUE_PRIORITY_NORMAL;

iree_hal_queue_family_spec_t QueueFamily(
    iree_hal_physical_device_affinity_t physical_device_affinity,
    uint32_t timestamp_valid_bits, uint64_t timestamp_frequency_hz) {
  iree_hal_queue_family_spec_t family = {
      .name = iree_make_cstring_view("test"),
      .provisioned_queue_count = 1,
      .priority_count = 1,
      .priorities = &kNormalQueuePriority,
      .timestamp_valid_bits = timestamp_valid_bits,
      .timestamp_frequency_hz = timestamp_frequency_hz,
      .physical_device_affinity = physical_device_affinity,
      .role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH,
  };
  return family;
}

TEST(QueryTimestampDomainTest, ReadsTheSelectedQueueFamily) {
  const std::array<iree_hal_queue_family_spec_t, 2> families = {
      QueueFamily(kFirstPhysicalDevice, 64, 25000000ull),
      QueueFamily(kSecondPhysicalDevice, 32, 100000000ull),
  };
  // Conversion follows the selected family even when another family has a
  // different device, frequency, and width.
  const iree_hal_streaming_timestamp_domain_t domain =
      iree_hal_streaming_query_timestamp_domain(&families[1]);
  EXPECT_EQ(100000000ull, domain.frequency_hz);
  EXPECT_EQ(32u, domain.valid_bits);
}

TEST(QueryTimestampDomainTest, RejectsAFamilySpanningTwoPhysicalDevices) {
  const iree_hal_queue_family_spec_t family = QueueFamily(
      kFirstPhysicalDevice | kSecondPhysicalDevice, 64, 100000000ull);
  const iree_hal_streaming_timestamp_domain_t domain =
      iree_hal_streaming_query_timestamp_domain(&family);
  EXPECT_EQ(0ull, domain.frequency_hz);
  EXPECT_EQ(0u, domain.valid_bits);
}

TEST(QueryTimestampDomainTest, RejectsFamilyFactsThatCannotConvertATick) {
  const std::array<iree_hal_queue_family_spec_t, 4> families = {
      QueueFamily(kFirstPhysicalDevice, 64, 0),
      QueueFamily(kFirstPhysicalDevice, 0, 100000000ull),
      QueueFamily(kFirstPhysicalDevice, 65, 100000000ull),
      QueueFamily(0, 64, 100000000ull),
  };
  for (const auto& family : families) {
    const iree_hal_streaming_timestamp_domain_t domain =
        iree_hal_streaming_query_timestamp_domain(&family);
    EXPECT_EQ(0ull, domain.frequency_hz);
    EXPECT_EQ(0u, domain.valid_bits);
  }
}

TEST(QueryTimestampDomainTest, RejectsAMissingFamily) {
  const iree_hal_streaming_timestamp_domain_t domain =
      iree_hal_streaming_query_timestamp_domain(nullptr);
  EXPECT_EQ(0ull, domain.frequency_hz);
  EXPECT_EQ(0u, domain.valid_bits);
}

//===----------------------------------------------------------------------===//
// Streaming events on a real task device
//===----------------------------------------------------------------------===//

struct FailingSnapshotAllocator {
  // Allocator that owns all context and completion-record storage.
  iree_allocator_t delegate = iree_allocator_system();
  // Armed only for the snapshot allocation being exercised.
  std::atomic<bool> fail_next_allocation = false;
  // Announces the failed allocation without timing-dependent polling.
  std::binary_semaphore failure_observed{0};

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<FailingSnapshotAllocator*>(self);
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      if (allocator->fail_next_allocation.exchange(false,
                                                   std::memory_order_acq_rel)) {
        allocator->failure_observed.release();
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "injected stream snapshot allocation failure");
      }
    }
    return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                   inout_ptr);
  }

  iree_allocator_t AsAllocator() {
    return iree_allocator_t{this, &FailingSnapshotAllocator::Control};
  }
};

class CpuStreamingContextTest : public ::testing::Test {
 protected:
  // Owns the task device and its runtime through fixture cleanup.
  hrx_device_group_t group_ = nullptr;

  struct Gate {
    // Semaphore blocking submitted test work.
    iree_hal_semaphore_t* semaphore;
    // Largest value cleanup must signal to release submitted work.
    uint64_t release_value;
    // Largest value the test has explicitly signaled.
    uint64_t signaled_value;
  };

  void SetUp() override {
    hrx_runtime_t runtime = nullptr;
    IREE_ASSERT_OK(HRX_CALL(
        hrx_runtime_create(nullptr, hrx_host_allocator_system(), &runtime)));
    const hrx_device_create_params_t params = {.uri = "task://"};
    hrx_status_t status = hrx_device_group_create(runtime, 1, &params, &group_);
    hrx_runtime_release(runtime);
    IREE_ASSERT_OK(HRX_CALL(status));
    hrx_device_t hrx_device = hrx_device_group_get_device(group_, 0);

    // Stands in for the registry entry global initialization builds around an
    // enumerated accelerator: contexts take their HAL device from the entry and
    // graphs carve their node storage out of its block pool.
    memset(&device_entry_, 0, sizeof(device_entry_));
    device_entry_.hrx_device = hrx_device;
    device_entry_.hal_device = hrx_device_hal(hrx_device);
    iree_slim_mutex_initialize(&device_entry_.primary_context_mutex);
    iree_slim_mutex_initialize(&device_entry_.graph_memory_mutex);
    iree_arena_block_pool_initialize(/*block_size=*/64 * 1024,
                                     iree_allocator_system(),
                                     &device_entry_.block_pool);

    iree_hal_streaming_context_flags_t context_flags = {
        .scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO,
    };
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entry_, context_flags, snapshot_allocator_.AsAllocator(),
        &context_));
  }

  void TearDown() override {
    // Fatal assertions return from a test body. Release every remaining gate
    // before context teardown so a diagnostic failure cannot turn into a hang.
    IREE_EXPECT_OK(ReleaseAllGates());
    iree_hal_streaming_context_release(context_);
    for (iree_host_size_t i = 0; i < gate_count_; ++i) {
      iree_hal_semaphore_release(gates_[i].semaphore);
    }
    iree_arena_block_pool_deinitialize(&device_entry_.block_pool);
    iree_slim_mutex_deinitialize(&device_entry_.graph_memory_mutex);
    iree_slim_mutex_deinitialize(&device_entry_.primary_context_mutex);
    hrx_device_group_release(group_);
    group_ = nullptr;
  }

  iree_status_t CreateGate(uint64_t release_value,
                           iree_hal_semaphore_t** out_semaphore) {
    IREE_ASSERT_ARGUMENT(out_semaphore);
    *out_semaphore = nullptr;
    if (IREE_UNLIKELY(gate_count_ >= gates_.size())) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "test gate capacity exhausted");
    }
    iree_hal_semaphore_t* semaphore = nullptr;
    iree_status_t status = iree_hal_semaphore_create(
        context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore);
    if (iree_status_is_ok(status)) {
      gates_[gate_count_].semaphore = semaphore;
      gates_[gate_count_].release_value = release_value;
      gates_[gate_count_].signaled_value = 0;
      ++gate_count_;
      *out_semaphore = semaphore;
    }
    return status;
  }

  iree_status_t SignalGate(iree_hal_semaphore_t* semaphore,
                           uint64_t new_value) {
    for (iree_host_size_t i = 0; i < gate_count_; ++i) {
      if (gates_[i].semaphore == semaphore) {
        iree_status_t status = iree_hal_semaphore_signal(semaphore, new_value,
                                                         /*frontier=*/nullptr);
        if (iree_status_is_ok(status)) {
          gates_[i].signaled_value =
              iree_max(gates_[i].signaled_value, new_value);
        }
        return status;
      }
    }
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "test gate is not registered");
  }

  iree_status_t ReleaseAllGates() {
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0; i < gate_count_; ++i) {
      if (gates_[i].signaled_value >= gates_[i].release_value) {
        continue;
      }
      iree_status_t signal_status = iree_hal_semaphore_signal(
          gates_[i].semaphore, gates_[i].release_value, /*frontier=*/nullptr);
      if (iree_status_is_ok(signal_status)) {
        gates_[i].signaled_value = gates_[i].release_value;
      }
      status = iree_status_join(status, signal_status);
    }
    return status;
  }

  iree_status_t CreateNonBlockingStream(
      iree_hal_streaming_context_t* context,
      iree_hal_streaming_stream_t** out_stream) {
    IREE_ASSERT_ARGUMENT(context);
    IREE_ASSERT_ARGUMENT(out_stream);
    *out_stream = nullptr;
    const iree_hal_queue_family_t* queue_family =
        iree_hal_device_queue_family(context->device, /*family_ordinal=*/0);
    if (IREE_UNLIKELY(!queue_family)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "task device has no queue family");
    }
    iree_hal_queue_params_t queue_params;
    iree_hal_queue_params_initialize(&queue_params);
    iree_hal_queue_t* queue = nullptr;
    iree_status_t status =
        iree_hal_queue_acquire(queue_family, &queue_params, &queue);
    iree_hal_streaming_stream_t* stream = nullptr;
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_stream_create(
          context, queue, IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING,
          /*priority=*/0, iree_allocator_system(), &stream);
    }
    iree_hal_queue_release(queue);
    if (iree_status_is_ok(status)) {
      *out_stream = stream;
    }
    return status;
  }

  // Registry entry backing |context_|; outlives every context created from it.
  iree_hal_streaming_device_t device_entry_ = {};
  // Lives through context teardown and every asynchronous completion callback.
  FailingSnapshotAllocator snapshot_allocator_;
  // Streaming context created on |device_entry_|.
  iree_hal_streaming_context_t* context_ = nullptr;
  // Gates that teardown must release before destroying |context_|.
  std::array<Gate, 2> gates_ = {};
  // Number of initialized entries in |gates_|.
  iree_host_size_t gate_count_ = 0;
};

TEST_F(CpuStreamingContextTest, UsesTheTaskQueueNanosecondClock) {
  EXPECT_EQ(1000000000ull, context_->timestamp_domain.frequency_hz);
  EXPECT_EQ(64u, context_->timestamp_domain.valid_bits);
}

TEST_F(CpuStreamingContextTest, UntimedRecordOnAnEmptyStreamCompletes) {
  iree_hal_streaming_event_t* event = nullptr;
  iree_hal_streaming_recorded_point_t point = {};
  ScopeExit cleanup([&] {
    iree_hal_streaming_event_release_recorded_point(&point);
    iree_hal_streaming_event_release(event);
  });
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(iree_hal_streaming_event_record(
      event, context_->default_stream, /*capture_external_record=*/false));
  iree_hal_streaming_event_acquire_recorded_point(event, &point);
  EXPECT_NE(nullptr, point.semaphore);
  EXPECT_EQ(0u, point.value);
  EXPECT_EQ(0u, context_->default_stream->pending_value);
  EXPECT_EQ(nullptr, point.timestamp_slot);
  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(event));
  iree_hal_streaming_completion_state_t completion =
      IREE_HAL_STREAMING_COMPLETION_STATE_PENDING;
  IREE_ASSERT_OK(iree_hal_streaming_event_query(event, &completion));
  EXPECT_EQ(IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE, completion);
}

TEST_F(CpuStreamingContextTest,
       UntimedPointSurvivesRerecordAndStreamDestruction) {
  iree_hal_streaming_stream_t* stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  iree_hal_streaming_recorded_point_t first = {};
  iree_hal_streaming_recorded_point_t second = {};
  ScopeExit cleanup([&] {
    IREE_EXPECT_OK(ReleaseAllGates());
    iree_hal_streaming_event_release_recorded_point(&second);
    iree_hal_streaming_event_release_recorded_point(&first);
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(stream);
  });
  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &stream));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(CreateGate(/*release_value=*/2, &gate));
  uint64_t gate_value = 1;
  IREE_ASSERT_OK(iree_hal_streaming_stream_wait_semaphores(
      stream, {1, &gate, &gate_value}));
  const uint64_t first_tail = stream->pending_value;
  IREE_ASSERT_OK(iree_hal_streaming_event_record(
      event, stream, /*capture_external_record=*/false));
  iree_hal_streaming_event_acquire_recorded_point(event, &first);
  EXPECT_EQ(first_tail, first.value);
  EXPECT_EQ(first_tail, stream->pending_value);

  gate_value = 2;
  IREE_ASSERT_OK(iree_hal_streaming_stream_wait_semaphores(
      stream, {1, &gate, &gate_value}));
  const uint64_t second_tail = stream->pending_value;
  IREE_ASSERT_OK(iree_hal_streaming_event_record(
      event, stream, /*capture_external_record=*/false));
  iree_hal_streaming_event_acquire_recorded_point(event, &second);
  EXPECT_EQ(second_tail, second.value);
  EXPECT_EQ(second_tail, stream->pending_value);
  EXPECT_EQ(first.semaphore, second.semaphore);
  EXPECT_GT(second.value, first.value);
  EXPECT_EQ(stream->stream_id, first.ordered_after_stream_id);
  EXPECT_EQ(first.value, first.ordered_after_stream_value);
  EXPECT_EQ(second.value, second.ordered_after_stream_value);
  EXPECT_TRUE(first.requires_independent_queue);

  IREE_ASSERT_OK(SignalGate(gate, 1));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(first.semaphore, first.value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_streaming_completion_state_t completion =
      IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE;
  IREE_ASSERT_OK(iree_hal_streaming_event_query(event, &completion));
  EXPECT_EQ(IREE_HAL_STREAMING_COMPLETION_STATE_PENDING, completion);

  IREE_ASSERT_OK(SignalGate(gate, 2));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream));
  iree_hal_streaming_stream_release(stream);
  stream = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(event));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(first.semaphore, first.value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_F(CpuStreamingContextTest,
       RejectedCrossContextRecordPreservesUntimedPoint) {
  iree_hal_streaming_context_t* other_context = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  iree_hal_streaming_recorded_point_t before = {};
  iree_hal_streaming_recorded_point_t after = {};
  ScopeExit cleanup([&] {
    IREE_EXPECT_OK(ReleaseAllGates());
    iree_hal_streaming_event_release_recorded_point(&after);
    iree_hal_streaming_event_release_recorded_point(&before);
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_context_release(other_context);
  });
  IREE_ASSERT_OK(iree_hal_streaming_context_create(
      &device_entry_,
      {.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO},
      iree_allocator_system(), &other_context));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(CreateGate(/*release_value=*/1, &gate));
  uint64_t gate_value = 1;
  IREE_ASSERT_OK(iree_hal_streaming_stream_wait_semaphores(
      context_->default_stream, {1, &gate, &gate_value}));
  IREE_ASSERT_OK(iree_hal_streaming_event_record(
      event, context_->default_stream, /*capture_external_record=*/false));
  iree_hal_streaming_event_acquire_recorded_point(event, &before);
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_begin(other_context->default_stream));
  auto* pending_recording = other_context->default_stream->command_buffer;
  ASSERT_NE(nullptr, pending_recording);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INCOMPATIBLE,
      iree_hal_streaming_event_record(event, other_context->default_stream,
                                      /*capture_external_record=*/false));
  EXPECT_EQ(pending_recording, other_context->default_stream->command_buffer);
  EXPECT_EQ(0u, other_context->default_stream->pending_value);
  iree_hal_streaming_event_acquire_recorded_point(event, &after);
  EXPECT_EQ(before.semaphore, after.semaphore);
  EXPECT_EQ(before.value, after.value);
  EXPECT_EQ(before.ordered_after_stream_id, after.ordered_after_stream_id);
  iree_hal_streaming_completion_state_t completion =
      IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE;
  IREE_ASSERT_OK(iree_hal_streaming_event_query(event, &completion));
  EXPECT_EQ(IREE_HAL_STREAMING_COMPLETION_STATE_PENDING, completion);
  IREE_ASSERT_OK(SignalGate(gate, 1));
  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(event));
}

TEST_F(CpuStreamingContextTest, UntimedPointPropagatesHostOperationFailure) {
  struct HostOperation {
    // Announces that the host operation owns its pending stream point.
    std::binary_semaphore entered{0};
    // Holds completion until the event has recorded that stream point.
    std::binary_semaphore release{0};
  } operation;
  iree_hal_streaming_stream_t* stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  std::thread thread;
  bool released = false;
  ScopeExit cleanup([&] {
    if (thread.joinable()) {
      if (!released) {
        operation.release.release();
      }
      thread.join();
    }
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(stream);
  });
  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &stream));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  thread = std::thread([&] {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_DATA_LOSS,
        iree_hal_streaming_execute_host_operation(
            stream,
            [](void* user_data) -> iree_status_t {
              auto* operation = static_cast<HostOperation*>(user_data);
              operation->entered.release();
              operation->release.acquire();
              return iree_make_status(IREE_STATUS_DATA_LOSS,
                                      "host operation failed");
            },
            &operation));
  });
  operation.entered.acquire();
  IREE_ASSERT_OK(iree_hal_streaming_event_record(
      event, stream, /*capture_external_record=*/false));
  iree_hal_streaming_completion_state_t completion =
      IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE;
  IREE_ASSERT_OK(iree_hal_streaming_event_query(event, &completion));
  EXPECT_EQ(IREE_HAL_STREAMING_COMPLETION_STATE_PENDING, completion);
  operation.release.release();
  released = true;
  thread.join();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_streaming_event_synchronize(event));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_streaming_event_query(event, &completion));
}

TEST_F(CpuStreamingContextTest, RepeatedStreamJoinReusesAcceptedSourcePoint) {
  iree_hal_streaming_stream_t* consumer = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_stream_create(
      context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NONE,
      /*priority=*/0, iree_allocator_system(), &consumer));
  ScopeExit release_consumer([&] {
    IREE_EXPECT_OK(ReleaseAllGates());
    iree_hal_streaming_stream_release(consumer);
  });
  iree_hal_streaming_stream_t* producer = context_->default_stream;
  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(CreateGate(/*release_value=*/2, &gate));
  uint64_t gate_value = 1;
  IREE_ASSERT_OK(iree_hal_streaming_stream_wait_semaphores(
      producer, {1, &gate, &gate_value}));
  IREE_ASSERT_OK(iree_hal_streaming_stream_wait_stream(consumer, producer));
  const uint64_t first_join = consumer->pending_value;
  EXPECT_GT(first_join, 0u);

  // The producer is still pending. Completion is not required to reuse an
  // accepted dependency, and a later consumer operation keeps that ordering.
  IREE_ASSERT_OK(iree_hal_streaming_stream_wait_semaphores(
      consumer, {1, &gate, &gate_value}));
  const uint64_t consumer_tail = consumer->pending_value;
  EXPECT_GT(consumer_tail, first_join);
  IREE_ASSERT_OK(iree_hal_streaming_stream_wait_stream(consumer, producer));
  EXPECT_EQ(consumer_tail, consumer->pending_value);

  // A newer source point must still create a new join.
  gate_value = 2;
  IREE_ASSERT_OK(iree_hal_streaming_stream_wait_semaphores(
      producer, {1, &gate, &gate_value}));
  IREE_ASSERT_OK(iree_hal_streaming_stream_wait_stream(consumer, producer));
  EXPECT_GT(consumer->pending_value, consumer_tail);
  IREE_ASSERT_OK(SignalGate(gate, 2));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(consumer));
}

TEST_F(CpuStreamingContextTest, DirectRecordsMeasureTheTaskQueueInterval) {
  iree_hal_streaming_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_stream_create(
      context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NONE,
      /*priority=*/0, iree_allocator_system(), &stream));
  iree_hal_streaming_event_t* start = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_NONE, iree_allocator_system(),
      &start));
  iree_hal_streaming_event_t* stop = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_NONE, iree_allocator_system(),
      &stop));

  IREE_ASSERT_OK(iree_hal_streaming_event_record(
      start, stream, /*capture_external_record=*/false));
  IREE_ASSERT_OK(iree_hal_streaming_event_record(
      stop, stream, /*capture_external_record=*/false));
  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(stop));

  float ms = -1.0f;
  iree_hal_streaming_event_timing_t timing =
      IREE_HAL_STREAMING_EVENT_TIMING_UNTIMED;
  IREE_ASSERT_OK(
      iree_hal_streaming_event_elapsed_time(&ms, start, stop, &timing));
  EXPECT_EQ(IREE_HAL_STREAMING_EVENT_TIMING_MEASURED, timing);
  EXPECT_GE(ms, 0.0f);

  iree_hal_streaming_event_release(stop);
  iree_hal_streaming_event_release(start);
  iree_hal_streaming_stream_release(stream);
}

TEST_F(CpuStreamingContextTest, ContextRecordWaitsForEveryCurrentStream) {
  iree_hal_streaming_stream_t* first_stream = nullptr;
  iree_hal_streaming_stream_t* second_stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  ScopeExit cleanup([&] {
    IREE_EXPECT_OK(ReleaseAllGates());
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(second_stream);
    iree_hal_streaming_stream_release(first_stream);
  });

  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &first_stream));
  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &second_stream));
  ASSERT_NE(first_stream->queue, second_stream->queue);
  ASSERT_NE(first_stream->queue, context_->queue);
  ASSERT_NE(second_stream->queue, context_->queue);

  iree_hal_semaphore_t* first_gate = nullptr;
  IREE_ASSERT_OK(CreateGate(/*release_value=*/2, &first_gate));
  iree_hal_semaphore_t* second_gate = nullptr;
  IREE_ASSERT_OK(CreateGate(/*release_value=*/2, &second_gate));
  uint64_t gate_value = 1;
  const iree_hal_semaphore_list_t first_wait = {
      .count = 1,
      .semaphores = &first_gate,
      .payload_values = &gate_value,
  };
  const iree_hal_semaphore_list_t second_wait = {
      .count = 1,
      .semaphores = &second_gate,
      .payload_values = &gate_value,
  };
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_wait_semaphores(first_stream, first_wait));
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_wait_semaphores(second_stream, second_wait));

  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(iree_hal_streaming_context_record_event(context_, event));

  iree_hal_streaming_completion_state_t event_completion_state =
      IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE;
  IREE_ASSERT_OK(
      iree_hal_streaming_event_query(event, &event_completion_state));
  EXPECT_EQ(IREE_HAL_STREAMING_COMPLETION_STATE_PENDING,
            event_completion_state);
  IREE_ASSERT_OK(SignalGate(first_gate, gate_value));
  IREE_ASSERT_OK(
      iree_hal_streaming_event_query(event, &event_completion_state));
  EXPECT_EQ(IREE_HAL_STREAMING_COMPLETION_STATE_PENDING,
            event_completion_state);
  IREE_ASSERT_OK(SignalGate(second_gate, gate_value));
  IREE_ASSERT_OK(iree_hal_streaming_context_synchronize(context_));
  IREE_ASSERT_OK(
      iree_hal_streaming_event_query(event, &event_completion_state));
  EXPECT_EQ(IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE,
            event_completion_state);

  // Repeat with the opposite release order. The two rounds prove that the
  // fan-in waits for each stream rather than accidentally naming only one.
  gate_value = 2;
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_wait_semaphores(first_stream, first_wait));
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_wait_semaphores(second_stream, second_wait));
  IREE_ASSERT_OK(iree_hal_streaming_context_record_event(context_, event));
  IREE_ASSERT_OK(SignalGate(second_gate, gate_value));
  IREE_ASSERT_OK(
      iree_hal_streaming_event_query(event, &event_completion_state));
  EXPECT_EQ(IREE_HAL_STREAMING_COMPLETION_STATE_PENDING,
            event_completion_state);
  IREE_ASSERT_OK(SignalGate(first_gate, gate_value));
  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(event));
}

TEST_F(CpuStreamingContextTest, ContextWaitOrdersCurrentAndLaterStreams) {
  iree_hal_streaming_stream_t* source_stream = nullptr;
  iree_hal_streaming_stream_t* current_stream = nullptr;
  iree_hal_streaming_stream_t* later_stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  ScopeExit cleanup([&] {
    IREE_EXPECT_OK(ReleaseAllGates());
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(later_stream);
    iree_hal_streaming_stream_release(current_stream);
    iree_hal_streaming_stream_release(source_stream);
  });

  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &source_stream));
  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &current_stream));
  ASSERT_NE(source_stream->queue, current_stream->queue);

  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(CreateGate(/*release_value=*/1, &gate));
  uint64_t gate_value = 1;
  const iree_hal_semaphore_list_t gate_wait = {
      .count = 1,
      .semaphores = &gate,
      .payload_values = &gate_value,
  };
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_wait_semaphores(source_stream, gate_wait));

  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(iree_hal_streaming_event_record(
      event, source_stream, /*capture_external_record=*/false));
  IREE_ASSERT_OK(iree_hal_streaming_context_wait_event(context_, event));

  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &later_stream));
  ASSERT_NE(later_stream->queue, source_stream->queue);
  ASSERT_NE(later_stream->queue, current_stream->queue);

  iree_hal_streaming_completion_state_t current_completion_state =
      IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE;
  IREE_ASSERT_OK(iree_hal_streaming_stream_query(current_stream,
                                                 &current_completion_state));
  EXPECT_EQ(IREE_HAL_STREAMING_COMPLETION_STATE_PENDING,
            current_completion_state);
  iree_hal_streaming_completion_state_t later_completion_state =
      IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE;
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_query(later_stream, &later_completion_state));
  EXPECT_EQ(IREE_HAL_STREAMING_COMPLETION_STATE_PENDING,
            later_completion_state);

  IREE_ASSERT_OK(SignalGate(gate, gate_value));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(current_stream));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(later_stream));
}

TEST_F(CpuStreamingContextTest, CrossContextWaitOrdersCurrentAndLaterStreams) {
  iree_hal_streaming_context_t* target_context = nullptr;
  iree_hal_streaming_stream_t* source_stream = nullptr;
  iree_hal_streaming_stream_t* current_stream = nullptr;
  iree_hal_streaming_stream_t* later_stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  ScopeExit cleanup([&] {
    IREE_EXPECT_OK(ReleaseAllGates());
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(later_stream);
    iree_hal_streaming_stream_release(current_stream);
    iree_hal_streaming_stream_release(source_stream);
    iree_hal_streaming_context_release(target_context);
  });

  iree_hal_streaming_context_flags_t context_flags = {
      .scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO,
  };
  IREE_ASSERT_OK(iree_hal_streaming_context_create(
      &device_entry_, context_flags, iree_allocator_system(), &target_context));
  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &source_stream));
  IREE_ASSERT_OK(CreateNonBlockingStream(target_context, &current_stream));

  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(CreateGate(/*release_value=*/1, &gate));
  uint64_t gate_value = 1;
  const iree_hal_semaphore_list_t gate_wait = {
      .count = 1,
      .semaphores = &gate,
      .payload_values = &gate_value,
  };
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_wait_semaphores(source_stream, gate_wait));

  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(iree_hal_streaming_event_record(
      event, source_stream, /*capture_external_record=*/false));
  IREE_ASSERT_OK(iree_hal_streaming_context_wait_event(target_context, event));

  IREE_ASSERT_OK(CreateNonBlockingStream(target_context, &later_stream));
  iree_hal_streaming_completion_state_t current_completion_state =
      IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE;
  IREE_ASSERT_OK(iree_hal_streaming_stream_query(current_stream,
                                                 &current_completion_state));
  EXPECT_EQ(IREE_HAL_STREAMING_COMPLETION_STATE_PENDING,
            current_completion_state);
  iree_hal_streaming_completion_state_t later_completion_state =
      IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE;
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_query(later_stream, &later_completion_state));
  EXPECT_EQ(IREE_HAL_STREAMING_COMPLETION_STATE_PENDING,
            later_completion_state);

  IREE_ASSERT_OK(SignalGate(gate, gate_value));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(current_stream));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(later_stream));
}

TEST_F(CpuStreamingContextTest,
       SnapshotAllocationFailureStillDrainsEverySelectedTail) {
  iree_hal_streaming_stream_t* first_stream = nullptr;
  iree_hal_streaming_stream_t* second_stream = nullptr;
  iree_hal_semaphore_t* first_gate = nullptr;
  iree_hal_semaphore_t* second_gate = nullptr;
  std::thread synchronize_thread;
  std::atomic<bool> synchronize_returned = false;
  std::atomic<iree_status_code_t> synchronize_status_code = IREE_STATUS_UNKNOWN;
  ScopeExit cleanup([&] {
    IREE_EXPECT_OK(ReleaseAllGates());
    if (synchronize_thread.joinable()) {
      synchronize_thread.join();
    }
    iree_hal_streaming_stream_release(second_stream);
    iree_hal_streaming_stream_release(first_stream);
  });

  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &first_stream));
  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &second_stream));
  IREE_ASSERT_OK(CreateGate(/*release_value=*/1, &first_gate));
  IREE_ASSERT_OK(CreateGate(/*release_value=*/1, &second_gate));
  uint64_t first_gate_value = 1;
  const iree_hal_semaphore_list_t first_wait = {
      .count = 1,
      .semaphores = &first_gate,
      .payload_values = &first_gate_value,
  };
  uint64_t second_gate_value = 1;
  const iree_hal_semaphore_list_t second_wait = {
      .count = 1,
      .semaphores = &second_gate,
      .payload_values = &second_gate_value,
  };
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_wait_semaphores(first_stream, first_wait));
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_wait_semaphores(second_stream, second_wait));

  snapshot_allocator_.fail_next_allocation.store(true,
                                                 std::memory_order_release);
  synchronize_thread = std::thread([&] {
    iree_status_t status = iree_hal_streaming_context_synchronize(context_);
    synchronize_status_code.store(iree_status_code(status),
                                  std::memory_order_release);
    iree_status_ignore(status);
    synchronize_returned.store(true, std::memory_order_release);
  });

  snapshot_allocator_.failure_observed.acquire();
  EXPECT_FALSE(synchronize_returned.load(std::memory_order_acquire));

  IREE_EXPECT_OK(SignalGate(first_gate, first_gate_value));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize_flushed(first_stream));
  EXPECT_FALSE(synchronize_returned.load(std::memory_order_acquire));

  IREE_EXPECT_OK(SignalGate(second_gate, second_gate_value));
  synchronize_thread.join();
  EXPECT_TRUE(synchronize_returned.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED,
            synchronize_status_code.load(std::memory_order_acquire));
}

struct HostOperationGate {
  // Set after the operation begins executing.
  std::atomic<bool> entered = false;
  // Set by the test to let the operation complete.
  std::atomic<bool> release = false;
};

iree_status_t WaitInHostOperation(void* user_data) {
  auto* gate = static_cast<HostOperationGate*>(user_data);
  gate->entered.store(true, std::memory_order_release);
  while (!gate->release.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  return iree_ok_status();
}

uint64_t StreamPendingValue(iree_hal_streaming_stream_t* stream) {
  iree_slim_mutex_lock(&stream->mutex);
  const uint64_t value = stream->pending_value;
  iree_slim_mutex_unlock(&stream->mutex);
  return value;
}

TEST_F(CpuStreamingContextTest,
       HostOperationsReserveTheirTimelineBeforeWaiting) {
  iree_hal_streaming_stream_t* stream = nullptr;
  std::thread first_thread;
  std::thread second_thread;
  HostOperationGate first_gate;
  HostOperationGate second_gate;
  ScopeExit cleanup([&] {
    first_gate.release.store(true, std::memory_order_release);
    second_gate.release.store(true, std::memory_order_release);
    IREE_EXPECT_OK(ReleaseAllGates());
    if (first_thread.joinable()) {
      first_thread.join();
    }
    if (second_thread.joinable()) {
      second_thread.join();
    }
    iree_hal_streaming_stream_release(stream);
  });

  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &stream));
  iree_hal_semaphore_t* prior_gate = nullptr;
  IREE_ASSERT_OK(CreateGate(/*release_value=*/1, &prior_gate));
  uint64_t prior_value = 1;
  const iree_hal_semaphore_list_t prior_wait = {
      .count = 1,
      .semaphores = &prior_gate,
      .payload_values = &prior_value,
  };
  IREE_ASSERT_OK(iree_hal_streaming_stream_wait_semaphores(stream, prior_wait));

  std::atomic<iree_status_code_t> first_status_code = IREE_STATUS_UNKNOWN;
  std::atomic<bool> first_finished = false;
  first_thread = std::thread([&] {
    iree_status_t status = iree_hal_streaming_execute_host_operation(
        stream, WaitInHostOperation, &first_gate);
    first_status_code.store(iree_status_code(status),
                            std::memory_order_release);
    iree_status_ignore(status);
    first_finished.store(true, std::memory_order_release);
  });
  while (StreamPendingValue(stream) < 2 &&
         !first_finished.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  ASSERT_GE(StreamPendingValue(stream), 2u);

  std::atomic<iree_status_code_t> second_status_code = IREE_STATUS_UNKNOWN;
  std::atomic<bool> second_finished = false;
  second_thread = std::thread([&] {
    iree_status_t status = iree_hal_streaming_execute_host_operation(
        stream, WaitInHostOperation, &second_gate);
    second_status_code.store(iree_status_code(status),
                             std::memory_order_release);
    iree_status_ignore(status);
    second_finished.store(true, std::memory_order_release);
  });
  while (StreamPendingValue(stream) < 3 &&
         !second_finished.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  ASSERT_GE(StreamPendingValue(stream), 3u);

  IREE_ASSERT_OK(SignalGate(prior_gate, prior_value));
  while (!first_gate.entered.load(std::memory_order_acquire) &&
         !first_finished.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  ASSERT_TRUE(first_gate.entered.load(std::memory_order_acquire));
  EXPECT_FALSE(second_gate.entered.load(std::memory_order_acquire));
  first_gate.release.store(true, std::memory_order_release);
  while (!second_gate.entered.load(std::memory_order_acquire) &&
         !second_finished.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  ASSERT_TRUE(second_gate.entered.load(std::memory_order_acquire));
  second_gate.release.store(true, std::memory_order_release);

  first_thread.join();
  second_thread.join();
  EXPECT_EQ(IREE_STATUS_OK, first_status_code.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_OK, second_status_code.load(std::memory_order_acquire));
}

TEST_F(CpuStreamingContextTest, GraphDisabledTimingRecordsOnlySignal) {
  static constexpr iree_host_size_t kEventCount = 11;

  iree_hal_streaming_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_stream_create(
      context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NONE,
      /*priority=*/0, iree_allocator_system(), &stream));
  iree_hal_streaming_graph_t* graph = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));

  std::array<iree_hal_streaming_event_t*, kEventCount> events = {};
  for (iree_host_size_t i = 0; i < kEventCount; ++i) {
    IREE_ASSERT_OK(iree_hal_streaming_event_create(
        context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
        iree_allocator_system(), &events[i]));
    iree_hal_streaming_graph_node_t* node = nullptr;
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_event_node(
        graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
        IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD, events[i], &node));
  }

  iree_hal_streaming_graph_exec_t* exec = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));
  iree_hal_streaming_graph_exec_launch_result_t launch_result =
      IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_ERROR;
  IREE_ASSERT_OK(
      iree_hal_streaming_graph_exec_launch(exec, stream, &launch_result));
  EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS, launch_result);
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream));

  EXPECT_EQ(nullptr, context_->timestamp_pool.slabs)
      << "timing-disabled events acquired timestamp storage";
  for (iree_host_size_t i = 0; i < kEventCount; ++i) {
    iree_hal_streaming_completion_state_t completion_state =
        IREE_HAL_STREAMING_COMPLETION_STATE_PENDING;
    IREE_ASSERT_OK(
        iree_hal_streaming_event_query(events[i], &completion_state));
    EXPECT_EQ(IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE, completion_state)
        << "event " << i;
  }

  for (iree_host_size_t i = kEventCount; i > 0; --i) {
    iree_hal_streaming_event_release(events[i - 1]);
  }
  iree_hal_streaming_graph_exec_release(exec);
  iree_hal_streaming_graph_release(graph);
  iree_hal_streaming_stream_release(stream);
}

}  // namespace
