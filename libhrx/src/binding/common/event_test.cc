// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>

#include "common/internal.h"
#include "common/stream.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

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
  iree_hal_streaming_timestamp_domain_t domain = {};
  domain.frequency_hz = frequency_hz;
  domain.valid_bits = valid_bits;
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
  iree_hal_queue_family_spec_t family = {};
  family.name = iree_make_cstring_view("test");
  family.provisioned_queue_count = 1;
  family.priority_count = 1;
  family.priorities = &kNormalQueuePriority;
  family.timestamp_valid_bits = timestamp_valid_bits;
  family.timestamp_frequency_hz = timestamp_frequency_hz;
  family.physical_device_affinity = physical_device_affinity;
  family.role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH;
  return family;
}

iree_hal_device_timing_spec_t Timing(iree_hal_device_timing_spec_flags_t flags,
                                     uint32_t timestamp_valid_bits,
                                     uint64_t timestamp_frequency_hz) {
  iree_hal_device_timing_spec_t timing = {};
  timing.timestamp_valid_bits = timestamp_valid_bits;
  timing.timestamp_frequency_hz = timestamp_frequency_hz;
  timing.flags = flags;
  return timing;
}

// Builds a spec carrying only the two facets the gate reads and answers it.
iree_hal_streaming_timestamp_domain_t QueryDomain(
    const iree_hal_queue_family_spec_t* families, iree_host_size_t family_count,
    iree_hal_device_timing_spec_t timing) {
  iree_hal_device_queue_spec_t queues = {};
  queues.family_count = family_count;
  queues.families = families;
  iree_hal_device_spec_params_t params = {};
  params.queues = &queues;
  params.timing = &timing;
  iree_hal_device_spec_t* spec = NULL;
  IREE_EXPECT_OK(
      iree_hal_device_spec_create(&params, iree_allocator_system(), &spec));
  const iree_hal_streaming_timestamp_domain_t domain =
      iree_hal_streaming_query_timestamp_domain(spec);
  iree_hal_device_spec_release(spec);
  return domain;
}

// The facts a single-GPU device publishes, which is every device this layer
// builds: one family covering one physical device at 100 MHz.
iree_hal_streaming_timestamp_domain_t QueryOneFamilyDomain(
    uint32_t family_valid_bits, uint64_t family_frequency_hz,
    iree_hal_device_timing_spec_flags_t timing_flags) {
  const iree_hal_queue_family_spec_t family =
      QueueFamily(kFirstPhysicalDevice, family_valid_bits, family_frequency_hz);
  return QueryDomain(
      &family, 1, Timing(timing_flags, family_valid_bits, family_frequency_hz));
}

TEST(QueryTimestampDomainTest, ReadsTheFactsOfTheSingleQueueFamily) {
  const iree_hal_streaming_timestamp_domain_t domain = QueryOneFamilyDomain(
      64, 100000000ull, IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS);
  EXPECT_EQ(100000000ull, domain.frequency_hz);
  EXPECT_EQ(64u, domain.valid_bits);
}

// The family's own width is what a tick is converted with, so a narrow counter
// is carried through rather than rounded up to the device-scope summary.
TEST(QueryTimestampDomainTest, CarriesANarrowCounterWidth) {
  const iree_hal_streaming_timestamp_domain_t domain = QueryOneFamilyDomain(
      32, 100000000ull, IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS);
  EXPECT_EQ(100000000ull, domain.frequency_hz);
  EXPECT_EQ(32u, domain.valid_bits);
}

TEST(QueryTimestampDomainTest, RejectsADeviceNotAdvertisingTimestamps) {
  const iree_hal_streaming_timestamp_domain_t domain = QueryOneFamilyDomain(
      64, 100000000ull, IREE_HAL_DEVICE_TIMING_SPEC_FLAG_NONE);
  EXPECT_EQ(0ull, domain.frequency_hz);
  EXPECT_EQ(0u, domain.valid_bits);
}

// Two families may capture in two domains, and which one a record resolves to
// is internal to the implementation, so no pair of records is known comparable.
TEST(QueryTimestampDomainTest, RejectsADeviceReportingSeveralQueueFamilies) {
  const std::array<iree_hal_queue_family_spec_t, 2> families = {
      QueueFamily(kFirstPhysicalDevice, 64, 100000000ull),
      QueueFamily(kSecondPhysicalDevice, 64, 100000000ull),
  };
  const iree_hal_streaming_timestamp_domain_t domain =
      QueryDomain(families.data(), families.size(),
                  Timing(IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS, 64,
                         100000000ull));
  EXPECT_EQ(0ull, domain.frequency_hz);
  EXPECT_EQ(0u, domain.valid_bits);
}

// One family spanning two physical devices spans two independent counters.
TEST(QueryTimestampDomainTest, RejectsAFamilySpanningTwoPhysicalDevices) {
  const iree_hal_queue_family_spec_t family = QueueFamily(
      kFirstPhysicalDevice | kSecondPhysicalDevice, 64, 100000000ull);
  const iree_hal_streaming_timestamp_domain_t domain =
      QueryDomain(&family, 1,
                  Timing(IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS, 64,
                         100000000ull));
  EXPECT_EQ(0ull, domain.frequency_hz);
  EXPECT_EQ(0u, domain.valid_bits);
}

// The flag comes from the device-scope summary and the numbers from the family,
// which is sound only while the two describe the same domain; a device whose
// facets disagree names no numbers a tick can be converted with.
TEST(QueryTimestampDomainTest, RejectsFacetsThatDisagreeAboutTheDomain) {
  const iree_hal_queue_family_spec_t family =
      QueueFamily(kFirstPhysicalDevice, 64, 100000000ull);
  const iree_hal_streaming_timestamp_domain_t disagreeing_frequency =
      QueryDomain(&family, 1,
                  Timing(IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS, 64,
                         25000000ull));
  EXPECT_EQ(0ull, disagreeing_frequency.frequency_hz);
  EXPECT_EQ(0u, disagreeing_frequency.valid_bits);

  const iree_hal_streaming_timestamp_domain_t disagreeing_width =
      QueryDomain(&family, 1,
                  Timing(IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS, 32,
                         100000000ull));
  EXPECT_EQ(0ull, disagreeing_width.frequency_hz);
  EXPECT_EQ(0u, disagreeing_width.valid_bits);
}

// A domain is populated or zeroed as a unit, so facts that cannot convert a
// tick pair leave it zeroed rather than half-filled.
TEST(QueryTimestampDomainTest, RejectsFamilyFactsThatCannotConvertATick) {
  const iree_hal_streaming_timestamp_domain_t no_frequency =
      QueryOneFamilyDomain(64, 0ull,
                           IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS);
  EXPECT_EQ(0ull, no_frequency.frequency_hz);
  EXPECT_EQ(0u, no_frequency.valid_bits);

  const iree_hal_streaming_timestamp_domain_t no_width = QueryOneFamilyDomain(
      0, 100000000ull, IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS);
  EXPECT_EQ(0ull, no_width.frequency_hz);
  EXPECT_EQ(0u, no_width.valid_bits);

  const iree_hal_streaming_timestamp_domain_t too_wide = QueryOneFamilyDomain(
      65, 100000000ull, IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS);
  EXPECT_EQ(0ull, too_wide.frequency_hz);
  EXPECT_EQ(0u, too_wide.valid_bits);
}

TEST(QueryTimestampDomainTest, RejectsADevicePublishingNoFacts) {
  const iree_hal_streaming_timestamp_domain_t domain =
      iree_hal_streaming_query_timestamp_domain(NULL);
  EXPECT_EQ(0ull, domain.frequency_hz);
  EXPECT_EQ(0u, domain.valid_bits);
}

//===----------------------------------------------------------------------===//
// Streaming events on a real task device
//===----------------------------------------------------------------------===//

struct FailingSnapshotAllocator {
  iree_allocator_t delegate = iree_allocator_system();
  std::atomic<bool> fail_allocations = false;
  std::atomic<int> allocation_attempt_count = 0;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<FailingSnapshotAllocator*>(self);
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      allocator->allocation_attempt_count.fetch_add(1,
                                                    std::memory_order_acq_rel);
      if (allocator->fail_allocations.load(std::memory_order_acquire)) {
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
  struct Gate {
    // Semaphore blocking submitted test work.
    iree_hal_semaphore_t* semaphore;
    // Largest value cleanup must signal to release submitted work.
    uint64_t release_value;
    // Largest value the test has explicitly signaled.
    uint64_t signaled_value;
  };

  void SetUp() override {
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(/*flags=*/0)));
    hrx_device_t hrx_device = nullptr;
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(/*index=*/0, &hrx_device)));

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

    iree_hal_streaming_context_flags_t context_flags = {};
    context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entry_, context_flags, iree_allocator_system(), &context_));
  }

  void TearDown() override {
    // Fatal assertions return from a test body. Release every remaining gate
    // before context teardown so a diagnostic failure cannot turn into a hang.
    IREE_EXPECT_OK(ReleaseAllGates());
    // Graph-exec destruction is nonblocking. Drain its stream-ordered
    // retirement calls before releasing the fixture-owned device entry and
    // the arena block pool embedded in it.
    IREE_EXPECT_OK(iree_hal_streaming_context_synchronize(context_));
    iree_hal_streaming_context_release(context_);
    for (iree_host_size_t i = 0; i < gate_count_; ++i) {
      iree_hal_semaphore_release(gates_[i].semaphore);
    }
    iree_arena_block_pool_deinitialize(&device_entry_.block_pool);
    iree_slim_mutex_deinitialize(&device_entry_.graph_memory_mutex);
    iree_slim_mutex_deinitialize(&device_entry_.primary_context_mutex);
    IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
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
  // Streaming context created on |device_entry_|.
  iree_hal_streaming_context_t* context_ = nullptr;
  // Gates that teardown must release before destroying |context_|.
  std::array<Gate, 2> gates_ = {};
  // Number of initialized entries in |gates_|.
  iree_host_size_t gate_count_ = 0;
};

// The CPU device publishes no timing facet and its single queue family reports
// no frequency and no width, so a context created on it reads a zeroed domain.
// The fabricated specs above pin each rejection case; this fixture pins that
// the result is propagated through context creation. Device-timing CTS covers
// the positive path on devices that advertise a timestamp domain.
TEST_F(CpuStreamingContextTest, LeavesTheDomainZeroedOnTheCpuDevice) {
  EXPECT_EQ(0ull, context_->timestamp_domain.frequency_hz);
  EXPECT_EQ(0u, context_->timestamp_domain.valid_bits);
}

// A record on a context whose domain is zeroed captures no tick, so the pool it
// would have drawn a slot from stays empty and the interval between two such
// records names no clock to be measured on. This is the only reachable path to
// that outcome anywhere in this tree: every device libhrx builds a context on
// advertises a domain.
TEST_F(CpuStreamingContextTest, DirectRecordsOnAnUntimedDeviceGoUntimed) {
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

  IREE_ASSERT_OK(iree_hal_streaming_event_record(start, stream));
  IREE_ASSERT_OK(iree_hal_streaming_event_record(stop, stream));
  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(stop));

  EXPECT_EQ(nullptr, context_->timestamp_pool.slabs)
      << "a record on a device advertising no domain acquired a tick slot";

  float ms = -1.0f;
  iree_hal_streaming_event_timing_t timing =
      IREE_HAL_STREAMING_EVENT_TIMING_MEASURED;
  IREE_ASSERT_OK(
      iree_hal_streaming_event_elapsed_time(&ms, start, stop, &timing));
  EXPECT_EQ(IREE_HAL_STREAMING_EVENT_TIMING_UNSUPPORTED, timing);
  EXPECT_FLOAT_EQ(-1.0f, ms) << "an unmeasurable pair reported a duration";

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
      /*.count=*/1,
      /*.semaphores=*/&first_gate,
      /*.payload_values=*/&gate_value,
  };
  const iree_hal_semaphore_list_t second_wait = {
      /*.count=*/1,
      /*.semaphores=*/&second_gate,
      /*.payload_values=*/&gate_value,
  };
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_wait_semaphores(first_stream, first_wait));
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_wait_semaphores(second_stream, second_wait));

  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(iree_hal_streaming_context_record_event(context_, event));

  int event_status = 0;
  IREE_ASSERT_OK(iree_hal_streaming_event_query(event, &event_status));
  EXPECT_EQ(1, event_status);
  IREE_ASSERT_OK(SignalGate(first_gate, gate_value));
  IREE_ASSERT_OK(iree_hal_streaming_event_query(event, &event_status));
  EXPECT_EQ(1, event_status);
  IREE_ASSERT_OK(SignalGate(second_gate, gate_value));
  IREE_ASSERT_OK(iree_hal_streaming_context_synchronize(context_));
  IREE_ASSERT_OK(iree_hal_streaming_event_query(event, &event_status));
  EXPECT_EQ(0, event_status);

  // Repeat with the opposite release order. The two rounds prove that the
  // fan-in waits for each stream rather than accidentally naming only one.
  gate_value = 2;
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_wait_semaphores(first_stream, first_wait));
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_wait_semaphores(second_stream, second_wait));
  IREE_ASSERT_OK(iree_hal_streaming_context_record_event(context_, event));
  IREE_ASSERT_OK(SignalGate(second_gate, gate_value));
  IREE_ASSERT_OK(iree_hal_streaming_event_query(event, &event_status));
  EXPECT_EQ(1, event_status);
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
      /*.count=*/1,
      /*.semaphores=*/&gate,
      /*.payload_values=*/&gate_value,
  };
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_wait_semaphores(source_stream, gate_wait));

  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(iree_hal_streaming_event_record(event, source_stream));
  IREE_ASSERT_OK(iree_hal_streaming_context_wait_event(context_, event));

  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &later_stream));
  ASSERT_NE(later_stream->queue, source_stream->queue);
  ASSERT_NE(later_stream->queue, current_stream->queue);

  int current_status = 0;
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_query(current_stream, &current_status));
  EXPECT_EQ(1, current_status);
  int later_status = 0;
  IREE_ASSERT_OK(iree_hal_streaming_stream_query(later_stream, &later_status));
  EXPECT_EQ(1, later_status);

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

  iree_hal_streaming_context_flags_t context_flags = {};
  context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
  IREE_ASSERT_OK(iree_hal_streaming_context_create(
      &device_entry_, context_flags, iree_allocator_system(), &target_context));
  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &source_stream));
  IREE_ASSERT_OK(CreateNonBlockingStream(target_context, &current_stream));

  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(CreateGate(/*release_value=*/1, &gate));
  uint64_t gate_value = 1;
  const iree_hal_semaphore_list_t gate_wait = {
      /*.count=*/1,
      /*.semaphores=*/&gate,
      /*.payload_values=*/&gate_value,
  };
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_wait_semaphores(source_stream, gate_wait));

  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(iree_hal_streaming_event_record(event, source_stream));
  IREE_ASSERT_OK(iree_hal_streaming_context_wait_event(target_context, event));

  IREE_ASSERT_OK(CreateNonBlockingStream(target_context, &later_stream));
  int current_status = 0;
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_query(current_stream, &current_status));
  EXPECT_EQ(1, current_status);
  int later_status = 0;
  IREE_ASSERT_OK(iree_hal_streaming_stream_query(later_stream, &later_status));
  EXPECT_EQ(1, later_status);

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
  FailingSnapshotAllocator allocator;
  const iree_allocator_t original_allocator = context_->host_allocator;
  ScopeExit cleanup([&] {
    allocator.fail_allocations.store(false, std::memory_order_release);
    context_->host_allocator = original_allocator;
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
      /*.count=*/1,
      /*.semaphores=*/&first_gate,
      /*.payload_values=*/&first_gate_value,
  };
  uint64_t second_gate_value = 1;
  const iree_hal_semaphore_list_t second_wait = {
      /*.count=*/1,
      /*.semaphores=*/&second_gate,
      /*.payload_values=*/&second_gate_value,
  };
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_wait_semaphores(first_stream, first_wait));
  IREE_ASSERT_OK(
      iree_hal_streaming_stream_wait_semaphores(second_stream, second_wait));

  context_->host_allocator = allocator.AsAllocator();
  allocator.fail_allocations.store(true, std::memory_order_release);
  std::atomic<bool> synchronize_returned = false;
  std::atomic<iree_status_code_t> synchronize_status_code = IREE_STATUS_UNKNOWN;
  synchronize_thread = std::thread([&] {
    iree_status_t status = iree_hal_streaming_context_synchronize(context_);
    synchronize_status_code.store(iree_status_code(status),
                                  std::memory_order_release);
    iree_status_ignore(status);
    synchronize_returned.store(true, std::memory_order_release);
  });

  bool observed_snapshot_failure = false;
  for (int i = 0; i < 1000000; ++i) {
    if (allocator.allocation_attempt_count.load(std::memory_order_acquire) >
        0) {
      observed_snapshot_failure = true;
      break;
    }
    std::this_thread::yield();
  }
  if (!observed_snapshot_failure) {
    FAIL() << "context synchronization did not attempt a stream snapshot";
    return;
  }
  EXPECT_FALSE(synchronize_returned.load(std::memory_order_acquire));

  IREE_EXPECT_OK(SignalGate(first_gate, first_gate_value));
  for (int i = 0;
       i < 100000 && !synchronize_returned.load(std::memory_order_acquire);
       ++i) {
    std::this_thread::yield();
  }
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
      /*.count=*/1,
      /*.semaphores=*/&prior_gate,
      /*.payload_values=*/&prior_value,
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

// The records a graph launch enqueues run through the same helper as a direct
// record, so a launch recording eleven events on this device acquires no slot
// either.
TEST_F(CpuStreamingContextTest, AGraphLaunchOnAnUntimedDeviceGoesUntimed) {
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
        context_, IREE_HAL_STREAMING_EVENT_FLAG_NONE, iree_allocator_system(),
        &events[i]));
    iree_hal_streaming_graph_node_t* node = nullptr;
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_event_node(
        graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
        IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD, events[i], &node));
  }

  iree_hal_streaming_graph_exec_t* exec = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(exec, stream));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream));

  EXPECT_EQ(nullptr, context_->timestamp_pool.slabs)
      << "a launch on a device advertising no domain acquired tick slots";
  for (iree_host_size_t i = 0; i < kEventCount; ++i) {
    int event_status = -1;
    IREE_ASSERT_OK(iree_hal_streaming_event_query(events[i], &event_status));
    EXPECT_EQ(0, event_status) << "event " << i;
  }

  for (iree_host_size_t i = kEventCount; i > 0; --i) {
    iree_hal_streaming_event_release(events[i - 1]);
  }
  IREE_EXPECT_OK(iree_hal_streaming_graph_exec_destroy_handle(exec));
  iree_hal_streaming_graph_release(graph);
  iree_hal_streaming_stream_release(stream);
}

}  // namespace
