// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "common/internal.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

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

iree_hal_queue_family_spec_t QueueFamily(
    iree_hal_physical_device_affinity_t physical_device_affinity,
    uint32_t timestamp_valid_bits, uint64_t timestamp_frequency_hz) {
  iree_hal_queue_family_spec_t family = {};
  family.name = iree_make_cstring_view("test");
  family.queue_count = 1;
  family.priority_count = 1;
  family.timestamp_valid_bits = timestamp_valid_bits;
  family.timestamp_frequency_hz = timestamp_frequency_hz;
  family.physical_device_affinity = physical_device_affinity;
  family.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
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
// The gate over a real device
//===----------------------------------------------------------------------===//

// Runs the gate against the host CPU device, which is the one device reachable
// from here that a context can be built on, and the only device in this tree
// the gate rejects.
class CpuContextTimestampDomainTest : public ::testing::Test {
 protected:
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
    iree_hal_streaming_context_release(context_);
    iree_arena_block_pool_deinitialize(&device_entry_.block_pool);
    iree_slim_mutex_deinitialize(&device_entry_.graph_memory_mutex);
    iree_slim_mutex_deinitialize(&device_entry_.primary_context_mutex);
    IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
  }

  // Registry entry backing |context_|; outlives every context created from it.
  iree_hal_streaming_device_t device_entry_ = {};
  // Context the gate answered for when it was created.
  iree_hal_streaming_context_t* context_ = nullptr;
};

// The CPU device publishes no timing facet and its single queue family reports
// no frequency and no width, so a context created on it reads a zeroed domain.
// That is all this pins: a context whose domain was never assigned reads the
// same zeros, because the context allocation is zeroed, so it cannot see the
// gate run. Which condition turns which device away is pinned by the fabricated
// specs above; that the gate is wired into context creation at all is pinned on
// a device that does advertise a domain, by the device-timing tests in
// libhrx/cts/tests/hip/event_test.cpp, whose records capture no tick and whose
// elapsed time reports unsupported without it.
TEST_F(CpuContextTimestampDomainTest, LeavesTheDomainZeroedOnTheCpuDevice) {
  EXPECT_EQ(0ull, context_->timestamp_domain.frequency_hz);
  EXPECT_EQ(0u, context_->timestamp_domain.valid_bits);
}

// A record on a context whose domain is zeroed captures no tick, so the pool it
// would have drawn a slot from stays empty and the interval between two such
// records names no clock to be measured on. This is the only reachable path to
// that outcome anywhere in this tree: every device libhrx builds a context on
// advertises a domain.
TEST_F(CpuContextTimestampDomainTest, DirectRecordsOnAnUntimedDeviceGoUntimed) {
  iree_hal_streaming_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_stream_create(
      context_, IREE_HAL_STREAMING_STREAM_FLAG_NONE, /*priority=*/0,
      iree_allocator_system(), &stream));
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

// The records a graph launch enqueues run through the same helper as a direct
// record, so a launch recording eleven events on this device acquires no slot
// either.
TEST_F(CpuContextTimestampDomainTest,
       AGraphLaunchOnAnUntimedDeviceGoesUntimed) {
  static constexpr iree_host_size_t kEventCount = 11;

  iree_hal_streaming_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_stream_create(
      context_, IREE_HAL_STREAMING_STREAM_FLAG_NONE, /*priority=*/0,
      iree_allocator_system(), &stream));
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
  iree_hal_streaming_graph_exec_release(exec);
  iree_hal_streaming_graph_release(graph);
  iree_hal_streaming_stream_release(stream);
}

}  // namespace
