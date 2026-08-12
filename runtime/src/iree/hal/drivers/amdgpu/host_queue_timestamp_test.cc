// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// End-to-end tests for iree_hal_device_queue_timestamp on the amdgpu driver.
// Every case is parameterized over both production capture strategies, and
// cases independent of the strategy skip the second parameterization.
// GPU required.

#include "iree/hal/drivers/amdgpu/host_queue_timestamp.h"

#include <cstdint>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device_capabilities.h"
#include "iree/hal/drivers/amdgpu/transient_buffer.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/pm4_capabilities.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;

// Which device timestamp capture path a case runs the op through: the PM4
// timestamp packet the GPU ISA selects where one exists, or the builtin capture
// dispatch forced onto every queue.
enum class CaptureStrategy {
  kPm4Packet,
  kCaptureDispatch,
};

// Minimal logical-device harness (mirrors host_queue_submission_test.cc).
class TestLogicalDevice {
 public:
  ~TestLogicalDevice() {
    iree_hal_device_release(base_device_);
    iree_hal_device_group_release(device_group_);
  }

  iree_status_t Initialize(
      const iree_hal_amdgpu_logical_device_options_t* options,
      const iree_hal_amdgpu_libhsa_t* libhsa,
      const iree_hal_amdgpu_topology_t* topology,
      iree_allocator_t host_allocator, CaptureStrategy strategy) {
    IREE_RETURN_IF_ERROR(create_context_.Initialize(host_allocator));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
        IREE_SV("amdgpu"), options, libhsa, topology, create_context_.params(),
        host_allocator, &base_device_));
    // Host queues are created when the device is assigned a frontier by the
    // group, so the strategy override has to follow that.
    IREE_RETURN_IF_ERROR(iree_hal_device_group_create_from_device(
        base_device_, create_context_.frontier_tracker(), host_allocator,
        &device_group_));
    if (strategy == CaptureStrategy::kCaptureDispatch) {
      ForceCaptureDispatchStrategy();
    }
    return iree_ok_status();
  }

  iree_hal_device_t* base_device() const { return base_device_; }

  iree_hal_amdgpu_logical_device_t* logical_device() const {
    return (iree_hal_amdgpu_logical_device_t*)base_device_;
  }

  iree_hal_amdgpu_physical_device_t* first_physical_device() const {
    iree_hal_amdgpu_logical_device_t* device = this->logical_device();
    if (device->physical_device_count == 0) return NULL;
    return device->physical_devices[0];
  }

  iree_hal_amdgpu_host_queue_t* first_host_queue() const {
    iree_hal_amdgpu_physical_device_t* physical_device =
        this->first_physical_device();
    if (!physical_device || iree_hal_amdgpu_physical_device_host_queue_count(
                                physical_device) == 0) {
      return NULL;
    }
    return &physical_device->host_queues[0];
  }

 private:
  // Clears the PM4 timestamp strategy on every host queue so
  // iree_hal_amdgpu_host_queue_submit_timestamp takes the capture dispatch
  // branch. Submission-path readers hold submission_mutex, which this write
  // also takes, but iree_hal_amdgpu_host_queue_enable_profile_counters does
  // not: running before the device escapes Initialize is what keeps that safe.
  void ForceCaptureDispatchStrategy() {
    iree_hal_amdgpu_logical_device_t* device = this->logical_device();
    for (iree_host_size_t device_index = 0;
         device_index < device->physical_device_count; ++device_index) {
      iree_hal_amdgpu_physical_device_t* physical_device =
          device->physical_devices[device_index];
      const iree_host_size_t host_queue_count =
          iree_hal_amdgpu_physical_device_host_queue_count(physical_device);
      for (iree_host_size_t queue_index = 0; queue_index < host_queue_count;
           ++queue_index) {
        iree_hal_amdgpu_host_queue_t* queue =
            &physical_device->host_queues[queue_index];
        iree_slim_mutex_lock(&queue->locks.submission_mutex);
        queue->pm4_timestamp_strategy =
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_NONE;
        iree_slim_mutex_unlock(&queue->locks.submission_mutex);
      }
    }
  }

  iree::hal::cts::DeviceCreateContext create_context_;
  iree_hal_device_t* base_device_ = NULL;
  iree_hal_device_group_t* device_group_ = NULL;
};

class HostQueueTimestampTest
    : public ::testing::TestWithParam<CaptureStrategy> {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  static iree_status_t InitializeDeviceWithStrategy(
      TestLogicalDevice* out_device, CaptureStrategy strategy) {
    iree_hal_amdgpu_logical_device_options_t options;
    iree_hal_amdgpu_logical_device_options_initialize(&options);
    return out_device->Initialize(&options, &libhsa_, &topology_,
                                  host_allocator_, strategy);
  }

  iree_status_t InitializeDevice(TestLogicalDevice* out_device) const {
    return InitializeDeviceWithStrategy(out_device, GetParam());
  }

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t HostQueueTimestampTest::host_allocator_;
iree_hal_amdgpu_libhsa_t HostQueueTimestampTest::libhsa_;
iree_hal_amdgpu_topology_t HostQueueTimestampTest::topology_;

INSTANTIATE_TEST_SUITE_P(
    CaptureStrategies, HostQueueTimestampTest,
    ::testing::Values(CaptureStrategy::kPm4Packet,
                      CaptureStrategy::kCaptureDispatch),
    [](const ::testing::TestParamInfo<CaptureStrategy>& info) {
      return info.param == CaptureStrategy::kPm4Packet ? "Pm4Packet"
                                                       : "CaptureDispatch";
    });

// True when the GPU agent's ISA has a PM4 timestamp packet at all. Derived from
// the reported ISA rather than from the queue strategy under test, so a queue
// that fails to select PM4 on an ISA that has one fails instead of skipping.
static bool IsaProvidesPm4Timestamps(
    const iree_hal_amdgpu_physical_device_t* physical_device) {
  return iree_hal_amdgpu_select_pm4_timestamp_strategy(
             physical_device->agent_target->primary_isa.identity.version) !=
         IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_NONE;
}

// Running total of kernarg blocks |queue| has handed out. This is how a case
// observes which branch a capture took: the capture dispatch reserves one block
// for the target pointer while the PM4 packet encodes that pointer into the
// queue's IB slot and reserves none. The committed AQL packet cannot serve: the
// command processor overwrites its header once it consumes the packet.
static uint64_t ConsumedKernargBlocks(iree_hal_amdgpu_host_queue_t* queue) {
  return (uint64_t)iree_atomic_load(&queue->kernarg_ring.write_position,
                                    iree_memory_order_acquire);
}

static bool HostQueueHasPendingOps(iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  const bool has_pending_ops = queue->pending_head != NULL;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  return has_pending_ops;
}

static iree_status_t AllocateTimestampBufferSized(
    iree_hal_device_t* device, iree_device_size_t size,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  // DEVICE_LOCAL would need a fine-grained device pool only large-BAR and APU
  // GPUs expose; the host pool is device-visible, so the device still writes
  // here and the host maps it to read back.
  params.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, size, out_buffer);
}

static iree_status_t AllocateTimestampBuffer(iree_hal_device_t* device,
                                             iree_hal_buffer_t** out_buffer) {
  return AllocateTimestampBufferSized(device, sizeof(uint64_t), out_buffer);
}

// Written over a capture target before the capture runs so an untouched slot is
// distinguishable from a device tick. No reachable tick value collides with it.
static constexpr uint64_t kUnwrittenSentinel = 0xDEADBEEFDEADBEEFull;

static iree_status_t FillWithSentinel(iree_hal_buffer_t* buffer) {
  const iree_device_size_t length = iree_hal_buffer_byte_length(buffer);
  for (iree_device_size_t offset = 0;
       offset + sizeof(kUnwrittenSentinel) <= length;
       offset += sizeof(kUnwrittenSentinel)) {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_write(
        buffer, offset, &kUnwrittenSentinel, sizeof(kUnwrittenSentinel)));
  }
  return iree_ok_status();
}

// The returned list borrows the pointer slot |semaphore| owns, so both must
// outlive it.
static iree_hal_semaphore_list_t TimelinePoint(
    Ref<iree_hal_semaphore_t>& semaphore, uint64_t* value) {
  iree_hal_semaphore_list_t list = {
      /*count=*/1,
      /*semaphores=*/semaphore.out(),
      /*payload_values=*/value,
  };
  return list;
}

// Sleeps until |deadline_ns|. iree_wait_until can return early when the sleep
// is aborted, so resume until the deadline has actually passed.
static void SleepUntil(iree_time_t deadline_ns) {
  while (!iree_wait_until(deadline_ns) && iree_time_now() < deadline_ns) {
  }
}

// Reports the tick and the host-time bracket it was written in:
// *out_before_ns <= write <= *out_after_ns.
static iree_status_t CaptureBracketedTick(
    iree_hal_device_t* device, Ref<iree_hal_semaphore_t>& timeline,
    uint64_t value, iree_hal_buffer_t* target, iree_time_t* out_before_ns,
    iree_time_t* out_after_ns, uint64_t* out_tick) {
  *out_before_ns = iree_time_now();
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_timestamp(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      TimelinePoint(timeline, &value), target, /*target_offset=*/0,
      IREE_HAL_TIMESTAMP_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait(
      timeline, value, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  *out_after_ns = iree_time_now();
  return iree_hal_buffer_map_read(target, 0, out_tick, sizeof(*out_tick));
}

// Captures one tick on |test_device| and reports whether the submission ran
// through the PM4 timestamp packet. The test thread is the only submitter, so
// every kernarg block consumed across the submit belongs to this capture.
static iree_status_t CaptureOneTick(TestLogicalDevice* test_device,
                                    uint64_t* out_tick,
                                    bool* out_captured_with_pm4) {
  *out_tick = 0;
  *out_captured_with_pm4 = false;
  iree_hal_device_t* device = test_device->base_device();
  iree_hal_amdgpu_host_queue_t* queue = test_device->first_host_queue();
  if (!queue) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device has no host queue to capture on");
  }

  Ref<iree_hal_buffer_t> target;
  IREE_RETURN_IF_ERROR(AllocateTimestampBuffer(device, target.out()));
  Ref<iree_hal_semaphore_t> timeline;
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, timeline.out()));

  const uint64_t kernarg_blocks_before = ConsumedKernargBlocks(queue);
  uint64_t value = 1;
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_timestamp(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      TimelinePoint(timeline, &value), target, /*target_offset=*/0,
      IREE_HAL_TIMESTAMP_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait(
      timeline, value, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  *out_captured_with_pm4 =
      ConsumedKernargBlocks(queue) == kernarg_blocks_before;
  return iree_hal_buffer_map_read(target, 0, out_tick, sizeof(*out_tick));
}

// Host-side gap between the two captures; larger values tighten the brackets.
static constexpr iree_duration_t kTickRateMeasurementGapNs =
    200000000;  // 200ms

// Measures a bracket [*out_minimum_hz, *out_maximum_hz] around |device|'s tick
// rate from two captures kTickRateMeasurementGapNs apart. Each tick lands
// between its capture's enqueue and the return of its wait, so
//   inner = enqueue(second) - waited(first)  <= true interval
//   outer = waited(second)  - enqueue(first) >= true interval
// and the true rate lies in [delta/outer, delta/inner]. Latency only pushes
// those edges apart, so load can only widen the bracket.
static iree_status_t MeasureTickRateBracketHz(iree_hal_device_t* device,
                                              double* out_minimum_hz,
                                              double* out_maximum_hz) {
  *out_minimum_hz = 0.0;
  *out_maximum_hz = 0.0;

  // The tick subtraction below is unmasked, which is the wrap-correct delta
  // only because this domain is the full counter width.
  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(iree_hal_device_spec(device));
  if (!timing || timing->timestamp_valid_bits != 64) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "tick rate measurement requires a full-width "
                            "64-bit device timestamp domain");
  }

  // Allocated up front so no allocation lands inside either bracket.
  Ref<iree_hal_buffer_t> first_target;
  Ref<iree_hal_buffer_t> second_target;
  IREE_RETURN_IF_ERROR(AllocateTimestampBuffer(device, first_target.out()));
  IREE_RETURN_IF_ERROR(AllocateTimestampBuffer(device, second_target.out()));
  Ref<iree_hal_semaphore_t> timeline;
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, timeline.out()));

  iree_time_t first_before_ns = 0;
  iree_time_t first_after_ns = 0;
  uint64_t first_tick = 0;
  IREE_RETURN_IF_ERROR(CaptureBracketedTick(device, timeline, /*value=*/1,
                                            first_target, &first_before_ns,
                                            &first_after_ns, &first_tick));

  SleepUntil(iree_time_now() + kTickRateMeasurementGapNs);

  iree_time_t second_before_ns = 0;
  iree_time_t second_after_ns = 0;
  uint64_t second_tick = 0;
  IREE_RETURN_IF_ERROR(CaptureBracketedTick(device, timeline, /*value=*/2,
                                            second_target, &second_before_ns,
                                            &second_after_ns, &second_tick));

  if (second_tick <= first_tick) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "device clock did not advance across a %" PRIi64
                            "ns host interval",
                            (int64_t)kTickRateMeasurementGapNs);
  }
  const iree_duration_t inner_ns = second_before_ns - first_after_ns;
  const iree_duration_t outer_ns = second_after_ns - first_before_ns;
  // The sleep deadline is taken after first_after_ns, so a gap this short means
  // the host clock ran backwards.
  if (inner_ns < kTickRateMeasurementGapNs / 2) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "host gap between captures did not elapse; the "
                            "tick-rate bracket cannot be evaluated");
  }

  const uint64_t tick_delta = second_tick - first_tick;
  *out_minimum_hz = (double)tick_delta * 1.0e9 / (double)outer_ns;
  *out_maximum_hz = (double)tick_delta * 1.0e9 / (double)inner_ns;
  return iree_ok_status();
}

// The parameterization must move the strategy boundary and the submission must
// honor it, otherwise every case below silently tests one path twice.
TEST_P(HostQueueTimestampTest, CapturesThroughTheParameterizedStrategy) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(InitializeDevice(&test_device));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);
  const iree_hal_amdgpu_physical_device_t* physical_device =
      test_device.first_physical_device();
  ASSERT_NE(physical_device, nullptr);

  const bool expect_pm4 = GetParam() == CaptureStrategy::kPm4Packet;
  if (expect_pm4 && !IsaProvidesPm4Timestamps(physical_device)) {
    GTEST_SKIP() << "this GPU ISA has no PM4 timestamp packet, so both "
                    "parameterizations capture through the dispatch";
  }

  EXPECT_EQ(iree_hal_amdgpu_host_queue_can_use_pm4_timestamp(queue),
            expect_pm4);

  uint64_t tick = 0;
  bool captured_with_pm4 = false;
  IREE_ASSERT_OK(CaptureOneTick(&test_device, &tick, &captured_with_pm4));
  EXPECT_NE(tick, 0u);
  EXPECT_EQ(captured_with_pm4, expect_pm4);
}

TEST_P(HostQueueTimestampTest, DeviceSpecAdvertisesTimestamps) {
  if (GetParam() != CaptureStrategy::kPm4Packet) {
    GTEST_SKIP() << "the device spec is built at device creation from the GPU "
                    "agents' clock facts and reads no queue capture strategy, "
                    "so this case only needs to run once";
  }

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(InitializeDevice(&test_device));

  const iree_hal_device_spec_t* spec =
      iree_hal_device_spec(test_device.base_device());
  ASSERT_NE(spec, nullptr);
  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(spec);
  ASSERT_NE(timing, nullptr);
  EXPECT_TRUE(iree_all_bits_set(
      timing->flags, IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS));
  EXPECT_GT(timing->timestamp_frequency_hz, 0u);
  EXPECT_EQ(timing->timestamp_valid_bits, 64u);
}

// The system-scope HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY is a different domain
// whose rate need not match the agent's own wallclock rate.
TEST_P(HostQueueTimestampTest, DeviceSpecFrequencyMatchesAgentTickRate) {
  if (GetParam() != CaptureStrategy::kPm4Packet) {
    GTEST_SKIP() << "the device spec is built at device creation from the GPU "
                    "agents' clock facts and reads no queue capture strategy, "
                    "so this case only needs to run once";
  }

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(InitializeDevice(&test_device));

  const iree_hal_device_spec_t* spec =
      iree_hal_device_spec(test_device.base_device());
  ASSERT_NE(spec, nullptr);
  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(spec);
  ASSERT_NE(timing, nullptr);
  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(spec);
  ASSERT_NE(queues, nullptr);
  ASSERT_EQ(queues->family_count, topology_.gpu_agent_count);

  uint64_t first_agent_frequency_hz = 0;
  for (iree_host_size_t i = 0; i < topology_.gpu_agent_count; ++i) {
    uint64_t agent_frequency_hz = 0;
    IREE_ASSERT_OK(iree_hsa_agent_get_info(
        IREE_LIBHSA(&libhsa_), topology_.gpu_agents[i],
        (hsa_agent_info_t)HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY,
        &agent_frequency_hz));
    ASSERT_GT(agent_frequency_hz, 0u);
    EXPECT_EQ(queues->families[i].timestamp_frequency_hz, agent_frequency_hz);
    EXPECT_EQ(queues->families[i].timestamp_valid_bits, 64u);
    if (i == 0) first_agent_frequency_hz = agent_frequency_hz;
  }

  // The default topology only groups agents of the same model, so every agent
  // reports the same rate and agent 0's is the device-scope rate. They remain
  // separate timestamp domains; device_spec_builder_test.cc covers rejection of
  // physical devices whose rates disagree.
  EXPECT_EQ(timing->timestamp_frequency_hz, first_agent_frequency_hz);
}

// Largest factor by which the advertised frequency may fall outside the
// observed tick-rate bracket. It absorbs the drift between the agent's nominal
// advertised rate and the rate its counter runs at against the host monotonic
// clock, measured at 0.15% on gfx942, while still rejecting a rate wrong by a
// factor of two. Every latency term only widens the bracket, so this bound
// binds hardest on an idle machine.
static constexpr double kMaximumTickRateRatio = 1.05;

// A nonzero advertised frequency passes for any counter; only a measured rate
// catches a frequency sourced from the wrong domain.
TEST_P(HostQueueTimestampTest, AdvertisedFrequencyMatchesObservedTickRate) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(InitializeDevice(&test_device));
  iree_hal_device_t* device = test_device.base_device();

  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(iree_hal_device_spec(device));
  ASSERT_NE(timing, nullptr);
  const uint64_t advertised_hz = timing->timestamp_frequency_hz;
  ASSERT_GT(advertised_hz, 0u);

  double minimum_observed_hz = 0.0;
  double maximum_observed_hz = 0.0;
  IREE_ASSERT_OK(MeasureTickRateBracketHz(device, &minimum_observed_hz,
                                          &maximum_observed_hz));

  EXPECT_GE((double)advertised_hz, minimum_observed_hz / kMaximumTickRateRatio)
      << "advertised " << advertised_hz << " hz is below the observed rate of ["
      << minimum_observed_hz << ", " << maximum_observed_hz << "] hz";
  EXPECT_LE((double)advertised_hz, maximum_observed_hz * kMaximumTickRateRatio)
      << "advertised " << advertised_hz << " hz is above the observed rate of ["
      << minimum_observed_hz << ", " << maximum_observed_hz << "] hz";
}

TEST_P(HostQueueTimestampTest, WritesMonotonicDeviceTicks) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(InitializeDevice(&test_device));
  iree_hal_device_t* device = test_device.base_device();

  Ref<iree_hal_buffer_t> tick0;
  Ref<iree_hal_buffer_t> tick1;
  IREE_ASSERT_OK(AllocateTimestampBuffer(device, tick0.out()));
  IREE_ASSERT_OK(AllocateTimestampBuffer(device, tick1.out()));
  IREE_ASSERT_OK(FillWithSentinel(tick0));
  IREE_ASSERT_OK(FillWithSentinel(tick1));

  Ref<iree_hal_semaphore_t> timeline;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           timeline.out()));

  uint64_t v1 = 1;
  uint64_t v2 = 2;
  IREE_ASSERT_OK(iree_hal_device_queue_timestamp(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      TimelinePoint(timeline, &v1), tick0, /*target_offset=*/0,
      IREE_HAL_TIMESTAMP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_device_queue_timestamp(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, TimelinePoint(timeline, &v1),
      TimelinePoint(timeline, &v2), tick1, /*target_offset=*/0,
      IREE_HAL_TIMESTAMP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      timeline, 2ull, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint64_t start_tick = 0;
  uint64_t stop_tick = 0;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(tick0, 0, &start_tick, sizeof(start_tick)));
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(tick1, 0, &stop_tick, sizeof(stop_tick)));

  EXPECT_NE(start_tick, kUnwrittenSentinel);
  EXPECT_NE(stop_tick, kUnwrittenSentinel);
  EXPECT_NE(start_tick, 0u);
  EXPECT_NE(stop_tick, 0u);
  EXPECT_GE(stop_tick, start_tick);
}

TEST_P(HostQueueTimestampTest, TickAdvancesAcrossWork) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(InitializeDevice(&test_device));
  iree_hal_device_t* device = test_device.base_device();

  Ref<iree_hal_buffer_t> tick0;
  Ref<iree_hal_buffer_t> tick1;
  IREE_ASSERT_OK(AllocateTimestampBuffer(device, tick0.out()));
  IREE_ASSERT_OK(AllocateTimestampBuffer(device, tick1.out()));

  // Scratch buffer filled between the two captures to create measurable work.
  const iree_device_size_t kScratchBytes = 64 * 1024 * 1024;
  iree_hal_buffer_params_t scratch_params = {0};
  scratch_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  scratch_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  Ref<iree_hal_buffer_t> scratch;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), scratch_params, kScratchBytes,
      scratch.out()));

  Ref<iree_hal_semaphore_t> timeline;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           timeline.out()));

  uint64_t v1 = 1;
  uint64_t v2 = 2;
  uint64_t v3 = 3;
  const uint32_t pattern = 0xABCDu;
  IREE_ASSERT_OK(iree_hal_device_queue_timestamp(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      TimelinePoint(timeline, &v1), tick0, /*target_offset=*/0,
      IREE_HAL_TIMESTAMP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, TimelinePoint(timeline, &v1),
      TimelinePoint(timeline, &v2), scratch, /*target_offset=*/0, kScratchBytes,
      &pattern, /*pattern_length=*/sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_device_queue_timestamp(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, TimelinePoint(timeline, &v2),
      TimelinePoint(timeline, &v3), tick1, /*target_offset=*/0,
      IREE_HAL_TIMESTAMP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      timeline, 3ull, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint64_t start_tick = 0;
  uint64_t stop_tick = 0;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(tick0, 0, &start_tick, sizeof(start_tick)));
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(tick1, 0, &stop_tick, sizeof(stop_tick)));

  EXPECT_NE(start_tick, 0u);
  EXPECT_NE(stop_tick, 0u);
  EXPECT_GT(stop_tick, start_tick);
}

// Offset 4 plus 8 bytes fits the 16-byte buffer, so this exercises the
// alignment check in prepare_timestamp_target rather than the range check.
TEST_P(HostQueueTimestampTest, RejectsMisalignedOffset) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(InitializeDevice(&test_device));
  iree_hal_device_t* device = test_device.base_device();

  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(
      AllocateTimestampBufferSized(device, /*size=*/16, target.out()));

  Ref<iree_hal_semaphore_t> timeline;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           timeline.out()));

  uint64_t v1 = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_queue_timestamp(
          device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          TimelinePoint(timeline, &v1), target, /*target_offset=*/4,
          IREE_HAL_TIMESTAMP_FLAG_NONE));
}

// iree_hal_buffer_validate_range reports OUT_OF_RANGE rather than
// INVALID_ARGUMENT, and the seam forwards the code unchanged.
TEST_P(HostQueueTimestampTest, RejectsOutOfRangeOffset) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(InitializeDevice(&test_device));
  iree_hal_device_t* device = test_device.base_device();

  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(AllocateTimestampBuffer(device, target.out()));  // 8 bytes.

  Ref<iree_hal_semaphore_t> timeline;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           timeline.out()));

  uint64_t v1 = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_device_queue_timestamp(
          device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          TimelinePoint(timeline, &v1), target, /*target_offset=*/8,
          IREE_HAL_TIMESTAMP_FLAG_NONE));
}

// iree_hal_buffer_validate_usage reports PERMISSION_DENIED rather than
// INVALID_ARGUMENT, and the seam forwards the code unchanged.
TEST_P(HostQueueTimestampTest, RejectsBufferWithoutTransferTarget) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(InitializeDevice(&test_device));
  iree_hal_device_t* device = test_device.base_device();

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_MAPPING;
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), params, sizeof(uint64_t),
      target.out()));
  // Pin the premise: a widened usage mask leaves this case covering nothing.
  ASSERT_FALSE(iree_all_bits_set(iree_hal_buffer_allowed_usage(target),
                                 IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));

  Ref<iree_hal_semaphore_t> timeline;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           timeline.out()));

  uint64_t v1 = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_device_queue_timestamp(
          device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          TimelinePoint(timeline, &v1), target, /*target_offset=*/0,
          IREE_HAL_TIMESTAMP_FLAG_NONE));
}

// An empty wait list keeps the immediate submit path, which validates flags
// inline rather than at issue time.
TEST_P(HostQueueTimestampTest, RejectsUnsupportedFlags) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(InitializeDevice(&test_device));
  iree_hal_device_t* device = test_device.base_device();

  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(AllocateTimestampBuffer(device, target.out()));

  Ref<iree_hal_semaphore_t> timeline;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           timeline.out()));

  uint64_t v1 = 1;
  const iree_hal_timestamp_flags_t bad_flags =
      (iree_hal_timestamp_flags_t)(1ull << 0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_queue_timestamp(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                      iree_hal_semaphore_list_empty(),
                                      TimelinePoint(timeline, &v1), target,
                                      /*target_offset=*/0, bad_flags));
}

// A capture that ignored the target offset would leave a plausible tick at
// offset 0, which is why the slot there is checked too.
TEST_P(HostQueueTimestampTest, WritesAtNonZeroOffset) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(InitializeDevice(&test_device));
  iree_hal_device_t* device = test_device.base_device();

  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(
      AllocateTimestampBufferSized(device, /*size=*/16, target.out()));
  IREE_ASSERT_OK(FillWithSentinel(target));

  Ref<iree_hal_semaphore_t> timeline;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           timeline.out()));

  uint64_t v1 = 1;
  IREE_ASSERT_OK(iree_hal_device_queue_timestamp(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      TimelinePoint(timeline, &v1), target, /*target_offset=*/8,
      IREE_HAL_TIMESTAMP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      timeline, 1ull, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint64_t tick = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(target, /*source_offset=*/8, &tick,
                                          sizeof(tick)));
  EXPECT_NE(tick, kUnwrittenSentinel);
  EXPECT_NE(tick, 0u);

  uint64_t untouched = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(target, /*source_offset=*/0,
                                          &untouched, sizeof(untouched)));
  EXPECT_EQ(untouched, kUnwrittenSentinel);
}

// The capture waits on a value with no submitted signal, so resolve_waits
// cannot satisfy it. The wait semaphore is released purely from the host so the
// queue cannot elide the wait via same-queue producer-axis tracking.
TEST_P(HostQueueTimestampTest, DefersUntilWaitSignaled) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(InitializeDevice(&test_device));
  iree_hal_device_t* device = test_device.base_device();
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(AllocateTimestampBuffer(device, target.out()));
  IREE_ASSERT_OK(FillWithSentinel(target));

  Ref<iree_hal_semaphore_t> wait_sem;
  Ref<iree_hal_semaphore_t> signal_sem;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           wait_sem.out()));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           signal_sem.out()));

  uint64_t wait_value = 1;
  uint64_t signal_value = 1;
  IREE_ASSERT_OK(iree_hal_device_queue_timestamp(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, TimelinePoint(wait_sem, &wait_value),
      TimelinePoint(signal_sem, &signal_value), target, /*target_offset=*/0,
      IREE_HAL_TIMESTAMP_FLAG_NONE));

  // Pin the premise: an op that issued inline makes this a duplicate of
  // WritesMonotonicDeviceTicks.
  ASSERT_TRUE(HostQueueHasPendingOps(queue));

  IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_sem, wait_value,
                                           /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal_sem, signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  uint64_t tick = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(target, 0, &tick, sizeof(tick)));
  EXPECT_NE(tick, kUnwrittenSentinel);
  EXPECT_NE(tick, 0u);
}

// The PM4 strategy copies the command processor's GPU clock counter while the
// builtin capture dispatch reads the shader-visible steady counter. On one
// agent both must share an epoch and a rate or their ticks are not comparable.
// IREE_HAL_QUEUE_AFFINITY_ANY collapses to the first selected queue, so every
// capture below lands on one agent even on a box with several GPUs.
TEST_P(HostQueueTimestampTest, CaptureStrategiesAgreeOnTickOrderAndRate) {
  if (GetParam() != CaptureStrategy::kPm4Packet) {
    GTEST_SKIP() << "the cross-strategy comparison builds both devices itself "
                    "and only needs to run once";
  }

  TestLogicalDevice pm4_device;
  IREE_ASSERT_OK(
      InitializeDeviceWithStrategy(&pm4_device, CaptureStrategy::kPm4Packet));
  const iree_hal_amdgpu_physical_device_t* physical_device =
      pm4_device.first_physical_device();
  ASSERT_NE(physical_device, nullptr);
  if (!IsaProvidesPm4Timestamps(physical_device)) {
    GTEST_SKIP() << "this GPU ISA has no PM4 timestamp packet, so both devices "
                    "would run the capture dispatch and there is no second "
                    "clock source to compare against";
  }

  TestLogicalDevice dispatch_device;
  IREE_ASSERT_OK(InitializeDeviceWithStrategy(
      &dispatch_device, CaptureStrategy::kCaptureDispatch));

  uint64_t first_pm4_tick = 0;
  uint64_t dispatch_tick = 0;
  uint64_t second_pm4_tick = 0;
  bool pm4_device_used_pm4 = false;
  bool dispatch_device_used_pm4 = false;
  IREE_ASSERT_OK(
      CaptureOneTick(&pm4_device, &first_pm4_tick, &pm4_device_used_pm4));
  IREE_ASSERT_OK(CaptureOneTick(&dispatch_device, &dispatch_tick,
                                &dispatch_device_used_pm4));
  IREE_ASSERT_OK(
      CaptureOneTick(&pm4_device, &second_pm4_tick, &pm4_device_used_pm4));
  // Pin the premise: comparing two clock sources requires the two devices to
  // have captured through different strategies.
  ASSERT_TRUE(pm4_device_used_pm4);
  ASSERT_FALSE(dispatch_device_used_pm4);

  EXPECT_LE(first_pm4_tick, dispatch_tick)
      << "dispatch-captured tick precedes the PM4 tick captured before it";
  EXPECT_LE(dispatch_tick, second_pm4_tick)
      << "dispatch-captured tick follows the PM4 tick captured after it";

  double pm4_minimum_hz = 0.0;
  double pm4_maximum_hz = 0.0;
  double dispatch_minimum_hz = 0.0;
  double dispatch_maximum_hz = 0.0;
  IREE_ASSERT_OK(MeasureTickRateBracketHz(pm4_device.base_device(),
                                          &pm4_minimum_hz, &pm4_maximum_hz));
  IREE_ASSERT_OK(MeasureTickRateBracketHz(dispatch_device.base_device(),
                                          &dispatch_minimum_hz,
                                          &dispatch_maximum_hz));
  // Each bracket contains the true rate of the counter its strategy read, so
  // two strategies reading one counter must intersect and disjoint brackets
  // prove the rates differ. Capture cost only widens a bracket.
  EXPECT_LE(pm4_minimum_hz, dispatch_maximum_hz)
      << "capture strategies disagree on the device tick rate: PM4 ["
      << pm4_minimum_hz << ", " << pm4_maximum_hz << "] hz vs dispatch ["
      << dispatch_minimum_hz << ", " << dispatch_maximum_hz << "] hz";
  EXPECT_LE(dispatch_minimum_hz, pm4_maximum_hz)
      << "capture strategies disagree on the device tick rate: PM4 ["
      << pm4_minimum_hz << ", " << pm4_maximum_hz << "] hz vs dispatch ["
      << dispatch_minimum_hz << ", " << dispatch_maximum_hz << "] hz";
}

// A capture signaling only a device-local semaphore on one queue is the
// configuration in which the semaphore-derived release scope is AGENT, so the
// capture must take its scope from the target buffer instead. The emitted scope
// is not observable once the packet is consumed, so this pins only that the
// configuration is reachable; host_queue_policy_test.cc pins the rest.
TEST_P(HostQueueTimestampTest, CapturesThroughAnAgentScopeSignalList) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(InitializeDevice(&test_device));
  iree_hal_device_t* device = test_device.base_device();
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(AllocateTimestampBuffer(device, target.out()));
  IREE_ASSERT_OK(FillWithSentinel(target));

  // No HOST_INTERRUPT, EXPORTABLE, or EXPORTABLE_TIMEPOINTS, which force the
  // signal list to SYSTEM scope. The narrowed affinity is for the same reason:
  // a semaphore usable on any queue spans every physical device.
  Ref<iree_hal_semaphore_t> timeline;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, /*queue_affinity=*/1ull, 0ull,
      IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL, timeline.out()));

  uint64_t value = 1;
  const iree_hal_semaphore_list_t signal_list = TimelinePoint(timeline, &value);
  // Pin the premise: without an AGENT scope here the case covers nothing.
  ASSERT_EQ(
      iree_hal_amdgpu_host_queue_signal_list_release_scope(queue, signal_list),
      IREE_HSA_FENCE_SCOPE_AGENT);

  IREE_ASSERT_OK(iree_hal_device_queue_timestamp(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      signal_list, target, /*target_offset=*/0, IREE_HAL_TIMESTAMP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      timeline, value, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint64_t tick = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(target, 0, &tick, sizeof(tick)));
  EXPECT_NE(tick, kUnwrittenSentinel);
  EXPECT_NE(tick, 0u);
}

// Parameters for a transient capture target. MAPPING_SCOPED is what lets the
// tick be read back and the alignment is what the capture requires of its
// target.
static iree_hal_buffer_params_t TransientTimestampBufferParams() {
  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.min_alignment = sizeof(uint64_t);
  return params;
}

// A queue_alloca target reaches the capture as a transient wrapper rather than
// as an allocated buffer, so the capture has to resolve the staged backing
// behind it. A fresh allocation may hold anything, so the tick is pinned by two
// captures bracketing it instead of by a sentinel. Every step carries a
// semaphore edge because submission order alone does not order the timeline.
TEST_P(HostQueueTimestampTest, CapturesIntoQueueAllocaTarget) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(InitializeDevice(&test_device));
  iree_hal_device_t* device = test_device.base_device();

  Ref<iree_hal_buffer_t> before_target;
  Ref<iree_hal_buffer_t> after_target;
  IREE_ASSERT_OK(AllocateTimestampBuffer(device, before_target.out()));
  IREE_ASSERT_OK(AllocateTimestampBuffer(device, after_target.out()));

  Ref<iree_hal_semaphore_t> timeline;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           timeline.out()));

  uint64_t before_value = 1;
  IREE_ASSERT_OK(iree_hal_device_queue_timestamp(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      TimelinePoint(timeline, &before_value), before_target,
      /*target_offset=*/0, IREE_HAL_TIMESTAMP_FLAG_NONE));

  uint64_t alloca_value = 2;
  iree_hal_buffer_t* transient_target_ptr = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device, IREE_HAL_QUEUE_AFFINITY_ANY,
      TimelinePoint(timeline, &before_value),
      TimelinePoint(timeline, &alloca_value), /*pool=*/NULL,
      TransientTimestampBufferParams(), sizeof(uint64_t),
      IREE_HAL_ALLOCA_FLAG_NONE, &transient_target_ptr));
  Ref<iree_hal_buffer_t> transient_target(transient_target_ptr);
  // Pin the premise: an allocated buffer here covers none of the wrapper path.
  ASSERT_TRUE(iree_hal_amdgpu_transient_buffer_isa(transient_target));

  uint64_t capture_value = 3;
  IREE_ASSERT_OK(iree_hal_device_queue_timestamp(
      device, IREE_HAL_QUEUE_AFFINITY_ANY,
      TimelinePoint(timeline, &alloca_value),
      TimelinePoint(timeline, &capture_value), transient_target,
      /*target_offset=*/0, IREE_HAL_TIMESTAMP_FLAG_NONE));

  uint64_t after_value = 4;
  IREE_ASSERT_OK(iree_hal_device_queue_timestamp(
      device, IREE_HAL_QUEUE_AFFINITY_ANY,
      TimelinePoint(timeline, &capture_value),
      TimelinePoint(timeline, &after_value), after_target, /*target_offset=*/0,
      IREE_HAL_TIMESTAMP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(timeline, after_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  uint64_t before_tick = 0;
  uint64_t transient_tick = 0;
  uint64_t after_tick = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(before_target, 0, &before_tick,
                                          sizeof(before_tick)));
  IREE_ASSERT_OK(iree_hal_buffer_map_read(transient_target, 0, &transient_tick,
                                          sizeof(transient_tick)));
  IREE_ASSERT_OK(iree_hal_buffer_map_read(after_target, 0, &after_tick,
                                          sizeof(after_tick)));
  EXPECT_NE(transient_tick, 0u);
  EXPECT_LE(before_tick, transient_tick);
  EXPECT_LE(transient_tick, after_tick);

  uint64_t dealloca_value = 5;
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      TimelinePoint(timeline, &dealloca_value), transient_target,
      IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(timeline, dealloca_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
}

// Parameters for a transient target drawn from an explicit device-local pool.
// No capture ever lands in one of these, so neither mapping nor the capture
// alignment is requested.
static iree_hal_buffer_params_t DeviceLocalTransientBufferParams() {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  return params;
}

// Creates a pool holding exactly one allocation of |block_size| bytes, so a
// second allocation stays unstaged until the first one is released.
static iree_status_t CreateSingleBlockPool(iree_hal_device_t* device,
                                           iree_device_size_t block_size,
                                           iree_hal_pool_t** out_pool) {
  iree_hal_queue_pool_backend_t backend = {0};
  IREE_RETURN_IF_ERROR(iree_hal_device_query_queue_pool_backend(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, &backend));
  if (!backend.slab_provider || !backend.notification) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "queue pool backend query returned an incomplete backend bundle");
  }
  iree_hal_fixed_block_pool_options_t options = {};
  options.block_allocator_options.block_size = block_size;
  options.block_allocator_options.block_count = 1;
  options.block_allocator_options.frontier_capacity = 2;
  return iree_hal_fixed_block_pool_create(
      options, backend.slab_provider, backend.notification,
      iree_hal_pool_epoch_query_null(), iree_allocator_system(), out_pool);
}

// A queue_alloca target with no staged backing has no device pointer to write
// to, and the only thing that can make one appear is the allocation signal the
// caller failed to wait on. Naming the missing wait names the defect; naming an
// unbacked target describes a buffer the caller never passed.
TEST_P(HostQueueTimestampTest, RejectsUnstagedTargetWithoutAllocationWait) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(InitializeDevice(&test_device));
  iree_hal_device_t* device = test_device.base_device();

  const iree_device_size_t allocation_size = 4096;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateSingleBlockPool(device, allocation_size, pool.out()));

  // One timeline per wrapper: the blocked allocation completes only after the
  // held wrapper is deallocated, so a shared timeline would signal backwards.
  // The third carries the rejected capture's signal list, which never fires.
  Ref<iree_hal_semaphore_t> held_timeline;
  Ref<iree_hal_semaphore_t> blocked_timeline;
  Ref<iree_hal_semaphore_t> capture_timeline;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           held_timeline.out()));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           blocked_timeline.out()));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           capture_timeline.out()));

  uint64_t held_alloca_value = 1;
  iree_hal_buffer_t* held_target_ptr = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      TimelinePoint(held_timeline, &held_alloca_value), pool,
      DeviceLocalTransientBufferParams(), allocation_size,
      IREE_HAL_ALLOCA_FLAG_NONE, &held_target_ptr));
  Ref<iree_hal_buffer_t> held_target(held_target_ptr);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(held_timeline, held_alloca_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  uint64_t blocked_alloca_value = 1;
  iree_hal_buffer_t* blocked_target_ptr = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      TimelinePoint(blocked_timeline, &blocked_alloca_value), pool,
      DeviceLocalTransientBufferParams(), allocation_size,
      IREE_HAL_ALLOCA_FLAG_NONE, &blocked_target_ptr));
  Ref<iree_hal_buffer_t> blocked_target(blocked_target_ptr);
  // Pin the premise: a staged target makes this a duplicate of
  // CapturesIntoQueueAllocaTarget.
  ASSERT_EQ(iree_hal_amdgpu_transient_buffer_backing_buffer(blocked_target),
            nullptr);

  uint64_t capture_value = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_device_queue_timestamp(
          device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          TimelinePoint(capture_timeline, &capture_value), blocked_target,
          /*target_offset=*/0, IREE_HAL_TIMESTAMP_FLAG_NONE));

  // Release the held block so the blocked allocation can complete, then unwind
  // both wrappers through the queue rather than through wrapper destruction.
  uint64_t held_dealloca_value = 2;
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      TimelinePoint(held_timeline, &held_dealloca_value), held_target,
      IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(blocked_timeline, blocked_alloca_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  uint64_t blocked_dealloca_value = 2;
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      TimelinePoint(blocked_timeline, &blocked_dealloca_value), blocked_target,
      IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      blocked_timeline, blocked_dealloca_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));
}

}  // namespace
}  // namespace iree::hal::amdgpu
