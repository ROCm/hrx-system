// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// End-to-end tests for iree_hal_device_queue_capture_timestamp on the amdgpu
// driver: the device must write a real, monotonically increasing GPU clock tick
// into the caller buffer at the queue timeline point, and the tick delta must
// convert to a sane positive duration around real device work. GPU required.

#include <cstdint>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

class HostQueueTimestampCaptureTest : public ::testing::Test {
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

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t HostQueueTimestampCaptureTest::host_allocator_;
iree_hal_amdgpu_libhsa_t HostQueueTimestampCaptureTest::libhsa_;
iree_hal_amdgpu_topology_t HostQueueTimestampCaptureTest::topology_;

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
      iree_allocator_t host_allocator) {
    IREE_RETURN_IF_ERROR(create_context_.Initialize(host_allocator));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
        IREE_SV("amdgpu"), options, libhsa, topology, create_context_.params(),
        host_allocator, &base_device_));
    return iree_hal_device_group_create_from_device(
        base_device_, create_context_.frontier_tracker(), host_allocator,
        &device_group_);
  }

  iree_hal_device_t* base_device() const { return base_device_; }

 private:
  iree::hal::cts::DeviceCreateContext create_context_;
  iree_hal_device_t* base_device_ = NULL;
  iree_hal_device_group_t* device_group_ = NULL;
};

// Allocates an 8-byte host-visible device buffer usable as a timestamp target.
iree_status_t AllocateTimestampBuffer(iree_hal_device_t* device,
                                      iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, sizeof(uint64_t),
                                            out_buffer);
}

// Builds a single-semaphore timeline list pointing at |value|.
static iree_hal_semaphore_list_t TimelinePoint(iree_hal_semaphore_t** semaphore,
                                               uint64_t* value) {
  iree_hal_semaphore_list_t list = {
      /*count=*/1,
      /*semaphores=*/semaphore,
      /*payload_values=*/value,
  };
  return list;
}

// The device exposes a usable device-timestamp domain via the device spec.
TEST_F(HostQueueTimestampCaptureTest, DeviceSpecAdvertisesTimestamps) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  const iree_hal_device_spec_t* spec =
      iree_hal_device_spec(test_device.base_device());
  ASSERT_NE(spec, nullptr);
  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(spec);
  ASSERT_NE(timing, nullptr);
  EXPECT_TRUE(iree_all_bits_set(
      timing->flags, IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS));
  EXPECT_GT(timing->timestamp_frequency_hz, 0u);
}

// Two ordered captures write non-zero, monotonically non-decreasing ticks.
TEST_F(HostQueueTimestampCaptureTest, CaptureWritesMonotonicDeviceTicks) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_device_t* device = test_device.base_device();

  iree_hal_buffer_t* tick0 = NULL;
  iree_hal_buffer_t* tick1 = NULL;
  IREE_ASSERT_OK(AllocateTimestampBuffer(device, &tick0));
  IREE_ASSERT_OK(AllocateTimestampBuffer(device, &tick1));

  iree_hal_semaphore_t* timeline = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           &timeline));

  uint64_t v1 = 1;
  uint64_t v2 = 2;
  IREE_ASSERT_OK(iree_hal_device_queue_capture_timestamp(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      TimelinePoint(&timeline, &v1), tick0, /*target_offset=*/0,
      IREE_HAL_CAPTURE_TIMESTAMP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_device_queue_capture_timestamp(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, TimelinePoint(&timeline, &v1),
      TimelinePoint(&timeline, &v2), tick1, /*target_offset=*/0,
      IREE_HAL_CAPTURE_TIMESTAMP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      timeline, 2ull, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint64_t start_tick = 0;
  uint64_t stop_tick = 0;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(tick0, 0, &start_tick, sizeof(start_tick)));
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(tick1, 0, &stop_tick, sizeof(stop_tick)));

  EXPECT_NE(start_tick, 0u);
  EXPECT_NE(stop_tick, 0u);
  EXPECT_GE(stop_tick, start_tick);

  iree_hal_semaphore_release(timeline);
  iree_hal_buffer_release(tick1);
  iree_hal_buffer_release(tick0);
}

// A capture bracketing real device work yields a strictly positive elapsed
// time that converts to a sane number of milliseconds.
TEST_F(HostQueueTimestampCaptureTest, CaptureMeasuresElapsedAroundWork) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_device_t* device = test_device.base_device();

  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(iree_hal_device_spec(device));
  ASSERT_NE(timing, nullptr);
  const uint64_t frequency_hz = timing->timestamp_frequency_hz;
  ASSERT_GT(frequency_hz, 0u);

  iree_hal_buffer_t* tick0 = NULL;
  iree_hal_buffer_t* tick1 = NULL;
  IREE_ASSERT_OK(AllocateTimestampBuffer(device, &tick0));
  IREE_ASSERT_OK(AllocateTimestampBuffer(device, &tick1));

  // Scratch buffer filled between the two captures to create measurable work.
  const iree_device_size_t kScratchBytes = 64 * 1024 * 1024;
  iree_hal_buffer_params_t scratch_params = {0};
  scratch_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  scratch_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  iree_hal_buffer_t* scratch = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), scratch_params, kScratchBytes,
      &scratch));

  iree_hal_semaphore_t* timeline = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           &timeline));

  uint64_t v1 = 1;
  uint64_t v2 = 2;
  uint64_t v3 = 3;
  const uint32_t pattern = 0xABCDu;
  IREE_ASSERT_OK(iree_hal_device_queue_capture_timestamp(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      TimelinePoint(&timeline, &v1), tick0, /*target_offset=*/0,
      IREE_HAL_CAPTURE_TIMESTAMP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, TimelinePoint(&timeline, &v1),
      TimelinePoint(&timeline, &v2), scratch, /*target_offset=*/0, kScratchBytes,
      &pattern, /*pattern_length=*/sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_device_queue_capture_timestamp(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, TimelinePoint(&timeline, &v2),
      TimelinePoint(&timeline, &v3), tick1, /*target_offset=*/0,
      IREE_HAL_CAPTURE_TIMESTAMP_FLAG_NONE));
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
  ASSERT_GT(stop_tick, start_tick);
  const double elapsed_ms =
      (double)(stop_tick - start_tick) * 1000.0 / (double)frequency_hz;
  EXPECT_GT(elapsed_ms, 0.0);
  EXPECT_LT(elapsed_ms, 10000.0);

  iree_hal_semaphore_release(timeline);
  iree_hal_buffer_release(scratch);
  iree_hal_buffer_release(tick1);
  iree_hal_buffer_release(tick0);
}

}  // namespace
}  // namespace iree::hal::amdgpu
