// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

#include "hrx_test_fixture.hpp"

namespace {

uint32_t queue_count(hrx_device_t device) {
  uint32_t count = 0;
  REQUIRE_OK(hrx().device_get_property(device, HRX_DEVICE_PROPERTY_QUEUE_COUNT,
                                       &count, sizeof(count)));
  return count;
}

// Fills a buffer through |stream| and reads it back, so the queue the stream
// named is proven to be a queue that actually runs work.
void fill_and_verify(hrx_stream_t stream, uint32_t pattern) {
  constexpr size_t kByteCount = 1024;
  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().buffer_allocate(
      stream, kByteCount,
      HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &buf));

  REQUIRE_OK(hrx().stream_fill_buffer(stream, buf, 0, kByteCount, &pattern,
                                      sizeof(pattern)));
  REQUIRE_OK(hrx().stream_synchronize(stream));

  void* ptr = nullptr;
  REQUIRE_OK(hrx().buffer_map(buf, HRX_MAP_READ, 0, kByteCount, &ptr));
  const uint32_t* data = static_cast<const uint32_t*>(ptr);
  for (size_t i = 0; i < kByteCount / sizeof(uint32_t); ++i) {
    REQUIRE(data[i] == pattern);
  }
  REQUIRE_OK(hrx().buffer_unmap(buf));

  hrx().buffer_release(buf);
}

}  // namespace

TEST_CASE_METHOD(HrxTestFixture, "Stream create and release", "[stream]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));
  REQUIRE(stream != nullptr);
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream reports owning device", "[stream]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  hrx_device_t stream_device = nullptr;
  REQUIRE_OK(hrx().stream_get_device(stream, &stream_device));
  REQUIRE(stream_device == device_);

  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream flush with no work", "[stream]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));
  REQUIRE_OK(hrx().stream_flush(stream));
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream sync with no work", "[stream]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));
  REQUIRE_OK(hrx().stream_synchronize(stream));
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream wait with no work", "[stream]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));
  REQUIRE_OK(hrx().stream_wait(stream));
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream query empty is complete", "[stream]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));
  bool complete = false;
  REQUIRE_OK(hrx().stream_query(stream, &complete));
  REQUIRE(complete);
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream get_semaphore", "[stream][sync]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  hrx_semaphore_t sem = nullptr;
  REQUIRE_OK(hrx().stream_get_semaphore(stream, &sem));
  REQUIRE(sem != nullptr);

  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream get_timeline_position",
                 "[stream][sync]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  hrx_timeline_point_t pos;
  REQUIRE_OK(hrx().stream_get_timeline_position(stream, &pos));
  REQUIRE(pos.semaphore != nullptr);
  REQUIRE(pos.value == 0);

  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Device reports at least one queue",
                 "[stream][queue]") {
  REQUIRE(queue_count(device_) >= 1);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream create leaves the queue unbound",
                 "[stream][queue]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  hrx_queue_affinity_t affinity = ~hrx_queue_affinity_t{0};
  REQUIRE_OK(hrx().stream_get_queue_affinity(stream, &affinity));
  REQUIRE(affinity == 0);

  fill_and_verify(stream, 0x5EED5EEDu);
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream keeps the queue it was created on",
                 "[stream][queue]") {
  const uint32_t queues = queue_count(device_);
  for (uint32_t i = 0; i < queues; ++i) {
    const hrx_queue_affinity_t requested = hrx_queue_affinity_t{1} << i;
    hrx_stream_t stream = nullptr;
    REQUIRE_OK(hrx().stream_create_on_queue(device_, 0, requested, &stream));

    hrx_queue_affinity_t affinity = 0;
    REQUIRE_OK(hrx().stream_get_queue_affinity(stream, &affinity));
    REQUIRE(affinity == requested);

    fill_and_verify(stream, 0xA5000000u | i);
    hrx().stream_release(stream);
  }
}

TEST_CASE_METHOD(HrxTestFixture, "Streams on distinct queues both complete",
                 "[stream][queue]") {
  if (queue_count(device_) < 2) {
    SUCCEED("device has a single queue");
    return;
  }

  hrx_stream_t first = nullptr;
  hrx_stream_t second = nullptr;
  REQUIRE_OK(hrx().stream_create_on_queue(device_, 0, 1, &first));
  REQUIRE_OK(hrx().stream_create_on_queue(device_, 0, 2, &second));

  fill_and_verify(first, 0x11111111u);
  fill_and_verify(second, 0x22222222u);

  hrx().stream_release(second);
  hrx().stream_release(first);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream rejects a queue the device lacks",
                 "[stream][queue]") {
  const uint32_t queues = queue_count(device_);
  if (queues >= 64) {
    SUCCEED("every affinity bit names a real queue");
    return;
  }

  hrx_stream_t stream = nullptr;
  hrx_status_t status = hrx().stream_create_on_queue(
      device_, 0, hrx_queue_affinity_t{1} << queues, &stream);
  REQUIRE(!hrx_status_is_ok(status));
  REQUIRE(hrx().status_code(status) == HRX_STATUS_OUT_OF_RANGE);
  REQUIRE(stream == nullptr);
  hrx().status_ignore(status);
}
