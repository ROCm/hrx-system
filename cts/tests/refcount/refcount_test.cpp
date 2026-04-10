// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE_METHOD(HrxTestFixture, "Semaphore retain and release",
                 "[refcount][semaphore]") {
  hrx_semaphore_t sem = nullptr;
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &sem));

  // Retain adds a ref.
  hrx().semaphore_retain(sem);
  // First release — still alive.
  hrx().semaphore_release(sem);
  // Second release — destroyed. Should not crash.
  hrx().semaphore_release(sem);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream retain and release",
                 "[refcount][stream]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  hrx().stream_retain(stream);
  hrx().stream_release(stream);
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Buffer retain and release",
                 "[refcount][memory]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().buffer_allocate(
      stream, 256, HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &buf));

  hrx().buffer_retain(buf);
  hrx().buffer_release(buf);
  hrx().buffer_release(buf);

  hrx().stream_release(stream);
}
