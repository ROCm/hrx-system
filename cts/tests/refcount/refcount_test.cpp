// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE_METHOD(PyreTestFixture, "Semaphore retain and release",
                 "[refcount][semaphore]") {
  pyre_semaphore_t sem = nullptr;
  REQUIRE_OK(pyre().semaphore_create(device_, 0, &sem));

  // Retain adds a ref.
  REQUIRE_OK(pyre().semaphore_retain(sem));
  // First release — still alive.
  REQUIRE_OK(pyre().semaphore_release(sem));
  // Second release — destroyed. Should not crash.
  REQUIRE_OK(pyre().semaphore_release(sem));
}

TEST_CASE_METHOD(PyreTestFixture, "Stream retain and release",
                 "[refcount][stream]") {
  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));

  REQUIRE_OK(pyre().stream_retain(stream));
  REQUIRE_OK(pyre().stream_release(stream));
  REQUIRE_OK(pyre().stream_release(stream));
}

TEST_CASE_METHOD(PyreTestFixture, "Buffer retain and release",
                 "[refcount][memory]") {
  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));

  pyre_buffer_t buf = nullptr;
  REQUIRE_OK(pyre().buffer_allocate(stream, 256, PYRE_MEMORY_TYPE_HOST_LOCAL | PYRE_MEMORY_TYPE_DEVICE_VISIBLE, PYRE_BUFFER_USAGE_DEFAULT | PYRE_BUFFER_USAGE_MAPPING_SCOPED, &buf));

  REQUIRE_OK(pyre().buffer_retain(buf));
  REQUIRE_OK(pyre().buffer_release(buf));
  REQUIRE_OK(pyre().buffer_release(buf));

  REQUIRE_OK(pyre().stream_release(stream));
}

TEST_CASE_METHOD(PyreTestFixture, "NULL handle returns error",
                 "[refcount]") {
  pyre_status_t s;

  s = pyre().semaphore_retain(nullptr);
  REQUIRE(!pyre_status_is_ok(s));
  pyre().status_ignore(s);

  s = pyre().stream_retain(nullptr);
  REQUIRE(!pyre_status_is_ok(s));
  pyre().status_ignore(s);

  s = pyre().buffer_retain(nullptr);
  REQUIRE(!pyre_status_is_ok(s));
  pyre().status_ignore(s);
}
