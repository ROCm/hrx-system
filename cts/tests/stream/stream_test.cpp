// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE_METHOD(PyreTestFixture, "Stream create and release", "[stream]") {
  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));
  REQUIRE(stream != nullptr);
  pyre().stream_release(stream);
}

TEST_CASE_METHOD(PyreTestFixture, "Stream flush with no work", "[stream]") {
  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));
  REQUIRE_OK(pyre().stream_flush(stream));
  pyre().stream_release(stream);
}

TEST_CASE_METHOD(PyreTestFixture, "Stream sync with no work", "[stream]") {
  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));
  REQUIRE_OK(pyre().stream_synchronize(stream));
  pyre().stream_release(stream);
}

TEST_CASE_METHOD(PyreTestFixture, "Stream query empty is complete",
                 "[stream]") {
  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));
  bool complete = false;
  REQUIRE_OK(pyre().stream_query(stream, &complete));
  REQUIRE(complete);
  pyre().stream_release(stream);
}

TEST_CASE_METHOD(PyreTestFixture, "Stream get_semaphore", "[stream][sync]") {
  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));

  pyre_semaphore_t sem = nullptr;
  REQUIRE_OK(pyre().stream_get_semaphore(stream, &sem));
  REQUIRE(sem != nullptr);

  pyre().stream_release(stream);
}

TEST_CASE_METHOD(PyreTestFixture, "Stream get_timeline_position",
                 "[stream][sync]") {
  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));

  pyre_timeline_point_t pos;
  REQUIRE_OK(pyre().stream_get_timeline_position(stream, &pos));
  REQUIRE(pos.semaphore != nullptr);
  REQUIRE(pos.value == 0);

  pyre().stream_release(stream);
}
