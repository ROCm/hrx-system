// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>

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

TEST_CASE_METHOD(HrxTestFixture, "Stream query empty is complete",
                 "[stream]") {
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
