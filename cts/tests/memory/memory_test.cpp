// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>

TEST_CASE_METHOD(PyreTestFixture, "Buffer allocate and release",
                 "[memory][alloc]") {
  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));

  pyre_buffer_t buf = nullptr;
  REQUIRE_OK(pyre().buffer_allocate(stream, 4096, PYRE_MEMORY_HOST_LOCAL, &buf));
  REQUIRE(buf != nullptr);

  REQUIRE_OK(pyre().buffer_release(buf));
  REQUIRE_OK(pyre().stream_release(stream));
}

TEST_CASE_METHOD(PyreTestFixture, "Buffer map and unmap",
                 "[memory][map]") {
  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));

  pyre_buffer_t buf = nullptr;
  REQUIRE_OK(pyre().buffer_allocate(stream, 4096, PYRE_MEMORY_HOST_LOCAL, &buf));

  void* ptr = nullptr;
  REQUIRE_OK(pyre().buffer_map(buf, PYRE_MAP_WRITE, 0, 4096, &ptr));
  REQUIRE(ptr != nullptr);

  // Write pattern.
  memset(ptr, 0xAB, 4096);

  REQUIRE_OK(pyre().buffer_unmap(buf));
  REQUIRE_OK(pyre().buffer_release(buf));
  REQUIRE_OK(pyre().stream_release(stream));
}

TEST_CASE_METHOD(PyreTestFixture, "Buffer map read back written data",
                 "[memory][map]") {
  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));

  pyre_buffer_t buf = nullptr;
  REQUIRE_OK(pyre().buffer_allocate(stream, 256, PYRE_MEMORY_HOST_LOCAL, &buf));

  // Write.
  void* wptr = nullptr;
  REQUIRE_OK(pyre().buffer_map(buf, PYRE_MAP_WRITE, 0, 256, &wptr));
  memset(wptr, 0x42, 256);
  REQUIRE_OK(pyre().buffer_unmap(buf));

  // Read back.
  void* rptr = nullptr;
  REQUIRE_OK(pyre().buffer_map(buf, PYRE_MAP_READ, 0, 256, &rptr));
  unsigned char* data = (unsigned char*)rptr;
  for (int i = 0; i < 256; i++) {
    REQUIRE(data[i] == 0x42);
  }
  REQUIRE_OK(pyre().buffer_unmap(buf));

  REQUIRE_OK(pyre().buffer_release(buf));
  REQUIRE_OK(pyre().stream_release(stream));
}

TEST_CASE_METHOD(PyreTestFixture, "Zero-size buffer allocation fails",
                 "[memory][alloc]") {
  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));

  pyre_buffer_t buf = nullptr;
  pyre_status_t status =
      pyre().buffer_allocate(stream, 0, PYRE_MEMORY_HOST_LOCAL, &buf);
  REQUIRE(!pyre_status_is_ok(status));
  pyre().status_ignore(status);

  REQUIRE_OK(pyre().stream_release(stream));
}
