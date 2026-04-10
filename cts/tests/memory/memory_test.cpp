// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>

TEST_CASE_METHOD(HrxTestFixture, "Buffer allocate and release",
                 "[memory][alloc]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().buffer_allocate(
      stream, 4096, HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &buf));
  REQUIRE(buf != nullptr);

  hrx().buffer_release(buf);
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Buffer map and unmap", "[memory][map]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().buffer_allocate(
      stream, 4096, HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &buf));

  void *ptr = nullptr;
  REQUIRE_OK(hrx().buffer_map(buf, HRX_MAP_WRITE, 0, 4096, &ptr));
  REQUIRE(ptr != nullptr);

  // Write pattern.
  memset(ptr, 0xAB, 4096);

  REQUIRE_OK(hrx().buffer_unmap(buf));
  hrx().buffer_release(buf);
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Buffer map read back written data",
                 "[memory][map]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().buffer_allocate(
      stream, 256, HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &buf));

  // Write.
  void *wptr = nullptr;
  REQUIRE_OK(hrx().buffer_map(buf, HRX_MAP_WRITE, 0, 256, &wptr));
  memset(wptr, 0x42, 256);
  REQUIRE_OK(hrx().buffer_unmap(buf));

  // Read back.
  void *rptr = nullptr;
  REQUIRE_OK(hrx().buffer_map(buf, HRX_MAP_READ, 0, 256, &rptr));
  unsigned char *data = (unsigned char *)rptr;
  for (int i = 0; i < 256; i++) {
    REQUIRE(data[i] == 0x42);
  }
  REQUIRE_OK(hrx().buffer_unmap(buf));

  hrx().buffer_release(buf);
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Zero-size buffer allocation fails",
                 "[memory][alloc]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  hrx_buffer_t buf = nullptr;
  hrx_status_t status = hrx().buffer_allocate(
      stream, 0, HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &buf);
  REQUIRE(!hrx_status_is_ok(status));
  hrx().status_ignore(status);

  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Buffer get_size returns correct size",
                 "[memory][size]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().buffer_allocate(
      stream, 8192, HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &buf));

  size_t size = 0;
  REQUIRE_OK(hrx().buffer_get_size(buf, &size));
  REQUIRE(size == 8192);

  hrx().buffer_release(buf);
  hrx().stream_release(stream);
}
