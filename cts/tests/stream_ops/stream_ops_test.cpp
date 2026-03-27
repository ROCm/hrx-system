// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>

TEST_CASE_METHOD(PyreTestFixture, "Stream fill buffer",
                 "[stream_ops][fill]") {
  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));

  pyre_buffer_t buf = nullptr;
  REQUIRE_OK(pyre().buffer_allocate(stream, 1024, PYRE_MEMORY_HOST_LOCAL, &buf));

  uint32_t pattern = 0xCAFEBABE;
  REQUIRE_OK(pyre().stream_fill_buffer(
      stream, buf, 0, 1024, &pattern, sizeof(pattern)));
  REQUIRE_OK(pyre().stream_flush(stream));
  REQUIRE_OK(pyre().stream_synchronize(stream));

  // Verify.
  void* ptr = nullptr;
  REQUIRE_OK(pyre().buffer_map(buf, PYRE_MAP_READ, 0, 1024, &ptr));
  uint32_t* data = (uint32_t*)ptr;
  for (int i = 0; i < 256; i++) {
    REQUIRE(data[i] == 0xCAFEBABE);
  }
  REQUIRE_OK(pyre().buffer_unmap(buf));

  REQUIRE_OK(pyre().buffer_release(buf));
  REQUIRE_OK(pyre().stream_release(stream));
}

TEST_CASE_METHOD(PyreTestFixture, "Stream copy buffer",
                 "[stream_ops][copy]") {
  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));

  pyre_buffer_t src = nullptr;
  pyre_buffer_t dst = nullptr;
  REQUIRE_OK(pyre().buffer_allocate(stream, 512, PYRE_MEMORY_HOST_LOCAL, &src));
  REQUIRE_OK(pyre().buffer_allocate(stream, 512, PYRE_MEMORY_HOST_LOCAL, &dst));

  // Fill source and flush (intra-CB RAW hazard requires separate submission).
  uint32_t pattern = 0x12345678;
  REQUIRE_OK(pyre().stream_fill_buffer(
      stream, src, 0, 512, &pattern, sizeof(pattern)));
  REQUIRE_OK(pyre().stream_flush(stream));
  REQUIRE_OK(pyre().stream_synchronize(stream));

  // Copy src -> dst.
  REQUIRE_OK(pyre().stream_copy_buffer(stream, src, 0, dst, 0, 512));
  REQUIRE_OK(pyre().stream_flush(stream));
  REQUIRE_OK(pyre().stream_synchronize(stream));

  // Verify.
  void* ptr = nullptr;
  REQUIRE_OK(pyre().buffer_map(dst, PYRE_MAP_READ, 0, 512, &ptr));
  uint32_t* data = (uint32_t*)ptr;
  for (int i = 0; i < 128; i++) {
    REQUIRE(data[i] == 0x12345678);
  }
  REQUIRE_OK(pyre().buffer_unmap(dst));

  REQUIRE_OK(pyre().buffer_release(dst));
  REQUIRE_OK(pyre().buffer_release(src));
  REQUIRE_OK(pyre().stream_release(stream));
}

TEST_CASE_METHOD(PyreTestFixture, "Stream update buffer",
                 "[stream_ops][copy]") {
  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));

  pyre_buffer_t buf = nullptr;
  REQUIRE_OK(pyre().buffer_allocate(stream, 64, PYRE_MEMORY_HOST_LOCAL, &buf));

  // Upload host data.
  uint8_t host_data[64];
  memset(host_data, 0x77, sizeof(host_data));
  REQUIRE_OK(pyre().stream_update_buffer(stream, host_data, 64, buf, 0));
  REQUIRE_OK(pyre().stream_flush(stream));
  REQUIRE_OK(pyre().stream_synchronize(stream));

  // Verify.
  void* ptr = nullptr;
  REQUIRE_OK(pyre().buffer_map(buf, PYRE_MAP_READ, 0, 64, &ptr));
  uint8_t* data = (uint8_t*)ptr;
  for (int i = 0; i < 64; i++) {
    REQUIRE(data[i] == 0x77);
  }
  REQUIRE_OK(pyre().buffer_unmap(buf));

  REQUIRE_OK(pyre().buffer_release(buf));
  REQUIRE_OK(pyre().stream_release(stream));
}
