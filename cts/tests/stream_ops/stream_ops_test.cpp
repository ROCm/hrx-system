// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>

TEST_CASE_METHOD(HrxTestFixture, "Stream fill buffer", "[stream_ops][fill]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().buffer_allocate(
      stream, 1024, HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &buf));

  uint32_t pattern = 0xCAFEBABE;
  REQUIRE_OK(hrx().stream_fill_buffer(stream, buf, 0, 1024, &pattern,
                                      sizeof(pattern)));
  REQUIRE_OK(hrx().stream_flush(stream));
  REQUIRE_OK(hrx().stream_synchronize(stream));

  // Verify.
  void *ptr = nullptr;
  REQUIRE_OK(hrx().buffer_map(buf, HRX_MAP_READ, 0, 1024, &ptr));
  uint32_t *data = (uint32_t *)ptr;
  for (int i = 0; i < 256; i++) {
    REQUIRE(data[i] == 0xCAFEBABE);
  }
  REQUIRE_OK(hrx().buffer_unmap(buf));

  hrx().buffer_release(buf);
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream copy buffer", "[stream_ops][copy]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  hrx_buffer_t src = nullptr;
  hrx_buffer_t dst = nullptr;
  REQUIRE_OK(hrx().buffer_allocate(
      stream, 512, HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &src));
  REQUIRE_OK(hrx().buffer_allocate(
      stream, 512, HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &dst));

  // Fill source and flush (intra-CB RAW hazard requires separate submission).
  uint32_t pattern = 0x12345678;
  REQUIRE_OK(
      hrx().stream_fill_buffer(stream, src, 0, 512, &pattern, sizeof(pattern)));
  REQUIRE_OK(hrx().stream_flush(stream));
  REQUIRE_OK(hrx().stream_synchronize(stream));

  // Copy src -> dst.
  REQUIRE_OK(hrx().stream_copy_buffer(stream, src, 0, dst, 0, 512));
  REQUIRE_OK(hrx().stream_flush(stream));
  REQUIRE_OK(hrx().stream_synchronize(stream));

  // Verify.
  void *ptr = nullptr;
  REQUIRE_OK(hrx().buffer_map(dst, HRX_MAP_READ, 0, 512, &ptr));
  uint32_t *data = (uint32_t *)ptr;
  for (int i = 0; i < 128; i++) {
    REQUIRE(data[i] == 0x12345678);
  }
  REQUIRE_OK(hrx().buffer_unmap(dst));

  hrx().buffer_release(dst);
  hrx().buffer_release(src);
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream update buffer", "[stream_ops][copy]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().buffer_allocate(
      stream, 64, HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &buf));

  // Upload host data.
  uint8_t host_data[64];
  memset(host_data, 0x77, sizeof(host_data));
  REQUIRE_OK(hrx().stream_update_buffer(stream, host_data, 64, buf, 0));
  REQUIRE_OK(hrx().stream_flush(stream));
  REQUIRE_OK(hrx().stream_synchronize(stream));

  // Verify.
  void *ptr = nullptr;
  REQUIRE_OK(hrx().buffer_map(buf, HRX_MAP_READ, 0, 64, &ptr));
  uint8_t *data = (uint8_t *)ptr;
  for (int i = 0; i < 64; i++) {
    REQUIRE(data[i] == 0x77);
  }
  REQUIRE_OK(hrx().buffer_unmap(buf));

  hrx().buffer_release(buf);
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream execution barrier",
                 "[stream_ops][barrier]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().buffer_allocate(
      stream, 64, HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &buf));

  uint32_t first = 0x11223344;
  uint32_t second = 0x55667788;
  REQUIRE_OK(
      hrx().stream_fill_buffer(stream, buf, 0, 32, &first, sizeof(first)));
  REQUIRE_OK(hrx().stream_execution_barrier(stream));
  REQUIRE_OK(
      hrx().stream_fill_buffer(stream, buf, 32, 32, &second, sizeof(second)));
  REQUIRE_OK(hrx().stream_flush(stream));
  REQUIRE_OK(hrx().stream_synchronize(stream));

  void *ptr = nullptr;
  REQUIRE_OK(hrx().buffer_map(buf, HRX_MAP_READ, 0, 64, &ptr));
  const uint32_t *data = static_cast<const uint32_t *>(ptr);
  for (int i = 0; i < 8; ++i) {
    REQUIRE(data[i] == first);
    REQUIRE(data[8 + i] == second);
  }
  REQUIRE_OK(hrx().buffer_unmap(buf));

  hrx().buffer_release(buf);
  hrx().stream_release(stream);
}
