// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>

TEST_CASE_METHOD(PyreTestFixture, "synchronous_h2d basic",
                 "[transfer][h2d]") {
  pyre_allocator_t alloc = pyre().device_allocator(device_);

  pyre_buffer_params_t params = {};
  params.type = PYRE_MEMORY_TYPE_HOST_LOCAL | PYRE_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = PYRE_BUFFER_USAGE_DEFAULT | PYRE_BUFFER_USAGE_MAPPING_SCOPED;

  pyre_buffer_t buf = nullptr;
  REQUIRE_OK(pyre().allocator_allocate_buffer(alloc, params, 256, &buf));

  uint8_t src[256];
  memset(src, 0x42, sizeof(src));

  REQUIRE_OK(pyre().synchronous_h2d(device_, src, buf, 0, 256));

  // Verify via map.
  void* mapped = nullptr;
  REQUIRE_OK(pyre().buffer_map(buf, PYRE_MAP_READ, 0, 256, &mapped));
  REQUIRE(memcmp(mapped, src, 256) == 0);
  REQUIRE_OK(pyre().buffer_unmap(buf));

  pyre().buffer_release(buf);
}

TEST_CASE_METHOD(PyreTestFixture, "synchronous_d2h basic",
                 "[transfer][d2h]") {
  pyre_allocator_t alloc = pyre().device_allocator(device_);

  pyre_buffer_params_t params = {};
  params.type = PYRE_MEMORY_TYPE_HOST_LOCAL | PYRE_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = PYRE_BUFFER_USAGE_DEFAULT | PYRE_BUFFER_USAGE_MAPPING_SCOPED;

  pyre_buffer_t buf = nullptr;
  REQUIRE_OK(pyre().allocator_allocate_buffer(alloc, params, 256, &buf));

  // Write via map.
  void* mapped = nullptr;
  REQUIRE_OK(pyre().buffer_map(buf, PYRE_MAP_WRITE, 0, 256, &mapped));
  memset(mapped, 0x99, 256);
  REQUIRE_OK(pyre().buffer_unmap(buf));

  // Read back via synchronous_d2h.
  uint8_t dst[256] = {};
  REQUIRE_OK(pyre().synchronous_d2h(device_, buf, 0, dst, 256));

  for (int i = 0; i < 256; i++) {
    REQUIRE(dst[i] == 0x99);
  }

  pyre().buffer_release(buf);
}

TEST_CASE_METHOD(PyreTestFixture, "synchronous h2d then d2h roundtrip",
                 "[transfer][roundtrip]") {
  pyre_allocator_t alloc = pyre().device_allocator(device_);

  pyre_buffer_params_t params = {};
  params.type = PYRE_MEMORY_TYPE_HOST_LOCAL | PYRE_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = PYRE_BUFFER_USAGE_DEFAULT | PYRE_BUFFER_USAGE_MAPPING_SCOPED;

  pyre_buffer_t buf = nullptr;
  REQUIRE_OK(pyre().allocator_allocate_buffer(alloc, params, 1024, &buf));

  // Write pattern.
  uint32_t pattern[256];
  for (int i = 0; i < 256; i++) pattern[i] = (uint32_t)i * 0x01010101;

  REQUIRE_OK(pyre().synchronous_h2d(device_, pattern, buf, 0, sizeof(pattern)));

  // Read back.
  uint32_t readback[256] = {};
  REQUIRE_OK(pyre().synchronous_d2h(device_, buf, 0, readback, sizeof(readback)));

  REQUIRE(memcmp(pattern, readback, sizeof(pattern)) == 0);

  pyre().buffer_release(buf);
}

TEST_CASE_METHOD(PyreTestFixture, "synchronous_h2d out of range fails",
                 "[transfer][error]") {
  pyre_allocator_t alloc = pyre().device_allocator(device_);

  pyre_buffer_params_t params = {};
  params.type = PYRE_MEMORY_TYPE_HOST_LOCAL | PYRE_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = PYRE_BUFFER_USAGE_DEFAULT | PYRE_BUFFER_USAGE_MAPPING_SCOPED;

  pyre_buffer_t buf = nullptr;
  REQUIRE_OK(pyre().allocator_allocate_buffer(alloc, params, 64, &buf));

  uint8_t src[128];
  // Try to write 128 bytes into a 64-byte buffer.
  pyre_status_t s = pyre().synchronous_h2d(device_, src, buf, 0, 128);
  REQUIRE(!pyre_status_is_ok(s));
  pyre().status_ignore(s);

  pyre().buffer_release(buf);
}
