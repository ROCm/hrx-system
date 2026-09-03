// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>

#include "hrx_test_fixture.hpp"

namespace {

struct StreamTransferResources {
  ~StreamTransferResources() {
    if (stream) hrx().stream_release(stream);
    if (target_buffer) hrx().buffer_release(target_buffer);
    if (source_buffer) hrx().buffer_release(source_buffer);
  }

  hrx_stream_t stream = nullptr;
  hrx_buffer_t source_buffer = nullptr;
  hrx_buffer_t target_buffer = nullptr;
};

}  // namespace

TEST_CASE_METHOD(HrxTestFixture, "synchronous_h2d basic", "[transfer][h2d]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  hrx_buffer_params_t params = {};
  params.type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED;

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().allocator_allocate_buffer(alloc, params, 256, &buf));

  uint8_t src[256];
  memset(src, 0x42, sizeof(src));

  REQUIRE_OK(hrx().synchronous_h2d(device_, src, buf, 0, 256));

  // Verify via map.
  void* mapped = nullptr;
  REQUIRE_OK(hrx().buffer_map(buf, HRX_MAP_READ, 0, 256, &mapped));
  REQUIRE(memcmp(mapped, src, 256) == 0);
  REQUIRE_OK(hrx().buffer_unmap(buf));

  hrx().buffer_release(buf);
}

TEST_CASE_METHOD(HrxTestFixture, "synchronous_d2h basic", "[transfer][d2h]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  hrx_buffer_params_t params = {};
  params.type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED;

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().allocator_allocate_buffer(alloc, params, 256, &buf));

  // Write via map.
  void* mapped = nullptr;
  REQUIRE_OK(hrx().buffer_map(buf, HRX_MAP_WRITE, 0, 256, &mapped));
  memset(mapped, 0x99, 256);
  REQUIRE_OK(hrx().buffer_unmap(buf));

  // Read back via synchronous_d2h.
  uint8_t dst[256] = {};
  REQUIRE_OK(hrx().synchronous_d2h(device_, buf, 0, dst, 256));

  for (int i = 0; i < 256; i++) {
    REQUIRE(dst[i] == 0x99);
  }

  hrx().buffer_release(buf);
}

TEST_CASE_METHOD(HrxTestFixture, "synchronous h2d then d2h roundtrip",
                 "[transfer][roundtrip]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  hrx_buffer_params_t params = {};
  params.type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED;

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().allocator_allocate_buffer(alloc, params, 1024, &buf));

  // Write pattern.
  uint32_t pattern[256];
  for (int i = 0; i < 256; i++) pattern[i] = (uint32_t)i * 0x01010101;

  REQUIRE_OK(hrx().synchronous_h2d(device_, pattern, buf, 0, sizeof(pattern)));

  // Read back.
  uint32_t readback[256] = {};
  REQUIRE_OK(
      hrx().synchronous_d2h(device_, buf, 0, readback, sizeof(readback)));

  REQUIRE(memcmp(pattern, readback, sizeof(pattern)) == 0);

  hrx().buffer_release(buf);
}

TEST_CASE_METHOD(HrxTestFixture,
                 "stream transfers order intervening command-buffer work",
                 "[transfer][stream]") {
  static constexpr size_t kBufferSize = 257;
  std::array<uint8_t, kBufferSize> source = {};
  std::array<uint8_t, kBufferSize> target = {};
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>((i * 29 + 17) & 0xFF);
  }

  StreamTransferResources resources;
  REQUIRE_OK(hrx().stream_create(device_, /*flags=*/0, &resources.stream));
  REQUIRE_OK(hrx().buffer_allocate(
      resources.stream, kBufferSize, HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_BUFFER_USAGE_DEFAULT, &resources.source_buffer));
  REQUIRE_OK(hrx().buffer_allocate(
      resources.stream, kBufferSize, HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_BUFFER_USAGE_DEFAULT, &resources.target_buffer));

  REQUIRE_OK(hrx().stream_copy_h2d(resources.stream, source.data(),
                                   resources.source_buffer, 0, source.size()));
  REQUIRE_OK(hrx().stream_copy_buffer(resources.stream, resources.source_buffer,
                                      0, resources.target_buffer, 0,
                                      source.size()));
  REQUIRE_OK(hrx().stream_copy_d2h(resources.stream, resources.target_buffer, 0,
                                   target.data(), target.size()));
  REQUIRE_OK(hrx().stream_synchronize(resources.stream));
  REQUIRE(target == source);
}

TEST_CASE_METHOD(HrxTestFixture, "synchronous_h2d out of range fails",
                 "[transfer][error]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  hrx_buffer_params_t params = {};
  params.type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED;

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().allocator_allocate_buffer(alloc, params, 64, &buf));

  uint8_t src[128];
  // Try to write 128 bytes into a 64-byte buffer.
  hrx_status_t s = hrx().synchronous_h2d(device_, src, buf, 0, 128);
  REQUIRE(!hrx_status_is_ok(s));
  hrx().status_ignore(s);

  hrx().buffer_release(buf);
}
