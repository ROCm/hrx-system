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
    if (stream) {
      ReportHrxTestFailure("stream transfer completion",
                           hrx().stream_synchronize(stream));
      hrx().stream_release(stream);
    }
    if (target_buffer) {
      hrx().buffer_release(target_buffer);
    }
    if (source_buffer) {
      hrx().buffer_release(source_buffer);
    }
  }

  // Owns accepted transfers through completion, including assertion cleanup.
  hrx_stream_t stream = nullptr;
  // Retained source allocation.
  hrx_buffer_t source_buffer = nullptr;
  // Retained destination allocation.
  hrx_buffer_t target_buffer = nullptr;
};

}  // namespace

TEST_CASE_METHOD(HrxTestFixture, "synchronous_h2d_linear basic",
                 "[transfer][h2d]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  hrx_buffer_params_t params = {
      .type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      .usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED,
  };

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().allocator_allocate_buffer(alloc, params, 256, &buf));

  uint8_t src[256];
  memset(src, 0x42, sizeof(src));

  REQUIRE_OK(hrx().synchronous_h2d_linear(transfer_queue_, src, buf, 0, 256));

  // Verify via map.
  void* mapped = nullptr;
  REQUIRE_OK(hrx().buffer_map(buf, HRX_MAP_READ, 0, 256, &mapped));
  REQUIRE(memcmp(mapped, src, 256) == 0);
  REQUIRE_OK(hrx().buffer_unmap(buf));

  hrx().buffer_release(buf);
}

TEST_CASE_METHOD(HrxTestFixture, "synchronous_d2h_linear basic",
                 "[transfer][d2h]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  hrx_buffer_params_t params = {
      .type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      .usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED,
  };

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().allocator_allocate_buffer(alloc, params, 256, &buf));

  // Write via map.
  void* mapped = nullptr;
  REQUIRE_OK(hrx().buffer_map(buf, HRX_MAP_WRITE, 0, 256, &mapped));
  memset(mapped, 0x99, 256);
  REQUIRE_OK(hrx().buffer_unmap(buf));

  // Read back via synchronous_d2h_linear.
  uint8_t dst[256] = {};
  REQUIRE_OK(hrx().synchronous_d2h_linear(transfer_queue_, buf, 0, dst, 256));

  for (int i = 0; i < 256; i++) {
    REQUIRE(dst[i] == 0x99);
  }

  hrx().buffer_release(buf);
}

TEST_CASE_METHOD(HrxTestFixture, "synchronous h2d then d2h roundtrip",
                 "[transfer][roundtrip]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  hrx_buffer_params_t params = {
      .type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      .usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED,
  };

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().allocator_allocate_buffer(alloc, params, 1024, &buf));

  // Write pattern.
  uint32_t pattern[256];
  for (int i = 0; i < 256; i++) {
    pattern[i] = (uint32_t)i * 0x01010101;
  }

  REQUIRE_OK(hrx().synchronous_h2d_linear(transfer_queue_, pattern, buf, 0,
                                          sizeof(pattern)));

  // Read back.
  uint32_t readback[256] = {};
  REQUIRE_OK(hrx().synchronous_d2h_linear(transfer_queue_, buf, 0, readback,
                                          sizeof(readback)));

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
  REQUIRE_OK(
      hrx().stream_create(transfer_queue_, /*flags=*/0, &resources.stream));
  REQUIRE_OK(hrx().buffer_allocate(
      resources.stream, kBufferSize, HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_BUFFER_USAGE_DEFAULT, &resources.source_buffer));
  REQUIRE_OK(hrx().buffer_allocate(
      resources.stream, kBufferSize, HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_BUFFER_USAGE_DEFAULT, &resources.target_buffer));

  REQUIRE_OK(hrx().stream_copy_h2d_linear(resources.stream, source.data(),
                                          resources.source_buffer, 0,
                                          source.size()));
  REQUIRE_OK(hrx().stream_copy_buffer_linear(
      resources.stream, resources.source_buffer, 0, resources.target_buffer, 0,
      source.size()));
  REQUIRE_OK(hrx().stream_copy_d2h_linear(resources.stream,
                                          resources.target_buffer, 0,
                                          target.data(), target.size()));
  REQUIRE_OK(hrx().stream_synchronize(resources.stream));
  REQUIRE(target == source);
}

TEST_CASE_METHOD(HrxTestFixture, "synchronous_h2d_linear out of range fails",
                 "[transfer][error]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  hrx_buffer_params_t params = {
      .type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      .usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED,
  };

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().allocator_allocate_buffer(alloc, params, 64, &buf));

  uint8_t src[128];
  // Try to write 128 bytes into a 64-byte buffer.
  hrx_status_t s =
      hrx().synchronous_h2d_linear(transfer_queue_, src, buf, 0, 128);
  REQUIRE(!hrx_status_is_ok(s));
  hrx().status_ignore(s);

  // Neither API selects a hidden device queue when execution is unspecified.
  s = hrx().synchronous_h2d_linear(nullptr, src, buf, 0, 4);
  CHECK(hrx().status_code(s) == HRX_STATUS_INVALID_ARGUMENT);
  hrx().status_ignore(s);
  s = hrx().synchronous_d2h_linear(nullptr, buf, 0, src, 4);
  CHECK(hrx().status_code(s) == HRX_STATUS_INVALID_ARGUMENT);
  hrx().status_ignore(s);
  s = hrx().stream_copy_h2d_linear(nullptr, src, buf, 0, 4);
  CHECK(hrx().status_code(s) == HRX_STATUS_INVALID_ARGUMENT);
  hrx().status_ignore(s);
  s = hrx().stream_copy_d2h_linear(nullptr, buf, 0, src, 4);
  CHECK(hrx().status_code(s) == HRX_STATUS_INVALID_ARGUMENT);
  hrx().status_ignore(s);

  hrx().buffer_release(buf);
}

TEST_CASE_METHOD(
    HrxTestFixture,
    "rectangular host and stream copies preserve independent layouts",
    "[transfer][stream][geometry]") {
  std::array<uint8_t, 256> source;
  std::array<uint8_t, 256> padding;
  std::array<uint8_t, 256> target;
  std::array<uint8_t, 256> observed;
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>(i * 11 + 7);
  }
  padding.fill(0xCD);
  StreamTransferResources resources;
  REQUIRE_OK(hrx().stream_create(transfer_queue_, 0, &resources.stream));
  REQUIRE_OK(hrx().buffer_allocate(
      resources.stream, source.size(), HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_BUFFER_USAGE_DEFAULT, &resources.source_buffer));
  REQUIRE_OK(hrx().buffer_allocate(
      resources.stream, target.size(), HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_BUFFER_USAGE_DEFAULT, &resources.target_buffer));

  const hrx_transfer_region_t upload = {
      .row_length = 9,
      .row_count = 3,
      .slice_count = 2,
      .source_strides = {13, 52},
      .target_strides = {20, 80},
  };
  const hrx_transfer_region_t copy = {
      .row_length = 9,
      .row_count = 3,
      .slice_count = 2,
      .source_strides = {20, 80},
      .target_strides = {24, 104},
  };
  const hrx_transfer_region_t download = {
      .row_length = 9,
      .row_count = 3,
      .slice_count = 2,
      .source_strides = {24, 104},
      .target_strides = {11, 44},
  };
  for (bool asynchronous : {false, true}) {
    CAPTURE(asynchronous);
    target = padding;
    REQUIRE_OK(hrx().synchronous_h2d_linear(transfer_queue_, padding.data(),
                                            resources.source_buffer, 0,
                                            padding.size()));
    REQUIRE_OK(hrx().synchronous_h2d_linear(transfer_queue_, padding.data(),
                                            resources.target_buffer, 0,
                                            padding.size()));
    if (asynchronous) {
      REQUIRE_OK(hrx().stream_copy_h2d(resources.stream, source.data() + 3,
                                       resources.source_buffer, 5, &upload));
    } else {
      REQUIRE_OK(hrx().synchronous_h2d(transfer_queue_, source.data() + 3,
                                       resources.source_buffer, 5, &upload));
    }
    REQUIRE_OK(hrx().stream_copy_buffer(resources.stream,
                                        resources.source_buffer, 5,
                                        resources.target_buffer, 7, &copy));
    if (asynchronous) {
      REQUIRE_OK(hrx().stream_copy_d2h(resources.stream,
                                       resources.target_buffer, 7,
                                       target.data() + 2, &download));
    } else {
      REQUIRE_OK(hrx().stream_synchronize(resources.stream));
      REQUIRE_OK(hrx().synchronous_d2h(transfer_queue_, resources.target_buffer,
                                       7, target.data() + 2, &download));
    }
    REQUIRE_OK(hrx().stream_synchronize(resources.stream));
    auto expected_host = padding;
    auto expected_device = padding;
    for (size_t z = 0; z < 2; ++z) {
      for (size_t y = 0; y < 3; ++y) {
        for (size_t x = 0; x < 9; ++x) {
          const uint8_t value = source[3 + z * 52 + y * 13 + x];
          expected_host[2 + z * 44 + y * 11 + x] = value;
          expected_device[7 + z * 104 + y * 24 + x] = value;
        }
      }
    }
    REQUIRE(target == expected_host);
    REQUIRE_OK(hrx().synchronous_d2h_linear(transfer_queue_,
                                            resources.target_buffer, 0,
                                            observed.data(), observed.size()));
    REQUIRE(observed == expected_device);
  }
}

TEST_CASE_METHOD(HrxTestFixture,
                 "rectangular updates capture useful bytes before return",
                 "[transfer][stream][geometry]") {
  std::array<uint8_t, 128> source;
  std::array<uint8_t, 128> observed;
  std::array<uint8_t, 128> expected;
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>(i + 1);
  }
  expected.fill(0xCA);
  StreamTransferResources resources;
  REQUIRE_OK(hrx().stream_create(transfer_queue_, 0, &resources.stream));
  REQUIRE_OK(hrx().buffer_allocate(
      resources.stream, observed.size(), HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_BUFFER_USAGE_DEFAULT, &resources.target_buffer));
  const uint8_t padding = 0xCA;
  REQUIRE_OK(hrx().stream_fill_buffer_linear(
      resources.stream, resources.target_buffer, 0, observed.size(), &padding,
      sizeof(padding)));
  const hrx_transfer_region_t update = {
      .row_length = 4,
      .row_count = 3,
      .slice_count = 2,
      .source_strides = {12, 48},
      .target_strides = {16, 64},
  };
  for (size_t z = 0; z < 2; ++z) {
    for (size_t y = 0; y < 3; ++y) {
      for (size_t x = 0; x < 4; ++x) {
        expected[8 + z * 64 + y * 16 + x] = source[z * 48 + y * 12 + x];
      }
    }
  }
  REQUIRE_OK(hrx().stream_update_buffer(resources.stream, source.data(),
                                        source.size(), resources.target_buffer,
                                        8, &update));
  source.fill(0xFF);
  REQUIRE_OK(hrx().stream_copy_d2h_linear(resources.stream,
                                          resources.target_buffer, 0,
                                          observed.data(), observed.size()));
  REQUIRE_OK(hrx().stream_synchronize(resources.stream));
  REQUIRE(observed == expected);

  const hrx_fill_region_t fill = {
      .row_length = 8,
      .row_count = 3,
      .slice_count = 2,
      .target_strides = {16, 64},
  };
  const uint32_t pattern = 0x7953A216u;
  const auto* pattern_bytes = reinterpret_cast<const uint8_t*>(&pattern);
  for (size_t pattern_size : {1u, 2u, 4u}) {
    CAPTURE(pattern_size);
    REQUIRE_OK(hrx().stream_fill_buffer(resources.stream,
                                        resources.target_buffer, 8, &pattern,
                                        pattern_size, &fill));
    for (size_t z = 0; z < 2; ++z) {
      for (size_t y = 0; y < 3; ++y) {
        for (size_t x = 0; x < 8; ++x) {
          expected[8 + z * 64 + y * 16 + x] = pattern_bytes[x % pattern_size];
        }
      }
    }
    REQUIRE_OK(hrx().stream_copy_d2h_linear(resources.stream,
                                            resources.target_buffer, 0,
                                            observed.data(), observed.size()));
    REQUIRE_OK(hrx().stream_synchronize(resources.stream));
    REQUIRE(observed == expected);
  }
}

TEST_CASE_METHOD(HrxTestFixture,
                 "rectangular transfers validate addressed windows",
                 "[transfer][geometry][error]") {
  std::array<uint8_t, 64> host = {};
  StreamTransferResources resources;
  REQUIRE_OK(hrx().stream_create(transfer_queue_, 0, &resources.stream));
  REQUIRE_OK(hrx().buffer_allocate(
      resources.stream, host.size(), HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_BUFFER_USAGE_DEFAULT, &resources.target_buffer));
  const auto check_out_of_range = [](hrx_status_t status) {
    CHECK(hrx().status_code(status) == HRX_STATUS_OUT_OF_RANGE);
    hrx().status_ignore(status);
  };
  hrx_transfer_region_t region = {
      .row_length = 4,
      .row_count = 3,
      .slice_count = 1,
      .source_strides = {12, 0},
      .target_strides = {32, 0},
  };
  check_out_of_range(hrx().synchronous_h2d(
      transfer_queue_, host.data(), resources.target_buffer, 0, &region));
  check_out_of_range(hrx().stream_copy_h2d(
      resources.stream, host.data(), resources.target_buffer, 0, &region));
  region.source_strides = {UINT64_MAX, 0};
  region.target_strides = {12, 0};
  check_out_of_range(hrx().synchronous_d2h(
      transfer_queue_, resources.target_buffer, 0, host.data(), &region));
  check_out_of_range(hrx().stream_copy_d2h(
      resources.stream, resources.target_buffer, 0, host.data(), &region));
  region.source_strides = {12, 0};
  // Twelve useful bytes still require a 28-byte source window.
  check_out_of_range(hrx().stream_update_buffer(
      resources.stream, host.data(), 12, resources.target_buffer, 0, &region));
  check_out_of_range(
      hrx().stream_copy_buffer(resources.stream, resources.target_buffer, 48,
                               resources.target_buffer, 0, &region));
  check_out_of_range(hrx().queue_copy(
      transfer_queue_, nullptr, nullptr, resources.target_buffer, 48,
      resources.target_buffer, 0, &region, nullptr));
  const hrx_fill_region_t fill = {
      .row_length = 4,
      .row_count = 3,
      .slice_count = 1,
      .target_strides = {32, 0},
  };
  const uint32_t pattern = 0;
  check_out_of_range(
      hrx().stream_fill_buffer(resources.stream, resources.target_buffer, 0,
                               &pattern, sizeof(pattern), &fill));
  check_out_of_range(hrx().queue_fill(transfer_queue_, nullptr, nullptr,
                                      resources.target_buffer, 0, &pattern,
                                      sizeof(pattern), &fill, nullptr));
}
