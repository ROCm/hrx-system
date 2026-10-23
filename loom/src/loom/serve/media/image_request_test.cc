// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/image_request.h"

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/test_allocator.h"

namespace {

constexpr char kRgbPam[] =
    "data:image/x-portable-arbitrarymap;base64,"
    "UDcKV0lEVEggMgpIRUlHSFQgMgpERVBUSCAzCk1BWFZBTCAyNTUKVFVQTFRZUEUgUkdC"
    "CkVOREhEUgr/AAAA/wAAAP////8=";
constexpr char kRgbaPam[] =
    "data:image/x-portable-arbitrarymap;base64,"
    "UDcKV0lEVEggMgpIRUlHSFQgMgpERVBUSCA0Ck1BWFZBTCAyNTUKVFVQTFRZUEUgUkdC"
    "X0FMUEhBCkVOREhEUgoKFB4AKDI8gEZQWv9kbnjI";
constexpr char kRgbGif[] =
    "data:image/gif;base64,R0lGODdhAgACAIEAAP///wD/AP8AAAAA/ywAAAAAAgACAAAIBwAF"
    "BBgAICAAOw==";
constexpr char kRgbWebP[] =
    "data:image/webp;base64,UklGRiwAAABXRUJQVlA4TB8AAAAvAUAAAB8gEEjeHzqN+RcQF"
    "Pwf3fxHZA/gBgwR/Q8BAA==";
constexpr char kRgbaWebP[] =
    "data:image/webp;base64,UklGRiwAAABXRUJQVlA4TB8AAAAvAUAAEB9wCo8reNRDIEA4Z"
    "89VwO2DbIAyVZhF9D92AA==";
constexpr loom_serve_media_image_limits_t kLimits = {
    .compressed_bytes = 2048,
    .width = 64,
    .height = 64,
    .pixels = 4096,
};

class ImageRequestTest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (auto& input : inputs_) {
      loom_serve_media_image_input_deinitialize(&input);
    }
    for (auto* packet : packets_) {
      iree_vm_buffer_release(packet);
    }
  }

  bool Open(size_t index, const char* url) {
    auto status = loom_serve_media_image_input_initialize(
        iree_make_cstring_view(url), &kLimits, iree_allocator_system(),
        &inputs_[index]);
    if (iree_status_is_unimplemented(status)) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
      return false;
    }
    IREE_EXPECT_OK(status);
    return inputs_[index].allocation != nullptr;
  }

  // Independent immutable codec inputs, held through batch realization.
  std::array<loom_serve_media_image_input_t, 5> inputs_ = {};
  // Source-visible packets released after assertions, including early failure.
  std::array<iree_vm_buffer_t*, 5> packets_ = {};
};

TEST_F(ImageRequestTest, MixedCodecsAndChannelsPreserveExactSourcePackets) {
  const char* urls[] = {kRgbaPam, kRgbGif, kRgbPam, kRgbaWebP, kRgbWebP};
  std::array<std::vector<uint8_t>, 5> metadata;
  std::vector<loom_serve_media_image_request_t> requests;
  for (size_t i = 0; i < IREE_ARRAYSIZE(urls); ++i) {
    const size_t index = requests.size();
    if (!Open(index, urls[i])) {
      continue;
    }
    metadata[index].assign(i * 17, uint8_t(i + 1));
    requests.push_back({
        .image = &inputs_[index],
        .metadata = iree_make_const_byte_span(metadata[index].data(),
                                              metadata[index].size()),
        .output_rows = uint32_t(i + 1),
    });
  }
  std::array<iree_status_t, 5> statuses = {};
  IREE_ASSERT_OK(loom_serve_media_image_request_create_batch(
      requests.size(), requests.data(), iree_allocator_system(),
      packets_.data(), statuses.data()));
  const std::vector<uint8_t> rgb = {255, 0, 0,   0,   255, 0,
                                    0,   0, 255, 255, 255, 255};
  const std::vector<uint8_t> rgba = {10, 20, 30, 0,   40,  50,  60,  128,
                                     70, 80, 90, 255, 100, 110, 120, 200};
  for (size_t i = 0; i < requests.size(); ++i) {
    SCOPED_TRACE(i);
    IREE_ASSERT_OK(statuses[i]);
    const auto info = inputs_[i].info;
    loom_serve_media_image_input_deinitialize(&inputs_[i]);
    const uint32_t metadata_length = uint32_t(metadata[i].size());
    const uint32_t pixel_offset = (64 + metadata_length + 15) & ~15u;
    iree_const_byte_span_t bytes;
    IREE_ASSERT_OK(iree_vm_buffer_map_read(
        packets_[i], 0, iree_vm_buffer_length(packets_[i]), &bytes));
    const size_t pixel_end = pixel_offset + info.pixel_length;
    EXPECT_EQ(bytes.data_length, (pixel_end + 15) & ~size_t{15});
    const uint32_t fields[] = {
        1,
        uint32_t(info.format),
        info.width,
        info.height,
        info.display_width,
        info.display_height,
        info.orientation,
        requests[i].output_rows,
        64,
        metadata_length,
        pixel_offset,
        uint32_t(info.pixel_length),
        uint32_t(info.pixel_format),
        0,
        0,
        0,
    };
    for (size_t field = 0; field < IREE_ARRAYSIZE(fields); ++field) {
      SCOPED_TRACE(field);
      EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data + field * 4),
                fields[field]);
    }
    for (size_t j = 0; j < metadata_length; ++j) {
      EXPECT_EQ(bytes.data[64 + j], metadata[i][j]);
    }
    for (size_t j = 64 + metadata_length; j < pixel_offset; ++j) {
      EXPECT_EQ(bytes.data[j], 0u);
    }
    EXPECT_EQ(
        std::vector<uint8_t>(bytes.data + pixel_offset, bytes.data + pixel_end),
        info.pixel_format == LOOM_SERVE_MEDIA_IMAGE_PIXEL_FORMAT_RGB8 ? rgb
                                                                      : rgba);
    for (size_t j = pixel_end; j < bytes.data_length; ++j) {
      EXPECT_EQ(bytes.data[j], 0u);
    }
    // Descriptor-only planning borrows no compressed data and emits exactly
    // the execution packet's prefix, including empty and unaligned metadata.
    iree_vm_buffer_t* descriptor = nullptr;
    IREE_ASSERT_OK(loom_serve_media_image_request_describe(
        &info, requests[i].metadata, requests[i].output_rows,
        iree_allocator_system(), &descriptor));
    iree_const_byte_span_t described;
    IREE_EXPECT_OK(iree_vm_buffer_map_read(
        descriptor, 0, iree_vm_buffer_length(descriptor), &described));
    EXPECT_EQ(described.data_length, pixel_offset);
    EXPECT_EQ(std::vector<uint8_t>(described.data,
                                   described.data + described.data_length),
              std::vector<uint8_t>(bytes.data, bytes.data + pixel_offset));
    iree_vm_buffer_release(descriptor);
  }
}

TEST_F(ImageRequestTest, LateCodecFailureReleasesOnlyItsPacket) {
  if (!Open(0, kRgbPam)) {
    return;
  }
  // Header inspection succeeds; realization rejects the truncated raster.
  const char* damaged =
      "data:image/x-portable-arbitrarymap;base64,"
      "UDcKV0lEVEggMgpIRUlHSFQgMgpERVBUSCAzCk1BWFZBTCAyNTUKVFVQTFRZUEUgUkdC"
      "CkVOREhEUgr/AAAA/wAAAP///w==";
  ASSERT_TRUE(Open(1, damaged));
  const loom_serve_media_image_request_t requests[] = {
      {&inputs_[0], {}, 1},
      {&inputs_[1], {}, 2},
      {&inputs_[0], {}, 3},
  };
  iree_status_t statuses[3];
  IREE_ASSERT_OK(loom_serve_media_image_request_create_batch(
      3, requests, iree_allocator_system(), packets_.data(), statuses));
  IREE_EXPECT_OK(statuses[0]);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, statuses[1]);
  IREE_EXPECT_OK(statuses[2]);
  EXPECT_NE(packets_[0], nullptr);
  EXPECT_EQ(packets_[1], nullptr);
  EXPECT_NE(packets_[2], nullptr);
}

TEST_F(ImageRequestTest, AllocationFailuresPreserveOrdinalsAndReleaseStorage) {
  if (!Open(0, kRgbPam)) {
    return;
  }
  const loom_serve_media_image_request_t requests[] = {
      {&inputs_[0], {}, 1},
      {&inputs_[0], {}, 2},
      {&inputs_[0], {}, 3},
  };
  // Scratch, three exact packets, and shared codec scratch are the five
  // caller-allocator operations. Every site is exercised independently.
  for (size_t fail_at = 1; fail_at <= 5; ++fail_at) {
    SCOPED_TRACE(fail_at);
    iree::vm::testing::CountingAllocator allocator(fail_at);
    iree_status_t statuses[3];
    auto status = loom_serve_media_image_request_create_batch(
        3, requests, allocator.allocator(), packets_.data(), statuses);
    const bool shared = fail_at == 1 || fail_at == 5;
    if (shared) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    } else {
      IREE_EXPECT_OK(status);
    }
    for (size_t i = 0; i < 3; ++i) {
      const bool failed_packet = fail_at == i + 2;
      if (failed_packet) {
        IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, statuses[i]);
      } else {
        IREE_EXPECT_OK(statuses[i]);
      }
      EXPECT_EQ(packets_[i] == nullptr, shared || failed_packet);
      iree_vm_buffer_release(std::exchange(packets_[i], nullptr));
    }
    EXPECT_EQ(allocator.free_count(), allocator.allocation_count() - 1);
  }
}

TEST(ImageRequestEmptyTest, EmptyCohortDoesNotAllocate) {
  IREE_EXPECT_OK(loom_serve_media_image_request_create_batch(
      0, nullptr, iree_allocator_null(), nullptr, nullptr));
}

}  // namespace
