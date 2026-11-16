// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/program_image.h"

#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

iree_const_byte_span_t Span(const std::vector<uint8_t>& bytes) {
  return iree_make_const_byte_span(bytes.data(), bytes.size());
}

class ProgramImageStoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_amdxdna_program_image_store_create(
        iree_allocator_system(), &store_));
  }
  void TearDown() override {
    iree_hal_amdxdna_program_image_store_release(store_);
  }

  iree_hal_amdxdna_program_image_store_stats_t Stats() {
    iree_hal_amdxdna_program_image_store_stats_t stats;
    iree_hal_amdxdna_program_image_store_query_stats(store_, &stats);
    return stats;
  }

  iree_hal_amdxdna_program_image_t* Acquire(const std::vector<uint8_t>& bytes) {
    iree_hal_amdxdna_program_image_t* image = nullptr;
    IREE_CHECK_OK(iree_hal_amdxdna_program_image_acquire(
        store_, Span(bytes), iree_allocator_system(), &image));
    return image;
  }

  iree_hal_amdxdna_program_image_store_t* store_ = nullptr;
};

TEST_F(ProgramImageStoreTest, IdenticalContentsShareOneImage) {
  std::vector<uint8_t> bytes(1000, 0x5A);
  std::vector<uint8_t> copy = bytes;
  iree_hal_amdxdna_program_image_t* first = Acquire(bytes);
  iree_hal_amdxdna_program_image_t* second = Acquire(copy);
  EXPECT_EQ(first, second);
  EXPECT_EQ(Stats().image_count, 1u);
  EXPECT_EQ(Stats().content_bytes, bytes.size());

  iree_const_byte_span_t contents =
      iree_hal_amdxdna_program_image_contents(first);
  ASSERT_EQ(contents.data_length, bytes.size());
  EXPECT_NE(contents.data, bytes.data());
  EXPECT_EQ(0, memcmp(contents.data, bytes.data(), bytes.size()));
  EXPECT_GE(iree_hal_amdxdna_program_image_host_bytes(first), bytes.size());

  iree_hal_amdxdna_program_image_release(first);
  EXPECT_EQ(Stats().image_count, 1u);
  iree_hal_amdxdna_program_image_release(second);
  EXPECT_EQ(Stats().image_count, 0u);
  EXPECT_EQ(Stats().content_bytes, 0u);
}

TEST_F(ProgramImageStoreTest, DistinctContentsAreNotShared) {
  std::vector<uint8_t> a(64, 0x01);
  std::vector<uint8_t> b = a;
  b.back() = 0x02;  // Same length, differs only in the last byte.
  std::vector<uint8_t> c(65, 0x01);  // Same prefix, different length.
  iree_hal_amdxdna_program_image_t* image_a = Acquire(a);
  iree_hal_amdxdna_program_image_t* image_b = Acquire(b);
  iree_hal_amdxdna_program_image_t* image_c = Acquire(c);
  EXPECT_NE(image_a, image_b);
  EXPECT_NE(image_a, image_c);
  EXPECT_NE(image_b, image_c);
  EXPECT_EQ(Stats().image_count, 3u);
  EXPECT_EQ(Stats().content_bytes, a.size() + b.size() + c.size());

  // Unlinking from the middle of the store keeps the others reachable.
  iree_hal_amdxdna_program_image_release(image_b);
  EXPECT_EQ(Acquire(a), image_a);
  EXPECT_EQ(Acquire(c), image_c);
  iree_hal_amdxdna_program_image_release(image_a);
  iree_hal_amdxdna_program_image_release(image_c);
  iree_hal_amdxdna_program_image_release(image_a);
  iree_hal_amdxdna_program_image_release(image_c);
  EXPECT_EQ(Stats().image_count, 0u);
}

TEST_F(ProgramImageStoreTest, ReleasedImageIsRecreated) {
  std::vector<uint8_t> bytes(32, 0x33);
  iree_hal_amdxdna_program_image_t* image = Acquire(bytes);
  iree_hal_amdxdna_program_image_release(image);
  EXPECT_EQ(Stats().image_count, 0u);
  image = Acquire(bytes);
  EXPECT_EQ(Stats().image_count, 1u);
  iree_hal_amdxdna_program_image_retain(image);
  iree_hal_amdxdna_program_image_release(image);
  EXPECT_EQ(Stats().image_count, 1u);
  iree_hal_amdxdna_program_image_release(image);
  EXPECT_EQ(Stats().image_count, 0u);
}

TEST_F(ProgramImageStoreTest, ImagesKeepStoreAlive) {
  std::vector<uint8_t> bytes(16, 0x44);
  iree_hal_amdxdna_program_image_t* image = Acquire(bytes);
  iree_hal_amdxdna_program_image_store_release(store_);
  store_ = nullptr;
  iree_const_byte_span_t contents =
      iree_hal_amdxdna_program_image_contents(image);
  EXPECT_EQ(0, memcmp(contents.data, bytes.data(), bytes.size()));
  iree_hal_amdxdna_program_image_release(image);
}

TEST_F(ProgramImageStoreTest, EmptyContentsYieldNullImage) {
  iree_hal_amdxdna_program_image_t* image = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_program_image_acquire(
      store_, iree_const_byte_span_empty(), iree_allocator_system(), &image));
  EXPECT_EQ(image, nullptr);
  EXPECT_EQ(iree_hal_amdxdna_program_image_contents(image).data_length, 0u);
  EXPECT_EQ(Stats().image_count, 0u);
  iree_hal_amdxdna_program_image_release(image);
}

TEST(ProgramImageTest, PrivateImagesAreNotShared) {
  std::vector<uint8_t> bytes(128, 0x77);
  iree_hal_amdxdna_program_image_t* first = nullptr;
  iree_hal_amdxdna_program_image_t* second = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_program_image_acquire(
      /*store=*/nullptr, Span(bytes), iree_allocator_system(), &first));
  IREE_ASSERT_OK(iree_hal_amdxdna_program_image_acquire(
      /*store=*/nullptr, Span(bytes), iree_allocator_system(), &second));
  EXPECT_NE(first, second);
  EXPECT_EQ(0, memcmp(iree_hal_amdxdna_program_image_contents(second).data,
                      bytes.data(), bytes.size()));
  iree_hal_amdxdna_program_image_release(first);
  iree_hal_amdxdna_program_image_release(second);
}

}  // namespace
