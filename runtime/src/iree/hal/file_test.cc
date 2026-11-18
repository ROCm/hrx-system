// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/file.h"

#include <array>

#include "iree/base/internal/debugging.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::iree::testing::status::StatusIs;

TEST(FileTransferRecord, LinearReadsOnlyItsPrefix) {
  iree_hal_file_transfer_record_t record = {};
  iree_unaligned_store_le_u64(&record.words[0], UINT64_C(0x1234FEDCBA987654));
  iree_unaligned_store_le_u64(&record.words[1], UINT64_C(0xABCD000012345678));
  iree_unaligned_store_le_u64(&record.words[2], 90);
  IREE_ASAN_POISON_MEMORY_REGION(&record.words[3], 5 * sizeof(uint64_t));
  iree_hal_file_transfer_t transfer;
  iree_status_t status =
      iree_hal_file_transfer_record_decode(&record, &transfer);
  IREE_ASAN_UNPOISON_MEMORY_REGION(&record.words[3], 5 * sizeof(uint64_t));
  IREE_ASSERT_OK(status);
  EXPECT_EQ(transfer.file_binding, 0x1234u);
  EXPECT_EQ(transfer.buffer_binding, 0xABCDu);
  EXPECT_EQ(transfer.file_offset, UINT64_C(0xFEDCBA987654));
  EXPECT_EQ(transfer.buffer_offset, 0x12345678u);
  EXPECT_EQ(transfer.row_length, 90u);
  EXPECT_EQ(transfer.row_count, 1u);
  EXPECT_EQ(transfer.slice_count, 1u);
}

TEST(FileTransferRecord, StridedPreservesBothCountHalvesAndByteFields) {
  iree_hal_file_transfer_record_t record = {};
  const uint64_t words[] = {
      UINT64_C(0xFFFF000000000007), UINT64_C(0x800000000000000B),
      UINT64_C(0x000100000000005A), UINT64_C(0x123400FEDCBA9876),
      UINT64_C(0x9ABC00ABCDEF1234), UINT64_C(0x567800123456789A),
      UINT64_C(0xDEF0009876543210),
  };
  for (size_t i = 0; i < IREE_ARRAYSIZE(words); ++i) {
    iree_unaligned_store_le_u64(&record.words[i], words[i]);
  }
  IREE_ASAN_POISON_MEMORY_REGION(&record.words[7], sizeof(uint64_t));
  iree_hal_file_transfer_t transfer;
  iree_status_t status =
      iree_hal_file_transfer_record_decode(&record, &transfer);
  IREE_ASAN_UNPOISON_MEMORY_REGION(&record.words[7], sizeof(uint64_t));
  IREE_ASSERT_OK(status);
  EXPECT_EQ(transfer.file_binding, 65535u);
  EXPECT_EQ(transfer.buffer_binding, 32768u);
  EXPECT_EQ(transfer.file_offset, 7u);
  EXPECT_EQ(transfer.buffer_offset, 11u);
  EXPECT_EQ(transfer.row_length, 90u);
  EXPECT_EQ(transfer.row_count, 0x9ABC1234u);
  EXPECT_EQ(transfer.slice_count, 0xDEF05678u);
  EXPECT_EQ(transfer.file_strides.row_stride, UINT64_C(0xFEDCBA9876));
  EXPECT_EQ(transfer.buffer_strides.row_stride, UINT64_C(0xABCDEF1234));
  EXPECT_EQ(transfer.file_strides.slice_stride, UINT64_C(0x123456789A));
  EXPECT_EQ(transfer.buffer_strides.slice_stride, UINT64_C(0x9876543210));
}

TEST(FileTransferRecord, EmptyIgnoresFormAndBindings) {
  iree_hal_file_transfer_record_t record = {};
  iree_unaligned_store_le_u64(&record.words[2], UINT64_C(0xFFFF000000000000));
  IREE_ASAN_POISON_MEMORY_REGION(&record.words[0], 2 * sizeof(uint64_t));
  IREE_ASAN_POISON_MEMORY_REGION(&record.words[3], 5 * sizeof(uint64_t));
  iree_hal_file_transfer_t transfer;
  iree_status_t status =
      iree_hal_file_transfer_record_decode(&record, &transfer);
  IREE_ASAN_UNPOISON_MEMORY_REGION(&record, sizeof(record));
  IREE_ASSERT_OK(status);
  EXPECT_EQ(transfer.row_length, 0u);
  IREE_EXPECT_OK(iree_hal_file_transfer_validate({}, {}, &transfer));

  iree_unaligned_store_le_u64(&record.words[2], UINT64_C(0x0001000000000001));
  iree_unaligned_store_le_u64(&record.words[0], UINT64_MAX);
  iree_unaligned_store_le_u64(&record.words[1], UINT64_MAX);
  // Either zero count suppresses the transfer, even with invalid binding IDs.
  for (size_t count_word : {3u, 5u}) {
    iree_unaligned_store_le_u64(&record.words[count_word], UINT64_C(1) << 48);
    IREE_ASSERT_OK(iree_hal_file_transfer_record_decode(&record, &transfer));
    EXPECT_EQ(transfer.row_length, 0u);
    record.words[count_word] = 0;
  }
}

TEST(FileTransferRecord, RejectsNonemptyUnknownFormAndUnrepresentableOffset) {
  iree_hal_file_transfer_record_t record = {};
  iree_hal_file_transfer_t transfer;
  for (uint64_t form : {2u, 255u, 65535u}) {
    iree_unaligned_store_le_u64(&record.words[2], (form << 48) | 90);
    EXPECT_THAT(
        iree::Status(iree_hal_file_transfer_record_decode(&record, &transfer)),
        StatusIs(iree::StatusCode::kInvalidArgument));
  }
  iree_unaligned_store_le_u64(&record.words[1], UINT64_C(0x0000FFFFFFFFFFFF));
  iree_unaligned_store_le_u64(&record.words[2], 1);
  iree::Status status(iree_hal_file_transfer_record_decode(&record, &transfer));
  if (sizeof(iree_device_size_t) < sizeof(uint64_t)) {
    EXPECT_THAT(status, StatusIs(iree::StatusCode::kOutOfRange));
  } else {
    IREE_ASSERT_OK(status);
    EXPECT_EQ(transfer.buffer_offset, UINT64_C(0xFFFFFFFFFFFF));
  }
}

// Geometry consumes already-qualified window lengths; it never queries the
// resources again. Nonzero origins ensure these checks remain binding-relative.
class FileTransferBoundsTest : public ::testing::Test {
 protected:
  iree_status_t Validate(const iree_hal_file_transfer_t& transfer) {
    return iree_hal_file_transfer_validate({1, &file_}, {1, &buffer_},
                                           &transfer);
  }

  // Window inside a larger file, whose neighboring data remains inaccessible.
  iree_hal_file_binding_t file_ = {nullptr, 4096, 256};
  // Window inside a larger payload allocation, with independent geometry.
  iree_hal_buffer_binding_t buffer_ = {nullptr, 128, 256};
};

TEST_F(FileTransferBoundsTest, ChecksLastAddressedByteWithoutExpandingRows) {
  iree_hal_file_transfer_t transfer = {
      .file_offset = 7,
      .buffer_offset = 11,
      .row_length = 9,
      .row_count = 3,
      .slice_count = 4,
      .file_strides = {16, 64},
      .buffer_strides = {24, 48},
  };
  // File end 7+9+2*16+3*64=240; buffer end 11+9+2*24+3*48=212.
  file_.length = 240;
  buffer_.length = 212;
  IREE_ASSERT_OK(Validate(transfer));
  --file_.length;
  EXPECT_THAT(iree::Status(Validate(transfer)),
              StatusIs(iree::StatusCode::kOutOfRange));
  ++file_.length;
  --buffer_.length;
  EXPECT_THAT(iree::Status(Validate(transfer)),
              StatusIs(iree::StatusCode::kOutOfRange));
}

TEST_F(FileTransferBoundsTest, IgnoresUnusedStridesAndEmptyOperands) {
  iree_hal_file_transfer_t transfer = {
      .file_offset = 7,
      .buffer_offset = 11,
      .row_length = 90,
      .row_count = 1,
      .slice_count = 1,
      .file_strides = {UINT64_MAX, UINT64_MAX},
      .buffer_strides = {UINT64_MAX, UINT64_MAX},
  };
  IREE_ASSERT_OK(Validate(transfer));
  transfer.file_binding = UINT32_MAX;
  transfer.buffer_binding = UINT32_MAX;
  transfer.file_offset = UINT64_MAX;
  transfer.buffer_offset = IREE_DEVICE_SIZE_MAX;
  for (uint64_t* extent :
       {&transfer.row_length, &transfer.row_count, &transfer.slice_count}) {
    const uint64_t saved = *extent;
    *extent = 0;
    IREE_ASSERT_OK(Validate(transfer));
    *extent = saved;
  }
}

TEST_F(FileTransferBoundsTest, ChecksIndicesAndFullWidthArithmetic) {
  iree_hal_file_transfer_t transfer = {0, 0, 0, 0, 1, 1, 1, {}, {}};
  for (uint32_t* index : {&transfer.file_binding, &transfer.buffer_binding}) {
    *index = 1;
    EXPECT_THAT(iree::Status(Validate(transfer)),
                StatusIs(iree::StatusCode::kOutOfRange));
    *index = 0;
  }
  file_.length = UINT64_MAX;
  buffer_.length = IREE_DEVICE_SIZE_MAX;
  transfer.file_offset = UINT64_MAX - 1;
  transfer.buffer_offset = IREE_DEVICE_SIZE_MAX - 1;
  IREE_ASSERT_OK(Validate(transfer));
  transfer.row_length = 2;
  EXPECT_THAT(iree::Status(Validate(transfer)),
              StatusIs(iree::StatusCode::kOutOfRange));

  transfer.file_offset = transfer.buffer_offset = 0;
  transfer.row_length = 1;
  transfer.row_count = UINT64_MAX;
  transfer.file_strides.row_stride = 0;
  transfer.buffer_strides.row_stride = 0;
  IREE_ASSERT_OK(Validate(transfer));
  transfer.file_strides.row_stride = 2;
  EXPECT_THAT(iree::Status(Validate(transfer)),
              StatusIs(iree::StatusCode::kOutOfRange));
  transfer.file_strides.row_stride = 0;
  transfer.slice_count = UINT64_MAX;
  transfer.buffer_strides.slice_stride = 2;
  EXPECT_THAT(iree::Status(Validate(transfer)),
              StatusIs(iree::StatusCode::kOutOfRange));
}

// Each alignment can dominate the continuation stride independently. Small
// capacities exercise the same splitting as a native maximum request size.
class FileTransferTest
    : public ::testing::TestWithParam<iree_async_file_transfer_requirements_t> {
 protected:
  // Actual storage large enough for each prepared request.
  alignas(64) std::array<uint8_t, 256> storage_ = {};
};

TEST_P(FileTransferTest, SplitsWithoutLosingContinuationAlignment) {
  const auto requirements = GetParam();
  uint64_t offset = 128;
  size_t completed = 0;
  while (completed < storage_.size()) {
    size_t length = 0;
    IREE_ASSERT_OK(iree_hal_file_prepare_transfer(
        &requirements, offset, storage_.data() + completed,
        storage_.size() - completed, 63, &length));
    ASSERT_GT(length, 0u);
    ASSERT_LE(length, 63u);
    EXPECT_EQ(length, 32u);
    completed += length;
    offset += length;
  }
  EXPECT_EQ(completed, storage_.size());
}

TEST_P(FileTransferTest, ReusesAlignedStagingStorageForLargeLogicalRange) {
  const auto requirements = GetParam();
  const uint64_t remaining_length = (uint64_t{1} << 33) + 32;
  size_t length = 0;
  IREE_ASSERT_OK(iree_hal_file_prepare_transfer(
      &requirements, uint64_t{1} << 34, storage_.data(), remaining_length,
      storage_.size() - 1, &length));
  EXPECT_EQ(length, 224u);
  // The next chunk can reuse the same staging allocation; its logical file
  // offset, not the buffer's capacity, advances across the large file.
  IREE_ASSERT_OK(iree_hal_file_prepare_transfer(
      &requirements, (uint64_t{1} << 34) + length, storage_.data(),
      remaining_length - length, storage_.size() - 1, &length));
  EXPECT_EQ(length, 224u);
}

TEST_P(FileTransferTest, RejectsInsufficientCapacityWithoutProgress) {
  size_t length = 99;
  const auto requirements = GetParam();
  EXPECT_THAT(
      iree::Status(iree_hal_file_prepare_transfer(
          &requirements, 0, storage_.data(), storage_.size(), 31, &length)),
      StatusIs(iree::StatusCode::kResourceExhausted));
  EXPECT_EQ(length, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    IndependentAlignment, FileTransferTest,
    ::testing::Values(iree_async_file_transfer_requirements_t{32, 8, 4},
                      iree_async_file_transfer_requirements_t{8, 32, 4},
                      iree_async_file_transfer_requirements_t{8, 4, 32}));

TEST(FileTransfer, FinalChunkOnlyNeedsRequestAlignment) {
  alignas(64) uint8_t storage[64];
  const iree_async_file_transfer_requirements_t requirements = {64, 16, 4};
  size_t length = 0;
  IREE_ASSERT_OK(iree_hal_file_prepare_transfer(&requirements, 16, storage, 12,
                                                sizeof(storage), &length));
  EXPECT_EQ(length, 12u);
}

TEST(FileTransfer, ChecksEveryContinuationConstraint) {
  alignas(64) uint8_t storage[64];
  const iree_async_file_transfer_requirements_t requirements = {16, 8, 4};
  size_t length = 0;
  IREE_ASSERT_OK(iree_hal_file_prepare_transfer(&requirements, 0, storage, 64,
                                                64, &length));
  EXPECT_EQ(length, 64u);
  for (size_t completed : {1, 4, 8}) {
    EXPECT_THAT(iree::Status(iree_hal_file_prepare_transfer(
                    &requirements, completed, storage + completed,
                    64 - completed, 64 - completed, &length)),
                StatusIs(iree::StatusCode::kInvalidArgument));
    EXPECT_EQ(length, 0u);
  }
  IREE_ASSERT_OK(iree_hal_file_prepare_transfer(&requirements, 16, storage + 16,
                                                48, 48, &length));
  EXPECT_EQ(length, 48u);
}

TEST(FileTransfer, EmptyAndByteGranularRequests) {
  uint8_t storage[7];
  const iree_async_file_transfer_requirements_t requirements = {1, 1, 1};
  size_t length = 99;
  IREE_ASSERT_OK(
      iree_hal_file_prepare_transfer(&requirements, 3, nullptr, 0, 0, &length));
  EXPECT_EQ(length, 0u);
  IREE_ASSERT_OK(iree_hal_file_prepare_transfer(&requirements, 3, storage + 1,
                                                6, 5, &length));
  EXPECT_EQ(length, 5u);
  EXPECT_THAT(iree::Status(iree_hal_file_prepare_transfer(
                  &requirements, 0, nullptr, 1, 1, &length)),
              StatusIs(iree::StatusCode::kInvalidArgument));
  EXPECT_EQ(length, 0u);
}

}  // namespace
