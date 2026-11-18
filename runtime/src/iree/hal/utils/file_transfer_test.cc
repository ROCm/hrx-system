// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/file_transfer.h"

#include <array>

#include "iree/base/internal/debugging.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::iree::testing::status::StatusIs;

TEST(FileTransferIterator, CollapsesContiguousTerabyteGeometry) {
  // Only descriptors exist here: no payload allocation or expanded row array.
  // On a 32-bit build the same geometry covers one GiB within its size limit.
  const uint64_t length =
      sizeof(iree_device_size_t) == 8 ? UINT64_C(1) << 40 : UINT64_C(1) << 30;
  const iree_hal_file_binding_t file = {nullptr, 17, length + 9};
  const iree_hal_buffer_binding_t buffer = {
      nullptr, 13, static_cast<iree_device_size_t>(length + 11)};
  const iree_hal_file_transfer_t transfer = {
      .file_offset = 9,
      .buffer_offset = 11,
      .row_length = 4096,
      .row_count = 256,
      .slice_count = length / (4096 * 256),
      .file_strides = {4096, 4096 * 256},
      .buffer_strides = {4096, 4096 * 256},
  };
  const iree_hal_file_transfer_batch_t batch = {
      .files = {1, &file},
      .buffers = {1, &buffer},
      .transfers =
          {
              .type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST,
              .host = {1, &transfer},
          },
  };
  IREE_ASSERT_OK(
      iree_hal_file_transfer_validate(batch.files, batch.buffers, &transfer));
  iree_hal_file_transfer_iterator_t iterator;
  IREE_ASSERT_OK(iree_hal_file_transfer_iterator_initialize(&batch, 0, nullptr,
                                                            &iterator));
  iree_hal_file_transfer_range_t range;
  IREE_ASSERT_OK(iree_hal_file_transfer_iterator_peek(&iterator, &range));
  EXPECT_EQ(range.length, length);
  EXPECT_EQ(range.file_offset, 9u);
  EXPECT_EQ(range.buffer_offset, 11u);
  iree_hal_file_transfer_iterator_advance(&iterator, 123);
  IREE_ASSERT_OK(iree_hal_file_transfer_iterator_peek(&iterator, &range));
  EXPECT_EQ(range.length, length - 123);
  EXPECT_EQ(range.file_offset, 132u);
  EXPECT_EQ(range.buffer_offset, 134u);
  iree_hal_file_transfer_iterator_advance(&iterator, range.length);
  IREE_ASSERT_OK(iree_hal_file_transfer_iterator_peek(&iterator, &range));
  EXPECT_EQ(range.length, 0u);
}

TEST(FileTransferIterator, VisitsOnlyUsefulRowsAndConsumesPartialRanges) {
  const iree_hal_file_binding_t file = {nullptr, 17, 512};
  const iree_hal_buffer_binding_t buffer = {nullptr, 13, 512};
  const iree_hal_file_transfer_t transfers[] = {
      {.file_binding = UINT32_MAX, .row_length = 0},
      {
          .file_offset = 5,
          .buffer_offset = 7,
          .row_length = 9,
          .row_count = 3,
          .slice_count = 2,
          .file_strides = {11, 37},
          .buffer_strides = {13, 43},
      },
      {.file_binding = UINT32_MAX, .row_length = 7, .row_count = 0},
  };
  const iree_hal_file_transfer_batch_t batch = {
      .files = {1, &file},
      .buffers = {1, &buffer},
      .transfers =
          {
              .type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST,
              .host = {IREE_ARRAYSIZE(transfers), transfers},
          },
  };
  for (const auto& transfer : transfers) {
    IREE_ASSERT_OK(
        iree_hal_file_transfer_validate(batch.files, batch.buffers, &transfer));
  }
  iree_hal_file_transfer_iterator_t iterator;
  IREE_ASSERT_OK(iree_hal_file_transfer_iterator_initialize(&batch, 0, nullptr,
                                                            &iterator));
  iree_hal_file_transfer_range_t range;
  for (uint64_t slice = 0; slice < 2; ++slice) {
    for (uint64_t row = 0; row < 3; ++row) {
      for (uint64_t part = 0; part < 9; part += 3) {
        IREE_ASSERT_OK(iree_hal_file_transfer_iterator_peek(&iterator, &range));
        EXPECT_EQ(range.file_offset, 5 + slice * 37 + row * 11 + part);
        EXPECT_EQ(range.buffer_offset, 7 + slice * 43 + row * 13 + part);
        EXPECT_EQ(range.length, 9 - part);
        iree_hal_file_transfer_iterator_advance(&iterator, 3);
      }
    }
  }
  IREE_ASSERT_OK(iree_hal_file_transfer_iterator_peek(&iterator, &range));
  EXPECT_EQ(range.length, 0u);
}

TEST(FileTransferIterator, DecodesOnceAndRejectsMalformedLateRecords) {
  const iree_hal_file_binding_t file = {nullptr, 0, 512};
  const iree_hal_buffer_binding_t buffer = {nullptr, 0, 512};
  std::array<iree_hal_file_transfer_record_t, 3> records = {};
  iree_unaligned_store_le_u64(&records[0].words[0], 5);
  iree_unaligned_store_le_u64(&records[0].words[1], 7);
  iree_unaligned_store_le_u64(&records[0].words[2], 90);
  iree_unaligned_store_le_u64(&records[1].words[2], UINT64_C(2) << 48 | 1);
  const iree_hal_file_transfer_batch_t batch = {
      .files = {1, &file},
      .buffers = {1, &buffer},
      .transfers =
          {
              .type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT,
              .indirect = {{nullptr, 0, sizeof(records)}, nullptr, 0},
          },
  };
  iree_hal_file_transfer_iterator_t iterator;
  IREE_ASSERT_OK(iree_hal_file_transfer_iterator_initialize(
      &batch, 2, records.data(), &iterator));
  iree_hal_file_transfer_range_t range;
  IREE_ASSERT_OK(iree_hal_file_transfer_iterator_peek(&iterator, &range));
  EXPECT_EQ(range.length, 90u);
  IREE_ASAN_POISON_MEMORY_REGION(&records[0], sizeof(records[0]));
  iree_hal_file_transfer_iterator_advance(&iterator, 30);
  iree_status_t status =
      iree_hal_file_transfer_iterator_peek(&iterator, &range);
  IREE_ASAN_UNPOISON_MEMORY_REGION(&records[0], sizeof(records[0]));
  IREE_ASSERT_OK(status);
  EXPECT_EQ(range.length, 60u);
  EXPECT_EQ(range.file_offset, 35u);
  EXPECT_EQ(range.buffer_offset, 37u);
  iree_hal_file_transfer_iterator_advance(&iterator, 60);
  EXPECT_THAT(
      iree::Status(iree_hal_file_transfer_iterator_peek(&iterator, &range)),
      StatusIs(iree::StatusCode::kInvalidArgument));
  EXPECT_EQ(range.length, 0u);
  EXPECT_THAT(iree::Status(iree_hal_file_transfer_iterator_initialize(
                  &batch, 4, records.data(), &iterator)),
              StatusIs(iree::StatusCode::kOutOfRange));

  // A zero live count never inspects even the first record.
  IREE_ASSERT_OK(iree_hal_file_transfer_iterator_initialize(&batch, 0, nullptr,
                                                            &iterator));
  IREE_ASSERT_OK(iree_hal_file_transfer_iterator_peek(&iterator, &range));
  EXPECT_EQ(range.length, 0u);
}

TEST(FileTransferIterator, AcquiresBoundedPagesWithoutReadingPastTheirPrefix) {
  const iree_hal_file_binding_t file = {nullptr, 0, 512};
  const iree_hal_buffer_binding_t buffer = {nullptr, 0, 512};
  const iree_hal_file_transfer_batch_t batch = {
      .files = {1, &file},
      .buffers = {1, &buffer},
      .transfers =
          {
              .type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT,
              .indirect = {{nullptr, 0, 4 * 64}, nullptr, 0},
          },
  };
  iree_hal_file_transfer_iterator_t iterator;
  IREE_ASSERT_OK(iree_hal_file_transfer_iterator_initialize(&batch, 4, nullptr,
                                                            &iterator));
  iree_hal_file_transfer_range_t range;
  IREE_ASSERT_OK(iree_hal_file_transfer_iterator_peek(&iterator, &range));
  EXPECT_EQ(range.length, 0u);
  EXPECT_EQ(iterator.index, 0u);
  // Empty records at the page boundary cannot make peek read the next page.
  std::array<iree_hal_file_transfer_record_t, 2> page = {};
  iree_unaligned_store_le_u64(&page[0].words[2], 90);
  iree_hal_file_transfer_iterator_acquire(&iterator, page.data(), page.size());
  IREE_ASSERT_OK(iree_hal_file_transfer_iterator_peek(&iterator, &range));
  EXPECT_EQ(range.length, 90u);
  iree_hal_file_transfer_iterator_advance(&iterator, 90);
  IREE_ASSERT_OK(iree_hal_file_transfer_iterator_peek(&iterator, &range));
  EXPECT_EQ(range.length, 0u);
  EXPECT_EQ(iterator.index, 2u);
  page = {};
  iree_unaligned_store_le_u64(&page[1].words[0], 180);
  iree_unaligned_store_le_u64(&page[1].words[1], 270);
  iree_unaligned_store_le_u64(&page[1].words[2], 90);
  iree_hal_file_transfer_iterator_acquire(&iterator, page.data(), page.size());
  IREE_ASSERT_OK(iree_hal_file_transfer_iterator_peek(&iterator, &range));
  EXPECT_EQ(range.file_offset, 180u);
  EXPECT_EQ(range.buffer_offset, 270u);
  iree_hal_file_transfer_iterator_advance(&iterator, 90);
  IREE_ASSERT_OK(iree_hal_file_transfer_iterator_peek(&iterator, &range));
  EXPECT_EQ(range.length, 0u);
  EXPECT_EQ(iterator.index, iterator.count);
}

}  // namespace
