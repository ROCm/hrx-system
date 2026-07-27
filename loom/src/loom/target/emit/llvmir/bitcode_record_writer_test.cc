// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/bitcode_record_writer.h"

#include <memory>
#include <string>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/emit/llvmir/bitcode_format.h"

namespace loom {
namespace {

using StreamPtr =
    std::unique_ptr<iree_io_stream_t, void (*)(iree_io_stream_t*)>;

StreamPtr CreateStream() {
  iree_io_stream_t* stream = NULL;
  IREE_CHECK_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE,
      1024, iree_allocator_system(), &stream));
  return StreamPtr(stream, iree_io_stream_release);
}

std::string StreamBytes(iree_io_stream_t* stream) {
  iree_io_stream_pos_t length = iree_io_stream_length(stream);
  IREE_ASSERT_GE(length, 0);
  std::string bytes((size_t)length, '\0');
  IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_CHECK_OK(iree_io_stream_read(stream, bytes.size(), bytes.data(), NULL));
  return bytes;
}

class LlvmIrBitcodeRecordWriterTest : public testing::Test {
 protected:
  void SetUp() override {
    stream_ = CreateStream();
    loom_llvmir_bitstream_writer_initialize(stream_.get(), &bitstream_);
    loom_llvmir_bitcode_record_writer_initialize(&bitstream_, &record_writer_);
  }

  StreamPtr stream_ = StreamPtr(NULL, iree_io_stream_release);
  loom_llvmir_bitstream_writer_t bitstream_;
  loom_llvmir_bitcode_record_writer_t record_writer_;
};

TEST_F(LlvmIrBitcodeRecordWriterTest, PatchesSubblockLengthOnExit) {
  IREE_ASSERT_OK(loom_llvmir_bitcode_record_writer_enter_subblock(
      &record_writer_, LOOM_LLVMIR_BITCODE_MODULE_BLOCK, 3));
  IREE_ASSERT_OK(loom_llvmir_bitcode_record_writer_exit_block(&record_writer_));
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_finish(&bitstream_));

  std::string bytes = StreamBytes(stream_.get());
  ASSERT_EQ(bytes.size(), 12u);
  EXPECT_EQ((uint8_t)bytes[0], 0x21);
  EXPECT_EQ((uint8_t)bytes[1], 0x0C);
  EXPECT_EQ((uint8_t)bytes[2], 0x00);
  EXPECT_EQ((uint8_t)bytes[3], 0x00);
  EXPECT_EQ((uint8_t)bytes[4], 0x01);
  EXPECT_EQ((uint8_t)bytes[5], 0x00);
  EXPECT_EQ((uint8_t)bytes[6], 0x00);
  EXPECT_EQ((uint8_t)bytes[7], 0x00);
}

TEST_F(LlvmIrBitcodeRecordWriterTest, WritesUnabbreviatedRecords) {
  uint64_t operands[] = {7, 63, 64};
  IREE_ASSERT_OK(loom_llvmir_bitcode_record_writer_enter_subblock(
      &record_writer_, LOOM_LLVMIR_BITCODE_MODULE_BLOCK, 3));
  IREE_ASSERT_OK(loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      &record_writer_, 5, operands, IREE_ARRAYSIZE(operands)));
  IREE_ASSERT_OK(loom_llvmir_bitcode_record_writer_exit_block(&record_writer_));
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_finish(&bitstream_));

  std::string bytes = StreamBytes(stream_.get());
  ASSERT_GE(bytes.size(), 12u);
  EXPECT_EQ((uint8_t)bytes[4], 0x02);
  EXPECT_EQ((uint8_t)bytes[5], 0x00);
  EXPECT_EQ((uint8_t)bytes[6], 0x00);
  EXPECT_EQ((uint8_t)bytes[7], 0x00);
}

TEST_F(LlvmIrBitcodeRecordWriterTest, WritesBlobRecords) {
  IREE_ASSERT_OK(loom_llvmir_bitcode_record_writer_enter_subblock(
      &record_writer_, LOOM_LLVMIR_BITCODE_STRTAB_BLOCK, 3));
  uint32_t blob_abbrev = 0;
  IREE_ASSERT_OK(loom_llvmir_bitcode_record_writer_define_blob_abbrev(
      &record_writer_, LOOM_LLVMIR_BITCODE_STRTAB_CODE_BLOB, &blob_abbrev));
  EXPECT_EQ(blob_abbrev, LOOM_LLVMIR_BITCODE_ABBREV_FIRST_APPLICATION);
  IREE_ASSERT_OK(loom_llvmir_bitcode_record_writer_write_blob_record(
      &record_writer_, blob_abbrev, IREE_SV("abc")));
  IREE_ASSERT_OK(loom_llvmir_bitcode_record_writer_exit_block(&record_writer_));
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_finish(&bitstream_));

  std::string bytes = StreamBytes(stream_.get());
  ASSERT_GE(bytes.size(), 16u);
  EXPECT_NE(bytes.find("abc"), std::string::npos);
}

TEST_F(LlvmIrBitcodeRecordWriterTest, RejectsUnbalancedExit) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_llvmir_bitcode_record_writer_exit_block(&record_writer_));
}

}  // namespace
}  // namespace loom
