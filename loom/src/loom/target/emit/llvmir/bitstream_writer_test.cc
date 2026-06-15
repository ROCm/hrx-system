// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/bitstream_writer.h"

#include <memory>
#include <string>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

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

class LlvmIrBitstreamWriterTest : public testing::Test {
 protected:
  void SetUp() override {
    stream_ = CreateStream();
    loom_llvmir_bitstream_writer_initialize(stream_.get(), &writer_);
  }

  StreamPtr stream_ = StreamPtr(NULL, iree_io_stream_release);
  loom_llvmir_bitstream_writer_t writer_;
};

TEST_F(LlvmIrBitstreamWriterTest, WritesBitsLeastSignificantBitFirst) {
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_write_bits(&writer_, 0b101, 3));
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_write_bits(&writer_, 0b11, 2));
  EXPECT_EQ(loom_llvmir_bitstream_writer_bit_offset(&writer_), 5u);
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_finish(&writer_));
  EXPECT_EQ(loom_llvmir_bitstream_writer_bit_offset(&writer_), 8u);
  EXPECT_EQ(StreamBytes(stream_.get()), std::string("\x1D", 1));
}

TEST_F(LlvmIrBitstreamWriterTest, WritesAcrossByteBoundaries) {
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_write_bits(&writer_, 0xAB, 8));
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_write_bits(&writer_, 0xCD, 8));
  EXPECT_EQ(loom_llvmir_bitstream_writer_bit_offset(&writer_), 16u);
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_finish(&writer_));
  EXPECT_EQ(StreamBytes(stream_.get()), std::string("\xAB\xCD", 2));
}

TEST_F(LlvmIrBitstreamWriterTest, RejectsBitFieldsThatDoNotFit) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_llvmir_bitstream_writer_write_bits(&writer_, 0b100, 2));
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_finish(&writer_));
  EXPECT_EQ(StreamBytes(stream_.get()), std::string());
}

TEST_F(LlvmIrBitstreamWriterTest, WritesVariableBitRateFields) {
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_write_vbr(&writer_, 291, 6));
  EXPECT_EQ(loom_llvmir_bitstream_writer_bit_offset(&writer_), 12u);
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_finish(&writer_));
  EXPECT_EQ(StreamBytes(stream_.get()), std::string("\x63\x02", 2));
}

TEST_F(LlvmIrBitstreamWriterTest, AlignsToThirtyTwoBits) {
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_write_bits(&writer_, 1, 1));
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_align32(&writer_));
  EXPECT_EQ(loom_llvmir_bitstream_writer_bit_offset(&writer_), 32u);
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_finish(&writer_));
  EXPECT_EQ(StreamBytes(stream_.get()), std::string("\x01\x00\x00\x00", 4));
}

TEST_F(LlvmIrBitstreamWriterTest, WritesBytePayloadsOnByteBoundaries) {
  const uint8_t bytes[] = {0x41, 0x00, 0x42};
  IREE_ASSERT_OK(
      loom_llvmir_bitstream_writer_write_bytes(&writer_, bytes, sizeof(bytes)));
  EXPECT_EQ(loom_llvmir_bitstream_writer_bit_offset(&writer_), 24u);
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_finish(&writer_));
  EXPECT_EQ(StreamBytes(stream_.get()), std::string("A\0B", 3));
}

TEST_F(LlvmIrBitstreamWriterTest, WritesBytePayloadsAtBitOffsets) {
  const uint8_t byte = 0x02;
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_write_bits(&writer_, 1, 1));
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_write_bytes(&writer_, &byte, 1));
  EXPECT_EQ(loom_llvmir_bitstream_writer_bit_offset(&writer_), 9u);
  IREE_ASSERT_OK(loom_llvmir_bitstream_writer_finish(&writer_));
  EXPECT_EQ(StreamBytes(stream_.get()), std::string("\x05\x00", 2));
}

}  // namespace
}  // namespace loom
