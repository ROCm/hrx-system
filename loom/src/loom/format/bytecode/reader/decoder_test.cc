// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/decoder.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

struct CapturedDiagnostic {
  // Structured error emitted by the decoder.
  const loom_error_def_t* error;
  // Absolute origin range emitted by the decoder.
  loom_source_range_t origin;
};

static iree_status_t CaptureDiagnostic(void* user_data,
                                       const loom_diagnostic_t* diagnostic) {
  CapturedDiagnostic* captured = static_cast<CapturedDiagnostic*>(user_data);
  captured->error = diagnostic->error;
  captured->origin = diagnostic->origin;
  return iree_ok_status();
}

static iree_status_t FailDiagnostic(void* user_data,
                                    const loom_diagnostic_t* diagnostic) {
  (void)user_data;
  (void)diagnostic;
  return iree_make_status(IREE_STATUS_DATA_LOSS, "diagnostic sink failure");
}

static iree_status_t DeferDiagnostic(void* user_data,
                                     const loom_diagnostic_t* diagnostic) {
  (void)user_data;
  (void)diagnostic;
  return iree_status_from_code(IREE_STATUS_DEFERRED);
}

class BytecodeDecoderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{CaptureDiagnostic, &captured_},
        IREE_SV("decoder_test.loombc"), &error_count_, &decoder_);
  }

  loom_bytecode_reader_decoder_t decoder_ = {};
  uint32_t error_count_ = 0;
  CapturedDiagnostic captured_ = {};
};

TEST_F(BytecodeDecoderTest, ReadsCanonicalScalarsAndVarints) {
  const uint8_t data[] = {
      0xAB, 0x34, 0x12, 0x78, 0x56, 0x34, 0x12, 0xEF, 0xCD, 0xAB,
      0x89, 0x67, 0x45, 0x23, 0x01, 0xAC, 0x02, 0x01, 0xDE, 0xAD,
  };
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(data, sizeof(data), 4096,
                                         IREE_SV("TEST"), &cursor);

  uint8_t value8 = 0;
  uint16_t value16 = 0;
  uint32_t value32 = 0;
  uint64_t value64 = 0;
  uint64_t unsigned_varint = 0;
  int64_t signed_varint = 0;
  iree_const_byte_span_t tail = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_bytecode_reader_read_u8(&decoder_, &cursor, &value8));
  IREE_ASSERT_OK(
      loom_bytecode_reader_read_u16_le(&decoder_, &cursor, &value16));
  IREE_ASSERT_OK(
      loom_bytecode_reader_read_u32_le(&decoder_, &cursor, &value32));
  IREE_ASSERT_OK(
      loom_bytecode_reader_read_u64_le(&decoder_, &cursor, &value64));
  IREE_ASSERT_OK(
      loom_bytecode_reader_read_uvarint(&decoder_, &cursor, &unsigned_varint));
  IREE_ASSERT_OK(
      loom_bytecode_reader_read_svarint(&decoder_, &cursor, &signed_varint));
  IREE_ASSERT_OK(loom_bytecode_reader_read_span(&decoder_, &cursor, 2, &tail));
  IREE_ASSERT_OK(
      loom_bytecode_reader_expect_empty(&decoder_, &cursor, IREE_SV("TEST")));

  EXPECT_EQ(value8, 0xABu);
  EXPECT_EQ(value16, 0x1234u);
  EXPECT_EQ(value32, 0x12345678u);
  EXPECT_EQ(value64, UINT64_C(0x0123456789ABCDEF));
  EXPECT_EQ(unsigned_varint, 300u);
  EXPECT_EQ(signed_varint, -1);
  ASSERT_EQ(tail.data_length, 2u);
  EXPECT_EQ(tail.data[0], 0xDEu);
  EXPECT_EQ(tail.data[1], 0xADu);
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeDecoderTest, TruncatedReadPreservesCursorAndAbsoluteOffset) {
  const uint8_t data[] = {0xAA};
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(data, sizeof(data), 4096,
                                         IREE_SV("TEST"), &cursor);

  uint16_t value = 0xCAFE;
  iree_status_t status =
      loom_bytecode_reader_read_u16_le(&decoder_, &cursor, &value);
  EXPECT_EQ(status, iree_status_from_code(IREE_STATUS_DEFERRED));
  iree_status_ignore(status);

  EXPECT_EQ(cursor.cursor.position, 0u);
  EXPECT_EQ(value, 0xCAFEu);
  EXPECT_EQ(error_count_, 1u);
  ASSERT_NE(captured_.error, nullptr);
  EXPECT_EQ(loom_error_def_domain(captured_.error), LOOM_ERROR_DOMAIN_BYTECODE);
  EXPECT_EQ(loom_error_def_code(captured_.error), 3u);
  EXPECT_EQ(captured_.origin.start, 4096u);
  EXPECT_EQ(captured_.origin.end, 4096u);
}

TEST_F(BytecodeDecoderTest, InvalidVarintPreservesCursorAndReportsStart) {
  const uint8_t data[] = {0x80, 0x00};
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(data, sizeof(data), 8192,
                                         IREE_SV("TEST"), &cursor);

  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED, loom_bytecode_reader_read_uvarint(
                                                  &decoder_, &cursor, &value));

  EXPECT_EQ(cursor.cursor.position, 0u);
  EXPECT_EQ(error_count_, 1u);
  ASSERT_NE(captured_.error, nullptr);
  EXPECT_EQ(loom_error_def_domain(captured_.error), LOOM_ERROR_DOMAIN_BYTECODE);
  EXPECT_EQ(loom_error_def_code(captured_.error), 8u);
  EXPECT_EQ(captured_.origin.start, 8192u);
}

TEST_F(BytecodeDecoderTest, TrailingBytesReportCurrentAbsoluteOffset) {
  const uint8_t data[] = {0x01, 0x02};
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(data, sizeof(data), 12288,
                                         IREE_SV("TEST"), &cursor);
  uint8_t value = 0;
  IREE_ASSERT_OK(loom_bytecode_reader_read_u8(&decoder_, &cursor, &value));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEFERRED,
      loom_bytecode_reader_expect_empty(&decoder_, &cursor, IREE_SV("table")));

  EXPECT_EQ(error_count_, 1u);
  ASSERT_NE(captured_.error, nullptr);
  EXPECT_EQ(loom_error_def_domain(captured_.error), LOOM_ERROR_DOMAIN_BYTECODE);
  EXPECT_EQ(loom_error_def_code(captured_.error), 6u);
  EXPECT_EQ(captured_.origin.start, 12289u);
  EXPECT_EQ(captured_.origin.end, 12290u);
}

TEST_F(BytecodeDecoderTest, RangeValidationRejectsOverflow) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEFERRED,
      loom_bytecode_reader_validate_range(&decoder_, IREE_SV("body"),
                                          UINT64_MAX - 3, 8, UINT64_MAX));

  EXPECT_EQ(error_count_, 1u);
  ASSERT_NE(captured_.error, nullptr);
  EXPECT_EQ(loom_error_def_domain(captured_.error), LOOM_ERROR_DOMAIN_BYTECODE);
  EXPECT_EQ(loom_error_def_code(captured_.error), 7u);
}

TEST_F(BytecodeDecoderTest, DiagnosticSinkFailurePropagates) {
  loom_bytecode_reader_decoder_initialize(
      loom_diagnostic_sink_t{FailDiagnostic, nullptr},
      IREE_SV("decoder_test.loombc"), &error_count_, &decoder_);
  const uint8_t data[] = {0xAA};
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(data, sizeof(data), 0, IREE_SV("TEST"),
                                         &cursor);

  uint16_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, loom_bytecode_reader_read_u16_le(
                                                   &decoder_, &cursor, &value));

  EXPECT_EQ(error_count_, 1u);
}

TEST_F(BytecodeDecoderTest, DiagnosticSinkCannotReturnReservedDeferred) {
  loom_bytecode_reader_decoder_initialize(
      loom_diagnostic_sink_t{DeferDiagnostic, nullptr},
      IREE_SV("decoder_test.loombc"), &error_count_, &decoder_);
  const uint8_t data[] = {0xAA};
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(data, sizeof(data), 0, IREE_SV("TEST"),
                                         &cursor);

  uint16_t value = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_bytecode_reader_read_u16_le(&decoder_, &cursor, &value));

  EXPECT_EQ(error_count_, 1u);
}

}  // namespace
}  // namespace loom
