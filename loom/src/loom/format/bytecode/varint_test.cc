// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/varint.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Unsigned varint length
//===----------------------------------------------------------------------===//

TEST(UvarintLength, SingleByte) {
  EXPECT_EQ(loom_uvarint_length(0), 1u);
  EXPECT_EQ(loom_uvarint_length(1), 1u);
  EXPECT_EQ(loom_uvarint_length(0x7F), 1u);
}

TEST(UvarintLength, TwoBytes) {
  EXPECT_EQ(loom_uvarint_length(0x80), 2u);
  EXPECT_EQ(loom_uvarint_length(0x3FFF), 2u);
}

TEST(UvarintLength, FiveBytes) {
  EXPECT_EQ(loom_uvarint_length(UINT32_MAX), 5u);
}

TEST(UvarintLength, TenBytes) {
  EXPECT_EQ(loom_uvarint_length(UINT64_MAX), 10u);
}

//===----------------------------------------------------------------------===//
// Signed varint length
//===----------------------------------------------------------------------===//

TEST(SvarintLength, Zero) { EXPECT_EQ(loom_svarint_length(0), 1u); }

TEST(SvarintLength, SmallPositive) {
  // Zigzag: 1 -> 2 (single byte).
  EXPECT_EQ(loom_svarint_length(1), 1u);
  // Zigzag: 63 -> 126 (single byte, max for 1-byte).
  EXPECT_EQ(loom_svarint_length(63), 1u);
  // Zigzag: 64 -> 128 (needs 2 bytes).
  EXPECT_EQ(loom_svarint_length(64), 2u);
}

TEST(SvarintLength, SmallNegative) {
  // Zigzag: -1 -> 1 (single byte).
  EXPECT_EQ(loom_svarint_length(-1), 1u);
  // Zigzag: -64 -> 127 (single byte, max negative for 1-byte).
  EXPECT_EQ(loom_svarint_length(-64), 1u);
  // Zigzag: -65 -> 129 (needs 2 bytes).
  EXPECT_EQ(loom_svarint_length(-65), 2u);
}

TEST(SvarintLength, Extremes) {
  EXPECT_EQ(loom_svarint_length(INT64_MAX), 10u);
  EXPECT_EQ(loom_svarint_length(INT64_MIN), 10u);
}

//===----------------------------------------------------------------------===//
// Unsigned varint encode + decode round-trip
//===----------------------------------------------------------------------===//

class UvarintRoundTripTest : public ::testing::TestWithParam<uint64_t> {};

TEST_P(UvarintRoundTripTest, RoundTrip) {
  uint64_t value = GetParam();

  // Encode.
  uint8_t raw_buffer[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t buffer = {raw_buffer, sizeof(raw_buffer)};
  iree_host_size_t encoded_length = 0;
  IREE_ASSERT_OK(loom_uvarint_encode(value, buffer, &encoded_length));
  ASSERT_GT(encoded_length, 0u);
  ASSERT_LE(encoded_length, LOOM_VARINT_MAX_LENGTH);

  // Verify encoded length matches prediction.
  EXPECT_EQ(encoded_length, loom_uvarint_length(value));

  // Decode.
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(raw_buffer, encoded_length, &cursor);
  uint64_t decoded = UINT64_MAX;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &decoded));
  EXPECT_EQ(decoded, value);
  EXPECT_EQ(cursor.position, encoded_length);
}

INSTANTIATE_TEST_SUITE_P(UvarintValues, UvarintRoundTripTest,
                         ::testing::Values(
                             // Boundary: 1-byte range [0, 127].
                             0ull, 1ull, 0x7Full,
                             // Boundary: 2-byte range [128, 16383].
                             0x80ull, 0x3FFFull,
                             // Boundary: 3-byte range.
                             0x4000ull, 0x1FFFFFull,
                             // Boundary: 4-byte range.
                             0x200000ull, 0x0FFFFFFFull,
                             // Boundary: 5-byte range (includes UINT32_MAX).
                             0x10000000ull, (uint64_t)UINT32_MAX,
                             // Large values.
                             (uint64_t)UINT32_MAX + 1, UINT64_MAX - 1,
                             UINT64_MAX));

//===----------------------------------------------------------------------===//
// Signed varint encode + decode round-trip
//===----------------------------------------------------------------------===//

class SvarintRoundTripTest : public ::testing::TestWithParam<int64_t> {};

TEST_P(SvarintRoundTripTest, RoundTrip) {
  int64_t value = GetParam();

  // Encode.
  uint8_t raw_buffer[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t buffer = {raw_buffer, sizeof(raw_buffer)};
  iree_host_size_t encoded_length = 0;
  IREE_ASSERT_OK(loom_svarint_encode(value, buffer, &encoded_length));
  ASSERT_GT(encoded_length, 0u);
  ASSERT_LE(encoded_length, LOOM_VARINT_MAX_LENGTH);

  // Verify encoded length matches prediction.
  EXPECT_EQ(encoded_length, loom_svarint_length(value));

  // Decode.
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(raw_buffer, encoded_length, &cursor);
  int64_t decoded = 0;
  IREE_ASSERT_OK(loom_svarint_decode(&cursor, &decoded));
  EXPECT_EQ(decoded, value);
  EXPECT_EQ(cursor.position, encoded_length);
}

INSTANTIATE_TEST_SUITE_P(
    SvarintValues, SvarintRoundTripTest,
    ::testing::Values(
        // Zero.
        0ll,
        // Small magnitudes (1-byte zigzag range: [-64, 63]).
        1ll, -1ll, 63ll, -64ll,
        // 2-byte boundary.
        64ll, -65ll,
        // Larger values.
        1000ll, -1000ll, (int64_t)INT32_MAX, (int64_t)INT32_MIN,
        // Extremes.
        INT64_MAX, INT64_MIN));

//===----------------------------------------------------------------------===//
// Zigzag encoding correctness
//===----------------------------------------------------------------------===//

TEST(SvarintZigzag, MappingIsCorrect) {
  // Verify the specific zigzag mapping:
  //   0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, -3 -> 5, 3 -> 6.
  auto zigzag = [](int64_t v) -> uint64_t {
    return ((uint64_t)v << 1) ^ (uint64_t)(v >> 63);
  };
  EXPECT_EQ(zigzag(0), 0u);
  EXPECT_EQ(zigzag(-1), 1u);
  EXPECT_EQ(zigzag(1), 2u);
  EXPECT_EQ(zigzag(-2), 3u);
  EXPECT_EQ(zigzag(2), 4u);
  EXPECT_EQ(zigzag(-3), 5u);
  EXPECT_EQ(zigzag(3), 6u);

  // Verify these small values encode as single bytes (all < 128).
  for (int64_t v = -64; v <= 63; ++v) {
    EXPECT_LT(zigzag(v), 128u)
        << "zigzag(" << v << ") = " << zigzag(v) << " should be < 128";
  }
  // And the next values outside that range need 2 bytes.
  EXPECT_GE(zigzag(64), 128u);
  EXPECT_GE(zigzag(-65), 128u);
}

//===----------------------------------------------------------------------===//
// Encode buffer bounds checking
//===----------------------------------------------------------------------===//

TEST(UvarintEncode, BufferTooSmall) {
  // Value 128 needs 2 bytes. Buffer has 1.
  uint8_t raw_buffer[1];
  iree_byte_span_t buffer = {raw_buffer, sizeof(raw_buffer)};
  iree_host_size_t length = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_uvarint_encode(128, buffer, &length));
}

TEST(UvarintEncode, BufferExactSize) {
  // Value 128 needs exactly 2 bytes. Buffer has 2.
  uint8_t raw_buffer[2];
  iree_byte_span_t buffer = {raw_buffer, sizeof(raw_buffer)};
  iree_host_size_t length = 0;
  IREE_ASSERT_OK(loom_uvarint_encode(128, buffer, &length));
  EXPECT_EQ(length, 2u);
}

TEST(UvarintEncode, ZeroLengthBuffer) {
  iree_byte_span_t buffer = iree_byte_span_empty();
  iree_host_size_t length = 0;
  // Even value 0 needs 1 byte.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_uvarint_encode(0, buffer, &length));
}

//===----------------------------------------------------------------------===//
// Decode error cases
//===----------------------------------------------------------------------===//

TEST(UvarintDecode, EmptyData) {
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(NULL, 0, &cursor);
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_uvarint_decode(&cursor, &value));
  // Cursor unchanged on failure.
  EXPECT_EQ(cursor.position, 0u);
}

TEST(UvarintDecode, TruncatedVarint) {
  // A byte with continuation bit set, then nothing.
  uint8_t data[] = {0x80};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_uvarint_decode(&cursor, &value));
  EXPECT_EQ(cursor.position, 0u);
}

TEST(UvarintDecode, TruncatedMultiByte) {
  // 3 continuation bytes, then data ends.
  uint8_t data[] = {0x80, 0x80, 0x80};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_uvarint_decode(&cursor, &value));
  EXPECT_EQ(cursor.position, 0u);
}

TEST(UvarintDecode, NonCanonicalZero) {
  uint8_t data[] = {0x80, 0x00};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_uvarint_decode(&cursor, &value));
  EXPECT_EQ(cursor.position, 0u);
}

TEST(UvarintDecode, NonCanonicalSmallValue) {
  uint8_t data[] = {0x81, 0x00};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_uvarint_decode(&cursor, &value));
  EXPECT_EQ(cursor.position, 0u);
}

TEST(SvarintDecode, NonCanonicalFails) {
  uint8_t data[] = {0x80, 0x00};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  int64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_svarint_decode(&cursor, &value));
  EXPECT_EQ(cursor.position, 0u);
}

TEST(UvarintDecode, Overflow10thByteTooLarge) {
  // 9 continuation bytes followed by a 10th byte with value > 1.
  // This would decode to a value > UINT64_MAX.
  uint8_t data[] = {0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_uvarint_decode(&cursor, &value));
  EXPECT_EQ(cursor.position, 0u);
}

TEST(UvarintDecode, OverflowTooManyBytes) {
  // 11 continuation bytes — exceeds maximum varint length.
  uint8_t data[] = {0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
                    0x80, 0x80, 0x80, 0x80, 0x00};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_uvarint_decode(&cursor, &value));
  EXPECT_EQ(cursor.position, 0u);
}

TEST(UvarintDecode, MaxValidValue) {
  // UINT64_MAX = 0xFFFFFFFFFFFFFFFF. Zigzag-free: all 64 bits set.
  // LEB128: 9 bytes of 0xFF followed by 0x01.
  uint8_t data[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  uint64_t value = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &value));
  EXPECT_EQ(value, UINT64_MAX);
  EXPECT_EQ(cursor.position, 10u);
}

//===----------------------------------------------------------------------===//
// Cursor fixed-width reads
//===----------------------------------------------------------------------===//

TEST(CursorReadU8, Basic) {
  uint8_t data[] = {0x42, 0xFF};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);

  uint8_t value = 0;
  IREE_ASSERT_OK(loom_bytecode_cursor_read_u8(&cursor, &value));
  EXPECT_EQ(value, 0x42);
  EXPECT_EQ(cursor.position, 1u);

  IREE_ASSERT_OK(loom_bytecode_cursor_read_u8(&cursor, &value));
  EXPECT_EQ(value, 0xFF);
  EXPECT_EQ(cursor.position, 2u);
}

TEST(CursorReadU8, PastEnd) {
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(NULL, 0, &cursor);
  uint8_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_bytecode_cursor_read_u8(&cursor, &value));
}

TEST(CursorReadU16, LittleEndian) {
  uint8_t data[] = {0x34, 0x12};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  uint16_t value = 0;
  IREE_ASSERT_OK(loom_bytecode_cursor_read_u16_le(&cursor, &value));
  EXPECT_EQ(value, 0x1234);
}

TEST(CursorReadU16, InsufficientData) {
  uint8_t data[] = {0x34};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  uint16_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_bytecode_cursor_read_u16_le(&cursor, &value));
}

TEST(CursorReadU16, PositionOverflowFailsWithoutAdvancing) {
  uint8_t data[] = {0x34};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  cursor.position = IREE_HOST_SIZE_MAX - 1;
  uint16_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_bytecode_cursor_read_u16_le(&cursor, &value));
  EXPECT_EQ(cursor.position, IREE_HOST_SIZE_MAX - 1);
}

TEST(CursorReadU32, LittleEndian) {
  uint8_t data[] = {0x78, 0x56, 0x34, 0x12};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  uint32_t value = 0;
  IREE_ASSERT_OK(loom_bytecode_cursor_read_u32_le(&cursor, &value));
  EXPECT_EQ(value, 0x12345678u);
}

TEST(CursorReadU64, LittleEndian) {
  uint8_t data[] = {0xEF, 0xCD, 0xAB, 0x90, 0x78, 0x56, 0x34, 0x12};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  uint64_t value = 0;
  IREE_ASSERT_OK(loom_bytecode_cursor_read_u64_le(&cursor, &value));
  EXPECT_EQ(value, 0x1234567890ABCDEFull);
}

//===----------------------------------------------------------------------===//
// Cursor span and skip
//===----------------------------------------------------------------------===//

TEST(CursorReadSpan, Basic) {
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);

  iree_const_byte_span_t span = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_bytecode_cursor_read_span(&cursor, 3, &span));
  EXPECT_EQ(span.data, data);
  EXPECT_EQ(span.data_length, 3u);
  EXPECT_EQ(cursor.position, 3u);
}

TEST(CursorReadSpan, PastEnd) {
  uint8_t data[] = {0x01};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  iree_const_byte_span_t span = iree_const_byte_span_empty();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_bytecode_cursor_read_span(&cursor, 2, &span));
  EXPECT_EQ(cursor.position, 0u);
}

TEST(CursorSkip, Basic) {
  uint8_t data[] = {0x01, 0x02, 0x03};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  IREE_ASSERT_OK(loom_bytecode_cursor_skip(&cursor, 2));
  EXPECT_EQ(cursor.position, 2u);
  uint8_t value = 0;
  IREE_ASSERT_OK(loom_bytecode_cursor_read_u8(&cursor, &value));
  EXPECT_EQ(value, 0x03);
}

TEST(CursorSkip, PastEnd) {
  uint8_t data[] = {0x01};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_bytecode_cursor_skip(&cursor, 2));
  EXPECT_EQ(cursor.position, 0u);
}

//===----------------------------------------------------------------------===//
// Cursor helpers
//===----------------------------------------------------------------------===//

TEST(CursorState, RemainingAndEmpty) {
  uint8_t data[] = {0x01, 0x02};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  EXPECT_EQ(loom_bytecode_cursor_remaining(&cursor), 2u);
  EXPECT_FALSE(loom_bytecode_cursor_is_empty(&cursor));

  uint8_t v;
  IREE_ASSERT_OK(loom_bytecode_cursor_read_u8(&cursor, &v));
  EXPECT_EQ(loom_bytecode_cursor_remaining(&cursor), 1u);

  IREE_ASSERT_OK(loom_bytecode_cursor_read_u8(&cursor, &v));
  EXPECT_EQ(loom_bytecode_cursor_remaining(&cursor), 0u);
  EXPECT_TRUE(loom_bytecode_cursor_is_empty(&cursor));
}

TEST(CursorState, InvalidPositionHasNoRemainingBytes) {
  uint8_t data[] = {0x01};
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(data, sizeof(data), &cursor);
  cursor.position = 2;
  EXPECT_EQ(loom_bytecode_cursor_remaining(&cursor), 0u);
  EXPECT_TRUE(loom_bytecode_cursor_is_empty(&cursor));
  EXPECT_FALSE(loom_bytecode_cursor_has_bytes(&cursor, 1));
}

//===----------------------------------------------------------------------===//
// Sequential decode from a stream of varints
//===----------------------------------------------------------------------===//

TEST(UvarintDecode, MultipleSequential) {
  // Encode several values into a single buffer, then decode them all.
  uint8_t raw_buffer[64];
  iree_host_size_t total_length = 0;

  uint64_t values[] = {0, 1, 127, 128, 16383, 16384, UINT32_MAX, UINT64_MAX};
  const iree_host_size_t value_count = sizeof(values) / sizeof(values[0]);

  // Encode all values sequentially.
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    iree_byte_span_t remaining = {raw_buffer + total_length,
                                  sizeof(raw_buffer) - total_length};
    iree_host_size_t length = 0;
    IREE_ASSERT_OK(loom_uvarint_encode(values[i], remaining, &length));
    total_length += length;
  }

  // Decode all values sequentially.
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(raw_buffer, total_length, &cursor);
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    uint64_t decoded = 0;
    IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &decoded));
    EXPECT_EQ(decoded, values[i]) << "mismatch at index " << i;
  }
  EXPECT_TRUE(loom_bytecode_cursor_is_empty(&cursor));
}

TEST(SvarintDecode, MultipleSequential) {
  uint8_t raw_buffer[64];
  iree_host_size_t total_length = 0;

  int64_t values[] = {0,   1,         -1,        63,        -64,      64,
                      -65, INT32_MAX, INT32_MIN, INT64_MAX, INT64_MIN};
  const iree_host_size_t value_count = sizeof(values) / sizeof(values[0]);

  for (iree_host_size_t i = 0; i < value_count; ++i) {
    iree_byte_span_t remaining = {raw_buffer + total_length,
                                  sizeof(raw_buffer) - total_length};
    iree_host_size_t length = 0;
    IREE_ASSERT_OK(loom_svarint_encode(values[i], remaining, &length));
    total_length += length;
  }

  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(raw_buffer, total_length, &cursor);
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    int64_t decoded = 0;
    IREE_ASSERT_OK(loom_svarint_decode(&cursor, &decoded));
    EXPECT_EQ(decoded, values[i]) << "mismatch at index " << i;
  }
  EXPECT_TRUE(loom_bytecode_cursor_is_empty(&cursor));
}

}  // namespace
}  // namespace loom
