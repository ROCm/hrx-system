// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/alignment.h"

#include "iree/testing/gtest.h"

namespace {

TEST(AlignmentTest, GenericLittleEndianRoundTrip) {
  int8_t i8 = 0;
  uint8_t u8 = 0;
  int16_t i16 = 0;
  uint16_t u16 = 0;
  int32_t i32 = 0;
  uint32_t u32 = 0;
  int64_t i64 = 0;
  uint64_t u64 = 0;
  float f32 = 0.0f;
  double f64 = 0.0;

  iree_unaligned_store_le(&i8, 0x12u);
  iree_unaligned_store_le(&u8, 0x23u);
  iree_unaligned_store_le(&i16, 0x1234u);
  iree_unaligned_store_le(&u16, 0x2345u);
  iree_unaligned_store_le(&i32, 0x12345678u);
  iree_unaligned_store_le(&u32, 0x23456789u);
  iree_unaligned_store_le(&i64, UINT64_C(0x123456789ABCDEF));
  iree_unaligned_store_le(&u64, UINT64_C(0x23456789ABCDEF01));
  iree_unaligned_store_le(&f32, 1.25f);
  iree_unaligned_store_le(&f64, 2.5);

  EXPECT_EQ(0x12u, iree_unaligned_load_le(&i8));
  EXPECT_EQ(0x23u, iree_unaligned_load_le(&u8));
  EXPECT_EQ(0x1234u, iree_unaligned_load_le(&i16));
  EXPECT_EQ(0x2345u, iree_unaligned_load_le(&u16));
  EXPECT_EQ(0x12345678u, iree_unaligned_load_le(&i32));
  EXPECT_EQ(0x23456789u, iree_unaligned_load_le(&u32));
  EXPECT_EQ(UINT64_C(0x123456789ABCDEF), iree_unaligned_load_le(&i64));
  EXPECT_EQ(UINT64_C(0x23456789ABCDEF01), iree_unaligned_load_le(&u64));
  EXPECT_EQ(1.25f, iree_unaligned_load_le(&f32));
  EXPECT_EQ(2.5, iree_unaligned_load_le(&f64));
}

TEST(AlignmentTest, UnalignedLeavesUseLittleEndianBytes) {
  uint8_t u8_storage[] = {0xCD, 0xCD, 0xCD};
  iree_unaligned_store_le_u8(u8_storage + 1, 0x23u);
  EXPECT_EQ(0xCDu, u8_storage[0]);
  EXPECT_EQ(0x23u, u8_storage[1]);
  EXPECT_EQ(0xCDu, u8_storage[2]);
  EXPECT_EQ(0x23u, iree_unaligned_load_le_u8(u8_storage + 1));

  uint8_t u16_storage[] = {0xCD, 0xCD, 0xCD, 0xCD};
  iree_unaligned_store_le_u16(u16_storage + 1, 0x2345u);
  EXPECT_EQ(0xCDu, u16_storage[0]);
  EXPECT_EQ(0x45u, u16_storage[1]);
  EXPECT_EQ(0x23u, u16_storage[2]);
  EXPECT_EQ(0xCDu, u16_storage[3]);
  EXPECT_EQ(0x2345u, iree_unaligned_load_le_u16(u16_storage + 1));

  uint8_t u32_storage[] = {0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD};
  iree_unaligned_store_le_u32(u32_storage + 1, UINT32_C(0x23456789));
  EXPECT_EQ(0xCDu, u32_storage[0]);
  EXPECT_EQ(0x89u, u32_storage[1]);
  EXPECT_EQ(0x67u, u32_storage[2]);
  EXPECT_EQ(0x45u, u32_storage[3]);
  EXPECT_EQ(0x23u, u32_storage[4]);
  EXPECT_EQ(0xCDu, u32_storage[5]);
  EXPECT_EQ(UINT32_C(0x23456789), iree_unaligned_load_le_u32(u32_storage + 1));

  uint8_t u64_storage[] = {0xCD, 0xCD, 0xCD, 0xCD, 0xCD,
                           0xCD, 0xCD, 0xCD, 0xCD, 0xCD};
  iree_unaligned_store_le_u64(u64_storage + 1, UINT64_C(0x23456789ABCDEF01));
  EXPECT_EQ(0xCDu, u64_storage[0]);
  EXPECT_EQ(0x01u, u64_storage[1]);
  EXPECT_EQ(0xEFu, u64_storage[2]);
  EXPECT_EQ(0xCDu, u64_storage[3]);
  EXPECT_EQ(0xABu, u64_storage[4]);
  EXPECT_EQ(0x89u, u64_storage[5]);
  EXPECT_EQ(0x67u, u64_storage[6]);
  EXPECT_EQ(0x45u, u64_storage[7]);
  EXPECT_EQ(0x23u, u64_storage[8]);
  EXPECT_EQ(0xCDu, u64_storage[9]);
  EXPECT_EQ(UINT64_C(0x23456789ABCDEF01),
            iree_unaligned_load_le_u64(u64_storage + 1));

  uint8_t f32_storage[] = {0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD};
  iree_unaligned_store_le_f32(f32_storage + 1, 1.25f);
  EXPECT_EQ(0xCDu, f32_storage[0]);
  EXPECT_EQ(0x00u, f32_storage[1]);
  EXPECT_EQ(0x00u, f32_storage[2]);
  EXPECT_EQ(0xA0u, f32_storage[3]);
  EXPECT_EQ(0x3Fu, f32_storage[4]);
  EXPECT_EQ(0xCDu, f32_storage[5]);
  EXPECT_EQ(1.25f, iree_unaligned_load_le_f32(f32_storage + 1));

  uint8_t f64_storage[] = {0xCD, 0xCD, 0xCD, 0xCD, 0xCD,
                           0xCD, 0xCD, 0xCD, 0xCD, 0xCD};
  iree_unaligned_store_le_f64(f64_storage + 1, 2.5);
  EXPECT_EQ(0xCDu, f64_storage[0]);
  EXPECT_EQ(0x00u, f64_storage[1]);
  EXPECT_EQ(0x00u, f64_storage[2]);
  EXPECT_EQ(0x00u, f64_storage[3]);
  EXPECT_EQ(0x00u, f64_storage[4]);
  EXPECT_EQ(0x00u, f64_storage[5]);
  EXPECT_EQ(0x00u, f64_storage[6]);
  EXPECT_EQ(0x04u, f64_storage[7]);
  EXPECT_EQ(0x40u, f64_storage[8]);
  EXPECT_EQ(0xCDu, f64_storage[9]);
  EXPECT_EQ(2.5, iree_unaligned_load_le_f64(f64_storage + 1));
}

}  // namespace
