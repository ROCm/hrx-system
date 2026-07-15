// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/alignment.h"

#include "iree/testing/gtest.h"

namespace {

TEST(AlignmentTest, GenericLoadLittleEndian) {
  int8_t i8 = 0x12;
  uint8_t u8 = 0x23;
  int16_t i16 = 0x1234;
  uint16_t u16 = 0x2345;
  int32_t i32 = 0x12345678;
  uint32_t u32 = 0x23456789u;
  int64_t i64 = INT64_C(0x123456789ABCDEF);
  uint64_t u64 = UINT64_C(0x23456789ABCDEF01);
  float f32 = 1.25f;
  double f64 = 2.5;

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

TEST(AlignmentTest, GenericStoreLittleEndian) {
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

  EXPECT_EQ(0x12, i8);
  EXPECT_EQ(0x23u, u8);
  EXPECT_EQ(0x1234, i16);
  EXPECT_EQ(0x2345u, u16);
  EXPECT_EQ(0x12345678, i32);
  EXPECT_EQ(0x23456789u, u32);
  EXPECT_EQ(INT64_C(0x123456789ABCDEF), i64);
  EXPECT_EQ(UINT64_C(0x23456789ABCDEF01), u64);
  EXPECT_EQ(1.25f, f32);
  EXPECT_EQ(2.5, f64);
}

TEST(AlignmentTest, StoreLittleEndianF64) {
  double value = 0.0;
  iree_unaligned_store_le_f64(&value, 2.5);
  EXPECT_EQ(2.5, value);
}

}  // namespace
