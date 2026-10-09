// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/lower/vector_carrier.h"

#include <cstdint>

#include "iree/testing/gtest.h"

namespace {

loom_type_t Vector1D(loom_scalar_type_t element_type, int64_t lane_count) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, element_type,
                             loom_dim_pack_static(lane_count),
                             /*encoding_id=*/0);
}

loom_type_t Vector2D(loom_scalar_type_t element_type, int64_t row_count,
                     int64_t column_count) {
  return loom_type_shaped_2d(LOOM_TYPE_VECTOR, element_type,
                             loom_dim_pack_static(row_count),
                             loom_dim_pack_static(column_count),
                             /*encoding_id=*/0);
}

TEST(VectorCarrierTest, MapsNumericAndAddressScalarFamily) {
  struct Case {
    loom_scalar_type_t element_type;
    uint8_t element_bit_count;
    uint16_t lanes_per_packet;
  };
  static constexpr Case kCases[] = {
      {LOOM_SCALAR_TYPE_INDEX, 32, 4},  {LOOM_SCALAR_TYPE_OFFSET, 32, 4},
      {LOOM_SCALAR_TYPE_I8, 8, 16},     {LOOM_SCALAR_TYPE_I16, 16, 8},
      {LOOM_SCALAR_TYPE_I32, 32, 4},    {LOOM_SCALAR_TYPE_I64, 64, 2},
      {LOOM_SCALAR_TYPE_F8E4M3, 8, 16}, {LOOM_SCALAR_TYPE_F8E5M2, 8, 16},
      {LOOM_SCALAR_TYPE_F16, 16, 8},    {LOOM_SCALAR_TYPE_BF16, 16, 8},
      {LOOM_SCALAR_TYPE_F32, 32, 4},    {LOOM_SCALAR_TYPE_F64, 64, 2},
  };
  for (const Case& test_case : kCases) {
    SCOPED_TRACE(loom_scalar_type_name(test_case.element_type));
    for (uint16_t packet_count : {1u, 2u, 3u, 4095u}) {
      const uint16_t lane_count = test_case.lanes_per_packet * packet_count;
      const loom_wasm_vector_carrier_t rank_one =
          loom_wasm_vector_carrier_for_type(
              Vector1D(test_case.element_type, lane_count));
      EXPECT_EQ(rank_one.element_bit_count, test_case.element_bit_count);
      EXPECT_EQ(rank_one.payload_byte_count, packet_count * 16u);
      EXPECT_EQ(rank_one.packet_count, packet_count);

      const loom_wasm_vector_carrier_t rank_two =
          loom_wasm_vector_carrier_for_type(
              Vector2D(test_case.element_type, packet_count,
                       test_case.lanes_per_packet));
      EXPECT_EQ(rank_two.element_bit_count, test_case.element_bit_count);
      EXPECT_EQ(rank_two.payload_byte_count, packet_count * 16u);
      EXPECT_EQ(rank_two.packet_count, packet_count);
    }
  }
}

TEST(VectorCarrierTest, RoundsPartialPayloadsToPackets) {
  const loom_wasm_vector_carrier_t bytes =
      loom_wasm_vector_carrier_for_type(Vector1D(LOOM_SCALAR_TYPE_I8, 17));
  EXPECT_EQ(bytes.payload_byte_count, 17);
  EXPECT_EQ(bytes.packet_count, 2);

  const loom_wasm_vector_carrier_t doubles =
      loom_wasm_vector_carrier_for_type(Vector2D(LOOM_SCALAR_TYPE_F64, 3, 3));
  EXPECT_EQ(doubles.payload_byte_count, 72);
  EXPECT_EQ(doubles.packet_count, 5);

  const loom_wasm_vector_carrier_t maximum =
      loom_wasm_vector_carrier_for_type(Vector1D(LOOM_SCALAR_TYPE_I8, 65535));
  EXPECT_EQ(maximum.payload_byte_count, 65535);
  EXPECT_EQ(maximum.packet_count, 4096);
}

TEST(VectorCarrierTest, RejectsDynamicEmptyAndOverBoundShapes) {
  const loom_type_t dynamic = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8,
      loom_dim_pack_dynamic(/*value_id=*/1), /*encoding_id=*/0);
  EXPECT_EQ(loom_wasm_vector_carrier_for_type(dynamic).packet_count, 0);
  EXPECT_EQ(loom_wasm_vector_carrier_for_type(Vector1D(LOOM_SCALAR_TYPE_I8, 0))
                .packet_count,
            0);
  EXPECT_EQ(
      loom_wasm_vector_carrier_for_type(Vector1D(LOOM_SCALAR_TYPE_I8, 65536))
          .packet_count,
      0);
}

TEST(VectorCarrierTest, KeepsCallableAbiAtExactV128) {
  EXPECT_TRUE(
      loom_wasm_vector_type_is_callable(Vector1D(LOOM_SCALAR_TYPE_I8, 16)));
  EXPECT_TRUE(
      loom_wasm_vector_type_is_callable(Vector1D(LOOM_SCALAR_TYPE_I1, 4)));
  EXPECT_TRUE(
      loom_wasm_vector_type_is_callable(Vector1D(LOOM_SCALAR_TYPE_I1, 2)));
  EXPECT_TRUE(
      loom_wasm_vector_type_is_callable(Vector1D(LOOM_SCALAR_TYPE_I1, 8)));
  EXPECT_TRUE(
      loom_wasm_vector_type_is_callable(Vector1D(LOOM_SCALAR_TYPE_I1, 16)));
  EXPECT_TRUE(
      loom_wasm_vector_type_is_callable(Vector1D(LOOM_SCALAR_TYPE_F64, 2)));

  EXPECT_FALSE(
      loom_wasm_vector_type_is_callable(Vector1D(LOOM_SCALAR_TYPE_INDEX, 4)));
  EXPECT_FALSE(
      loom_wasm_vector_type_is_callable(Vector1D(LOOM_SCALAR_TYPE_I8, 8)));
  EXPECT_FALSE(
      loom_wasm_vector_type_is_callable(Vector1D(LOOM_SCALAR_TYPE_I1, 3)));
  EXPECT_FALSE(
      loom_wasm_vector_type_is_callable(Vector2D(LOOM_SCALAR_TYPE_I8, 4, 4)));
}

}  // namespace
