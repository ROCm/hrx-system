// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/vector_carrier.h"

#include "iree/testing/gtest.h"
#include "loom/ir/types.h"

namespace {

TEST(VectorCarrierTest, ShapedAccumulatorsRetainEveryLogicalExtent) {
  const loom_scalar_type_t element_types[] = {
      LOOM_SCALAR_TYPE_F32,   LOOM_SCALAR_TYPE_I32,    LOOM_SCALAR_TYPE_I64,
      LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_OFFSET,
  };
  for (loom_scalar_type_t element_type : element_types) {
    SCOPED_TRACE(element_type);
    const uint32_t element_bits =
        loom_aie2p_scalar_type_physical_bit_count(element_type);
    for (uint32_t count = 1024 / element_bits + 1; count <= 2048 / element_bits;
         ++count) {
      SCOPED_TRACE(count);
      // Factorizations include singleton axes and partially occupied final
      // packets. Rank changes never change the ordered accumulator payload.
      for (uint32_t rows = 1; rows <= count; ++rows) {
        if (count % rows != 0) {
          continue;
        }
        const loom_type_t type = loom_type_shaped_2d(
            LOOM_TYPE_VECTOR, element_type, loom_dim_pack_static(rows),
            loom_dim_pack_static(count / rows), /*encoding_id=*/0);
        const auto carrier = loom_aie2p_vector_carrier_for_type(type);
        EXPECT_EQ(carrier.kind, LOOM_AIE2P_VECTOR_CARRIER_ACCUMULATOR);
        EXPECT_EQ(carrier.unit_count, 4u);
      }
    }
    const loom_type_t oversized = loom_type_shaped_2d(
        LOOM_TYPE_VECTOR, element_type, loom_dim_pack_static(2),
        loom_dim_pack_static(1024 / element_bits + 1), /*encoding_id=*/0);
    EXPECT_EQ(loom_aie2p_vector_carrier_for_type(oversized).kind,
              LOOM_AIE2P_VECTOR_CARRIER_NONE);
  }
}

TEST(VectorCarrierTest, NumericFormatsWithoutWideAccumulatorsStayUnsupported) {
  const loom_scalar_type_t element_types[] = {
      LOOM_SCALAR_TYPE_I8,     LOOM_SCALAR_TYPE_I16, LOOM_SCALAR_TYPE_F16,
      LOOM_SCALAR_TYPE_BF16,   LOOM_SCALAR_TYPE_F64, LOOM_SCALAR_TYPE_F8E4M3,
      LOOM_SCALAR_TYPE_F8E5M2,
  };
  for (loom_scalar_type_t element_type : element_types) {
    SCOPED_TRACE(element_type);
    const uint32_t element_bits =
        loom_aie2p_scalar_type_physical_bit_count(element_type);
    const loom_type_t type = loom_type_shaped_2d(
        LOOM_TYPE_VECTOR, element_type, loom_dim_pack_static(2),
        loom_dim_pack_static(1024 / element_bits), /*encoding_id=*/0);
    EXPECT_EQ(loom_aie2p_vector_carrier_for_type(type).kind,
              LOOM_AIE2P_VECTOR_CARRIER_NONE);
  }
}

}  // namespace
