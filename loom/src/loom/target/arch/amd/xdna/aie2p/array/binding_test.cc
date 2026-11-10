// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/binding.h"

#include "iree/testing/gtest.h"

namespace {

class Aie2pArrayBindingTest
    : public ::testing::TestWithParam<loom_xdna_tile_kind_t> {
 protected:
  bool Resolve(const loom_storage_geometry_t& geometry, uint8_t record_rank,
               uint32_t record_bytes, uint32_t record_count,
               loom_aie2p_array_binding_plan_t* plan,
               iree_string_view_t* reason) {
    const auto* family = loom_xdna_npu2_array_family();
    return loom_aie2p_array_resolve_binding_transfer(
        &geometry, record_rank, family, /*binding_view_byte_offset=*/128,
        /*partitioned=*/false, /*partition_lane=*/0, record_bytes, record_count,
        &loom_xdna_array_tile_kind_facts(family, GetParam())->dma, plan,
        reason);
  }
};

TEST_P(Aie2pArrayBindingTest, SingletonAxesPreserveEveryTransferredAddress) {
  // The same three-dimensional transfer retains its order and native packing
  // with singleton axes before, between, and after its varying coordinates.
  for (uint16_t bits : {1, 2, 4, 8, 16, 32}) {
    const uint64_t elements_per_word = 32 / bits;
    const uint32_t word_bytes = 4;
    for (uint64_t singleton_stride : {UINT64_C(0), UINT64_C(1), UINT64_MAX}) {
      SCOPED_TRACE(::testing::Message()
                   << "bits=" << bits << " stride=" << singleton_stride);
      const loom_storage_geometry_t geometry = {
          .layout_kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
          .element_bit_count = bits,
          .rank = 7,
          .axes = {{1, singleton_stride},
                   {2, 32 * elements_per_word},
                   {1, singleton_stride},
                   {3, 8 * elements_per_word},
                   {1, singleton_stride},
                   {2 * elements_per_word, 1},
                   {1, singleton_stride}},
      };
      loom_aie2p_array_binding_plan_t plan = {};
      iree_string_view_t reason;
      ASSERT_TRUE(
          Resolve(geometry, geometry.rank, 12 * word_bytes, 1, &plan, &reason));
      EXPECT_EQ(plan.binding_byte_offset, 128u);
      EXPECT_EQ(plan.binding_span_byte_length, 50 * word_bytes);
      EXPECT_EQ(plan.transfer_byte_length, 12 * word_bytes);
      EXPECT_EQ(plan.task_repeat_count, 1);
      ASSERT_EQ(plan.dma_dimension_count, 3);
      EXPECT_EQ(plan.dma_dimensions[0].step_size, word_bytes / 4);
      EXPECT_EQ(plan.dma_dimensions[0].wrap, 2);
      EXPECT_EQ(plan.dma_dimensions[1].step_size, 8 * word_bytes / 4);
      EXPECT_EQ(plan.dma_dimensions[1].wrap, 3);
      EXPECT_EQ(plan.dma_dimensions[2].step_size, 32 * word_bytes / 4);
      EXPECT_EQ(plan.dma_dimensions[2].wrap, 0);

      // Expand the hardware address dimensions independently and compare every
      // transferred native unit with the logical layout's element coordinates.
      for (uint32_t outer = 0; outer < 2; ++outer) {
        for (uint32_t middle = 0; middle < 3; ++middle) {
          for (uint32_t inner = 0; inner < 2; ++inner) {
            const uint64_t address =
                4 * (outer * plan.dma_dimensions[2].step_size +
                     middle * plan.dma_dimensions[1].step_size +
                     inner * plan.dma_dimensions[0].step_size);
            EXPECT_EQ(address, (32 * outer + 8 * middle + inner) * word_bytes);
          }
        }
      }
    }
  }
}

TEST_P(Aie2pArrayBindingTest, SingleCoordinateHasNoAddressDimensions) {
  const loom_storage_geometry_t geometry = {
      .layout_kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
      .element_bit_count = 32,
      .rank = 4,
      .axes = {{1, 0}, {1, UINT64_MAX}, {1, 1}, {1, 0}},
  };
  loom_aie2p_array_binding_plan_t plan = {};
  iree_string_view_t reason;
  ASSERT_TRUE(Resolve(geometry, geometry.rank, 4, 1, &plan, &reason));
  EXPECT_EQ(plan.binding_span_byte_length, 4u);
  EXPECT_EQ(plan.transfer_byte_length, 4u);
  EXPECT_EQ(plan.task_repeat_count, 1);
  EXPECT_EQ(plan.dma_dimension_count, 0);
}

TEST_P(Aie2pArrayBindingTest, SingletonSequenceAxesPreserveTaskRepetition) {
  const loom_storage_geometry_t geometry = {
      .layout_kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
      .element_bit_count = 32,
      .rank = 7,
      .axes = {{1, UINT64_MAX},
               {5, 0},
               {1, 0},
               {2, 8},
               {1, UINT64_MAX},
               {4, 1},
               {1, 0}},
  };
  loom_aie2p_array_binding_plan_t plan = {};
  iree_string_view_t reason;
  ASSERT_TRUE(Resolve(geometry, /*record_rank=*/4, 32, 5, &plan, &reason));
  EXPECT_EQ(plan.binding_span_byte_length, 48u);
  EXPECT_EQ(plan.transfer_byte_length, 32u);
  EXPECT_EQ(plan.task_repeat_count, 5);
  ASSERT_EQ(plan.dma_dimension_count, 2);
  EXPECT_EQ(plan.dma_dimensions[0].step_size, 1);
  EXPECT_EQ(plan.dma_dimensions[0].wrap, 4);
  EXPECT_EQ(plan.dma_dimensions[1].step_size, 8);
  EXPECT_EQ(plan.dma_dimensions[1].wrap, 0);
}

TEST_P(Aie2pArrayBindingTest, VaryingAxesStillRequireAddressDimensions) {
  const auto* dma = &loom_xdna_array_tile_kind_facts(
                         loom_xdna_npu2_array_family(), GetParam())
                         ->dma;
  loom_storage_geometry_t geometry = {
      .layout_kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
      .element_bit_count = 32,
      .rank = static_cast<uint8_t>(dma->address_dimension_count + 1),
  };
  for (uint8_t axis = 0; axis < geometry.rank; ++axis) {
    geometry.axes[axis] = {2, UINT64_C(1) << (2 * (geometry.rank - axis - 1))};
  }
  loom_aie2p_array_binding_plan_t plan = {};
  iree_string_view_t reason;
  EXPECT_FALSE(
      Resolve(geometry, geometry.rank, 4u << geometry.rank, 1, &plan, &reason));
  EXPECT_TRUE(iree_string_view_equal(
      reason, IREE_SV("the strided transfer requires more address dimensions "
                      "than DMA provides")));
}

TEST_P(Aie2pArrayBindingTest,
       RepeatedCoordinatesInsideARecordRemainUnsupported) {
  const loom_storage_geometry_t geometry = {
      .layout_kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
      .element_bit_count = 32,
      .rank = 3,
      .axes = {{2, 8}, {2, 0}, {2, 1}},
  };
  loom_aie2p_array_binding_plan_t plan = {};
  iree_string_view_t reason;
  EXPECT_FALSE(Resolve(geometry, geometry.rank, 32, 1, &plan, &reason));
  EXPECT_TRUE(iree_string_view_equal(
      reason, IREE_SV("only leading source sequence axes may use zero stride "
                      "for task repetition")));
}

INSTANTIATE_TEST_SUITE_P(AllDmaModules, Aie2pArrayBindingTest,
                         ::testing::Values(LOOM_XDNA_TILE_KIND_COMPUTE,
                                           LOOM_XDNA_TILE_KIND_MEMORY,
                                           LOOM_XDNA_TILE_KIND_SHIM_NOC));

}  // namespace
