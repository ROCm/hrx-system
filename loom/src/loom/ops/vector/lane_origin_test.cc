// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class VectorLaneOriginTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* tables = loom_vector_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_VECTOR,
                                                 tables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("lane_origin"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void Check(loom_scalar_type_t input_element,
             loom_scalar_type_t output_element, std::array<int64_t, 2> offsets,
             bool second_argument, uint8_t conversion_flags,
             bool expected_origin) {
    const auto vector_type = [](loom_scalar_type_t element, uint32_t count) {
      return loom_type_shaped_1d(LOOM_TYPE_VECTOR, element,
                                 loom_dim_pack_static(count), 0);
    };
    const loom_type_t input_type = vector_type(input_element, 128);
    const loom_type_t input_packet_type = vector_type(input_element, 32);
    const loom_type_t output_packet_type = vector_type(output_element, 32);
    const loom_type_t output_type = vector_type(output_element, 64);
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_value_id_t values[2];
    for (auto& value : values) {
      IREE_ASSERT_OK(loom_module_define_value(module_, input_type, &value));
      IREE_ASSERT_OK(
          loom_block_add_arg(module_, loom_module_block(module_), value));
    }
    loom_op_t* operations[5] = {};
    loom_value_id_t packets[2];
    for (uint32_t index = 0; index < 2; ++index) {
      IREE_ASSERT_OK(loom_vector_slice_build(
          &builder, values[second_argument ? index : 0], nullptr, 0,
          &offsets[index], 1, input_packet_type, LOOM_LOCATION_UNKNOWN,
          &operations[index * 2]));
      IREE_ASSERT_OK(loom_vector_extf_build(
          &builder, conversion_flags,
          loom_vector_slice_result(operations[index * 2]), input_packet_type,
          output_packet_type, LOOM_LOCATION_UNKNOWN,
          &operations[index * 2 + 1]));
      packets[index] = loom_vector_extf_result(operations[index * 2 + 1]);
    }
    IREE_ASSERT_OK(loom_vector_concat_build(&builder, 0, packets, 2,
                                            output_type, LOOM_LOCATION_UNKNOWN,
                                            &operations[4]));
    const loom_value_id_t result = loom_vector_concat_result(operations[4]);
    loom_value_fact_table_t facts = {};
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &module_->arena,
                                                    module_->values.count));
    for (const auto* operation : operations) {
      IREE_ASSERT_OK(
          loom_value_fact_table_compute_op(&facts, module_, operation));
    }
    if (conversion_flags == 0) {
      for (uint32_t index = 0; index < 2; ++index) {
        loom_value_fact_exact_lane_origin_t packet_origin = {};
        ASSERT_TRUE(loom_value_fact_table_query_exact_lane_origin(
            &facts, module_, packets[index], &packet_origin));
        EXPECT_EQ(packet_origin.source_value_id,
                  loom_vector_slice_result(operations[index * 2]));
        EXPECT_EQ(packet_origin.source_lane_offset, 0);
        EXPECT_EQ(packet_origin.source_lane_stride, 1);
      }
    }
    loom_value_fact_exact_lane_origin_t origin = {};
    ASSERT_EQ(loom_value_fact_table_query_exact_lane_origin(&facts, module_,
                                                            result, &origin),
              expected_origin);
    if (expected_origin) {
      EXPECT_EQ(origin.source_value_id, values[0]);
      EXPECT_EQ(origin.source_lane_offset, offsets[0]);
      EXPECT_EQ(origin.source_lane_stride, 1);
    }
  }

  // Arena blocks shared by the test module and its fact table.
  iree_arena_block_pool_t block_pool_ = {};
  // Vector operation registry.
  loom_context_t context_ = {};
  // Module containing the vector operations whose facts are queried.
  loom_module_t* module_ = nullptr;
};

TEST_F(VectorLaneOriginTest, BFloat16PacketsReassembleExactOrigin) {
  Check(LOOM_SCALAR_TYPE_BF16, LOOM_SCALAR_TYPE_F32, {0, 32}, false, 0, true);
}
TEST_F(VectorLaneOriginTest, Float16PacketsReassembleExactOrigin) {
  Check(LOOM_SCALAR_TYPE_F16, LOOM_SCALAR_TYPE_F32, {0, 32}, false, 0, true);
}
TEST_F(VectorLaneOriginTest, Float32PacketsReassembleExactOrigin) {
  Check(LOOM_SCALAR_TYPE_F32, LOOM_SCALAR_TYPE_F64, {0, 32}, false, 0, true);
}
TEST_F(VectorLaneOriginTest, ContiguousSubrangePreservesOffset) {
  Check(LOOM_SCALAR_TYPE_BF16, LOOM_SCALAR_TYPE_F32, {16, 48}, false, 0, true);
}
TEST_F(VectorLaneOriginTest, ReorderedPacketsHaveNoAffineOrigin) {
  Check(LOOM_SCALAR_TYPE_BF16, LOOM_SCALAR_TYPE_F32, {32, 0}, false, 0, false);
}
TEST_F(VectorLaneOriginTest, RepeatedPacketsHaveNoAffineOrigin) {
  Check(LOOM_SCALAR_TYPE_BF16, LOOM_SCALAR_TYPE_F32, {0, 0}, false, 0, false);
}
TEST_F(VectorLaneOriginTest, GapBetweenPacketsHasNoAffineOrigin) {
  Check(LOOM_SCALAR_TYPE_BF16, LOOM_SCALAR_TYPE_F32, {0, 64}, false, 0, false);
}
TEST_F(VectorLaneOriginTest, DistinctSourcesHaveNoSharedOrigin) {
  Check(LOOM_SCALAR_TYPE_BF16, LOOM_SCALAR_TYPE_F32, {0, 32}, true, 0, false);
}
TEST_F(VectorLaneOriginTest, FlushingPotentialSubnormalsIsNotExact) {
  Check(LOOM_SCALAR_TYPE_BF16, LOOM_SCALAR_TYPE_F32, {0, 32}, false,
        LOOM_VECTOR_FLOATCONVERSIONFLAGS_DAZ, false);
}

}  // namespace
}  // namespace loom
