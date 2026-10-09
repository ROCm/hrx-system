// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/storage_geometry.h"

#include <algorithm>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"

namespace loom {
namespace {

TEST(StorageGeometryTest, ScalarWidthsKeepLogicalElementsSeparateFromBits) {
  const loom_scalar_type_t types[] = {
      LOOM_SCALAR_TYPE_I1,  LOOM_SCALAR_TYPE_F8E4M3, LOOM_SCALAR_TYPE_F8E5M2,
      LOOM_SCALAR_TYPE_I8,  LOOM_SCALAR_TYPE_I16,    LOOM_SCALAR_TYPE_BF16,
      LOOM_SCALAR_TYPE_F16, LOOM_SCALAR_TYPE_I32,    LOOM_SCALAR_TYPE_F32,
      LOOM_SCALAR_TYPE_I64, LOOM_SCALAR_TYPE_F64,
  };
  for (loom_scalar_type_t element_type : types) {
    const loom_type_t type = loom_type_shaped_2d(LOOM_TYPE_VIEW, element_type,
                                                 loom_dim_pack_static(3),
                                                 loom_dim_pack_static(32), 0);
    loom_storage_geometry_t geometry;
    ASSERT_TRUE(loom_storage_geometry_query(nullptr, nullptr, type, &geometry));
    EXPECT_EQ(geometry.element_bit_count,
              loom_scalar_type_bitwidth(element_type));
    EXPECT_EQ(geometry.axes[0].element_stride, 32u);
    EXPECT_EQ(geometry.axes[1].element_stride, 1u);
    loom_storage_geometry_span_t span;
    ASSERT_TRUE(loom_storage_geometry_measure(&geometry, 0, &span));
    EXPECT_EQ(span.element_count, 96u);
    EXPECT_EQ(span.element_span, 96u);
    EXPECT_TRUE(span.dense);
  }
}

TEST(StorageGeometryTest,
     RetainedRecordsHaveIndependentSlotStrideAndFootprint) {
  const loom_storage_geometry_t geometry = {
      LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
      16,
      3,
      {{12, 64}, {16, 768}, {64, 1}},
  };
  loom_storage_geometry_span_t record;
  ASSERT_TRUE(loom_storage_geometry_measure(&geometry, 1, &record));
  EXPECT_EQ(record.element_count * 2, 2048u);
  EXPECT_EQ(record.element_span * 2, 23168u);
  EXPECT_EQ(geometry.axes[0].element_stride * 2, 128u);
  EXPECT_FALSE(record.dense);

  loom_storage_geometry_span_t storage;
  ASSERT_TRUE(loom_storage_geometry_measure(&geometry, 0, &storage));
  EXPECT_EQ(storage.element_count * 2, 24576u);
  EXPECT_EQ(storage.element_span * 2, 24576u);
  // The same envelope size as a dense layout does not imply the same walk.
  EXPECT_FALSE(storage.dense);
}

// Enumerate addresses in logical order rather than reproducing the closed-form
// span calculation. Includes repeated, padded, overlapping and permuted
// layouts.
TEST(StorageGeometryTest, RegionSpansMatchEnumeratedAddresses) {
  for (uint64_t rows : {0, 1, 2, 3}) {
    for (uint64_t columns : {0, 1, 2, 4}) {
      for (uint64_t row_stride : {0, 1, 2, 4, 9}) {
        for (uint64_t column_stride : {0, 1, 3}) {
          const loom_storage_geometry_t geometry = {
              LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
              8,
              2,
              {{rows, row_stride}, {columns, column_stride}},
          };
          for (uint8_t first_axis : {0, 1, 2}) {
            SCOPED_TRACE(::testing::Message()
                         << rows << "x" << columns << " strides=" << row_stride
                         << "," << column_stride << " first=" << +first_axis);
            std::vector<uint64_t> addresses;
            const uint64_t row_count = first_axis > 0 ? 1 : rows;
            const uint64_t column_count = first_axis > 1 ? 1 : columns;
            for (uint64_t row = 0; row < row_count; ++row) {
              for (uint64_t column = 0; column < column_count; ++column) {
                addresses.push_back(row * row_stride + column * column_stride);
              }
            }
            bool dense = true;
            for (size_t i = 0; i < addresses.size(); ++i) {
              dense &= addresses[i] == i;
            }
            loom_storage_geometry_span_t span;
            ASSERT_TRUE(
                loom_storage_geometry_measure(&geometry, first_axis, &span));
            EXPECT_EQ(span.element_count, addresses.size());
            EXPECT_EQ(span.element_span,
                      addresses.empty() ? 0u
                                        : *std::max_element(addresses.begin(),
                                                            addresses.end()) +
                                              1);
            EXPECT_EQ(span.dense, dense);
          }
        }
      }
    }
  }
}

TEST(StorageGeometryTest, UnrepresentableShapeOrSpanCannotWrap) {
  loom_storage_geometry_t geometry = {
      LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
      8,
      2,
      {{UINT64_MAX, 0}, {2, 1}},
  };
  loom_storage_geometry_span_t span;
  EXPECT_FALSE(loom_storage_geometry_measure(&geometry, 0, &span));
  geometry.axes[0] = {2, UINT64_MAX};
  EXPECT_FALSE(loom_storage_geometry_measure(&geometry, 0, &span));
  geometry.axes[1] = {0, 1};
  ASSERT_TRUE(loom_storage_geometry_measure(&geometry, 0, &span));
  EXPECT_EQ(span.element_count, 0u);
  EXPECT_EQ(span.element_span, 0u);
}

class StorageGeometryFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("geometry"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts_, &arena_, 3));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Backing storage shared by the module and fact arena.
  iree_arena_block_pool_t pool_;
  // Scope owning retained fact extensions.
  iree_arena_allocator_t arena_;
  // Type/module API owner; this fixture requires no operation dialects.
  loom_context_t context_;
  // Encoding query scope, with no authored operations.
  loom_module_t* module_ = nullptr;
  // Numeric dimension and layout facts supplied by specialization.
  loom_value_fact_table_t facts_ = {};
};

TEST_F(StorageGeometryFactsTest,
       SpecializationUsesRetainedDimensionAndLayoutFacts) {
  loom_type_t type = loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_BF16,
                                         loom_dim_pack_dynamic(1),
                                         loom_dim_pack_static(64), 0);
  loom_storage_geometry_t geometry;
  EXPECT_FALSE(
      loom_storage_geometry_query(&facts_.context, module_, type, &geometry));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&facts_, 1, loom_value_facts_exact_i64(16)));
  ASSERT_TRUE(
      loom_storage_geometry_query(&facts_.context, module_, type, &geometry));
  EXPECT_EQ(geometry.axes[0].extent, 16u);
  EXPECT_EQ(geometry.axes[0].element_stride, 64u);

  const loom_value_facts_t strides[] = {loom_value_facts_exact_i64(768),
                                        loom_value_facts_exact_i64(1)};
  loom_value_fact_encoding_summary_t summary = {
      .role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
      .address_layout = {LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED, 2, strides}};
  loom_value_facts_t layout;
  IREE_ASSERT_OK(loom_value_facts_make_encoding_summary(&facts_.context,
                                                        summary, &layout));
  IREE_ASSERT_OK(loom_value_fact_table_define(&facts_, 2, layout));
  type.encoding_id = 2;
  type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  ASSERT_TRUE(
      loom_storage_geometry_query(&facts_.context, module_, type, &geometry));
  loom_storage_geometry_span_t span;
  ASSERT_TRUE(loom_storage_geometry_measure(&geometry, 0, &span));
  EXPECT_EQ(span.element_count * 2, 2048u);
  EXPECT_EQ(span.element_span * 2, 23168u);
  EXPECT_FALSE(span.dense);

  // The retained geometry owns its axes after the query's fact scope changes.
  IREE_ASSERT_OK(loom_value_fact_table_define(&facts_, 1,
                                              loom_value_facts_make(1, 16, 1)));
  ASSERT_TRUE(loom_storage_geometry_measure(&geometry, 0, &span));
  EXPECT_EQ(span.element_span * 2, 23168u);
  EXPECT_FALSE(
      loom_storage_geometry_query(&facts_.context, module_, type, &geometry));
}

}  // namespace
}  // namespace loom
